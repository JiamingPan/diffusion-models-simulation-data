#!/usr/bin/env python
"""Prepare/train/sample three isolated L16 N=256 ablations. Never submits jobs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
NAME = "dit_l16_a40_init_patch_v1"
MATRIX_SHA = "8404ae7b4f607923251d430da2b78afe07871846269927c0a9e2b6e3d8c0e333"
ARMS = [("p8_zero", 8, "zero"), ("p4_native", 4, "native"), ("p4_zero", 4, "zero")]
TARGET = 300_000
# Operational-only difference: retain six complete safety/final checkpoints,
# instead of generating ~60 and deleting most. No checkpoint pruning.
CHECKPOINT_EPOCHS = 1563  # 50,016 nominal optimizer updates


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def publish_json(path, value):
    """One exclusive owner, valid JSON published only after its work completes."""
    from trace_trajectories import atomic_output
    with atomic_output(Path(path).parent / (Path(path).stem + "_record")) as pending:
        (pending / Path(path).name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def matrix_inputs(project):
    root = Path(project) / "results/dit_l16_sampler_matrix_v1"
    path = root / "plan/plan.json"
    if file_hash(path) != MATRIX_SHA:
        raise ValueError("original reviewed sampler matrix plan changed")
    plan = json.loads(path.read_text())
    noise_path = root / "plan/initial_noise.npz"
    if file_hash(noise_path) != plan["noise_file_sha256"]:
        raise ValueError("original paired noise artifact changed")
    with np.load(noise_path, allow_pickle=False) as data:
        noise = data["initial_noise"].copy()
    if (noise.shape != (128, 1, 128, 128) or noise.dtype != np.float32
            or not np.isfinite(noise).all() or array_hash(noise) != plan["noise_batch_sha256"]):
        raise ValueError("original paired noise fields do not match reviewed plan")
    baseline = next(row for row in plan["models"] if row["name"] == "dit_l16_fresh300k")
    if file_hash(baseline["config"]) != baseline["config_sha256"]:
        raise ValueError("existing fresh-300k baseline YAML changed")
    return root, plan, baseline, noise


def validate_baseline(config):
    kwargs = config["model"]["kwargs"]
    expected = {"num_layers": 16, "sample_size": 128, "patch_size": 8,
                "num_attention_heads": 12, "attention_head_dim": 64,
                "num_embeds_ada_norm": 1, "in_channels": 1, "out_channels": 1}
    if config["model"]["class"] != "DiTTransformer2DModel" or any(kwargs.get(k) != v for k, v in expected.items()):
        raise ValueError("baseline is not the expected L16/patch8 recipe")
    train = config["train"]
    if any(train.get(k) != v for k, v in {"batch_size": 2, "gradient_accumulation_steps": 4,
            "num_epochs": 9375, "mixed_precision": "fp16", "cfg_dropout": 0.,
            "conditioning": "discrete"}.items()):
        raise ValueError("baseline training budget/conditioning differs")
    if (config.get("augmentations") or config["data"].get("constant_label") != 0
            or config["data"].get("label_path") is not None):
        raise ValueError("expected no augmentation and the null-class contract")


def arm_configs(baseline, checkpoint_root):
    validate_baseline(baseline)
    rows = []
    for name, patch, initialization in ARMS:
        config = deepcopy(baseline)
        target = Path(checkpoint_root) / f"{name}_seed123_checkpoints"
        config["io"]["output_dir"] = str(target)
        config["model"]["kwargs"]["patch_size"] = patch
        config["train"]["checkpoint_every_n_epochs"] = CHECKPOINT_EPOCHS
        rows.append({"name": name, "patch_size": patch, "initialization": initialization,
            "seed": 123, "target_updates": TARGET, "dataset_size": 256,
            "checkpoint_dir": str(target), "expected_checkpoint": str(target / "checkpoint-epoch-9374"),
            "config_data": config})
    return rows


def prepare(project, output):
    from trace_trajectories import atomic_output
    _, matrix, baseline, noise = matrix_inputs(project)
    if "/scratch/huterer_root/huterer0/jiamingp/" not in str(output):
        raise ValueError("use the dedicated scratch experiment root, not /home or an existing run")
    base = yaml.safe_load(Path(baseline["config"]).read_text())
    rows = arm_configs(base, Path(output) / "checkpoints")
    with atomic_output(Path(output) / "plan") as pending:
        for row in rows:
            config = row.pop("config_data")
            relative = f"{row['name']}.yaml"
            path = pending / relative
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            row.update(config=str(Path(output) / "plan" / relative), config_sha256=file_hash(path))
        np.savez_compressed(pending / "initial_noise.npz", initial_noise=noise)
        plan = {"status": "prepared", "protocol": NAME, "arms": rows,
            "code_revision": os.environ.get("EXPECTED_COMMIT"),
            "baseline": baseline, "matrix_plan_sha256": MATRIX_SHA,
            "reference_sha256": matrix["reference_sha256"], "noise_batch_sha256": array_hash(noise),
            "noise_file_sha256": file_hash(pending / "initial_noise.npz"),
            "checkpoint_cadence_updates": CHECKPOINT_EPOCHS * 32,
            "checkpoint_retention": "all six per arm; no deletion or automatic resume",
            "quality_rule": "cos<0.8 is low similarity, NOT verified invalidity; inspect conditional spectra/boundaries and maps"}
        (pending / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"ABLATION PLAN PREPARED: 3 fresh N=256 arms; target=300000 updates, A40 only")
    print(f"ABLATION_PLAN_SHA256={file_hash(Path(output) / 'plan/plan.json')}")


def load_plan(output):
    path = Path(output) / "plan/plan.json"
    expected = os.environ.get("ABLATION_PLAN_SHA256")
    if not expected or file_hash(path) != expected:
        raise ValueError("set ABLATION_PLAN_SHA256 to the reviewed plan hash")
    plan = json.loads(path.read_text())
    if (plan.get("status") != "prepared" or plan.get("protocol") != NAME
            or plan.get("code_revision") != os.environ.get("EXPECTED_COMMIT")
            or [(r["name"], r["patch_size"], r["initialization"]) for r in plan["arms"]] != ARMS):
        raise ValueError("ablation plan/code/arm coverage changed")
    source = yaml.safe_load(Path(plan["baseline"]["config"]).read_text())
    if file_hash(plan["baseline"]["config"]) != plan["baseline"]["config_sha256"]:
        raise ValueError("baseline config changed")
    expected_rows = arm_configs(source, Path(output) / "checkpoints")
    for row, wanted in zip(plan["arms"], expected_rows):
        if any(row.get(k) != v for k, v in wanted.items() if k != "config_data"):
            raise ValueError("frozen arm identity/budget changed")
        if (Path(row["config"]).resolve() != (Path(output) / "plan" / f"{row['name']}.yaml").resolve()
                or file_hash(row["config"]) != row["config_sha256"]
                or yaml.safe_load(Path(row["config"]).read_text()) != wanted["config_data"]):
            raise ValueError("frozen arm config changed")
    noise_path = Path(output) / "plan/initial_noise.npz"
    if file_hash(noise_path) != plan["noise_file_sha256"]:
        raise ValueError("frozen paired noise changed")
    with np.load(noise_path, allow_pickle=False) as data:
        noise = data["initial_noise"].copy()
    if noise.shape != (128, 1, 128, 128) or array_hash(noise) != plan["noise_batch_sha256"]:
        raise ValueError("paired noise shape/hash changed")
    return plan, noise


def runtime():
    # Import compatibility from the ORIGINAL pinned runtime before new adapter modules.
    from simdiff_eval.torch_compat import install_torch_backend_compat
    install_torch_backend_compat(entry_point=__name__)
    import sample_cosmodiff as sc
    sc._reject_known_bad_runtime()
    sc._install_sklearn_roc_curve_stub()
    import torch
    import diffusers
    if diffusers.__version__ != "0.35.1":
        raise RuntimeError("this init ablation is verified against diffusers 0.35.1; reconcile rather than silently changing it")
    return torch, diffusers


def single_a40(torch):
    for key in ("WORLD_SIZE", "SLURM_NTASKS"):
        if int(os.environ.get(key, "1")) != 1:
            raise RuntimeError("single-process training only; DDP is not approved")
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        if int(os.environ.get(key, "0")) != 0:
            raise RuntimeError("a nonzero rank must not train/write this experiment")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA GPU required")
    name = torch.cuda.get_device_name(0)
    if "A40" not in name:
        raise RuntimeError(f"approved A40 experiment, found {name}")
    print(f"[ablation] GPU={name} VRAM_GiB={torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}", flush=True)


def cpu_preflight(output):
    """Exercise the actual installed model/optimizer API, without a GPU or data scan."""
    plan, _ = load_plan(output)
    torch, _ = runtime()
    from cosmodiff import optim, utils
    from dit_zero_init import zero_modulation_and_output
    from accelerate import Accelerator
    pin = Path(os.environ["COSMODIFF_PIN_ROOT"]).resolve()
    for module in (optim, utils):
        if pin not in Path(inspect.getfile(module)).resolve().parents:
            raise RuntimeError("preflight imports escaped the immutable cosmodiff pin")
    baseline = yaml.safe_load(Path(plan["arms"][0]["config"]).read_text())
    # bind(), not bind_partial(): all required native trainer arguments must work.
    inspect.signature(optim.train).bind(object(), object(), optimizer=object(),
        noise_scheduler=object(), lr_scheduler=object(), output_dir="/not-written",
        **baseline["train"])
    for name, patch, initialization in ARMS:
        tiny = deepcopy(baseline)
        tiny["global"]["device"] = "cpu"
        tiny["model"]["kwargs"].update(sample_size=16, patch_size=patch,
            num_attention_heads=2, attention_head_dim=8, norm_num_groups=4)
        # Preserve caller RNG. These models exist only for API/gradient checks.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            built = utils.parse_config_model(tiny)
            model, optimizer = built["model"], built["optimizer"]
            if initialization == "zero":
                zero_modulation_and_output(model, fresh=True)
            scheduler = built["noise_scheduler"]
            x = torch.randn(2, 1, 16, 16)
            noise = torch.randn_like(x)
            timesteps = torch.tensor([249, 399], dtype=torch.long)
            noisy = scheduler.add_noise(x, noise, timesteps)
            target = scheduler.get_velocity(x, noise, timesteps)
            prediction = model(noisy, timestep=timesteps,
                class_labels=torch.zeros(2, dtype=torch.long), return_dict=False)[0]
            if initialization == "zero" and bool(torch.any(prediction != 0).item()):
                raise RuntimeError("zero-init native model has nonzero initial output")
            abar = scheduler.alphas_cumprod[timesteps]
            snr = abar / (1-abar)
            loss = ((snr.clamp(max=tiny["train"]["min_snr_gamma"]) / (snr+1))
                    * ((prediction-target)**2).mean(dim=(1, 2, 3))).mean()
            loss.backward()
            grad = model.proj_out_2.weight.grad
            if grad is None or not bool(torch.isfinite(grad).all().item()) or not bool(torch.any(grad != 0).item()):
                raise RuntimeError("native first-step head gradients are absent/nonfinite")
            optimizer.step()
            if not bool(torch.isfinite(model.proj_out_2.weight).all().item()):
                raise RuntimeError("native first CPU update is nonfinite")
        print(f"NATIVE CPU API/GRADIENT SMOKE PASSED: {name}", flush=True)
    if not hasattr(torch.optim.AdamW, "register_step_post_hook") or not hasattr(Accelerator, "backward"):
        raise RuntimeError("native training first-step observation API is unavailable")
    print("ABLATION PREFLIGHT PASSED; TINY CPU MODEL ONLY; NO DATA LOAD, GPU TRAINING OR JOB SUBMISSION")


def reference_for(row, plan):
    from evaluate_late_start import load_reference
    reference = load_reference(Path(row["config"]), None)
    if reference.shape != (256, 1, 128, 128) or array_hash(reference) != plan["reference_sha256"]:
        raise ValueError("ablation training reference differs from the frozen baseline")
    return reference


def train(output, index):
    plan, _ = load_plan(output)
    if not 0 <= index < 3:
        raise ValueError("arm index must be 0..2")
    row = plan["arms"][index]
    checkpoint_dir = Path(row["checkpoint_dir"])
    # Even a partial prior launch is preserved; do not restart/overwrite implicitly.
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    torch, diffusers = runtime()
    single_a40(torch)
    from cosmodiff import optim, utils
    pin = Path(os.environ["COSMODIFF_PIN_ROOT"]).resolve()
    for module in (optim, utils):
        if pin not in Path(inspect.getfile(module)).resolve().parents:
            raise RuntimeError("cosmodiff did not resolve inside the immutable pin")
    from run_cosmodiff_train_with_dit_resume import install_constant_label_adapter, validate_scientific_checkpoint
    from run_cosmodiff_train_fresh_seeded import seed_everything
    from dit_zero_init import zero_modulation_and_output
    from accelerate import Accelerator
    config = yaml.safe_load(Path(row["config"]).read_text())
    reference = reference_for(row, plan)
    seed_everything(123)
    install_constant_label_adapter(utils)
    dataset = utils.parse_config_data(config)["data"]
    actual = dataset.arrays.detach().cpu().numpy()
    # Torch and NumPy tanh/log may round differently. Compare every element,
    # preserve order, and record both hashes; never substitute the IO reference.
    if (actual.shape != reference.shape or actual.dtype != np.float32
            or not np.allclose(actual, reference, rtol=0, atol=2e-6)):
        raise ValueError("native training tensors disagree with the frozen evaluation subset")
    if dataset.labels.dtype != torch.long or tuple(dataset.labels.shape) != (256,):
        raise ValueError("null class labels must be one long integer per image")
    built = utils.parse_config_model(config)
    model, optimizer = built["model"], built["optimizer"]
    init = (zero_modulation_and_output(model, fresh=True) if row["initialization"] == "zero"
            else {"kind": "native_diffusers", "zeroed_tensors": 0})
    provenance = {"status": "started", "arm": row, "initialization": init,
        "training_tensor_sha256": array_hash(actual), "evaluation_reference_sha256": array_hash(reference),
        "reference_max_abs_delta": float(np.max(np.abs(actual - reference))),
        "torch": str(torch.__version__), "diffusers": diffusers.__version__,
        "gpu": torch.cuda.get_device_name(0), "parameters": sum(p.numel() for p in model.parameters()),
        "initial_projection_sha256": array_hash(model.proj_out_2.weight.detach().cpu().numpy()),
        "train_source_sha256": file_hash(inspect.getfile(optim)), "plan_sha256": file_hash(Path(output) / "plan/plan.json")}
    (checkpoint_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    publish_json(checkpoint_dir / "initialization.json", provenance)
    print(f"[ablation] init={init['kind']} zeroed_tensors={init['zeroed_tensors']} parameters={provenance['parameters']}", flush=True)
    print(f"[ablation] subset_sha256={array_hash(actual)} reference_max_abs_delta={provenance['reference_max_abs_delta']}", flush=True)
    count, first_loss = 0, None
    started = time.monotonic()
    original_backward = Accelerator.backward

    def checked_backward(self, loss, *args, **kwargs):
        nonlocal first_loss
        if not bool(torch.isfinite(loss).all().item()):
            raise RuntimeError("non-finite training loss; preserving partial checkpoints")
        if first_loss is None:
            first_loss = float(loss.detach().item())
        return original_backward(self, loss, *args, **kwargs)

    def after_step(_optimizer, _args, _kwargs):
        nonlocal count
        count += 1
        if count == 1:
            head = model.proj_out_2.weight.detach()
            if not bool(torch.isfinite(head).all().item()) or (row["initialization"] == "zero" and not bool(torch.any(head != 0).item())):
                raise RuntimeError("first optimizer update did not produce a finite, learning output head")
            torch.cuda.synchronize()
            print(f"[ablation] FIRST_OPTIMIZER_STEP_OK loss={first_loss:.8g} peak_VRAM_GiB={torch.cuda.max_memory_allocated()/2**30:.3f}", flush=True)
        if count == 100:
            elapsed = time.monotonic() - started
            print(f"[ablation] 100 successful steps in {elapsed:.1f}s; rough_300k_hours={elapsed*3000/3600:.1f}", flush=True)

    hook = optimizer.register_step_post_hook(after_step)
    Accelerator.backward = checked_backward
    try:
        result = optim.train(dataset, model, optimizer=optimizer,
            noise_scheduler=built["noise_scheduler"], lr_scheduler=built["lr_scheduler"],
            output_dir=str(checkpoint_dir), **config["train"])
    finally:
        Accelerator.backward = original_backward
        hook.remove()
    if len(result["metrics"]["loss"]) != TARGET * 4 or not 0 < count <= TARGET:
        raise RuntimeError(f"nominal/successful update budget mismatch: microbatches={len(result['metrics']['loss'])}, successful_steps={count}; do not mark complete")
    skipped = TARGET - count
    if skipped:
        print(f"QUALIFICATION: fp16 AMP skipped {skipped} of {TARGET} scheduled updates; preserve baseline epoch budget, do not silently extend training.", flush=True)
    # Free live state before actually tensor-loading the saved checkpoint.
    del result, built, optimizer, model, dataset
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    state = validate_scientific_checkpoint(Path(row["expected_checkpoint"]),
        optimizer_steps_per_epoch=32, microbatches_per_optimizer_step=4,
        expected_ema_step=TARGET * 4 - config["train"]["ema_burn_in"],
        expected_ema_sigma_rels=config["train"]["ema_sigma_rels"],
        expected_ema_burn_in=config["train"]["ema_burn_in"])
    final = Path(row["expected_checkpoint"])
    saved_hashes = {str(path.relative_to(final)): file_hash(path) for path in final.rglob("*") if path.is_file()}
    publish_json(checkpoint_dir / "complete.json", {"status": "complete", "arm": row,
        "nominal_optimizer_steps": TARGET, "successful_optimizer_steps": count,
        "amp_skipped_optimizer_steps": skipped, "first_loss": first_loss,
        "checkpoint_files_sha256": saved_hashes,
        "plan_sha256": file_hash(Path(output) / "plan/plan.json"), "checkpoint_validation": state})
    print(f"ABLATION TRAIN COMPLETE: {row['name']}; validated 300000 nominal updates, successful={count}, AMP_skipped={skipped}", flush=True)


def sample(output, index):
    from trace_trajectories import atomic_output
    from evaluate_late_start import boundary_ratio, max_cosine, radial_power
    plan, noise = load_plan(output)
    if not 0 <= index < 3:
        raise ValueError("arm index must be 0..2")
    row = plan["arms"][index]
    report = json.loads((Path(row["checkpoint_dir"]) / "complete_record/complete.json").read_text())
    if report.get("status") != "complete" or report.get("arm") != row or report.get("plan_sha256") != file_hash(Path(output) / "plan/plan.json"):
        raise ValueError("training completion is missing or differs from reviewed plan")
    final = Path(row["expected_checkpoint"])
    current_hashes = {str(path.relative_to(final)): file_hash(path) for path in final.rglob("*") if path.is_file()}
    if not current_hashes or current_hashes != report["checkpoint_files_sha256"]:
        raise ValueError("validated final checkpoint artifacts changed")
    torch, _ = runtime()
    single_a40(torch)
    import sample_cosmodiff as sc
    reference = reference_for(row, plan)
    model, scheduler = sc._load_for_sampling(Path(row["expected_checkpoint"]), Path(row["config"]))
    model.to("cuda").eval()
    scheduler = sc.build_inference_scheduler(scheduler, "DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++", "solver_order": 2})
    with atomic_output(Path(output) / "samples" / row["name"]) as pending:
        with torch.no_grad():
            samples = sc.generate_samples(model, scheduler, batch_size=128, image_shape=(1, 128, 128),
                num_steps=50, device=torch.device("cuda"), initial_noise=torch.from_numpy(noise),
                generator=torch.Generator(device="cpu").manual_seed(124), model_batch_size=2,
                class_labels=torch.zeros(128, dtype=torch.long)).float().cpu().numpy()
        if samples.shape != noise.shape or not np.isfinite(samples).all():
            raise ValueError("sample fields are invalid/nonfinite")
        cos = max_cosine(samples, reference)
        power, k = radial_power(samples)
        real_power, _ = radial_power(reference)
        # Check BOTH grids for patch4 arms. Keep patch8 comparable to old figures.
        import evaluate_late_start as ev
        original_patch = ev.PATCH
        try:
            ev.PATCH = row["patch_size"]
            own_boundary = ev.boundary_ratio(samples)
            real_boundary = ev.boundary_ratio(reference)
        finally:
            ev.PATCH = original_patch
        np.savez_compressed(pending / "samples.npz", samples=samples, initial_noise=noise,
            max_cos=cos, boundary8=boundary_ratio(samples), boundary_model_grid=own_boundary,
            real_boundary_model_grid=real_boundary, power=power, reference_power=real_power, k=k,
            config_sha256=row["config_sha256"], checkpoint=row["expected_checkpoint"],
            reference_sha256=plan["reference_sha256"], initial_noise_batch_sha256=array_hash(noise),
            scheduler_config_json=json.dumps(dict(scheduler.config)), raw_weights=True,
            model_batch_size=2, seed=123, step_seed=124, num_steps=50)
        hi = (k >= 32) & (k <= 64)
        populations = {}
        for group, mask in {"near_copy": cos > .98, "low_similarity": cos < .8,
                             "intermediate": (cos >= .8) & (cos <= .98)}.items():
            populations[group] = {"n": int(mask.sum()),
                "boundary_grid_median": float(np.median(own_boundary[mask])) if mask.any() else None,
                "pk_ratio_hi": float((power[mask].mean(axis=0)/real_power.mean(axis=0))[hi].mean()) if mask.any() else None}
        (pending / "complete.json").write_text(json.dumps({"status": "complete", "arm": row,
            "n": 128, "populations": populations, "plan_sha256": file_hash(Path(output) / "plan/plan.json"),
            "noise_sha256": array_hash(noise), "samples_sha256": file_hash(pending / "samples.npz"),
            "quality_caveat": "Low similarity is NOT an invalidity label; spectra are normalized-field diagnostics.",
            "numerical_caveat": "model microbatch2 on A40 vs old matrix8; inspect repeatability before causal claims."}, indent=2, allow_nan=False) + "\n")
    print(f"ABLATION DPM50 COMPLETE: {row['name']} {populations}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--train", type=int)
    mode.add_argument("--sample", type=int)
    parser.add_argument("--project-dir", type=Path, default=Path("/home/jiamingp/diffusion_models_repo"))
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.project_dir, args.out_dir)
    elif args.preflight:
        cpu_preflight(args.out_dir)
    elif args.train is not None:
        train(args.out_dir, args.train)
    else:
        sample(args.out_dir, args.sample)


if __name__ == "__main__":
    main()
