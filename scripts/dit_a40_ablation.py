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

from dit_ablation_data import CONTRACT as DATA_CONTRACT, audit_native_dataset, load_native_training_reference

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
NAME = "dit_l16_a40_init_patch_v1"
MATRIX_SHA = "8404ae7b4f607923251d430da2b78afe07871846269927c0a9e2b6e3d8c0e333"
ARMS = [("p8_zero", 8, "zero"), ("p4_native", 4, "native"), ("p4_zero", 4, "zero")]
TARGET = 300_000
# Operational-only difference: retain six complete safety/final checkpoints,
# instead of generating ~60 and deleting most. No checkpoint pruning.
CHECKPOINT_EPOCHS = 1563  # 50,016 nominal optimizer updates
RUNTIME_KEYS = ("python", "torch", "numpy", "diffusers", "huggingface-hub")
TESTED_DIFFUSERS = ("0.38.0",)


def validate_runtime_versions(expected, actual=None):
    """Match the frozen, working baseline; never change shared packages."""
    if (not isinstance(expected, dict) or set(expected) != set(RUNTIME_KEYS)
            or any(not isinstance(expected[key], str) or not expected[key] for key in RUNTIME_KEYS)):
        raise ValueError("frozen matrix runtime versions are missing/incomplete; reprepare an isolated plan")
    if expected["diffusers"] not in TESTED_DIFFUSERS:
        raise RuntimeError(f"matrix diffusers {expected['diffusers']} has not been tested for this ablation; "
                           f"tested versions: {TESTED_DIFFUSERS}; do not reinstall shared packages")
    if actual is not None:
        changed = {key: {"expected": expected[key], "actual": actual.get(key)}
                   for key in RUNTIME_KEYS if actual.get(key) != expected[key]}
        if changed:
            raise RuntimeError(f"ablation runtime differs from frozen working matrix: {changed}; "
                               "do not reinstall shared packages or bypass this check")


def read_matrix_plan(path):
    if file_hash(path) != MATRIX_SHA:
        raise ValueError("original reviewed sampler matrix plan changed")
    matrix = json.loads(Path(path).read_text())
    validate_runtime_versions(matrix.get("runtime_versions"))
    return matrix


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
    plan = read_matrix_plan(path)
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
    matrix_root, matrix, baseline, noise = matrix_inputs(project)
    if "/scratch/huterer_root/huterer0/jiamingp/" not in str(output):
        raise ValueError("use the dedicated scratch experiment root, not /home or an existing run")
    base = yaml.safe_load(Path(baseline["config"]).read_text())
    training_reference, training_metadata = load_native_training_reference(base)
    if training_reference.shape != (256, 1, 128, 128):
        raise ValueError("native retained training subset must contain 256 128x128 maps")
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
            "matrix_plan_path": str(matrix_root / "plan/plan.json"),
            "runtime_versions": deepcopy(matrix["runtime_versions"]),
            "training_data_reference": training_metadata,
            "training_reference_sha256": array_hash(training_reference),
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
    validate_runtime_versions(plan.get("runtime_versions"))
    data_reference = plan.get("training_data_reference", {})
    if (data_reference.get("contract") != DATA_CONTRACT
            or data_reference.get("shape") != [256, 1, 128, 128]
            or data_reference.get("dtype") != "float32"
            or data_reference.get("reference_sha256") != plan.get("training_reference_sha256")
            or not isinstance(data_reference.get("selected_raw_sha256"), str)):
        raise ValueError("missing slice-first data contract; preserve the old plan and reprepare in a new root")
    matrix = read_matrix_plan(plan["matrix_plan_path"])
    if (plan.get("matrix_plan_sha256") != MATRIX_SHA
            or plan["runtime_versions"] != matrix["runtime_versions"]):
        raise ValueError("ablation runtime contract differs from the original frozen matrix")
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


def runtime(expected_versions):
    validate_runtime_versions(expected_versions)
    # Import compatibility from the ORIGINAL pinned runtime before new adapter modules.
    from simdiff_eval.torch_compat import install_torch_backend_compat
    install_torch_backend_compat(entry_point=__name__)
    import sample_cosmodiff as sc
    sc._reject_known_bad_runtime()
    sc._install_sklearn_roc_curve_stub()
    import torch
    import diffusers
    actual = {"python": sys.version.split()[0], "torch": str(torch.__version__),
              "numpy": np.__version__, "diffusers": diffusers.__version__,
              "huggingface-hub": sc.importlib.metadata.version("huggingface-hub")}
    print(f"[ablation] runtime expected={expected_versions} actual={actual}", flush=True)
    validate_runtime_versions(expected_versions, actual)
    print("[ablation] FROZEN MATRIX RUNTIME MATCH PASSED", flush=True)
    return torch, diffusers


def single_process():
    for key in ("WORLD_SIZE", "SLURM_NTASKS"):
        if int(os.environ.get(key, "1")) != 1:
            raise RuntimeError("single-process training only; DDP is not approved")
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        if int(os.environ.get(key, "0")) != 0:
            raise RuntimeError("a nonzero rank must not train/write this experiment")


def single_a40(torch):
    single_process()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA GPU required")
    name = torch.cuda.get_device_name(0)
    if "A40" not in name:
        raise RuntimeError(f"approved A40 experiment, found {name}")
    print(f"[ablation] GPU={name} VRAM_GiB={torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}", flush=True)


def cpu_preflight(output):
    """Exercise the actual installed model/optimizer API, without a GPU or data scan."""
    plan, _ = load_plan(output)
    torch, _ = runtime(plan["runtime_versions"])
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
    legacy = load_reference(Path(row["config"]), None)
    if legacy.shape != (256, 1, 128, 128) or array_hash(legacy) != plan["reference_sha256"]:
        raise ValueError("legacy evaluation reference differs from the frozen baseline")
    config = yaml.safe_load(Path(row["config"]).read_text())
    reference, metadata = load_native_training_reference(config)
    if (metadata != plan["training_data_reference"]
            or array_hash(reference) != plan["training_reference_sha256"]):
        raise ValueError("slice-first training reference/raw selection changed")
    return reference, legacy


def checked_dataset(utils, config, reference, plan, torch):
    from run_cosmodiff_train_with_dit_resume import install_constant_label_adapter
    if not config["data"].get("keep_on_cpu"):
        raise ValueError("native data preflight requires keep_on_cpu; do not silently change the recipe")
    install_constant_label_adapter(utils)
    parsed = utils.parse_config_data(config)
    audit = audit_native_dataset(parsed, reference, plan["training_data_reference"], torch)
    return parsed["data"], audit


def require_data_preflight(output, plan):
    path = Path(output) / "data_preflight_record/data_preflight.json"
    if not path.is_file():
        raise ValueError("run the real-data CPU preflight before GPU training; no completed receipt exists")
    report = json.loads(path.read_text())
    arms = report.get("arms") if isinstance(report, dict) else None
    if (not isinstance(report, dict) or not isinstance(arms, list)
            or not all(isinstance(row, dict) for row in arms)
            or report.get("status") != "complete" or report.get("plan_sha256") != file_hash(Path(output) / "plan/plan.json")
            or report.get("code_revision") != plan["code_revision"]
            or report.get("runtime_versions") != plan["runtime_versions"]
            or report.get("training_reference_sha256") != plan["training_reference_sha256"]
            or report.get("legacy_reference_sha256") != plan["reference_sha256"]
            or [r.get("arm") for r in arms] != [r[0] for r in ARMS]):
        raise ValueError("real-data CPU preflight receipt differs from the frozen plan")
    return report


def rescore_saved_baseline(plan, reference):
    """Reuse baseline samples; compare normalization without any sampling."""
    from evaluate_late_start import boundary_ratio, max_cosine, radial_power
    source = Path(plan["matrix_plan_path"]).parents[1] / "tasks/dit_l16_fresh300k__dpm50"
    report = json.loads((source / "complete.json").read_text())
    path = source / "samples.npz"
    if (report.get("status") != "complete" or report.get("plan_sha256") != MATRIX_SHA
            or report.get("task") != {"model": "dit_l16_fresh300k", "condition": "dpm50"}
            or report.get("reference_sha256") != plan["reference_sha256"]
            or file_hash(path) != report.get("artifacts_sha256", {}).get("samples.npz")):
        raise ValueError("saved baseline is not a completed, unchanged frozen matrix task")
    with np.load(path, allow_pickle=False) as data:
        samples = data["samples"].copy()
        if str(data["initial_noise_batch_sha256"].item()) != plan["noise_batch_sha256"]:
            raise ValueError("saved baseline noise pairing differs")
    if samples.shape != (128, 1, 128, 128) or not np.isfinite(samples).all():
        raise ValueError("saved baseline samples have invalid shape/values")
    cosine = max_cosine(samples, reference)
    power, k = radial_power(samples)
    real_power, _ = radial_power(reference)
    boundaries = boundary_ratio(samples)
    populations = {}
    for group, mask in {"near_copy": cosine > .98, "low_similarity": cosine < .8,
                         "intermediate": (cosine >= .8) & (cosine <= .98)}.items():
        hi = (k >= 32) & (k <= 64)
        populations[group] = {"n": int(mask.sum()),
            "boundary8_median": float(np.median(boundaries[mask])) if mask.any() else None,
            "pk_ratio_hi": float((power[mask].mean(axis=0) / real_power.mean(axis=0))[hi].mean()) if mask.any() else None}
    return {"model": "dit_l16_fresh300k", "condition": "dpm50", "source_samples_sha256": file_hash(path),
        "reference_sha256": array_hash(reference), "data_reference_contract": DATA_CONTRACT,
        "n": len(samples), "populations": populations,
        "quality_caveat": "Low training cosine is not an invalidity verdict; baseline sampling used microbatch8 vs new arms2."}


def data_preflight(output):
    """Validate actual retained data/labels before committing any training GPU."""
    single_process()
    plan, _ = load_plan(output)
    torch, _ = runtime(plan["runtime_versions"])
    from cosmodiff import utils
    pin = Path(os.environ["COSMODIFF_PIN_ROOT"]).resolve()
    if pin not in Path(inspect.getfile(utils)).resolve().parents:
        raise RuntimeError("data preflight escaped the immutable cosmodiff pin")
    rows = []
    for row in plan["arms"]:
        reference, legacy = reference_for(row, plan)
        config = yaml.safe_load(Path(row["config"]).read_text())
        dataset, audit = checked_dataset(utils, config, reference, plan, torch)
        audit.update(arm=row["name"], legacy_reference_max_abs_delta=float(np.max(np.abs(reference-legacy))))
        rows.append(audit)
        print(f"NATIVE REAL-DATA CPU MATCH PASSED: {row['name']} {audit}", flush=True)
        del dataset
    if len({r["training_tensor_sha256"] for r in rows}) != 1:
        raise ValueError("the three arms must receive byte-identical native training data")
    baseline = rescore_saved_baseline(plan, reference)
    print(f"SAVED BASELINE RESCORED WITH RETAINED-SLICE NORMALIZATION: {baseline}", flush=True)
    publish_json(Path(output) / "data_preflight.json", {"status": "complete", "arms": rows,
        "plan_sha256": file_hash(Path(output) / "plan/plan.json"), "code_revision": plan["code_revision"],
        "runtime_versions": plan["runtime_versions"], "training_reference_sha256": plan["training_reference_sha256"],
        "legacy_reference_sha256": plan["reference_sha256"], "utils_source_sha256": file_hash(inspect.getfile(utils)),
        "saved_baseline": baseline,
        "qualification": "Legacy IO fits normalization before z-thinning; retained-slice native normalization is the training/evaluation contract. Old matrix power ratios need rescoring; native training data is never replaced."})
    print("REAL-DATA CPU PREFLIGHT COMPLETE; ALL THREE ARMS MATCH; NO GPU/MODEL LOAD OR TRAINING", flush=True)


def train(output, index):
    plan, _ = load_plan(output)
    if not 0 <= index < 3:
        raise ValueError("arm index must be 0..2")
    row = plan["arms"][index]
    preflight = require_data_preflight(output, plan)
    checkpoint_dir = Path(row["checkpoint_dir"])
    # Even a partial prior launch is preserved; do not restart/overwrite implicitly.
    if checkpoint_dir.exists():
        raise FileExistsError(f"preserving prior arm output: {checkpoint_dir}")
    torch, diffusers = runtime(plan["runtime_versions"])
    single_a40(torch)
    from cosmodiff import optim, utils
    pin = Path(os.environ["COSMODIFF_PIN_ROOT"]).resolve()
    for module in (optim, utils):
        if pin not in Path(inspect.getfile(module)).resolve().parents:
            raise RuntimeError("cosmodiff did not resolve inside the immutable pin")
    from run_cosmodiff_train_with_dit_resume import validate_scientific_checkpoint
    from run_cosmodiff_train_fresh_seeded import seed_everything
    from dit_zero_init import zero_modulation_and_output
    from accelerate import Accelerator
    config = yaml.safe_load(Path(row["config"]).read_text())
    reference, legacy_reference = reference_for(row, plan)
    seed_everything(123)
    dataset, data_audit = checked_dataset(utils, config, reference, plan, torch)
    actual = dataset.arrays.detach().cpu().numpy()
    if (data_audit["training_tensor_sha256"] != preflight["arms"][index]["training_tensor_sha256"]
            or file_hash(inspect.getfile(utils)) != preflight["utils_source_sha256"]):
        raise ValueError("native dataset/source changed after the real-data CPU preflight")
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    built = utils.parse_config_model(config)
    model, optimizer = built["model"], built["optimizer"]
    init = (zero_modulation_and_output(model, fresh=True) if row["initialization"] == "zero"
            else {"kind": "native_diffusers", "zeroed_tensors": 0})
    provenance = {"status": "started", "arm": row, "initialization": init,
        "training_tensor_sha256": array_hash(actual), "evaluation_reference_sha256": array_hash(reference),
        "data_audit": data_audit, "legacy_reference_sha256": array_hash(legacy_reference),
        "legacy_reference_max_abs_delta": float(np.max(np.abs(actual-legacy_reference))),
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
    torch, _ = runtime(plan["runtime_versions"])
    single_a40(torch)
    import sample_cosmodiff as sc
    reference, _ = reference_for(row, plan)
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
            reference_sha256=plan["training_reference_sha256"], legacy_reference_sha256=plan["reference_sha256"],
            data_reference_contract=DATA_CONTRACT, initial_noise_batch_sha256=array_hash(noise),
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
    mode.add_argument("--data-preflight", action="store_true")
    mode.add_argument("--train", type=int)
    mode.add_argument("--sample", type=int)
    parser.add_argument("--project-dir", type=Path, default=Path("/home/jiamingp/diffusion_models_repo"))
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.project_dir, args.out_dir)
    elif args.preflight:
        cpu_preflight(args.out_dir)
    elif args.data_preflight:
        data_preflight(args.out_dir)
    elif args.train is not None:
        train(args.out_dir, args.train)
    else:
        sample(args.out_dir, args.sample)


if __name__ == "__main__":
    main()
