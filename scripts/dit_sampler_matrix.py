#!/usr/bin/env python
"""Immutable, paired raw-weight sampler matrix; no training or job submission."""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import sample_cosmodiff as sc
from trace_trajectories import (array_hash, atomic_output, make_initial_noise,
                                validate_completed_trajectory, write_csv)
from resolve_dit_high_noise_models import resolve

PROTOCOL = "dit_sampler_matrix_v1"
STEP_SEED = 124  # Distinct from initial seed 123: do not reuse its first Gaussian draw.
CONDITIONS = [
    {"name": "dpm50", "scheduler": "DPMSolverMultistepScheduler", "steps": 50,
     "kwargs": {"algorithm_type": "dpmsolver++", "solver_order": 2}},
    {"name": "dpm200", "scheduler": "DPMSolverMultistepScheduler", "steps": 200,
     "kwargs": {"algorithm_type": "dpmsolver++", "solver_order": 2}},
    {"name": "ddpm500", "scheduler": "DDPMScheduler", "steps": 500, "kwargs": {}},
    {"name": "sde_dpmpp50", "scheduler": "DPMSolverMultistepScheduler", "steps": 50,
     "kwargs": {"algorithm_type": "sde-dpmsolver++", "solver_order": 2}},
]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions():
    versions = {"python": sys.version.split()[0], "torch": str(torch.__version__), "numpy": np.__version__}
    for name in ("diffusers", "huggingface-hub"):
        try:
            versions[name] = sc.importlib.metadata.version(name)
        except sc.importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def raw_files(checkpoint):
    checkpoint = Path(checkpoint)
    config = checkpoint / "config.json"
    weights = sorted([*checkpoint.glob("diffusion_pytorch_model*.safetensors"),
                      *checkpoint.glob("diffusion_pytorch_model*.bin"),
                      *checkpoint.glob("pytorch_model*.bin")])
    if not config.is_file() or not weights or any(p.stat().st_size == 0 for p in weights):
        raise FileNotFoundError(f"selected exact checkpoint lacks raw config/weights: {checkpoint}")
    json.loads(config.read_text())
    # Detect incomplete sharded checkpoints rather than guessing a different epoch.
    for index in checkpoint.glob("*.index.json"):
        shards = json.loads(index.read_text()).get("weight_map", {})
        for name in set(shards.values()):
            if Path(name).name != name or not (checkpoint / name).is_file():
                raise FileNotFoundError(f"missing/invalid model shard: {name}")
    return [config, *weights, *sorted(checkpoint.glob("*.index.json"))]


def select_models(rows):
    selected, omitted = [], []
    for row in rows:
        if not Path(row["config"]).is_file():
            if row["name"] == "dit_l12_200k":
                omitted.append({"name": row["name"], "reason": "matching run config is missing"})
                continue
            raise FileNotFoundError(f"missing matching run config: {row['config']}")
        try:
            raw_files(row["checkpoint"])
        except FileNotFoundError as exc:
            if row["name"] != "dit_l12_200k":
                raise
            omitted.append({"name": row["name"], "reason": str(exc)})
            continue
        selected.append(row)
    required = {"dit_l8_200k", "dit_l16_fresh300k", "dit_l16_seed456_500k"}
    if not required.issubset({r["name"] for r in selected}):
        raise ValueError("matrix is missing a mandatory checkpoint")
    return selected, omitted


def extend_noise(original, n=128, seed=123):
    if original.dtype != np.float32 or original.shape != (32, 1, 128, 128):
        raise ValueError("original trajectory noise must be 32 float32 128x128 fields")
    if n != 128 or seed != 123:
        raise ValueError("matrix requires 128 draws and seed 123")
    noise = make_initial_noise(n, original.shape[1:], seed).numpy()
    if not np.array_equal(noise[:len(original)], original):
        raise ValueError("CPU RNG does not reproduce original 32 trajectory noise fields")
    return noise


def prepare(project, output):
    models, omitted = select_models(resolve(project, allow_missing_l12=True))
    trace_root = project / "results/dit_l16_trajectory_screen"
    original = reference_hash = None
    traces = {}
    for name in ("dit_l8_200k", "dit_l16_fresh300k", "dit_l16_seed456_500k"):
        path = trace_root / name / f"{name}_trajectory.npz"
        validate_completed_trajectory(path)
        report = json.loads(path.with_suffix(".json").read_text())
        matching_model = next(r for r in models if r["name"] == name)
        scheduler_config = report.get("scheduler_config", {})
        if (report.get("raw_weights") is not True or report.get("num_steps") != 50
                or scheduler_config.get("algorithm_type") != "dpmsolver++"
                or scheduler_config.get("solver_order") != 2
                or report.get("config_sha256") != file_hash(matching_model["config"])
                or Path(report.get("checkpoint", "")).resolve() != Path(matching_model["checkpoint"]).resolve()):
            raise ValueError("original control checkpoint/config/scheduler differs from matrix")
        with np.load(path, allow_pickle=False) as data:
            current, ref = data["initial_noise"].copy(), str(data["reference_sha256"].item())
            if int(data["seed"]) != 123:
                raise ValueError("original trajectory seed differs")
        if original is not None and (not np.array_equal(original, current) or reference_hash != ref):
            raise ValueError("original trajectory noise/reference differs across models")
        original, reference_hash = current, ref
        traces[name] = {"path": str(path.resolve()), "sha256": file_hash(path)}
    noise = extend_noise(original)
    for row in models:
        row["config_sha256"] = file_hash(row["config"])
        row["checkpoint_sha256"] = {p.name: file_hash(p) for p in raw_files(row["checkpoint"])}
    tasks = [{"model": row["name"], "condition": condition["name"]}
             for row in models for condition in CONDITIONS]
    with atomic_output(output / "plan") as pending:
        np.savez_compressed(pending / "initial_noise.npz", initial_noise=noise,
                            initial_noise_sha256=np.asarray([array_hash(x) for x in noise]))
        plan = {"status": "prepared", "protocol": PROTOCOL, "n": 128, "seed": 123,
                "step_seed": STEP_SEED, "models": models, "omitted": omitted, "tasks": tasks,
                "conditions": CONDITIONS, "reference_sha256": reference_hash, "traces": traces,
                "noise_file_sha256": file_hash(pending / "initial_noise.npz"),
                "noise_batch_sha256": array_hash(noise), "original32_sha256": array_hash(original),
                "code_revision": os.environ.get("EXPECTED_COMMIT"),
                "runtime_versions": runtime_versions()}
        (pending / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for row in omitted:
        print(f"OMITTED: {row['name']}: {row['reason']}")
    print(f"MATRIX PLAN PREPARED: {len(tasks)} tasks; first 32 noise fields byte-identical")
    print(f"PLAN_SHA256={file_hash(output / 'plan/plan.json')}")
    print(f"ARRAY=0-{len(tasks) - 1}%2")


def load_plan(output):
    path = output / "plan/plan.json"
    plan = json.loads(path.read_text())
    if plan.get("status") != "prepared" or plan.get("protocol") != PROTOCOL:
        raise ValueError("invalid matrix plan")
    if plan["conditions"] != CONDITIONS or plan["n"] != 128 or plan["seed"] != 123 or plan["step_seed"] != STEP_SEED:
        raise ValueError("matrix protocol changed")
    if plan["runtime_versions"] != runtime_versions():
        raise ValueError("Python/torch/numpy/diffusers/hub versions changed since preparation")
    expected = os.environ.get("MATRIX_PLAN_SHA256")
    if not expected:
        raise ValueError("set MATRIX_PLAN_SHA256 to the reviewed frozen plan hash")
    if file_hash(path) != expected:
        raise ValueError("frozen matrix plan hash changed")
    code = os.environ.get("EXPECTED_COMMIT")
    if code and plan["code_revision"] != code:
        raise ValueError("matrix plan belongs to different adapter code")
    expected_tasks = [{"model": r["name"], "condition": c["name"]}
                      for r in plan["models"] for c in CONDITIONS]
    if plan["tasks"] != expected_tasks or len({r["name"] for r in plan["models"]}) != len(plan["models"]):
        raise ValueError("frozen matrix task/model coverage changed")
    noise_path = path.parent / "initial_noise.npz"
    if file_hash(noise_path) != plan["noise_file_sha256"]:
        raise ValueError("matrix noise file hash changed")
    with np.load(noise_path, allow_pickle=False) as data:
        noise = data["initial_noise"].copy()
        if data["initial_noise_sha256"].tolist() != [array_hash(x) for x in noise]:
            raise ValueError("matrix per-sample noise hashes changed")
    if noise.dtype != np.float32 or noise.shape != (128, 1, 128, 128):
        raise ValueError("matrix noise shape/dtype changed")
    if array_hash(noise) != plan["noise_batch_sha256"] or array_hash(noise[:32]) != plan["original32_sha256"]:
        raise ValueError("matrix noise/prefix does not match frozen plan")
    return plan, noise


def validate_inputs(row):
    if file_hash(row["config"]) != row["config_sha256"]:
        raise ValueError(f"frozen config changed: {row['name']}")
    files = raw_files(row["checkpoint"])
    actual = {p.name: file_hash(p) for p in files}
    if actual != row["checkpoint_sha256"]:
        raise ValueError(f"frozen checkpoint changed: {row['name']}")


def scheduler_for(config, condition):
    base = sc._load_scheduler_from_config(Path(config))
    if base.config.prediction_type != "v_prediction" or base.config.num_train_timesteps != 500:
        raise ValueError("matrix requires the configured 500-timestep v-prediction training scheduler")
    result = sc.build_inference_scheduler(base, condition["scheduler"], condition["kwargs"])
    if (condition["name"] in ("ddpm500", "sde_dpmpp50")
            and "generator" not in inspect.signature(result.step).parameters):
        raise ValueError("stochastic scheduler lacks an explicit generator; upgrade/reconcile runtime")
    return result


def runtime_preflight(output):
    plan, _ = load_plan(output)
    sc._reject_known_bad_runtime()
    sc._install_sklearn_roc_curve_stub()
    # Actual installed schedulers, including the cosine/zero-SNR endpoint.
    class ToyModel:
        def eval(self):
            return self

        def __call__(self, x, **kwargs):
            return (x * .1,)

    for condition in CONDITIONS:
        scheduler = scheduler_for(plan["models"][0]["config"], condition)
        with torch.no_grad():
            result = sc.generate_samples(ToyModel(), scheduler, batch_size=2,
                image_shape=(1, 4, 4), num_steps=condition["steps"], device=torch.device("cpu"),
                initial_noise=make_initial_noise(2, (1, 4, 4), 123),
                generator=torch.Generator(device="cpu").manual_seed(STEP_SEED),
                class_labels=torch.zeros(2, dtype=torch.long))
        if not torch.isfinite(result).all():
            raise ValueError("scheduler smoke test produced non-finite values")
        print(f"SCHEDULER CPU SMOKE PASSED: {condition['name']} {condition['steps']} steps")
    ode = scheduler_for(plan["models"][0]["config"], CONDITIONS[0])
    sde = scheduler_for(plan["models"][0]["config"], CONDITIONS[-1])
    ode.set_timesteps(50)
    sde.set_timesteps(50)
    if not torch.equal(ode.timesteps, sde.timesteps) or not torch.equal(ode.sigmas, sde.sigmas):
        raise ValueError("equal-step ODE/SDE arms have different timesteps/sigmas")
    print("ODE/SDE EQUAL-STEP SCHEDULE MATCH PASSED")
    print("MATRIX RUNTIME PREFLIGHT PASSED; NO MODEL LOAD OR GPU SAMPLING")


def inspect_trajectories(project):
    """Read existing paired CSVs; print no causal verdict or synthetic-as-real data."""
    root = project / "results/dit_l16_trajectory_screen"
    complete = json.loads((root / "comparison/complete.json").read_text())
    if complete.get("status") != "complete" or complete.get("paired_samples") != 32:
        raise ValueError("original paired trajectory comparison is incomplete")
    fields = ["cohort_model", "cohort_outcome", "label", "step", "t", "snr", "n",
              "majority", "persistence", "patch_cos", "top_gap", "boundary"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    with (root / "comparison/paired_by_l16_outcome.csv").open() as f:
        for row in csv.DictReader(f):
            if (row["cohort_outcome"] in ("copy", "junk")
                    and row["label"] in (row["cohort_model"], "dit_l8_200k")
                    and int(row["step"]) in (5, 10, 15, 20, 25, 30, 40, 49)):
                writer.writerow({key: row[key] for key in fields})
    for name in ("dit_l8_200k", "dit_l16_fresh300k", "dit_l16_seed456_500k"):
        gallery = root / name / f"{name}_gallery.png"
        if not gallery.is_file():
            raise FileNotFoundError(f"missing completed gallery: {gallery}")
        print(f"GALLERY: {gallery}")


def run_task(output, index, eval_root, device):
    plan, noise = load_plan(output)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("array task index outside frozen matrix plan")
    task = plan["tasks"][index]
    row = next(r for r in plan["models"] if r["name"] == task["model"])
    condition = next(c for c in CONDITIONS if c["name"] == task["condition"])
    validate_inputs(row)
    target = output / "tasks" / f"{row['name']}__{condition['name']}"
    if target.exists():
        raise FileExistsError(f"refusing to repeat/overwrite completed task: {target}")
    sc._reject_known_bad_runtime()
    sc._install_sklearn_roc_curve_stub()
    sc._ensure_cosmodiff_on_path(Path.cwd())
    model, _ = sc._load_for_sampling(Path(row["checkpoint"]), Path(row["config"]))
    model.to(device).eval()
    scheduler = scheduler_for(row["config"], condition)
    print(f"MATRIX TASK: {task}; noise_sha256={array_hash(noise)}", flush=True)
    with atomic_output(target) as pending:
        with torch.no_grad():
            samples = sc.generate_samples(model, scheduler, batch_size=128, image_shape=(1, 128, 128),
                num_steps=condition["steps"], device=device,
                generator=torch.Generator(device="cpu").manual_seed(plan["step_seed"]),
                initial_noise=torch.from_numpy(noise), model_batch_size=8,
                class_labels=torch.zeros(128, dtype=torch.long)).float().cpu().numpy()
        sample_path = pending / "samples.npz"
        audit = sc.scheduler_audit_metadata(scheduler, condition["steps"])
        np.savez_compressed(sample_path, samples=samples, protocol=PROTOCOL,
            matrix_condition=condition["name"], label=row["name"],
            scheduler_config_json=json.dumps(dict(scheduler.config)),
            num_steps=condition["steps"], seed=123, step_seed=STEP_SEED, ema_sigma_rel=-1.,
            resolved_checkpoint=row["checkpoint"], config_sha256=row["config_sha256"],
            reference_sha256=plan["reference_sha256"], initial_noise=noise,
            initial_noise_sha256=np.asarray([array_hash(x) for x in noise]),
            initial_noise_batch_sha256=array_hash(noise), **audit)
        # A real child process using the existing evaluator, with explicit matrix mode.
        subprocess.run([sys.executable, str(SCRIPTS / "evaluate_late_start.py"),
            "--protocol", "sampler-matrix", "--config", row["config"], "--eval-root", str(eval_root),
            "--samples", str(sample_path), "--out", str(pending / "summary.csv")],
            check=True, env=os.environ.copy())
        from evaluate_late_start import load_reference, max_cosine
        reference = load_reference(Path(row["config"]), eval_root)
        cos = max_cosine(samples, reference)
        outcome = np.where(cos > .98, "copy", np.where(cos < .8, "junk", "other"))
        write_csv(pending / "outcomes.csv", [{"sample": i, "initial_noise_sha256": array_hash(noise[i]),
            "outcome": str(outcome[i]), "max_cos": float(cos[i])} for i in range(128)])
        qualification = []
        if condition["name"] == "dpm50" and row["name"] in plan["traces"]:
            trace = plan["traces"][row["name"]]
            if file_hash(trace["path"]) != trace["sha256"]:
                raise ValueError("original trajectory artifact changed")
            with np.load(trace["path"], allow_pickle=False) as data:
                same = np.array_equal(outcome[:32], data["outcome"])
                delta = float(np.max(np.abs(samples[:32] - data["final"])))
            print(f"DPM50 ORIGINAL32 CONTROL: same_outcomes={same} max_abs_delta={delta}", flush=True)
            if not same or delta > 1e-4:
                qualification.append("DPM50 first-32 outputs differ from the previous run; assess runtime/numerical repeatability before interpreting mechanism.")
        report = {"status": "complete", "protocol": PROTOCOL, "task": task, "n": 128,
            "plan_sha256": file_hash(output / "plan/plan.json"),
            "noise_batch_sha256": array_hash(noise), "reference_sha256": plan["reference_sha256"],
            "condition": condition, "scheduler_config": dict(scheduler.config),
            "input": row, "raw_weights": True, "class_label": 0, "model_batch_size": 8,
            "python": sys.executable, "torch_version": torch.__version__,
            "diffusers_version": sc.importlib.metadata.version("diffusers"),
            "runtime_versions": runtime_versions(),
            "scheduler_module_sha256": file_hash(inspect.getfile(type(scheduler))),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "seed": 123, "step_seed": STEP_SEED, "initial_rng": "CPU float32, separate from step RNG",
            "qualification": qualification,
            "artifacts_sha256": {p.name: file_hash(p) for p in pending.iterdir()}}
        (pending / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
        for warning in qualification:
            print(f"QUALIFICATION: {warning}", flush=True)
    print(f"MATRIX TASK COMPLETE: {target}", flush=True)


def summarize(output):
    plan, noise = load_plan(output)
    summaries, paired = [], [{"sample": i, "initial_noise_sha256": array_hash(noise[i])} for i in range(128)]
    qualifications = []
    for task in plan["tasks"]:
        name = f"{task['model']}__{task['condition']}"
        directory = output / "tasks" / name
        report = json.loads((directory / "complete.json").read_text())
        condition = next(c for c in CONDITIONS if c["name"] == task["condition"])
        model = next(r for r in plan["models"] if r["name"] == task["model"])
        if (report.get("status") != "complete" or report.get("protocol") != PROTOCOL
                or report["task"] != task or report["n"] != 128
                or report["condition"] != condition or report["input"] != model
                or report["runtime_versions"] != plan["runtime_versions"]
                or report["seed"] != 123 or report["step_seed"] != STEP_SEED or report["raw_weights"] is not True
                or report["plan_sha256"] != file_hash(output / "plan/plan.json")
                or report["noise_batch_sha256"] != array_hash(noise)
                or report["reference_sha256"] != plan["reference_sha256"]):
            raise ValueError(f"incomplete/unpaired matrix task: {name}")
        if not {"samples.npz", "summary.csv", "outcomes.csv"}.issubset(report["artifacts_sha256"]):
            raise ValueError("matrix completion record lacks required artifact hashes")
        for file, expected in report["artifacts_sha256"].items():
            if Path(file).name != file or file_hash(directory / file) != expected:
                raise ValueError(f"matrix artifact changed: {name}/{file}")
        from evaluate_late_start import validate_matrix_provenance
        with np.load(directory / "samples.npz", allow_pickle=False) as data:
            provenance = validate_matrix_provenance(data, reference_sha256=plan["reference_sha256"],
                                                   config_sha256=model["config_sha256"])
            if (provenance["checkpoint"] != model["checkpoint"]
                    or provenance["matrix_condition"] != task["condition"]):
                raise ValueError("sample checkpoint/condition differs from frozen task")
            if (data["samples"].shape != noise.shape
                    or not np.isfinite(data["samples"]).all()):
                raise ValueError("completed samples have invalid shape/non-finite fields")
            if not np.array_equal(data["initial_noise"], noise):
                raise ValueError("actual saved task noise differs from reviewed plan")
        with (directory / "summary.csv").open() as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 1:
            raise ValueError("matrix task summary must contain exactly one row")
        if rows[0].get("file") != "samples.npz" or any(
                str(rows[0].get(key)) != str(value) for key, value in provenance.items()):
            raise ValueError("task summary provenance differs from scored samples")
        summaries.append({"model": task["model"], **rows[0]})
        with (directory / "outcomes.csv").open() as f:
            outcomes = list(csv.DictReader(f))
        if len(outcomes) != 128:
            raise ValueError("matrix task outcomes must contain exactly 128 rows")
        for i, r in enumerate(outcomes):
            if int(r["sample"]) != i or r["initial_noise_sha256"] != paired[i]["initial_noise_sha256"]:
                raise ValueError("per-sample matrix pairing/order changed")
            cosine = float(r["max_cos"])
            expected_outcome = "copy" if cosine > .98 else "junk" if cosine < .8 else "other"
            if not np.isfinite(cosine) or r["outcome"] != expected_outcome:
                raise ValueError("per-sample outcome is inconsistent with its cosine")
            paired[i][f"{name}_outcome"] = r["outcome"]
            paired[i][f"{name}_max_cos"] = float(r["max_cos"])
        if int(rows[0]["n"]) != 128:
            raise ValueError("task summary denominator differs")
        for oc in ("copy", "junk", "other"):
            if int(rows[0][f"{oc}_count"]) != sum(r["outcome"] == oc for r in outcomes):
                raise ValueError("summary counts do not reconcile with per-sample outcomes")
        qualifications.extend(report["qualification"])
    with atomic_output(output / "comparison") as pending:
        write_csv(pending / "summary.csv", summaries)
        write_csv(pending / "paired_outcomes.csv", paired)
        write_csv(pending / "summary_original32.csv", [{"model": task["model"], "condition": task["condition"],
            "n": 32, **{f"{oc}_count": sum(r[f"{task['model']}__{task['condition']}_outcome"] == oc for r in paired[:32])
                         for oc in ("copy", "junk", "other")}} for task in plan["tasks"]])
        (pending / "complete.json").write_text(json.dumps({"status": "complete", "tasks": len(summaries),
            "n": 128, "original32_pairing": True, "plan_sha256": file_hash(output / "plan/plan.json"),
            "qualification": qualifications,
            "artifacts_sha256": {p.name: file_hash(p) for p in pending.iterdir()}}, indent=2) + "\n")
    print(f"MATRIX COMPARISON COMPLETE: {len(summaries)} tasks; 128 paired draws including original32")
    for warning in sorted(set(qualifications)):
        print(f"QUALIFICATION: {warning}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--runtime-preflight", action="store_true")
    mode.add_argument("--task", type=int)
    mode.add_argument("--summarize", action="store_true")
    mode.add_argument("--inspect-trajectories", action="store_true")
    p.add_argument("--project-dir", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--eval-root", type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.inspect_trajectories:
        if args.project_dir is None:
            p.error("--inspect-trajectories requires --project-dir")
        inspect_trajectories(args.project_dir.resolve())
        return
    if args.out_dir is None:
        p.error("matrix modes require --out-dir")
    if args.prepare:
        if args.project_dir is None:
            p.error("--prepare requires --project-dir")
        prepare(args.project_dir.resolve(), args.out_dir.resolve())
    elif args.runtime_preflight:
        runtime_preflight(args.out_dir.resolve())
    elif args.summarize:
        summarize(args.out_dir.resolve())
    else:
        if args.eval_root is None:
            p.error("--task requires --eval-root")
        run_task(args.out_dir.resolve(), args.task, args.eval_root.resolve(), torch.device(args.device))


if __name__ == "__main__":
    main()
