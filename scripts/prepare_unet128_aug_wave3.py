#!/usr/bin/env python
"""Plan augmentation wave 3 (ledger unet128_aug_wave3): patch-orbit collapse noise sigma_c test.

18 fresh UNet-128 unconditional runs, recipe of wave 2 (scripts/prepare_unet128_aug_wave2.py) except
the training-set size and the augmentation:

  jitter4   roll (dy, dx) uniform on {0,1,2,3}^2       (16 views)     N = 16, 32, 64, 128, 256
  shiftonly wave-2 ShiftOnly, roll uniform on 0..127   (16384 views)  N = 4, 8, 16, 32
  d4shift   wave-1 SymmetryPreserving (D4 x roll)      (131072 views) N = 4, 8, 16, 32
  d4only    wave-2 D4Only                              (8 views)      N = 16, 32
  stride32  roll (dy, dx) uniform on {0,32,64,96}^2    (16 views)     N = 16, 64, 256

Array-task order = PLAN (most informative first; they start first at %2).

Data: N <= 64 use the 4 source boxes of the N = 64 config (LH/CV x z = 0, 1, sim 0 of each),
thinned along z: zthin 128/64/32/16/8 -> 1/2/4/8/16 slices per box (N = 4/8/16/32/64).
N = 128 and 256 use the d2p07 / d2p08 baseline configs' data section unchanged (as wave 2).
Normalization is fitted on each run's own subset (unchanged per-run recipe).

Budget: cosmodiff.optim.train builds DataLoader(batch_size=32, shuffle, no drop_last) and loops
num_epochs epochs, one optimizer step per batch (gradient_accumulation_steps 1), so
updates/epoch = ceil(N / 32) and the effective batch is min(32, N). num_epochs is set so that
updates = 200000 for every run; checkpoint_every_n_epochs keeps one checkpoint per 20000 updates.
For N = 128 / 256 this reproduces the baseline num_epochs / checkpoint cadence exactly.

Each run gets its own config under local/unet128_aug_wave3/configs/, identical to its base config
(d2p06 for N <= 64, d2p07 for N = 128, d2p08 for N = 256) except io.output_dir, data.zthin,
train.num_epochs, train.checkpoint_every_n_epochs (only io.output_dir for N = 128 / 256).

Steps (never overwrite):
  --write-configs  write missing configs; an existing config must equal the planned one (else error)
  (then scripts/verify_unet128_aug_wave3_data.py --write  -> data_reference.json, CPU)
  --write-runs     write runs.json (training/sampling) and eval_runs.json (evaluate_run --runs-json),
                   dataset hashes from data_reference.json; create the scratch log directory.
No jobs are submitted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import yaml

WAVE = "unet128_aug_wave3"
FIG2 = "local/nf_generalize_fig2/configs"
# base config per training-set size: (path, sha256, baseline run name)
BASES = {
    64: (f"{FIG2}/nf_fig2_u128_d2p06_noaug_200k.yaml",
         "7fadfb10d759beb1e0996f0a20fca575d0d08bcd8208d8ad1803954d50556b30", "nf_fig2_u128_d2p06_noaug_200k"),
    128: (f"{FIG2}/nf_fig2_u128_d2p07_noaug_200k.yaml",
          "f108bffae5249b90041f0250f4c29e60e64744273571f1a86ef0cbee4364accc", "nf_fig2_u128_d2p07_noaug_200k"),
    256: (f"{FIG2}/nf_fig2_u128_d2p08_noaug_200k.yaml",
          "df53ece5e452a3fe02cb65b1de8b555c4c96a8869d0d204a75249764e3b26fe3", "nf_fig2_u128_d2p08_noaug_200k"),
}
# N = 64 base (kept for the runner / verify cross-check against the wave-1 dataset hash)
BASE_CONFIG, BASE_CONFIG_SHA256, BASE_RUN = BASES[64]
CHECKPOINT_ROOT = f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{WAVE}"
LOG_ROOT = f"{CHECKPOINT_ROOT}/logs"
SEED = 123
TOTAL_UPDATES = 200_000
UPDATES_PER_CHECKPOINT = 20_000
ZTHIN = {4: 128, 8: 64, 16: 32, 32: 16, 64: 8, 128: 8, 256: 8}
TAGS = {4: "d2p02", 8: "d2p03", 16: "d2p04", 32: "d2p05", 64: "d2p06", 128: "d2p07", 256: "d2p08"}
# arm_tag -> runner --arm (key of simdiff_eval.aug_wave3_transforms.ARMS)
ARMS = {"jitter4": "jitter4", "stride32": "stride32", "shiftonly": "shift_only",
        "d4shift": "d4_periodic_shift", "d4only": "d4_only"}
VIEWS = {"jitter4": 16, "stride32": 16, "shiftonly": 128 * 128, "d4shift": 8 * 128 * 128, "d4only": 8}
# array-task order: most informative first, then the rest grouped by arm
PLAN = [("jitter4", 16), ("stride32", 16), ("shiftonly", 4), ("d4shift", 4), ("jitter4", 32), ("jitter4", 64),
        ("jitter4", 128), ("jitter4", 256),
        ("shiftonly", 8), ("shiftonly", 16), ("shiftonly", 32),
        ("d4shift", 8), ("d4shift", 16), ("d4shift", 32),
        ("d4only", 16), ("d4only", 32),
        ("stride32", 64), ("stride32", 256)]
ALLOWED_CHANGES = {("io", "output_dir"), ("data", "zthin"), ("train", "num_epochs"),
                   ("train", "checkpoint_every_n_epochs")}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_for(n: int) -> tuple[str, str, str]:
    """(base config path, sha256, baseline run) for training-set size n."""
    return BASES[n] if n in (128, 256) else BASES[64]


def allowed_changes(n: int) -> set:
    return {("io", "output_dir")} if n in (128, 256) else ALLOWED_CHANGES


def sample_label(arm_tag: str) -> str:
    return f"dpm50_w3_{arm_tag}"


def run_name(arm_tag: str, n: int) -> str:
    return f"nf_fig2_u128_{TAGS[n]}_{arm_tag}_200k"


def budget(n: int, batch_size: int) -> dict:
    per_epoch = math.ceil(n / batch_size)       # DataLoader without drop_last
    if TOTAL_UPDATES % per_epoch or UPDATES_PER_CHECKPOINT % per_epoch:
        raise ValueError(f"N={n}: {TOTAL_UPDATES} updates not a whole number of epochs")
    return {"effective_batch": min(batch_size, n), "updates_per_epoch": per_epoch,
            "num_epochs": TOTAL_UPDATES // per_epoch, "checkpoint_every_n_epochs": UPDATES_PER_CHECKPOINT // per_epoch,
            "total_updates": TOTAL_UPDATES // per_epoch * per_epoch}


def config_diff(a: dict, b: dict, prefix=()) -> set:
    keys = set(a) | set(b)
    out = set()
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out |= config_diff(va, vb, prefix + (k,))
        elif va != vb:
            out.add(prefix + (k,))
    return out


def load_base(project_dir: Path, n: int) -> dict:
    path, sha, _ = base_for(n)
    if sha256_file(project_dir / path) != sha:
        raise ValueError(f"{path} changed")
    base = yaml.safe_load((project_dir / path).read_text())
    assert base["train"]["batch_size"] == 32 and base["train"]["gradient_accumulation_steps"] == 1
    assert base["data"]["zthin"] == 8 and base["data"]["seed"] is None
    assert not base.get("augmentations")
    if n <= 64:
        assert base["data"]["n_samples"] == [1, 1, 1, 1]
    return base


def build_configs(project_dir: Path) -> list[tuple[str, dict]]:
    assert len(PLAN) == len(set(PLAN)) == 18
    out = []
    for arm_tag, n in PLAN:
        base = load_base(project_dir, n)
        name = run_name(arm_tag, n)
        b = budget(n, base["train"]["batch_size"])
        cfg = copy.deepcopy(base)
        cfg["io"]["output_dir"] = f"{CHECKPOINT_ROOT}/{name}_checkpoints"
        cfg["data"]["zthin"] = ZTHIN[n]
        cfg["train"]["num_epochs"] = b["num_epochs"]
        cfg["train"]["checkpoint_every_n_epochs"] = b["checkpoint_every_n_epochs"]
        assert config_diff(base, cfg) <= allowed_changes(n), (name, config_diff(base, cfg))
        out.append((name, cfg))
    return out


def config_path(name: str) -> str:
    return f"local/{WAVE}/configs/{name}.yaml"


def build_runs(project_dir: Path, data_ref: dict) -> tuple[list[dict], list[dict]]:
    runs, evals = [], []
    for (arm_tag, n), (name, cfg) in zip(PLAN, build_configs(project_dir)):
        path = config_path(name)
        on_disk = yaml.safe_load((project_dir / path).read_text())
        assert on_disk == cfg, f"{path} differs from the planned config"
        ref = data_ref["configs"][path]
        assert ref["N"] == n and ref["config_sha256"] == sha256_file(project_dir / path), path
        b = budget(n, cfg["train"]["batch_size"])
        base_path, base_sha, base_run = base_for(n)
        sample = f"results/{WAVE}/samples/{name}_seed{{seed}}_{sample_label(arm_tag)}.npz"
        runs.append({"run_name": name, "arm": ARMS[arm_tag], "arm_tag": arm_tag, "views": VIEWS[arm_tag],
                     "dataset_size": n, "dataset_tag": TAGS[n], "zthin": ZTHIN[n],
                     **b, "batch_size_config": cfg["train"]["batch_size"],
                     "config": path, "config_sha256": ref["config_sha256"], "dataset_sha256": ref["dataset_sha256"],
                     "base_config": base_path, "base_config_sha256": base_sha,
                     "baseline_run_name": base_run,
                     "output_dir": cfg["io"]["output_dir"], "training_seed": SEED, "augmentation_seed": SEED + 1000,
                     "sample_label": sample_label(arm_tag), "sample_path": sample})
        evals.append(dict(run_name=name, arm=arm_tag, dataset_size=n, config=path, sample_path=sample))
    return runs, evals


def check_new_paths(project_dir: Path, runs: list[dict]) -> None:
    names = [r["run_name"] for r in runs]
    assert len(set(names)) == len(names)
    for r in runs:
        for p in (Path(r["output_dir"]), project_dir / r["sample_path"].format(seed=SEED)):
            if p.exists():
                raise FileExistsError(p)
    if (project_dir / f"results/{WAVE}/evaluate_run_v1").exists():
        raise FileExistsError(project_dir / f"results/{WAVE}/evaluate_run_v1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write-configs", action="store_true")
    parser.add_argument("--write-runs", action="store_true")
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    local = project_dir / "local" / WAVE
    configs = build_configs(project_dir)
    if args.write_configs:
        (local / "configs").mkdir(parents=True, exist_ok=True)
        for name, cfg in configs:
            dest = project_dir / config_path(name)
            if dest.exists():
                if yaml.safe_load(dest.read_text()) != cfg:
                    raise FileExistsError(f"{dest} exists and differs from the planned config")
                print("Kept (identical)", config_path(name), sha256_file(dest))
                continue
            with dest.open("x") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print("Wrote", config_path(name), sha256_file(dest))
    for i, ((arm_tag, n), (name, cfg)) in enumerate(zip(PLAN, configs)):
        b = budget(n, cfg["train"]["batch_size"])
        print(f"{i:2d} {name}: arm={arm_tag} N={n} zthin={ZTHIN[n]} batch={b['effective_batch']} "
              f"updates/epoch={b['updates_per_epoch']} num_epochs={b['num_epochs']} updates={b['total_updates']} "
              f"ckpt_every={b['checkpoint_every_n_epochs']} label={sample_label(arm_tag)}")
    if args.write_runs:
        data_ref = json.loads((local / "data_reference.json").read_text())
        if data_ref.get("status") != "VERIFIED":
            raise SystemExit("data_reference.json is not VERIFIED")
        runs, evals = build_runs(project_dir, data_ref)
        check_new_paths(project_dir, runs)
        for fname, obj in (("runs.json", runs), ("eval_runs.json", evals)):
            with (local / fname).open("x") as f:
                f.write(json.dumps(obj, indent=2) + "\n")
            print("Wrote", local / fname, sha256_file(local / fname))
        Path(LOG_ROOT).mkdir(parents=True, exist_ok=True)
        print("Log directory", LOG_ROOT)


if __name__ == "__main__":
    main()
