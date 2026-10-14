#!/usr/bin/env python
"""CPU preflight for ledger nf_cond_omsigc_n1024_n4096_aug_finetune (no GPU).

Same checks as scripts/preflight_nf_cond_omsigc_n256_aug_finetune.py, generalised to the two training runs of this
ledger entry (N = 1024 from checkpoint-epoch-6249, N = 4096 from checkpoint-epoch-1562). The N = 256 script is
hard-wired to one config / one source checkpoint / 200000 base updates, so it cannot be reused with arguments.

run            (default) Static checks on local/nf_cond_omsigc_n1024_n4096_aug_finetune (written by
               scripts/prepare_nf_cond_omsigc_n1024_n4096_aug_finetune.py), per training run:
                 * config differs from the baseline config only in io.output_dir, train.num_epochs,
                   train.checkpoint_every_n_epochs, augmentations (= the planned D4 + roll block);
                 * source = the baseline final checkpoint, no augmentations.pkl; output dir does not exist;
                 * the cosmodiff save rule (optim.py:580, start_epoch = source epoch + 1) saves every manifest
                   checkpoint epoch, with the manifest update count; sample files and eval dir do not exist.
               Build private runtimes (stock train / sample patch sets); then, per training run, run
               scripts/finetune_cosmodiff_resume.py --check-out for --check-epochs (default 1 epoch: 32 updates at
               N = 1024, 128 at N = 4096) on CPU from the real baseline checkpoint into a temporary dir and require
               the same resume properties as the N = 256 preflight (config pipeline used, every recorded fetch is a
               D4 x roll image of its stored map and none equals it, two fetches differ, labels unchanged,
               optimizer / scheduler counters continue the source by <= the check's updates, betas / wd kept,
               new checkpoint carries augmentations.pkl).
               Writes local/nf_cond_omsigc_n1024_n4096_aug_finetune/preflight_receipt.json (never overwrites) on PASS.
build-runtime  Used by the batch scripts with --verify-receipt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cosmodiff_private_runtime as cpr  # noqa: E402

LEDGER_RUN = "nf_cond_omsigc_n1024_n4096_aug_finetune"
LOCAL = ROOT / f"local/{LEDGER_RUN}"
MANIFEST = LOCAL / "manifest.json"
BASE_MANIFEST = ROOT / "local/nf_conditional_omsig_continuous_200k/manifest.json"
RECEIPT = LOCAL / "preflight_receipt.json"
EVAL_OUT = ROOT / f"results/{LEDGER_RUN}/evaluate_run_v1"
PATCH_SETS = dict(cpr.BASE_PATCH_SETS)
EXPECTED_AUG = {"RandomDihedral2D": {"dims": [-2, -1], "p": 1.0}, "RandomRoll": {"size": 128, "dims": [-2, -1]}}
ALLOWED_CONFIG_DIFF = {"io.output_dir", "train.num_epochs", "train.checkpoint_every_n_epochs", "augmentations"}


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and k != "augmentations":
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def training_runs(rows: list[dict]) -> list[str]:
    """Training runs in manifest order (= train array task order)."""
    seen = []
    for r in rows:
        if r["training_run_name"] not in seen:
            seen.append(r["training_run_name"])
    return seen


def static_checks() -> tuple[dict, list[str]]:
    import yaml
    from sample_nf_conditional_bias_probe import output_path_for

    fails = []
    rows = json.loads(MANIFEST.read_text())
    base = {r["run_name"]: r for r in json.loads(BASE_MANIFEST.read_text())}
    out = {"manifest": str(MANIFEST), "manifest_sha256": cpr.sha256(MANIFEST), "training_runs": {}, "rows": []}
    for run in training_runs(rows):
        grp = [r for r in rows if r["training_run_name"] == run]
        configs = {r["config"] for r in grp}
        srcs = {r["resume_from_checkpoint"] for r in grp}
        bases = {r["baseline_run_name"] for r in grp}
        if len(configs) != 1 or len(srcs) != 1 or len(bases) != 1:
            fails.append(f"{run}: rows must share one config, one source checkpoint and one baseline")
        cp = ROOT / sorted(configs)[0]
        b = base[sorted(bases)[0]]
        cfg, bcfg = yaml.safe_load(cp.read_text()), yaml.safe_load((ROOT / b["config"]).read_text())
        fc, fb = flatten(cfg), flatten(bcfg)
        diff = sorted(k for k in set(fc) | set(fb) if fc.get(k, "<missing>") != fb.get(k, "<missing>"))
        src = Path(sorted(srcs)[0])
        t = {"config": str(cp), "config_sha256": cpr.sha256(cp), "baseline_run_name": b["run_name"],
             "baseline_config": b["config"], "baseline_actual_updates": int(b["actual_updates"]),
             "config_diff": {k: [fb.get(k, "<missing>"), fc.get(k, "<missing>")] for k in diff},
             "source_checkpoint": str(src), "source_is_baseline_final": str(src) == b["requested_checkpoint"],
             "source_has_augmentations_pkl": (src / "augmentations.pkl").exists(),
             "output_dir": cfg["io"]["output_dir"], "output_dir_exists": os.path.exists(cfg["io"]["output_dir"])}
        if set(diff) != ALLOWED_CONFIG_DIFF or cfg.get("augmentations") != EXPECTED_AUG:
            fails.append(f"{run}: config diff {diff} / augmentations differ from the plan")
        if not t["source_is_baseline_final"] or t["source_has_augmentations_pkl"] or not src.is_dir():
            fails.append(f"{run}: source checkpoint is not the baseline final checkpoint without augmentations.pkl")
        if t["output_dir_exists"]:
            fails.append(f"{run}: output dir exists")
        spe, every = int(b["steps_per_epoch"]), int(cfg["train"]["checkpoint_every_n_epochs"])
        n_ep = int(cfg["train"]["num_epochs"])
        start = int(src.name.split("-")[-1]) + 1
        last = start + n_ep - 1
        saved = [e for e in range(start, start + n_ep) if (e + 1) % every == 0 or e == last]
        t["steps_per_epoch"], t["start_epoch"], t["last_epoch"] = spe, start, last
        t["checkpoint_epochs_that_will_be_saved"] = saved
        t["updates_planned"] = n_ep * spe
        out["training_runs"][run] = t
        for r in grp:
            ep = int(r["checkpoint_epoch"])
            sp = output_path_for(ROOT, r, 123, 64, None)
            rec = {"run_name": r["run_name"], "training_run_name": run, "checkpoint_epoch": ep,
                   "updates_after_source": (ep - start + 1) * spe, "requested_checkpoint": r["requested_checkpoint"],
                   "sample_path": str(sp), "sample_exists": sp.exists()}
            out["rows"].append(rec)
            if ep not in saved or rec["updates_after_source"] != r["finetune_updates"]:
                fails.append(f"{r['run_name']}: checkpoint epoch {ep} not saved or update count wrong")
            if r["requested_checkpoint"] != f"{cfg['io']['output_dir']}/checkpoint-epoch-{ep}" or sp.exists():
                fails.append(f"{r['run_name']}: checkpoint path inconsistent or sample exists")
        if sorted(int(r["checkpoint_epoch"]) for r in grp) != saved:
            fails.append(f"{run}: saved epochs {saved} != manifest epochs")
    out["eval_out_exists"] = EVAL_OUT.exists()
    if EVAL_OUT.exists():
        fails.append(f"{EVAL_OUT} exists")
    return out, fails


def check_report(rep: dict, base_updates: int, run: str) -> list[str]:
    f = []
    if rep["stock_load_checkpoint_augmentations"] != "None":
        f.append(f"{run}: stock loader returned augmentations (expected None)")
    if not rep["dataset_pipeline_after_train_is_config_pipeline"]:
        f.append(f"{run}: dataset pipeline after resume is not the config pipeline")
    n = rep["train_fetches_recorded"]
    if n == 0 or rep["train_fetches_in_d4_shift_orbit"] != n or rep["train_fetches_equal_stored"] != 0:
        f.append(f"{run}: training fetches are not D4 x roll images of the stored maps (or equal the stored map)")
    if not rep["post_fetch_index0_twice_differ"] or not rep["labels_unchanged"]:
        f.append(f"{run}: two fetches of index 0 are identical, or the label changed")
    s, nw, u = rep["source_checkpoint"], rep["new_checkpoint"], rep["updates_in_check"]
    # Source counters sit slightly below the batch count: accelerate does not step on fp16-overflow batches.
    if s["scheduler_last_epoch"] != int(s["adam_step"]) or not 0.99 * base_updates <= s["adam_step"] <= base_updates:
        f.append(f"{run}: source counters {s} inconsistent with ~{base_updates} updates")
    if nw["adam_step"] - s["adam_step"] > u or nw["adam_step"] - s["adam_step"] < 1 \
            or nw["scheduler_last_epoch"] - s["scheduler_last_epoch"] != nw["adam_step"] - s["adam_step"]:
        f.append(f"{run}: resumed counters {nw} do not continue the source counters {s} by <= {u} steps: "
                 "optimizer/scheduler state not restored")
    if (nw["weight_decay"], nw["betas"]) != (s["weight_decay"], s["betas"]):
        f.append(f"{run}: weight decay / betas changed on resume")
    if not nw["has_augmentations_pkl"]:
        f.append(f"{run}: new checkpoint has no augmentations.pkl")
    return f


def cmd_run(receipt: Path, check_epochs: int) -> int:
    if receipt.exists():
        raise SystemExit(f"Refusing to overwrite {receipt}")
    static, fails = static_checks()
    reps = {}
    with tempfile.TemporaryDirectory(prefix="ft_preflight_", dir=os.environ.get("TMPDIR")) as tmp:
        tmp = Path(tmp)
        runtimes = {ps: cpr.build_runtime(tmp / f"runtime_{ps}", pl) for ps, pl in PATCH_SETS.items()}
        rows = json.loads(MANIFEST.read_text())
        for i, run in enumerate(training_runs(rows)):
            row = next(r for r in rows if r["training_run_name"] == run)
            rep_path = tmp / f"check_{i}.json"
            cmd = [sys.executable, str(ROOT / "scripts/finetune_cosmodiff_resume.py"),
                   "--config", str(ROOT / row["config"]), "--resume-from", row["resume_from_checkpoint"],
                   "--check-out", str(rep_path), "--check-output-dir", str(tmp / f"check_out_{i}"),
                   "--check-epochs", str(check_epochs)]
            print(f"[preflight] resume check {run}", file=sys.stderr, flush=True)
            r = subprocess.run(cmd, env=cpr.runtime_env(tmp / "runtime_train"), capture_output=True, text=True)
            if r.returncode != 0 or not rep_path.exists():
                print(r.stdout[-4000:], r.stderr[-6000:], file=sys.stderr)
                fails.append(f"{run}: resume check failed rc={r.returncode}")
                reps[run] = {}
            else:
                reps[run] = json.loads(rep_path.read_text())
                fails += check_report(reps[run], static["training_runs"][run]["baseline_actual_updates"], run)
    for v in runtimes.values():
        v.pop("dest")
    report = {"status": "PASS" if not fails else "FAIL", "ledger_run": LEDGER_RUN,
              "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "host": socket.gethostname(), "python": sys.executable,
              "project_git_head": cpr.git(["rev-parse", "--short", "HEAD"], ROOT),
              "patch_sets": PATCH_SETS, "check_epochs": check_epochs, "fails": fails, "static": static,
              "runtime": runtimes, "resume_check": reps}
    text = json.dumps(report, indent=2)
    print(text)
    if fails:
        print(f"\nPREFLIGHT FAILED ({len(fails)}); receipt not written", file=sys.stderr)
        return 1
    with open(receipt, "x") as f:
        f.write(text + "\n")
    print(f"\nWrote {receipt}", file=sys.stderr)
    return 0


def cmd_build_runtime(dest: Path, patch_set: str, verify: str | None) -> int:
    prov = cpr.build_runtime(dest, PATCH_SETS[patch_set])
    print(json.dumps({k: v for k, v in prov.items() if k != "patches"}, indent=2))
    if verify:
        rec = json.loads(Path(verify).read_text())
        if rec.get("status") != "PASS":
            print(f"receipt {verify} is not PASS", file=sys.stderr)
            return 1
        bad = cpr.verify_against_receipt(prov, rec, patch_set)
        if bad:
            print(f"runtime differs from the preflight receipt: {bad}", file=sys.stderr)
            return 1
        print("runtime matches preflight receipt", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run")
    p.add_argument("--receipt", default=str(RECEIPT))
    p.add_argument("--check-epochs", type=int, default=1)
    p = sub.add_parser("static")
    p = sub.add_parser("build-runtime")
    p.add_argument("--dest", required=True)
    p.add_argument("--patch-set", choices=sorted(PATCH_SETS), required=True)
    p.add_argument("--verify-receipt")
    a = ap.parse_args()
    if a.cmd == "build-runtime":
        return cmd_build_runtime(Path(a.dest), a.patch_set, a.verify_receipt)
    if a.cmd == "static":
        static, fails = static_checks()
        print(json.dumps({"fails": fails, "static": static}, indent=2))
        return 1 if fails else 0
    return cmd_run(Path(getattr(a, "receipt", RECEIPT)), getattr(a, "check_epochs", 1))


if __name__ == "__main__":
    raise SystemExit(main())
