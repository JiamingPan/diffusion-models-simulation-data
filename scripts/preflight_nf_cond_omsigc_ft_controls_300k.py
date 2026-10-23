#!/usr/bin/env python
"""CPU preflight for ledger nf_cond_omsigc_ft_controls_300k (no GPU). Mirrors
scripts/preflight_nf_cond_omsigc_n1024_n4096_aug_finetune.py, extended to the three arms and two segments.

Needs the opt-in flags of scripts/finetune_cosmodiff_resume.py (--allow-no-augmentations,
--allow-checkpoint-augmentations); until that change is applied (APPROVE EDIT) the static check reports it as a FAIL.

static         Static checks on local/nf_cond_omsigc_ft_controls_300k (written by
               scripts/prepare_nf_cond_omsigc_ft_controls_300k.py), per training segment:
                 * config differs from the baseline config only in io.output_dir, train.num_epochs,
                   train.checkpoint_every_n_epochs (+ augmentations for B = identity block, C = D4 + roll block);
                   A has no augmentations key; C differs from the aug-arm config only in io.output_dir and
                   train.num_epochs;
                 * sources: seg1 = baseline final checkpoint without augmentations.pkl; seg2 (A, B) = the last
                   checkpoint of the matching seg1 output dir (must not exist yet); C = the aug-arm ft050k checkpoint
                   with augmentations.pkl and random_states_0.pkl;
                 * wrapper flags per segment; output dirs do not exist;
                 * the cosmodiff save rule (optim.py:580) saves exactly the manifest checkpoint epochs with the
                   manifest update counts; sample files and the eval dir do not exist;
                 * the fine-tune wrapper accepts the two opt-in flags.
run (default)  static + build private runtimes (stock train / sample patch sets) +
               (1) pipeline check of every config on EVERY training item through the stock ArrayDataset:
                   A: pipeline None, two reads identical and equal to the stored map;
                   B: identity pipeline, augmented image == stored image bitwise for all items, two reads identical;
                      records whether one call advances the CPU torch RNG (ledger: it should);
                   C: D4 x roll pipeline, first 256 items are D4 x roll images of the stored map, none equal;
               (2) a --check-epochs (default 1) CPU resume check per segment with the real flags:
                   seg1 A / B and C from the real sources; seg2 A / B CHAINED from the seg1 check's own checkpoint
                   (so the seg1 -> seg2 hand-off, including B's identity augmentations.pkl, is exercised);
                   requires: stock loader result as expected per arm, config pipeline used, fetches as in (1),
                   labels unchanged, optimizer / scheduler counters continue the source by <= the check's updates,
                   betas / wd kept, the new checkpoint name is source epoch + check epochs, augmentations.pkl present
                   (B, C) / absent (A).
               Writes local/nf_cond_omsigc_ft_controls_300k/preflight_receipt.json (never overwrites) on PASS.
pipeline-check Internal (runs inside the private runtime).
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
import prepare_nf_cond_omsigc_ft_controls_300k as prep  # noqa: E402

LEDGER_RUN = prep.LEDGER_RUN
LOCAL = prep.LOCAL
MANIFEST = LOCAL / "manifest.json"
RECEIPT = LOCAL / "preflight_receipt.json"
EVAL_OUT = ROOT / f"results/{LEDGER_RUN}/evaluate_run_v1"
WRAPPER = ROOT / "scripts/finetune_cosmodiff_resume.py"
PATCH_SETS = dict(cpr.BASE_PATCH_SETS)
BASE_DIFF = {"io.output_dir", "train.num_epochs", "train.checkpoint_every_n_epochs"}
EXPECTED_DESC = {  # pipeline_description() of the config pipelines
    "noaug": None,
    "idaug": [["RandomDihedral2D", {"dims": [-2, -1], "p": 0.0}],
              ["RandomRoll", {"dims": [-2, -1], "ndim": 2, "size": 1}]],
    "d4shift": [["RandomDihedral2D", {"dims": [-2, -1], "p": 1.0}],
                ["RandomRoll", {"dims": [-2, -1], "ndim": 2, "size": 128}]],
}


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and k != "augmentations":
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def cfg_diff(a: dict, b: dict) -> list[str]:
    fa, fb = flatten(a), flatten(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k, "<missing>") != fb.get(k, "<missing>"))


def wrapper_has_flags() -> bool:
    text = WRAPPER.read_text()
    return "--allow-no-augmentations" in text and "--allow-checkpoint-augmentations" in text


def training_runs(rows: list[dict]) -> list[str]:
    """Training runs ordered by train_task (= train array task id)."""
    tasks = {}
    for r in rows:
        tasks.setdefault(int(r["train_task"]), r["training_run_name"])
    if sorted(tasks) != list(range(len(tasks))):
        raise SystemExit(f"train_task ids {sorted(tasks)} are not 0..{len(tasks) - 1}")
    return [tasks[i] for i in range(len(tasks))]


def static_checks() -> tuple[dict, list[str]]:
    import yaml
    from sample_nf_conditional_bias_probe import output_path_for

    fails = []
    rows = json.loads(MANIFEST.read_text())
    segs = {s["training_run_name"]: s for s in prep.plan()}
    out = {"manifest": str(MANIFEST), "manifest_sha256": cpr.sha256(MANIFEST), "training_runs": {}, "rows": [],
           "wrapper": str(WRAPPER), "wrapper_sha256": cpr.sha256(WRAPPER), "wrapper_has_optin_flags": wrapper_has_flags()}
    if not out["wrapper_has_optin_flags"]:
        fails.append(f"{WRAPPER} lacks --allow-no-augmentations / --allow-checkpoint-augmentations "
                     "(proposed diff not applied)")
    order = training_runs(rows)
    if order != [s["training_run_name"] for s in prep.plan()]:
        fails.append(f"manifest training-run order {order} != prepare plan order")
    for run in order:
        grp = [r for r in rows if r["training_run_name"] == run]
        s = segs.get(run)
        if s is None:
            fails.append(f"{run}: not in the prepare plan")
            continue
        keys = ("config", "resume_from_checkpoint", "baseline_run_name", "arm", "segment", "train_task")
        if any(len({json.dumps(r[k]) for r in grp}) != 1 for k in keys + ("finetune_flags",)):
            fails.append(f"{run}: rows disagree on {keys} / finetune_flags")
        r0 = grp[0]
        cp = ROOT / r0["config"]
        cfg = yaml.safe_load(cp.read_text())
        bcfg = yaml.safe_load((ROOT / json_row(s["baseline_run_name"])["config"]).read_text())
        diff = cfg_diff(bcfg, cfg)
        src = Path(r0["resume_from_checkpoint"])
        arm, seg = r0["arm"], int(r0["segment"])
        t = {"train_task": int(r0["train_task"]), "arm": arm, "segment": seg, "dataset_size": s["dataset_size"],
             "config": str(cp), "config_sha256": cpr.sha256(cp), "baseline_run_name": r0["baseline_run_name"],
             "baseline_actual_updates": s["baseline_actual_updates"], "config_diff_vs_baseline": diff,
             "augmentations": cfg.get("augmentations", "<missing>"), "source_checkpoint": str(src),
             "source_exists": src.is_dir(), "source_has_augmentations_pkl": (src / "augmentations.pkl").exists(),
             "finetune_flags": r0["finetune_flags"], "depends_on_training_run": r0["depends_on_training_run"],
             "output_dir": cfg["io"]["output_dir"], "output_dir_exists": os.path.exists(cfg["io"]["output_dir"])}
        want_diff = BASE_DIFF | ({"augmentations"} if arm in ("idaug", "d4shift") else set())
        if set(diff) != want_diff:
            fails.append(f"{run}: config diff vs baseline {diff} != {sorted(want_diff)}")
        if arm == "noaug" and "augmentations" in cfg:
            fails.append(f"{run}: A config has an augmentations key")
        if arm in ("idaug", "d4shift") and cfg.get("augmentations") != prep.AUGS[arm]:
            fails.append(f"{run}: augmentations {cfg.get('augmentations')} != {prep.AUGS[arm]}")
        if arm == "d4shift":
            acfg = yaml.safe_load((ROOT / s["template_config"]).read_text())
            t["config_diff_vs_aug_arm"] = cfg_diff(acfg, cfg)
            if set(t["config_diff_vs_aug_arm"]) != {"io.output_dir", "train.num_epochs"}:
                fails.append(f"{run}: config diff vs aug-arm config {t['config_diff_vs_aug_arm']}")
        if r0["finetune_flags"] != prep.flags_for(arm, seg) or r0["finetune_flags"] != s["finetune_flags"]:
            fails.append(f"{run}: finetune_flags {r0['finetune_flags']} != plan {prep.flags_for(arm, seg)}")
        if str(src) != s["source_checkpoint"]:
            fails.append(f"{run}: source {src} != plan {s['source_checkpoint']}")
        if seg == 1:
            b = json_row(r0["baseline_run_name"])
            if str(src) != b["requested_checkpoint"] or not src.is_dir() or t["source_has_augmentations_pkl"]:
                fails.append(f"{run}: seg1 source is not the baseline final checkpoint without augmentations.pkl")
        elif arm == "d4shift":
            if not (src.is_dir() and t["source_has_augmentations_pkl"] and (src / "random_states_0.pkl").is_file()):
                fails.append(f"{run}: C source missing augmentations.pkl / random_states_0.pkl")
        else:
            dep = segs[r0["depends_on_training_run"]]
            if str(src) != f"{dep['output_dir']}/checkpoint-epoch-{dep['last_epoch']}":
                fails.append(f"{run}: seg2 source is not the last checkpoint of {dep['training_run_name']}")
            if src.exists():
                fails.append(f"{run}: seg2 source {src} already exists")
        if t["output_dir_exists"]:
            fails.append(f"{run}: output dir exists")
        spe, every = int(s["steps_per_epoch"]), int(cfg["train"]["checkpoint_every_n_epochs"])
        n_ep = int(cfg["train"]["num_epochs"])
        src_epoch = int(src.name.split("-")[-1])
        saved = prep.saved_epochs(src_epoch, n_ep, every)
        t.update(steps_per_epoch=spe, start_epoch=src_epoch + 1, last_epoch=saved[-1] if saved else None,
                 checkpoint_epochs_that_will_be_saved=saved, updates_planned=n_ep * spe)
        out["training_runs"][run] = t
        base_epoch = prep.BASES[s["dataset_size"]][2]
        for r in grp:
            ep = int(r["checkpoint_epoch"])
            sp = output_path_for(ROOT, r, 123, 64, None)
            rec = {"run_name": r["run_name"], "training_run_name": run, "checkpoint_epoch": ep,
                   "updates_after_baseline": (ep - base_epoch) * spe, "requested_checkpoint": r["requested_checkpoint"],
                   "sample_path": str(sp), "sample_exists": sp.exists()}
            out["rows"].append(rec)
            if ep not in saved or rec["updates_after_baseline"] != r["finetune_updates"]:
                fails.append(f"{r['run_name']}: checkpoint epoch {ep} not saved or update count wrong")
            if r["requested_checkpoint"] != f"{cfg['io']['output_dir']}/checkpoint-epoch-{ep}" or sp.exists():
                fails.append(f"{r['run_name']}: checkpoint path inconsistent or sample exists")
        if sorted(int(r["checkpoint_epoch"]) for r in grp) != saved:
            fails.append(f"{run}: saved epochs {saved} != manifest epochs")
    if len(rows) != 11 or len({r["run_name"] for r in rows}) != 11:
        fails.append(f"manifest has {len(rows)} rows / {len({r['run_name'] for r in rows})} names, expected 11")
    out["eval_out_exists"] = EVAL_OUT.exists()
    if EVAL_OUT.exists():
        fails.append(f"{EVAL_OUT} exists")
    return out, fails


_BASE_ROWS = None


def json_row(run_name: str) -> dict:
    global _BASE_ROWS
    if _BASE_ROWS is None:
        _BASE_ROWS = {r["run_name"]: r for r in json.loads(prep.BASE_MANIFEST.read_text())}
    return _BASE_ROWS[run_name]


def cmd_pipeline_check(config: str, arm: str, out: str, orbit_items: int) -> int:
    """Runs inside the private runtime: stock parse_config_data + ArrayDataset on every training item."""
    from simdiff_eval.torch_compat import install_torch_backend_compat

    install_torch_backend_compat(entry_point="scripts.preflight_nf_cond_omsigc_ft_controls_300k")
    import torch
    import yaml
    from cosmodiff.utils import parse_config_data
    from finetune_cosmodiff_resume import d4_shift_match, pipeline_description

    cfg = yaml.safe_load(Path(config).read_text())
    ds = parse_config_data(cfg)["data"]
    n = len(ds)
    rep = {"config": config, "arm": arm, "n_items": n, "pipeline": repr(ds.augmentations),
           "pipeline_description": pipeline_description(ds.augmentations)}
    eq1 = eq2 = same = 0
    for i in range(n):
        a, b = ds[i]["images"], ds[i]["images"]
        eq1 += bool(torch.equal(a, ds.arrays[i]))
        eq2 += bool(torch.equal(b, ds.arrays[i]))
        same += bool(torch.equal(a, b))
    rep.update(first_read_equal_stored=eq1, second_read_equal_stored=eq2, two_reads_identical=same)
    if ds.augmentations is not None:
        st = torch.get_rng_state()
        ds[0]
        rep["one_call_advances_cpu_rng"] = not torch.equal(st, torch.get_rng_state())
    if arm == "d4shift":
        m = [d4_shift_match(ds[i]["images"], ds.arrays[i]) for i in range(min(orbit_items, n))]
        rep["orbit_items_checked"] = len(m)
        rep["orbit_items_in_d4_shift_orbit"] = sum(x is not None for x in m)
    Path(out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


def check_pipeline(rep: dict, arm: str, run: str) -> list[str]:
    f = []
    n = rep.get("n_items", 0)
    if n == 0 or rep.get("pipeline_description") != EXPECTED_DESC[arm]:
        f.append(f"{run}: pipeline {rep.get('pipeline_description')} != expected {EXPECTED_DESC[arm]}")
    if arm in ("noaug", "idaug"):
        if not (rep.get("first_read_equal_stored") == rep.get("second_read_equal_stored")
                == rep.get("two_reads_identical") == n):
            f.append(f"{run}: not every item equals its stored map on two reads ({rep})")
    else:
        # One D4 x roll draw is the identity with probability 1/(8 * 128 * 128); allow <= 2 chance hits on all items.
        if rep.get("first_read_equal_stored", 99) > 2 or rep.get("two_reads_identical", 99) > 2:
            f.append(f"{run}: D4 x roll fetches equal the stored map or repeat more often than chance")
        if rep.get("orbit_items_in_d4_shift_orbit") != rep.get("orbit_items_checked"):
            f.append(f"{run}: D4 x roll fetches outside the orbit")
    return f


def check_report(rep: dict, base_updates: int, run: str, arm: str, src_epoch: int, check_epochs: int,
                 src_has_aug: bool) -> list[str]:
    f = []
    want_stock = EXPECTED_DESC[arm] if src_has_aug else None
    if rep.get("stock_load_checkpoint_augmentations_description") != want_stock:
        f.append(f"{run}: stock loader returned {rep.get('stock_load_checkpoint_augmentations_description')}, "
                 f"expected {want_stock}")
    if rep.get("config_pipeline_description") != EXPECTED_DESC[arm]:
        f.append(f"{run}: config pipeline {rep.get('config_pipeline_description')} != {EXPECTED_DESC[arm]}")
    if not rep["dataset_pipeline_after_train_is_config_pipeline"]:
        f.append(f"{run}: dataset pipeline after resume is not the config pipeline")
    n = rep["train_fetches_recorded"]
    if n == 0:
        f.append(f"{run}: no training fetches recorded")
    if arm in ("noaug", "idaug"):
        if rep["train_fetches_equal_stored"] != n or rep["post_fetch_index0_twice_differ"] \
                or rep["post_fetch_index0_maxabs_diff"] != 0.0:
            f.append(f"{run}: training fetches differ from the stored maps, or two fetches differ")
    else:
        if rep["train_fetches_in_d4_shift_orbit"] != n or rep["train_fetches_equal_stored"] != 0 \
                or not rep["post_fetch_index0_twice_differ"]:
            f.append(f"{run}: training fetches are not D4 x roll images of the stored maps (or equal it)")
    if not rep["labels_unchanged"]:
        f.append(f"{run}: label changed")
    s, nw, u = rep["source_checkpoint"], rep["new_checkpoint"], rep["updates_in_check"]
    # Source counters sit slightly below the batch count: accelerate does not step on fp16-overflow batches.
    if s["scheduler_last_epoch"] != int(s["adam_step"]) or not 0.99 * base_updates <= s["adam_step"] <= base_updates:
        f.append(f"{run}: source counters {s} inconsistent with ~{base_updates} updates")
    if nw["adam_step"] - s["adam_step"] > u or nw["adam_step"] - s["adam_step"] < 1 \
            or nw["scheduler_last_epoch"] - s["scheduler_last_epoch"] != nw["adam_step"] - s["adam_step"]:
        f.append(f"{run}: resumed counters {nw} do not continue the source counters {s} by <= {u} steps")
    if (nw["weight_decay"], nw["betas"]) != (s["weight_decay"], s["betas"]):
        f.append(f"{run}: weight decay / betas changed on resume")
    if nw["has_augmentations_pkl"] != (arm != "noaug"):
        f.append(f"{run}: new checkpoint augmentations.pkl present={nw['has_augmentations_pkl']} for arm {arm}")
    if rep["new_checkpoint_name"] != f"checkpoint-epoch-{src_epoch + check_epochs:04d}":
        f.append(f"{run}: new checkpoint {rep['new_checkpoint_name']} != epoch {src_epoch + check_epochs}")
    return f


def cmd_run(receipt: Path, check_epochs: int, orbit_items: int) -> int:
    if receipt.exists():
        raise SystemExit(f"Refusing to overwrite {receipt}")
    static, fails = static_checks()
    if not static["wrapper_has_optin_flags"]:
        print(json.dumps({"fails": fails}, indent=2))
        print("\nPREFLIGHT FAILED: apply the proposed wrapper diff first; receipt not written", file=sys.stderr)
        return 1
    rows = json.loads(MANIFEST.read_text())
    reps, pipes = {}, {}
    with tempfile.TemporaryDirectory(prefix="ftctl_preflight_", dir=os.environ.get("TMPDIR")) as tmp:
        tmp = Path(tmp)
        runtimes = {ps: cpr.build_runtime(tmp / f"runtime_{ps}", pl) for ps, pl in PATCH_SETS.items()}
        env = cpr.runtime_env(tmp / "runtime_train")
        check_ckpt = {}  # training run -> checkpoint written by its CPU check (seg2 chains from it)
        for i, run in enumerate(training_runs(rows)):
            t = static["training_runs"][run]
            row = next(r for r in rows if r["training_run_name"] == run)
            arm = t["arm"]
            if t["segment"] == 1 or arm == "d4shift":
                pj = tmp / f"pipe_{i}.json"
                cmd = [sys.executable, str(Path(__file__).resolve()), "pipeline-check", "--config",
                       str(ROOT / row["config"]), "--arm", arm, "--out", str(pj), "--orbit-items", str(orbit_items)]
                print(f"[preflight] pipeline check {run}", file=sys.stderr, flush=True)
                r = subprocess.run(cmd, env=env, capture_output=True, text=True)
                if r.returncode != 0 or not pj.exists():
                    print(r.stdout[-4000:], r.stderr[-6000:], file=sys.stderr)
                    fails.append(f"{run}: pipeline check failed rc={r.returncode}")
                    pipes[run] = {}
                else:
                    pipes[run] = json.loads(pj.read_text())
                    fails += check_pipeline(pipes[run], arm, run)
            if t["segment"] == 2 and arm != "d4shift":
                dep = t["depends_on_training_run"]
                if dep not in check_ckpt:
                    fails.append(f"{run}: no seg1 check checkpoint to chain from")
                    reps[run] = {}
                    continue
                src, base_updates = check_ckpt[dep]
                src_has_aug = arm == "idaug"
            else:
                src = Path(row["resume_from_checkpoint"])
                base_updates = int(t["baseline_actual_updates"])
                if arm == "d4shift":
                    base_updates += int(next(r for r in json.loads(prep.AUG_MANIFEST.read_text())
                                             if r["requested_checkpoint"] == str(src))["finetune_updates"])
                src_has_aug = t["source_has_augmentations_pkl"]
            rep_path, out_dir = tmp / f"check_{i}.json", tmp / f"check_out_{i}"
            cmd = [sys.executable, str(WRAPPER), "--config", str(ROOT / row["config"]), "--resume-from", str(src),
                   "--check-out", str(rep_path), "--check-output-dir", str(out_dir),
                   "--check-epochs", str(check_epochs), *row["finetune_flags"]]
            print(f"[preflight] resume check {run} from {src}", file=sys.stderr, flush=True)
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if r.returncode != 0 or not rep_path.exists():
                print(r.stdout[-4000:], r.stderr[-6000:], file=sys.stderr)
                fails.append(f"{run}: resume check failed rc={r.returncode}")
                reps[run] = {}
                continue
            reps[run] = json.loads(rep_path.read_text())
            reps[run]["chained_from_check_checkpoint"] = t["segment"] == 2 and arm != "d4shift"
            src_epoch = int(src.name.split("-")[-1])
            fails += check_report(reps[run], base_updates, run, arm, src_epoch, check_epochs, src_has_aug)
            check_ckpt[run] = (out_dir / reps[run]["new_checkpoint_name"],
                               base_updates + int(reps[run]["updates_in_check"]))
        # Negative controls: the wrapper must still refuse the mismatched mode (default behaviour unchanged).
        neg = {}
        a_cfg = ROOT / next(r for r in rows if r["arm"] == "noaug")["config"]
        a_src = next(r for r in rows if r["arm"] == "noaug")["resume_from_checkpoint"]
        c_row = next(r for r in rows if r["arm"] == "d4shift")
        for name, cmd in {
            "noaug_config_without_flag": [str(WRAPPER), "--config", str(a_cfg), "--resume-from", a_src],
            "aug_checkpoint_without_flag": [str(WRAPPER), "--config", str(ROOT / c_row["config"]),
                                            "--resume-from", c_row["resume_from_checkpoint"]],
            "aug_checkpoint_with_noaug_flag": [str(WRAPPER), "--config", str(ROOT / c_row["config"]),
                                               "--resume-from", c_row["resume_from_checkpoint"],
                                               "--allow-no-augmentations"],
        }.items():
            r = subprocess.run([sys.executable, *cmd, "--check-out", str(tmp / f"neg_{name}.json"),
                                "--check-output-dir", str(tmp / f"neg_out_{name}")],
                               env=env, capture_output=True, text=True)
            neg[name] = {"rc": r.returncode, "stderr_tail": r.stderr.strip()[-300:],
                         "output_dir_created": (tmp / f"neg_out_{name}").exists()}
            if r.returncode == 0 or neg[name]["output_dir_created"]:
                fails.append(f"negative control {name}: wrapper did not refuse")
    for v in runtimes.values():
        v.pop("dest")
    report = {"status": "PASS" if not fails else "FAIL", "ledger_run": LEDGER_RUN,
              "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "host": socket.gethostname(), "python": sys.executable,
              "project_git_head": cpr.git(["rev-parse", "--short", "HEAD"], ROOT),
              "patch_sets": PATCH_SETS, "check_epochs": check_epochs, "fails": fails, "static": static,
              "runtime": runtimes, "pipeline_check": pipes, "resume_check": reps, "negative_controls": neg}
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
        if cpr.sha256(WRAPPER) != rec["static"]["wrapper_sha256"]:
            print(f"{WRAPPER} changed since the preflight receipt", file=sys.stderr)
            return 1
        print("runtime and fine-tune wrapper match preflight receipt", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run")
    p.add_argument("--receipt", default=str(RECEIPT))
    p.add_argument("--check-epochs", type=int, default=1)
    p.add_argument("--orbit-items", type=int, default=256)
    sub.add_parser("static")
    p = sub.add_parser("pipeline-check")
    p.add_argument("--config", required=True)
    p.add_argument("--arm", choices=sorted(EXPECTED_DESC), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--orbit-items", type=int, default=256)
    p = sub.add_parser("build-runtime")
    p.add_argument("--dest", required=True)
    p.add_argument("--patch-set", choices=sorted(PATCH_SETS), required=True)
    p.add_argument("--verify-receipt")
    a = ap.parse_args()
    if a.cmd == "build-runtime":
        return cmd_build_runtime(Path(a.dest), a.patch_set, a.verify_receipt)
    if a.cmd == "pipeline-check":
        return cmd_pipeline_check(a.config, a.arm, a.out, a.orbit_items)
    if a.cmd == "static":
        static, fails = static_checks()
        print(json.dumps({"fails": fails, "static": static}, indent=2))
        return 1 if fails else 0
    return cmd_run(Path(getattr(a, "receipt", RECEIPT)), getattr(a, "check_epochs", 1), getattr(a, "orbit_items", 256))


if __name__ == "__main__":
    raise SystemExit(main())
