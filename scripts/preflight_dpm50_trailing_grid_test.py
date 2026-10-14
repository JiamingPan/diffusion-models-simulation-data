#!/usr/bin/env python
"""CPU preflight for ledger dpm50_trailing_grid_test (no GPU, no sampling).

The only change against the original DPM-50 sampling of the three arms is the inference timestep_spacing
(leading -> trailing). The override is added by scripts/patch_cosmodiff_timestep_spacing.py to a PRIVATE runtime
copy of cosmodiff (scripts/cosmodiff_private_runtime.py); the shared cosmo_diffusion_main checkout is never patched.

Patch sets (everything else is the runtime the arm's leading-grid samples were drawn with):
  sample      stock sample patch set + timestep_spacing                  arms a, c (non-CFG checkpoints; their
              leading samples came from the same five patches applied in place by
              scripts/slurm/sample_nf_conditional_omsig_continuous_array.sbatch, job 62232390)
  sample_cfg  stock sample patch set + cfg_null_flag + timestep_spacing  arm b (CFG null-flag checkpoint; leading
              samples from scripts/slurm/sample_nf_cond_omsigc_cfg_array.sbatch, sample patch set + cfg_null_flag)

make-manifest  Write local/dpm50_trailing_grid_test/manifest.json from the reference manifest rows (refuses an
               existing file). Rows = reference rows with a new sample_path (results/dpm50_trailing_grid_test/samples/,
               spelled dpm50_trailing), guidance_grid [null] for the CFG row, and provenance keys.
run            (default) Static checks + runtime checks; writes local/dpm50_trailing_grid_test/preflight_receipt.json
               (never overwrites) only if all pass:
                 * ledger entry planned; manifest rows = reference rows + allowed changes; configs are the reference
                   configs; sample paths new, under results/dpm50_trailing_grid_test/samples/, spell dpm50_trailing;
                   reference samples exist; checkpoints exist and carry the expected scheduler config (leading, 500);
                   evaluation / re-score output dirs do not exist;
                 * scripts/sample_nf_conditional_bias_probe.py has the --timestep-spacing pass-through (needs the
                   proposed diff, APPROVE EDIT);
                 * reference runtimes (patch sets without timestep_spacing) hash-match the receipts the reference
                   samples' runtimes were verified against (CFG: local/nf_cond_omsigc_cfg_200k; stock sample set:
                   local/nf_cond_omsigc_n1024_n4096_aug_finetune), and the new runtimes differ from them only in
                   scripts/cosmodiff_sample.py;
                 * in each new runtime, for each arm's real checkpoint scheduler: default grid = [450, ..., 9]
                   (unchanged), trailing grid = [499, ..., 9], 50 steps.
build-runtime  Build a private runtime at --dest for --patch-set {sample,sample_cfg}; with --verify-receipt exit 1
               unless its file hashes equal the receipt's. Used by the sample batch script.
"""
from __future__ import annotations

import argparse
import copy
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

LEDGER_RUN = "dpm50_trailing_grid_test"
LOCAL = ROOT / f"local/{LEDGER_RUN}"
MANIFEST = LOCAL / "manifest.json"
RECEIPT = LOCAL / "preflight_receipt.json"
OUT = ROOT / f"results/{LEDGER_RUN}"
SAMPLES_REL = f"results/{LEDGER_RUN}/samples"
EVAL_OUT = OUT / "evaluate_run_trailing"
RESCORE_OUT = OUT / "evaluate_run_leading_ref_a"
SAMPLE_LABEL = "dpm50_trailing"
PATCH_SETS = {
    "sample": cpr.BASE_PATCH_SETS["sample"] + ["timestep_spacing"],
    "sample_cfg": cpr.BASE_PATCH_SETS["sample"] + ["cfg_null_flag", "timestep_spacing"],
}
REFERENCE_PATCH_SETS = {k: [p for p in v if p != "timestep_spacing"] for k, v in PATCH_SETS.items()}
REFERENCE_RECEIPTS = {  # receipt, runtime key whose patch list equals REFERENCE_PATCH_SETS[k]
    "sample": (ROOT / "local/nf_cond_omsigc_n1024_n4096_aug_finetune/preflight_receipt.json", "sample"),
    "sample_cfg": (ROOT / "local/nf_cond_omsigc_cfg_200k/preflight_receipt.json", "sample"),
}
CONT = "local/nf_conditional_omsig_continuous_200k/manifest.json"
CFG = "local/nf_cond_omsigc_cfg_200k/manifest.json"
ARMS = [
    {"arm": "a", "run_name": "nf_cond_omsigc_hi_u128_d2p15_n32768_fresh200k", "reference_manifest": CONT,
     "patch_set": "sample", "reference_eval": f"results/{LEDGER_RUN}/evaluate_run_leading_ref_a",
     "reference_eval_run_name": "nf_cond_omsigc_hi_u128_d2p15_n32768_fresh200k"},
    {"arm": "b", "run_name": "nf_cond_omsigc_hi_u128_d2p15_n32768_cfg10flag_fresh200k", "reference_manifest": CFG,
     "patch_set": "sample_cfg",
     "reference_eval": "results/nf_cond_omsigc_cfg_200k/evaluate_run_v1/nf_cond_omsigc_hi_u128_d2p15_n32768_cfg10flag_fresh200k",
     "reference_eval_run_name": "nf_cond_omsigc_hi_u128_d2p15_n32768_cfg10flag_fresh200k__noguidance"},
    {"arm": "c", "run_name": "nf_cond_omsigc_hi_u128_d2p12_n4096_fresh200k", "reference_manifest": CONT,
     "patch_set": "sample", "reference_eval": "results/nf_cond_omsigc_n1024_n4096_aug_finetune/evaluate_run_plus0",
     "reference_eval_run_name": "nf_cond_omsigc_hi_u128_d2p12_n4096_fresh200k"},
]
ALLOWED_ROW_CHANGES = {"sample_path", "guidance_grid", "note"}
ADDED_ROW_KEYS = {"ledger_run", "arm", "reference_manifest", "reference_sample_path", "reference_eval",
                  "reference_eval_run_name", "timestep_spacing", "sample_label", "runtime_patch_set"}
EXPECTED_SCHED = {"_class_name": "DDPMScheduler", "num_train_timesteps": 500, "beta_schedule": "squaredcos_cap_v2",
                  "rescale_betas_zero_snr": True, "prediction_type": "v_prediction", "timestep_spacing": "leading",
                  "steps_offset": 0}
LEADING_50 = [450 - 9 * i for i in range(50)]
TRAILING_50 = [499 - 10 * i for i in range(50)]


def ckpt_dir(row: dict) -> Path:
    """Final checkpoint dir; training zero-pads epochs below 1000 (manifest may hold the unpadded name)."""
    p = Path(row["requested_checkpoint"])
    if p.is_dir():
        return p
    head, ep = str(p).rsplit("-", 1)
    return Path(f"{head}-{int(ep):04d}")


def reference_rows() -> dict:
    out = {}
    for arm in ARMS:
        rows = {r["run_name"]: r for r in json.loads((ROOT / arm["reference_manifest"]).read_text())}
        out[arm["run_name"]] = rows[arm["run_name"]]
    return out


def cmd_make_manifest() -> int:
    from sample_nf_conditional_bias_probe import output_path_for

    if MANIFEST.exists():
        raise SystemExit(f"Refusing to overwrite {MANIFEST}")
    refs = reference_rows()
    rows = []
    for arm in ARMS:
        ref = refs[arm["run_name"]]
        row = copy.deepcopy(ref)
        suffix = "_{guidance}" if "{guidance}" in ref["sample_path"] else ""
        row["sample_path"] = (f"{SAMPLES_REL}/{arm['run_name']}_seed{{seed}}_{SAMPLE_LABEL}_heldout_k{{k}}{suffix}.npz")
        if "guidance_grid" in ref:
            row["guidance_grid"] = [None]
        row["note"] = (f"{ref.get('note', '')} Ledger {LEDGER_RUN} arm {arm['arm']}: same checkpoint, labels, seed 123, "
                       "k = 64, DPMSolverMultistep 50 steps, batch 8, raw weights, no guidance; only change "
                       "timestep_spacing trailing.").strip()
        row.update({"ledger_run": LEDGER_RUN, "arm": arm["arm"], "reference_manifest": arm["reference_manifest"],
                    "reference_sample_path": str(output_path_for(ROOT, ref, 123, 64, None).relative_to(ROOT)),
                    "reference_eval": arm["reference_eval"], "reference_eval_run_name": arm["reference_eval_run_name"],
                    "timestep_spacing": "trailing", "sample_label": SAMPLE_LABEL,
                    "runtime_patch_set": arm["patch_set"]})
        rows.append(row)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "x") as f:
        f.write(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {MANIFEST} ({len(rows)} rows)")
    return 0


def static_checks() -> tuple[dict, list[str]]:
    from sample_nf_conditional_bias_probe import output_path_for
    import sample_nf_conditional_bias_probe as wrapper

    fails = []
    led = subprocess.run([sys.executable, str(ROOT / "scripts/ledger.py"), "check-planned", "--run", LEDGER_RUN],
                         capture_output=True, text=True)
    out = {"ledger_check_planned_rc": led.returncode, "manifest": str(MANIFEST), "manifest_sha256": cpr.sha256(MANIFEST),
           "wrapper": str(ROOT / "scripts/sample_nf_conditional_bias_probe.py"),
           "wrapper_sha256": cpr.sha256(ROOT / "scripts/sample_nf_conditional_bias_probe.py"), "arms": []}
    if led.returncode != 0:
        fails.append(f"ledger entry {LEDGER_RUN} is not planned: {led.stdout.strip()} {led.stderr.strip()}")
    if not hasattr(wrapper, "timestep_spacing_cli_args"):
        fails.append("scripts/sample_nf_conditional_bias_probe.py has no --timestep-spacing pass-through "
                     "(apply the proposed diff after APPROVE EDIT)")
    elif wrapper.timestep_spacing_cli_args(None) != [] or \
            wrapper.timestep_spacing_cli_args("trailing") != ["--timestep_spacing", "trailing"]:
        fails.append("wrapper timestep_spacing_cli_args does not behave as specified")
    rows = json.loads(MANIFEST.read_text())
    refs = reference_rows()
    if [r["run_name"] for r in rows] != [a["run_name"] for a in ARMS]:
        fails.append("manifest run order != arms a, b, c")
    for row, arm in zip(rows, ARMS):
        ref = refs[arm["run_name"]]
        rec = {"arm": arm["arm"], "run_name": row["run_name"], "patch_set": row.get("runtime_patch_set")}
        out["arms"].append(rec)
        changed = sorted(k for k in set(row) & set(ref) if row[k] != ref[k])
        added, removed = sorted(set(row) - set(ref)), sorted(set(ref) - set(row))
        rec.update({"changed_keys": changed, "added_keys": added, "removed_keys": removed})
        if not set(changed) <= ALLOWED_ROW_CHANGES or set(added) != ADDED_ROW_KEYS or removed:
            fails.append(f"{arm['arm']}: manifest row differs from the reference beyond allowed keys")
        if row.get("runtime_patch_set") != arm["patch_set"] or row.get("timestep_spacing") != "trailing":
            fails.append(f"{arm['arm']}: patch set / timestep_spacing fields wrong")
        cp = ROOT / row["config"]
        rec["config"], rec["config_sha256"] = str(cp), cpr.sha256(cp)
        new = output_path_for(ROOT, row, 123, 64, None)
        old = output_path_for(ROOT, ref, 123, 64, None)
        rec.update({"sample_path": str(new), "reference_sample_path": str(old),
                    "reference_sample_exists": old.exists()})
        if not str(new.relative_to(ROOT)).startswith(SAMPLES_REL + "/") or f"_{SAMPLE_LABEL}_" not in new.name:
            fails.append(f"{arm['arm']}: sample path {new} not under {SAMPLES_REL} or not spelled {SAMPLE_LABEL}")
        if new.exists():
            fails.append(f"{arm['arm']}: sample file exists {new}")
        if not old.exists():
            fails.append(f"{arm['arm']}: reference sample missing {old}")
        if row["reference_sample_path"] != str(old.relative_to(ROOT)):
            fails.append(f"{arm['arm']}: reference_sample_path field != reference manifest path")
        ck = ckpt_dir(row)
        rec["checkpoint"] = str(ck)
        sc = ck / "scheduler_config.json"
        if not sc.is_file():
            fails.append(f"{arm['arm']}: checkpoint scheduler config missing {sc}")
            continue
        sched = json.loads(sc.read_text())
        rec["scheduler_config_sha256"] = cpr.sha256(sc)
        bad = {k: sched.get(k) for k, v in EXPECTED_SCHED.items() if sched.get(k) != v}
        if bad:
            fails.append(f"{arm['arm']}: checkpoint scheduler config differs: {bad}")
    # The arm-a re-score (CPU, existing leading samples) may run before or after this preflight: recorded only.
    for p in (EVAL_OUT, RESCORE_OUT):
        out[f"exists:{p}"] = p.exists()
    if EVAL_OUT.exists():
        fails.append(f"{EVAL_OUT} exists")
    return out, fails


def child(out_json: Path) -> None:
    """Runs inside a new private runtime: grids for each arm's real checkpoint scheduler."""
    import importlib.util

    from simdiff_eval.torch_compat import install_torch_backend_compat

    install_torch_backend_compat(entry_point="scripts.preflight_dpm50_trailing_grid_test.child")
    from diffusers import DDPMScheduler, DPMSolverMultistepScheduler

    import cosmodiff

    rt = Path(os.environ["COSMODIFF_DIR"])
    script = rt / "scripts/cosmodiff_sample.py"
    spec = importlib.util.spec_from_file_location("patched_cosmodiff_sample", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    patch_set = os.environ["DPM50_TRAILING_PATCH_SET"]
    res = {"cosmodiff_file": cosmodiff.__file__, "patch_set": patch_set, "arms": [], "fails": []}
    if not cosmodiff.__file__.startswith(str(rt)):
        res["fails"].append("cosmodiff not imported from the private runtime")
    if ("cfg_null=_cfg_null" in script.read_text()) != (patch_set == "sample_cfg"):
        res["fails"].append("cfg_null pass-through presence does not match the patch set")
    rows = json.loads(MANIFEST.read_text())
    for row in rows:
        if row["runtime_patch_set"] != patch_set:
            continue
        ddpm = DDPMScheduler.from_pretrained(str(ckpt_dir(row)))
        stock = DPMSolverMultistepScheduler.from_config(ddpm.config)
        stock.set_timesteps(50)
        grids = {}
        for name, ov in (("default", None), ("trailing", "trailing")):
            s = DPMSolverMultistepScheduler.from_config(ddpm.config, **mod.timestep_spacing_kwargs(ov))
            grids[name] = mod.inference_timestep_grid(s, 50)
        rec = {"arm": row["arm"], "run_name": row["run_name"], "stock_first_last": [int(stock.timesteps[0]),
                                                                                    int(stock.timesteps[-1])],
               "default_first_last_n": [grids["default"][0], grids["default"][-1], len(grids["default"])],
               "trailing_first_last_n": [grids["trailing"][0], grids["trailing"][-1], len(grids["trailing"])]}
        if [int(t) for t in stock.timesteps] != LEADING_50 or grids["default"] != LEADING_50:
            res["fails"].append(f"{row['arm']}: default grid is not the stock leading grid")
        if grids["trailing"] != TRAILING_50:
            res["fails"].append(f"{row['arm']}: trailing grid != [499, ..., 9]")
        res["arms"].append(rec)
    out_json.write_text(json.dumps(res, indent=2))


def cmd_run(receipt: Path) -> int:
    if receipt.exists():
        raise SystemExit(f"Refusing to overwrite {receipt}")
    static, fails = static_checks()
    runtimes, ref_runtimes, dyn = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="dpm50_trailing_preflight_") as tmp:
        tmp = Path(tmp)
        for ps in PATCH_SETS:
            ref_runtimes[ps] = cpr.build_runtime(tmp / f"ref_{ps}", REFERENCE_PATCH_SETS[ps])
            runtimes[ps] = cpr.build_runtime(tmp / f"runtime_{ps}", PATCH_SETS[ps])
            rpath, rkey = REFERENCE_RECEIPTS[ps]
            rrec = json.loads(rpath.read_text())
            if rrec["runtime"][rkey]["patch_list"] != REFERENCE_PATCH_SETS[ps]:
                fails.append(f"{ps}: reference receipt {rpath} patch list differs")
            bad = cpr.verify_against_receipt(ref_runtimes[ps], rrec, rkey)
            if bad:
                fails.append(f"{ps}: reference runtime differs from {rpath} in {bad}")
            h_new, h_ref = runtimes[ps]["sha256"], ref_runtimes[ps]["sha256"]
            diff = sorted(k for k in set(h_new) | set(h_ref) if h_new.get(k) != h_ref.get(k))
            runtimes[ps]["differs_from_reference_runtime"] = diff
            if diff != ["scripts/cosmodiff_sample.py"]:
                fails.append(f"{ps}: new runtime differs from the reference runtime in {diff}")
            cj = tmp / f"child_{ps}.json"
            env = cpr.runtime_env(tmp / f"runtime_{ps}")
            env["DPM50_TRAILING_PATCH_SET"] = ps
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child", "--out", str(cj)],
                               env=env, capture_output=True, text=True)
            if r.returncode != 0 or not cj.exists():
                print(r.stdout[-4000:], r.stderr[-4000:], file=sys.stderr)
                fails.append(f"{ps}: child failed rc={r.returncode}")
                continue
            dyn[ps] = json.loads(cj.read_text())
            fails += [f"{ps}: {f}" for f in dyn[ps].pop("fails")]
    for v in list(runtimes.values()) + list(ref_runtimes.values()):
        v.pop("dest", None)
    report = {"status": "PASS" if not fails else "FAIL", "ledger_run": LEDGER_RUN,
              "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "host": socket.gethostname(), "python": sys.executable,
              "project_git_head": cpr.git(["rev-parse", "--short", "HEAD"], ROOT),
              "patch_sets": PATCH_SETS, "reference_patch_sets": REFERENCE_PATCH_SETS,
              "reference_receipts": {k: [str(p), key] for k, (p, key) in REFERENCE_RECEIPTS.items()},
              "fails": fails, "static": static, "runtime": runtimes, "reference_runtime": ref_runtimes,
              "grid_checks": dyn}
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
    sub.add_parser("make-manifest")
    p = sub.add_parser("build-runtime")
    p.add_argument("--dest", required=True)
    p.add_argument("--patch-set", choices=sorted(PATCH_SETS), required=True)
    p.add_argument("--verify-receipt")
    p = sub.add_parser("child")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "child":
        child(Path(a.out))
        return 0
    if a.cmd == "make-manifest":
        return cmd_make_manifest()
    if a.cmd == "build-runtime":
        return cmd_build_runtime(Path(a.dest), a.patch_set, a.verify_receipt)
    return cmd_run(Path(getattr(a, "receipt", RECEIPT)))


if __name__ == "__main__":
    raise SystemExit(main())
