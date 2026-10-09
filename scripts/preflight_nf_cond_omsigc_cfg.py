#!/usr/bin/env python
"""CPU preflight for ledger nf_cond_omsigc_cfg (no GPU, no training, no sampling).

run            (default) Static checks on local/nf_cond_omsigc_cfg_200k (written by
               scripts/prepare_nf_cond_omsigc_cfg.py), build private patched runtimes (train / sample patch sets
               + cfg_null_flag) in a temporary dir, and in a child process under the train runtime check that
               parse_config_data / parse_config_model give what the configs promise. Writes
               local/nf_cond_omsigc_cfg_200k/preflight_receipt.json (never overwrites) only if all checks pass.
                 * each config differs from its baseline only in io.output_dir, data.label_path,
                   model.kwargs.encoder_hid_dim (3), train.cfg_dropout (0.1), train.cfg_null (flag),
                   generate.continuous_labels;
                 * label files = baseline labels with an appended zero column (train and held-out);
                 * dataset images identical to the baseline dataset (same file, same normalization), dataset labels
                   (N, 3) float32 with flag column 0; model UNet2DConditionModel with encoder_hid_proj 3 -> 32;
                 * patched train()/generate() take cfg_null; train() refuses the config with cfg_null removed;
                 * manifest rows = baseline rows + allowed changes; no checkpoint dir, sample file (45) or
                   evaluation dir exists.
build-runtime  Build a private runtime at --dest for --patch-set {train,sample}; with --verify-receipt exit 1
               unless its file hashes equal the receipt's. Used by the batch scripts.
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

LEDGER_RUN = "nf_cond_omsigc_cfg"
LOCAL = ROOT / "local/nf_cond_omsigc_cfg_200k"
MANIFEST = LOCAL / "manifest.json"
TASKS = LOCAL / "sample_tasks.json"
BASE_MANIFEST = ROOT / "local/nf_conditional_omsig_continuous_200k/manifest.json"
RECEIPT = LOCAL / "preflight_receipt.json"
EVAL_OUT = ROOT / "results/nf_cond_omsigc_cfg_200k/evaluate_run_v1"
SAMPLES_OUT = ROOT / "results/nf_cond_omsigc_cfg_200k/samples"
PATCH_SETS = {k: v + ["cfg_null_flag"] for k, v in cpr.BASE_PATCH_SETS.items()}
ALLOWED_CONFIG_DIFF = {"io.output_dir", "data.label_path", "model.kwargs.encoder_hid_dim", "train.cfg_dropout",
                       "train.cfg_null", "generate.continuous_labels"}
EXPECTED_VALUES = {"model.kwargs.encoder_hid_dim": 3, "train.cfg_dropout": 0.1, "train.cfg_null": "flag"}
ALLOWED_ROW_CHANGES = {"run_name", "config", "checkpoint_dir", "requested_checkpoint", "sample_path",
                       "train_label_path", "heldout_sample_params_norm_path", "note"}
ADDED_ROW_KEYS = {"baseline_run_name", "cfg_dropout", "cfg_null", "label_dim", "label_columns", "guidance_grid"}
TRAIN_ORDER = [256, 32768, 1024, 64, 4096]


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def static_checks() -> tuple[dict, list[str]]:
    import numpy as np
    import yaml
    from sample_nf_conditional_bias_probe import output_path_for

    fails = []
    rows = json.loads(MANIFEST.read_text())
    base = {r["run_name"]: r for r in json.loads(BASE_MANIFEST.read_text())}
    out = {"manifest": str(MANIFEST), "manifest_sha256": cpr.sha256(MANIFEST),
           "sample_tasks": str(TASKS), "sample_tasks_sha256": cpr.sha256(TASKS),
           "baseline_manifest": str(BASE_MANIFEST), "baseline_manifest_sha256": cpr.sha256(BASE_MANIFEST),
           "train_order": [int(r["dataset_size"]) for r in rows], "runs": []}
    if out["train_order"] != TRAIN_ORDER:
        fails.append(f"manifest order {out['train_order']} != {TRAIN_ORDER}")
    for r in rows:
        b = base.get(r.get("baseline_run_name"))
        rec = {"run_name": r["run_name"], "baseline_run_name": r.get("baseline_run_name")}
        out["runs"].append(rec)
        if b is None:
            fails.append(f"{r['run_name']}: baseline row missing")
            continue
        changed = sorted(k for k in set(r) & set(b) if r[k] != b[k])
        added, removed = sorted(set(r) - set(b)), sorted(set(b) - set(r))
        rec.update({"manifest_changed_keys": changed, "manifest_added_keys": added, "manifest_removed_keys": removed})
        if not set(changed) <= ALLOWED_ROW_CHANGES or set(added) != ADDED_ROW_KEYS or removed:
            fails.append(f"{r['run_name']}: manifest row differs beyond allowed keys")
        cp, bp = ROOT / r["config"], ROOT / b["config"]
        cfg, bcfg = yaml.safe_load(cp.read_text()), yaml.safe_load(bp.read_text())
        fc, fb = flatten(cfg), flatten(bcfg)
        diff = sorted(k for k in set(fc) | set(fb) if fc.get(k, "<missing>") != fb.get(k, "<missing>"))
        rec.update({"config": str(cp), "config_sha256": cpr.sha256(cp), "baseline_config": str(bp),
                    "baseline_config_sha256": cpr.sha256(bp), "config_diff_keys": diff,
                    "config_diff": {k: [fb.get(k, "<missing>"), fc.get(k, "<missing>")] for k in diff}})
        if set(diff) != ALLOWED_CONFIG_DIFF:
            fails.append(f"{r['run_name']}: config diff keys {diff} != {sorted(ALLOWED_CONFIG_DIFF)}")
        for k, v in EXPECTED_VALUES.items():
            if fc.get(k) != v:
                fails.append(f"{r['run_name']}: {k} = {fc.get(k)!r}, expected {v!r}")
        if fc["data.label_path"] != r["train_label_path"] or fc["generate.continuous_labels"] != r["heldout_sample_params_norm_path"]:
            fails.append(f"{r['run_name']}: config label paths != manifest label paths")
        if fc["io.output_dir"] != r["checkpoint_dir"]:
            fails.append(f"{r['run_name']}: io.output_dir != checkpoint_dir")
        if r["requested_checkpoint"] != f"{r['checkpoint_dir']}/checkpoint-epoch-{int(r['checkpoint_epoch'])}":
            fails.append(f"{r['run_name']}: requested_checkpoint inconsistent")
        if int(cfg["train"]["num_epochs"]) - 1 != int(r["checkpoint_epoch"]):
            fails.append(f"{r['run_name']}: checkpoint_epoch != num_epochs - 1")
        for new_key, old_key in (("train_label_path", "train_label_path"),
                                 ("heldout_sample_params_norm_path", "heldout_sample_params_norm_path")):
            a, o = np.load(r[new_key]), np.load(b[old_key])
            ok = a.shape == (o.shape[0], 3) and a.dtype == np.float32 and np.array_equal(a[:, :2], o) \
                and bool((a[:, 2] == 0).all())
            rec[f"{new_key}_ok"] = bool(ok)
            rec[f"{new_key}_sha256"] = cpr.sha256(r[new_key])
            if not ok:
                fails.append(f"{r['run_name']}: {new_key} is not baseline + zero flag column")
        exists = {"checkpoint_dir": os.path.exists(r["checkpoint_dir"])}
        for w in r["guidance_grid"]:
            exists[f"sample_{w}"] = output_path_for(ROOT, r, 123, 64, w).exists()
        rec["exists"] = exists
        if any(exists.values()):
            fails.append(f"{r['run_name']}: output exists {exists}")
        ev = LOCAL / f"eval_manifests/{r['run_name']}.json"
        rec["eval_manifest"], rec["eval_manifest_sha256"] = str(ev), cpr.sha256(ev)
        ev_rows = json.loads(ev.read_text())
        if [e["sample_path"] for e in ev_rows] != [str(output_path_for(ROOT, r, 123, 64, w).relative_to(ROOT))
                                                   for w in r["guidance_grid"]]:
            fails.append(f"{r['run_name']}: eval manifest sample paths != sampler paths")
    tasks = json.loads(TASKS.read_text())
    out["n_sample_tasks"] = len(tasks)
    if len(tasks) != 45 or len({t["sample_path"] for t in tasks}) != 45:
        fails.append("sample_tasks.json must list 45 distinct sample paths")
    for p in (EVAL_OUT, SAMPLES_OUT):
        out[f"exists:{p}"] = p.exists()
        if p.exists():
            fails.append(f"{p} exists")
    return out, fails


def child(out_json: Path) -> None:
    from simdiff_eval.torch_compat import install_torch_backend_compat

    install_torch_backend_compat(entry_point="scripts.preflight_nf_cond_omsigc_cfg.child")
    import inspect

    import torch
    import yaml

    import cosmodiff
    from cosmodiff import optim
    from cosmodiff.utils import parse_config_data, parse_config_model

    res = {"cosmodiff_file": cosmodiff.__file__, "runs": [], "fails": []}
    if not cosmodiff.__file__.startswith(os.environ["COSMODIFF_DIR"]):
        res["fails"].append("cosmodiff not imported from the private runtime")
    for fn in (optim.train, optim.generate):
        if "cfg_null" not in inspect.signature(fn).parameters:
            res["fails"].append(f"{fn.__name__} has no cfg_null parameter")
    base = {r["run_name"]: r for r in json.loads(BASE_MANIFEST.read_text())}
    for row in json.loads(MANIFEST.read_text()):
        cfg = yaml.safe_load((ROOT / row["config"]).read_text())
        bcfg = yaml.safe_load((ROOT / base[row["baseline_run_name"]]["config"]).read_text())
        new = parse_config_data(cfg)["data"]
        old = parse_config_data(bcfg)["data"]
        rec = {"run_name": row["run_name"], "n": len(new), "label_shape": list(new.labels.shape),
               "label_dtype": str(new.labels.dtype),
               "images_equal_baseline": bool(torch.equal(new.arrays, old.arrays)),
               "labels_first2_equal_baseline": bool(torch.equal(new.labels[:, :2], old.labels)),
               "flag_column_zero": bool((new.labels[:, 2] == 0).all()),
               "augmentations": repr(new.augmentations)}
        del old
        mcfg = json.loads(json.dumps(cfg))
        mcfg.setdefault("global", {})["device"] = "cpu"  # yaml says cuda; the preflight runs on a CPU node
        m = parse_config_model(mcfg)["model"]
        rec.update({"model_class": type(m).__name__, "encoder_hid_dim": int(m.config.encoder_hid_dim),
                    "encoder_hid_proj": [int(m.encoder_hid_proj.in_features), int(m.encoder_hid_proj.out_features)],
                    "n_params": int(sum(p.numel() for p in m.parameters()))})
        try:
            t = dict(cfg["train"])
            t.pop("cfg_null")
            t.update(num_epochs=0, force_cpu=True, mixed_precision="no")  # must raise before any work
            optim.train(new, m, output_dir="/nonexistent/should_not_be_created", **t)
            rec["train_refuses_without_cfg_null"] = False
        except ValueError:
            rec["train_refuses_without_cfg_null"] = True
        fails = [k for k in ("images_equal_baseline", "labels_first2_equal_baseline", "flag_column_zero",
                             "train_refuses_without_cfg_null") if not rec[k]]
        if rec["label_shape"] != [int(row["dataset_size"]), 3] or rec["label_dtype"] != "torch.float32":
            fails.append("labels not (N, 3) float32")
        if rec["encoder_hid_proj"] != [3, 32] or rec["model_class"] != "UNet2DConditionModel":
            fails.append("model is not UNet2DConditionModel with encoder_hid_proj 3 -> 32")
        if new.augmentations is not None:
            fails.append("unexpected augmentations")
        rec["fails"] = fails
        res["fails"] += [f"{row['run_name']}: {f}" for f in fails]
        res["runs"].append(rec)
        del new
    out_json.write_text(json.dumps(res, indent=2))


def cmd_run(receipt: Path) -> int:
    if receipt.exists():
        raise SystemExit(f"Refusing to overwrite {receipt}")
    static, fails = static_checks()
    with tempfile.TemporaryDirectory(prefix="cfg_preflight_") as tmp:
        tmp = Path(tmp)
        runtimes = {ps: cpr.build_runtime(tmp / f"runtime_{ps}", pl) for ps, pl in PATCH_SETS.items()}
        cj = tmp / "child.json"
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child", "--out", str(cj)],
                           env=cpr.runtime_env(tmp / "runtime_train"), capture_output=True, text=True)
        if r.returncode != 0 or not cj.exists():
            print(r.stdout[-4000:], r.stderr[-4000:], file=sys.stderr)
            fails.append(f"child failed rc={r.returncode}")
            dyn = {}
        else:
            dyn = json.loads(cj.read_text())
            fails += dyn.pop("fails")
    for v in runtimes.values():
        v.pop("dest")
    report = {"status": "PASS" if not fails else "FAIL", "ledger_run": LEDGER_RUN,
              "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "host": socket.gethostname(), "python": sys.executable,
              "project_git_head": cpr.git(["rev-parse", "--short", "HEAD"], ROOT),
              "patch_sets": PATCH_SETS, "fails": fails, "static": static, "runtime": runtimes, "torch_checks": dyn}
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
    if a.cmd == "build-runtime":
        return cmd_build_runtime(Path(a.dest), a.patch_set, a.verify_receipt)
    return cmd_run(Path(getattr(a, "receipt", RECEIPT)))


if __name__ == "__main__":
    raise SystemExit(main())
