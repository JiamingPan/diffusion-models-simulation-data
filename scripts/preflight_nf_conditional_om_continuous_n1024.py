#!/usr/bin/env python
"""CPU preflight for sampling + evaluating the Omega_m-only conditional model at N = 1024
(nf_cond_omc_hi_u128_d2p10_n1024_fresh200k). No GPU, no sampling, no training.

Subcommands
-----------
run            (default) Check, without touching the shared venv or the shared cosmodiff checkout:
                 * manifest row RUN_INDEX is RUN_NAME with dataset_size 1024, and the same name is at the
                   same index of `prepare_nf_conditional_label_sweep_configs.py --variant om_continuous
                   --print-runs`;
                 * the final checkpoint (requested_checkpoint) exists and loads as UNet2DConditionModel;
                 * the seed-123 k=64 sample file and the eval output dir do not exist;
                 * a private patched cosmodiff copy (patch set "sample", same five patches as
                   sample_nf_conditional_om_continuous_array.sbatch) builds, cosmodiff imports from it,
                   cosmodiff_sample.py --help runs under it, and the sampling wrapper imports;
                 * pyvenv.cfg and 00-cosmodiff-base-venv.pth of the venv are byte-identical with the
                   same mtime before and after this preflight.
               Writes the receipt (never overwrites) only if every check passes; prints it either way.
build-runtime  Build the private runtime at --dest (refuses an existing --dest) and, with
               --verify-receipt, exit 1 unless its cosmodiff file hashes equal the receipt's.
               Used by sample_nf_conditional_om_continuous_n1024_safe.sbatch.
child          Internal: torch-side checks, run under the private runtime.
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# Private-runtime builder shared with the d4shift launchers (copies cosmo_diffusion_main sources to
# --dest and patches only the copy).
from preflight_nf_cond_omsigc_d4shift import (  # noqa: E402
    COSMODIFF_SRC,
    build_runtime,
    git,
    runtime_env,
    runtime_hashes,
    sha256,
)

LOCAL = ROOT / "local/nf_conditional_om_continuous_200k"
MANIFEST = LOCAL / "manifest.json"
RECEIPT = LOCAL / "preflight_n1024_safe_receipt.json"
RUN_NAME = "nf_cond_omc_hi_u128_d2p10_n1024_fresh200k"
RUN_INDEX = 2
DATASET_SIZE = 1024
EVAL_OUT = ROOT / "results/nf_conditional_om_continuous_200k/calibration_vgg_n1024"
D4SHIFT_RECEIPT = ROOT / "local/nf_cond_omsigc_d4shift_200k/preflight_receipt.json"
VENV = Path("/home/jiamingp/venvs/cosmodiff_nf_class")
VENV_FILES = [VENV / "pyvenv.cfg", VENV / "lib/python3.10/site-packages/00-cosmodiff-base-venv.pth"]
PATCH_SET = "sample"
SEED = 123
K = 64


def venv_state() -> dict:
    out = {}
    for f in VENV_FILES:
        st = f.stat()
        out[str(f)] = {"sha256": sha256(f), "mtime_ns": st.st_mtime_ns, "size": st.st_size}
    return out


def static_checks() -> tuple[dict, list[str]]:
    from sample_nf_conditional_bias_probe import output_path_for

    fails = []
    rows = json.loads(MANIFEST.read_text())
    printed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_nf_conditional_label_sweep_configs.py"),
         "--variant", "om_continuous", "--print-runs"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    row = rows[RUN_INDEX]
    out = {"manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
           "print_runs": printed, "manifest_run_names": [r["run_name"] for r in rows],
           "run_index": RUN_INDEX, "row": row}
    if row["run_name"] != RUN_NAME or int(row["dataset_size"]) != DATASET_SIZE:
        fails.append(f"manifest row {RUN_INDEX} is {row['run_name']} N={row['dataset_size']}")
    if len(printed) <= RUN_INDEX or printed[RUN_INDEX] != RUN_NAME:
        fails.append(f"--print-runs index {RUN_INDEX} is not {RUN_NAME}: {printed}")
    if printed != out["manifest_run_names"]:
        fails.append("--print-runs order differs from manifest order")
    ckpt = Path(row["requested_checkpoint"])
    out["checkpoint"] = str(ckpt)
    out["checkpoint_files"] = sorted(p.name for p in ckpt.iterdir()) if ckpt.is_dir() else []
    if not (ckpt / "diffusion_pytorch_model.safetensors").is_file():
        fails.append(f"checkpoint weights missing: {ckpt}")
    else:
        out["checkpoint_weights_sha256"] = sha256(ckpt / "diffusion_pytorch_model.safetensors")
    cfg_path = ROOT / row["config"]
    out["config"] = str(cfg_path)
    out["config_sha256"] = sha256(cfg_path)
    sample = output_path_for(ROOT, row, SEED, K, None)
    out["sample_path"] = str(sample)
    out["sample_exists"] = sample.exists()
    out["eval_out_dir"] = str(EVAL_OUT)
    out["eval_out_dir_exists"] = EVAL_OUT.exists()
    if sample.exists():
        fails.append(f"sample file exists: {sample}")
    if EVAL_OUT.exists():
        fails.append(f"eval dir exists: {EVAL_OUT}")
    for p in ("heldout_indices_path", "heldout_sample_params_norm_path", "heldout_raw_params_path"):
        if not Path(row[p]).is_file():
            fails.append(f"{p} missing: {row[p]}")
    return out, fails


def child(out_json: Path) -> None:
    from simdiff_eval.torch_compat import install_torch_backend_compat

    install_torch_backend_compat(entry_point="scripts.preflight_nf_conditional_om_continuous_n1024.child")

    import diffusers
    import torch
    from diffusers import DPMSolverMultistepScheduler, UNet2DConditionModel

    import cosmodiff
    import sample_nf_conditional_bias_probe as snb

    rt = Path(os.environ["COSMODIFF_DIR"]).resolve()
    res = {"cosmodiff_file": cosmodiff.__file__, "sample_wrapper_file": snb.__file__,
           "torch": torch.__version__, "diffusers": diffusers.__version__,
           "dpm_scheduler": DPMSolverMultistepScheduler.__name__, "fails": []}
    if not str(Path(cosmodiff.__file__).resolve()).startswith(str(rt)):
        res["fails"].append("cosmodiff not imported from the private runtime")
    row = json.loads(MANIFEST.read_text())[RUN_INDEX]
    model = UNet2DConditionModel.from_pretrained(row["requested_checkpoint"])
    res["unet_class"] = type(model).__name__
    res["unet_n_params"] = int(sum(p.numel() for p in model.parameters()))
    res["unet_encoder_hid_dim"] = model.config.get("encoder_hid_dim")
    res["unet_cross_attention_dim"] = model.config.get("cross_attention_dim")
    help_run = subprocess.run([sys.executable, str(rt / "scripts/cosmodiff_sample.py"), "--help"],
                              capture_output=True, text=True)
    res["cosmodiff_sample_help_rc"] = help_run.returncode
    res["cosmodiff_sample_help_head"] = help_run.stdout[:600]
    if help_run.returncode != 0:
        res["fails"].append(f"cosmodiff_sample.py --help rc={help_run.returncode}: {help_run.stderr[-800:]}")
    for flag in ("--checkpoint_epoch", "--scheduler", "--num_steps", "--seed", "--filepath"):
        if flag not in help_run.stdout:
            res["fails"].append(f"cosmodiff_sample.py --help lacks {flag}")
    out_json.write_text(json.dumps(res, indent=2))


def cmd_run(args) -> int:
    receipt = Path(args.receipt)
    if receipt.exists():
        raise SystemExit(f"Refusing to overwrite {receipt}")
    venv_before = venv_state()
    static, fails = static_checks()
    with tempfile.TemporaryDirectory(prefix="omc_n1024_preflight_") as tmp:
        tmp = Path(tmp)
        runtime = build_runtime(tmp / "runtime_sample", PATCH_SET)
        child_json = tmp / "child.json"
        env = runtime_env(tmp / "runtime_sample")
        env["OMP_NUM_THREADS"] = "2"
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child", "--out", str(child_json)],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0 or not child_json.exists():
            print(r.stdout[-4000:], r.stderr[-4000:], file=sys.stderr)
            fails.append(f"child failed rc={r.returncode}")
            dyn = {}
        else:
            dyn = json.loads(child_json.read_text())
            fails += dyn.pop("fails")
    runtime.pop("dest")
    shared_now = runtime_hashes(COSMODIFF_SRC)
    priv = runtime["sha256"]
    compare = {"private_vs_shared_checkout_differs": sorted(
        k for k in set(priv) | set(shared_now) if priv.get(k) != shared_now.get(k))}
    if D4SHIFT_RECEIPT.exists():
        d4 = json.loads(D4SHIFT_RECEIPT.read_text())["runtime"][PATCH_SET]["sha256"]
        compare["private_vs_d4shift_receipt_sample_differs"] = sorted(
            k for k in set(priv) | set(d4) if priv.get(k) != d4.get(k))
    venv_after = venv_state()
    if venv_after != venv_before:
        fails.append("venv pyvenv.cfg / 00-cosmodiff-base-venv.pth changed during preflight")
    report = {
        "status": "PASS" if not fails else "FAIL",
        "purpose": "sample + evaluate nf_cond_omc_hi_u128_d2p10_n1024_fresh200k (Omega_m-only, N=1024)",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": sys.executable,
        "project_git_head": git(["rev-parse", "--short", "HEAD"], ROOT),
        "seed": SEED,
        "samples_per_cosmology": K,
        "fails": fails,
        "static": static,
        "runtime": {PATCH_SET: runtime},
        "runtime_comparison": compare,
        "torch_checks": dyn,
        "venv_before": venv_before,
        "venv_after": venv_after,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if fails:
        print(f"\nPREFLIGHT FAILED ({len(fails)}); receipt not written", file=sys.stderr)
        return 1
    with open(receipt, "x") as f:
        f.write(text + "\n")
    print(f"\nWrote {receipt}", file=sys.stderr)
    return 0


def cmd_build_runtime(args) -> int:
    prov = build_runtime(Path(args.dest), PATCH_SET)
    print(json.dumps({k: v for k, v in prov.items() if k != "patches"}, indent=2))
    if args.verify_receipt:
        rec = json.loads(Path(args.verify_receipt).read_text())
        if rec.get("status") != "PASS":
            print(f"receipt {args.verify_receipt} status is not PASS", file=sys.stderr)
            return 1
        want = rec["runtime"][PATCH_SET]["sha256"]
        if prov["sha256"] != want:
            bad = sorted(k for k in set(want) | set(prov["sha256"]) if want.get(k) != prov["sha256"].get(k))
            print(f"cosmodiff runtime differs from the preflight receipt: {bad}", file=sys.stderr)
            return 1
        print("runtime matches preflight receipt", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p_run = sub.add_parser("run")
    p_run.add_argument("--receipt", default=str(RECEIPT))
    p_b = sub.add_parser("build-runtime")
    p_b.add_argument("--dest", required=True)
    p_b.add_argument("--verify-receipt")
    p_c = sub.add_parser("child")
    p_c.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "child":
        child(Path(args.out))
        return 0
    if args.cmd == "build-runtime":
        return cmd_build_runtime(args)
    if args.cmd is None:
        args.receipt = str(RECEIPT)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
