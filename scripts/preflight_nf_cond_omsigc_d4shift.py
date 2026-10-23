#!/usr/bin/env python
"""CPU preflight for ledger entry nf_cond_omsigc_d4shift (no GPU, no training, no sampling).

Subcommands
-----------
run            (default) Build a private, patched copy of the cosmodiff runtime in a temporary
               directory, then check in a child process that uses that runtime:
                 * each new config differs from its baseline config
                   (local/nf_conditional_omsig_continuous_200k) only in io.output_dir and the
                   top-level augmentations block, which is exactly
                   {RandomDihedral2D: {dims: [-2, -1], p: 1.0}, RandomRoll: {size: 128, dims: [-2, -1]}};
                 * each new manifest row differs from its baseline row only in run_name, config,
                   checkpoint_dir, requested_checkpoint, sample_path (plus the added keys
                   baseline_run_name, augmentations);
                 * parse_config_data gives identical stored images and labels for new and baseline config;
                 * 8 fetches of index 0 are pairwise different, each is exactly
                   roll(g(x0), s) for a D4 element g and a periodic shift s of the stored map x0,
                   and the label is unchanged;
                 * no checkpoint dir, sample file or evaluation dir of the new runs exists.
               Writes the receipt (never overwrites) only if every check passes; prints it either way.
build-runtime  Build the private runtime at --dest (refuses an existing --dest) for the given
               --patch-set; with --verify-receipt, exit 1 unless the cosmodiff file hashes equal those
               recorded in the receipt. Used by the training and sampling batch scripts so the shared
               cosmo_diffusion_main checkout is never patched in place.
child          Internal: the torch-side checks, run under the private runtime.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
COSMODIFF_SRC = Path("/home/jiamingp/Diffusion_model/cosmo_diffusion_main")
LOCAL = ROOT / "local/nf_cond_omsigc_d4shift_200k"
MANIFEST = LOCAL / "manifest.json"
BASE_MANIFEST = ROOT / "local/nf_conditional_omsig_continuous_200k/manifest.json"
RECEIPT = LOCAL / "preflight_receipt.json"
EVAL_OUT = ROOT / "results/nf_cond_omsigc_d4shift_200k/evaluate_run_v1"
EXPECTED_AUG = {"RandomDihedral2D": {"dims": [-2, -1], "p": 1.0},
                "RandomRoll": {"size": 128, "dims": [-2, -1]}}
ALLOWED_CONFIG_DIFF = {"io.output_dir"}
ALLOWED_ROW_CHANGES = {"run_name", "config", "checkpoint_dir", "requested_checkpoint", "sample_path"}
ADDED_ROW_KEYS = {"baseline_run_name", "augmentations"}
# Same patch lists as the baseline launchers (train_/sample_nf_conditional_omsig_continuous_array),
# applied to a private copy instead of the shared checkout.
PATCH_SETS = {
    "train": ["continuous_labels", "direct_unet_checkpoint", "numpy_compat"],
    "sample": ["package_metadata", "continuous_labels", "direct_unet_checkpoint", "numpy_compat",
               "sampling_compat"],
}
SEED = 123
N_FETCH = 8
SAMPLE_K = 64


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args, cwd) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def resolve(p) -> Path:
    p = Path(str(p))
    return p if p.is_absolute() else ROOT / p


# ------------------------------------------------------------------ runtime
def runtime_hashes(src: Path) -> dict:
    files = sorted(src.glob("cosmodiff/*.py")) + sorted(src.glob("scripts/cosmodiff_*.py"))
    return {str(f.relative_to(src)): sha256(f) for f in files}


def build_runtime(dest: Path, patch_set: str) -> dict:
    """Copy cosmodiff/*.py, cosmodiff/data and the two entry scripts (no .bak files, so the
    patch scripts start from the current sources), patch the copy, write the sklearn stub and
    the diffusers sitecustomize. Returns provenance."""
    if dest.exists():
        raise SystemExit(f"Refusing to reuse existing runtime dir {dest}")
    src = dest / "cosmodiff_src"
    stubs = dest / "stubs"
    (src / "cosmodiff").mkdir(parents=True)
    (src / "scripts").mkdir()
    for f in sorted((COSMODIFF_SRC / "cosmodiff").glob("*.py")):
        shutil.copy2(f, src / "cosmodiff" / f.name)
    shutil.copytree(COSMODIFF_SRC / "cosmodiff/data", src / "cosmodiff/data")
    for name in ("cosmodiff_train.py", "cosmodiff_sample.py"):
        shutil.copy2(COSMODIFF_SRC / "scripts" / name, src / "scripts" / name)
    patch_log = []
    for p in PATCH_SETS[patch_set]:
        r = subprocess.run([sys.executable, str(ROOT / f"scripts/patch_cosmodiff_{p}.py"), str(src)],
                           capture_output=True, text=True)
        patch_log.append({"patch": p, "rc": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()})
        if r.returncode != 0:
            raise SystemExit(f"patch {p} failed on {src}: {r.stderr}")
    (stubs / "sklearn/metrics").mkdir(parents=True)
    (stubs / "sklearn/__init__.py").write_text("from . import metrics\n")
    (stubs / "sklearn/metrics/__init__.py").write_text(
        "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed')\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/write_diffusers_runtime_sitecustomize.py"),
                    str(stubs / "sitecustomize.py")], check=True, capture_output=True, text=True)
    return {"dest": str(dest), "patch_set": patch_set, "patches": patch_log,
            "cosmodiff_source": str(COSMODIFF_SRC),
            "cosmodiff_source_git_head": git(["rev-parse", "HEAD"], COSMODIFF_SRC),
            "cosmodiff_source_git_status": git(["status", "--short", "--untracked-files=no"], COSMODIFF_SRC),
            "sha256": runtime_hashes(src)}


def runtime_env(dest: Path) -> dict:
    env = dict(os.environ)
    keep = [e for e in env.get("PYTHONPATH", "").split(":") if e and "site-packages" not in e
            and "dist-packages" not in e]
    env["PYTHONPATH"] = ":".join([str(dest / "stubs"), str(dest / "cosmodiff_src"), str(ROOT),
                                  str(ROOT / "scripts"), *keep])
    env["COSMODIFF_DIR"] = str(dest / "cosmodiff_src")
    env["COSMODIFF_STUB_SKLEARN"] = "1"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "8")
    return env


# ------------------------------------------------------------------ static checks
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
    import yaml

    fails = []
    rows = json.loads(MANIFEST.read_text())
    base = {r["run_name"]: r for r in json.loads(BASE_MANIFEST.read_text())}
    out = {"manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
           "baseline_manifest": str(BASE_MANIFEST), "baseline_manifest_sha256": sha256(BASE_MANIFEST),
           "runs": []}
    for r in rows:
        b = base.get(r.get("baseline_run_name"))
        rec = {"run_name": r["run_name"], "baseline_run_name": r.get("baseline_run_name")}
        if b is None:
            fails.append(f"{r['run_name']}: baseline row missing")
            out["runs"].append(rec)
            continue
        # manifest row
        changed = sorted(k for k in set(r) & set(b) if r[k] != b[k])
        added, removed = sorted(set(r) - set(b)), sorted(set(b) - set(r))
        rec["manifest_changed_keys"], rec["manifest_added_keys"], rec["manifest_removed_keys"] = changed, added, removed
        if not set(changed) <= ALLOWED_ROW_CHANGES or set(added) != ADDED_ROW_KEYS or removed:
            fails.append(f"{r['run_name']}: manifest row differs beyond allowed keys")
        if r.get("augmentations") != EXPECTED_AUG:
            fails.append(f"{r['run_name']}: manifest augmentations field differs")
        # config
        cp, bp = resolve(r["config"]), resolve(b["config"])
        cfg, bcfg = yaml.safe_load(cp.read_text()), yaml.safe_load(bp.read_text())
        fc, fb = flatten({k: v for k, v in cfg.items() if k != "augmentations"}), flatten(bcfg)
        diff = sorted(k for k in set(fc) | set(fb) if fc.get(k, "<missing>") != fb.get(k, "<missing>"))
        rec.update({"config": str(cp), "config_sha256": sha256(cp), "baseline_config": str(bp),
                    "baseline_config_sha256": sha256(bp), "config_diff_keys": diff + ["augmentations"],
                    "baseline_has_augmentations": "augmentations" in bcfg,
                    "augmentations": cfg.get("augmentations")})
        if set(diff) != ALLOWED_CONFIG_DIFF or "augmentations" in bcfg or cfg.get("augmentations") != EXPECTED_AUG:
            fails.append(f"{r['run_name']}: config differs beyond io.output_dir + augmentations")
        out_dir = cfg["io"]["output_dir"]
        rec["output_dir"] = out_dir
        if out_dir != r["checkpoint_dir"]:
            fails.append(f"{r['run_name']}: io.output_dir != manifest checkpoint_dir")
        if r["requested_checkpoint"] != f"{out_dir}/checkpoint-epoch-{int(r['checkpoint_epoch'])}":
            fails.append(f"{r['run_name']}: requested_checkpoint inconsistent")
        if int(cfg["train"]["num_epochs"]) - 1 != int(r["checkpoint_epoch"]):
            fails.append(f"{r['run_name']}: checkpoint_epoch != num_epochs - 1")
        if cfg["data"]["img_path"] != r["prepared_image_path"] or cfg["data"]["label_path"] != r["train_label_path"]:
            fails.append(f"{r['run_name']}: config data paths != manifest data paths")
        # nothing of the new run may exist yet
        from sample_nf_conditional_bias_probe import output_path_for
        sample = output_path_for(ROOT, r, SEED, SAMPLE_K, None)
        exist = {"checkpoint_dir": os.path.exists(out_dir), "sample_path": sample.exists()}
        rec["sample_path_seed123_k64"] = str(sample)
        rec["exists"] = exist
        if any(exist.values()):
            fails.append(f"{r['run_name']}: output already exists: {exist}")
        out["runs"].append(rec)
    out["eval_out_dir"] = str(EVAL_OUT)
    out["eval_out_dir_exists"] = EVAL_OUT.exists()
    if EVAL_OUT.exists():
        fails.append(f"{EVAL_OUT} exists")
    return out, fails


# ------------------------------------------------------------------ child (torch)
def child(out_json: Path) -> None:
    from simdiff_eval.torch_compat import install_torch_backend_compat

    install_torch_backend_compat(entry_point="scripts.preflight_nf_cond_omsigc_d4shift.child")

    import numpy as np
    import torch
    import yaml

    import cosmodiff
    from cosmodiff.augment import RandomDihedral2D, RandomRoll
    from cosmodiff.utils import parse_config_data

    rt = os.environ["COSMODIFF_DIR"]
    res = {"cosmodiff_file": cosmodiff.__file__, "torch": torch.__version__,
           "torch_num_threads": torch.get_num_threads(), "runs": [], "fails": []}
    if not str(Path(cosmodiff.__file__).resolve()).startswith(str(Path(rt).resolve())):
        res["fails"].append("cosmodiff not imported from the private runtime")

    def d4(x):  # independent enumeration: rot90^k of x and of x transposed
        for t in (False, True):
            y = x.transpose(-2, -1) if t else x
            for k in range(4):
                yield f"{'T' if t else 'I'}r{k}", torch.rot90(y, k, dims=(-2, -1))

    def decompose(img, x0):
        a = img[0].double().numpy()
        fa = np.fft.fft2(a)
        hits = []
        for name, gx in d4(x0):
            g = gx[0].double().numpy()
            c = np.fft.ifft2(fa * np.conj(np.fft.fft2(g))).real
            s = np.unravel_index(int(np.argmax(c)), c.shape)
            if torch.equal(torch.roll(gx, (int(s[0]), int(s[1])), dims=(-2, -1)), img):
                hits.append({"d4": name, "shift": [int(s[0]), int(s[1])]})
        return hits

    for row in json.loads(MANIFEST.read_text()):
        base_rows = {r["run_name"]: r for r in json.loads(BASE_MANIFEST.read_text())}
        b = base_rows[row["baseline_run_name"]]
        cfg = yaml.safe_load(resolve(row["config"]).read_text())
        bcfg = yaml.safe_load(resolve(b["config"]).read_text())
        new, old = parse_config_data(cfg)["data"], parse_config_data(bcfg)["data"]
        rec = {"run_name": row["run_name"], "n_images": int(len(new)), "image_shape": list(new.arrays.shape),
               "label_shape": list(new.labels.shape), "image_dtype": str(new.arrays.dtype),
               "image_file_sha256": sha256(row["prepared_image_path"]),
               "label_file_sha256": sha256(row["train_label_path"])}
        fails = []
        if len(new) != int(row["dataset_size"]):
            fails.append("dataset size != manifest dataset_size")
        rec["images_equal_baseline"] = bool(torch.equal(new.arrays, old.arrays))
        rec["labels_equal_baseline"] = bool(torch.equal(new.labels, old.labels))
        raw_lab = torch.as_tensor(np.asarray(np.load(row["train_label_path"])), dtype=new.labels.dtype)
        rec["labels_equal_label_file"] = bool(raw_lab.shape == new.labels.shape and torch.equal(raw_lab, new.labels))
        for k in ("images_equal_baseline", "labels_equal_baseline", "labels_equal_label_file"):
            if not rec[k]:
                fails.append(k)
        rec["baseline_augmentations"] = repr(old.augmentations)
        rec["augmentations"] = repr(new.augmentations)
        mods = list(new.augmentations) if new.augmentations is not None else []
        ok_pipe = (old.augmentations is None and len(mods) == 2
                   and type(mods[0]) is RandomDihedral2D and tuple(mods[0].dims) == (-2, -1) and mods[0].p == 1.0
                   and type(mods[1]) is RandomRoll and mods[1].size == 128 and tuple(mods[1].dims) == (-2, -1))
        rec["pipeline_ok"] = bool(ok_pipe)
        if not ok_pipe:
            fails.append("augmentation pipeline not [RandomDihedral2D(dims=(-2,-1), p=1), RandomRoll(size=128, dims=(-2,-1))]")
        # fetches of index 0
        x0 = new.arrays[0]
        base_item = old[0]
        if not torch.equal(base_item["images"], x0):
            fails.append("baseline fetch of index 0 is not the stored map")
        torch.manual_seed(SEED)
        fetches, recs = [], []
        for i in range(N_FETCH):
            item = new[0]
            img = item["images"]
            hits = decompose(img, x0) if img.shape == x0.shape else []
            label_same = bool(torch.equal(item["labels"], new.labels[0]) and torch.equal(item["labels"], old.labels[0]))
            recs.append({"fetch": i, "label_unchanged": label_same, "n_d4_shift_matches": len(hits),
                         "match": hits[0] if hits else None, "equals_stored_map": bool(torch.equal(img, x0))})
            if not hits:
                fails.append(f"fetch {i} is not a D4 element composed with a periodic roll of the stored map")
            if not label_same:
                fails.append(f"fetch {i} label changed")
            fetches.append(img.clone())
        distinct = sum(1 for i in range(N_FETCH) if not any(torch.equal(fetches[i], fetches[j]) for j in range(i)))
        rec["fetches"], rec["n_distinct_fetches"] = recs, distinct
        rec["label0"] = [float(v) for v in new.labels[0].reshape(-1)]
        if distinct != N_FETCH:
            fails.append(f"only {distinct} of {N_FETCH} fetches are distinct")
        rec["fails"] = fails
        res["fails"] += [f"{row['run_name']}: {f}" for f in fails]
        res["runs"].append(rec)
    out_json.write_text(json.dumps(res, indent=2))


# ------------------------------------------------------------------ main
def cmd_run(args) -> int:
    receipt = Path(args.receipt)
    if receipt.exists():
        raise SystemExit(f"Refusing to overwrite {receipt}")
    static, fails = static_checks()
    with tempfile.TemporaryDirectory(prefix="d4shift_preflight_") as tmp:
        tmp = Path(tmp)
        runtimes = {ps: build_runtime(tmp / f"runtime_{ps}", ps) for ps in PATCH_SETS}
        child_json = tmp / "child.json"
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child", "--out", str(child_json)],
                           env=runtime_env(tmp / "runtime_train"), capture_output=True, text=True)
        if r.returncode != 0 or not child_json.exists():
            print(r.stdout[-4000:], r.stderr[-4000:], file=sys.stderr)
            fails.append(f"child failed rc={r.returncode}")
            dyn = {}
        else:
            dyn = json.loads(child_json.read_text())
            fails += dyn.pop("fails")
    for ps in runtimes:
        runtimes[ps].pop("dest")
    report = {
        "status": "PASS" if not fails else "FAIL",
        "ledger_run": "nf_cond_omsigc_d4shift",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": sys.executable,
        "project_git_head": git(["rev-parse", "--short", "HEAD"], ROOT),
        "seed_for_fetches": SEED,
        "n_fetches": N_FETCH,
        "expected_augmentations": EXPECTED_AUG,
        "fails": fails,
        "static": static,
        "runtime": runtimes,
        "torch_checks": dyn,
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
    prov = build_runtime(Path(args.dest), args.patch_set)
    print(json.dumps({k: v for k, v in prov.items() if k != "patches"}, indent=2))
    if args.verify_receipt:
        rec = json.loads(Path(args.verify_receipt).read_text())
        if rec.get("status") != "PASS":
            print(f"receipt {args.verify_receipt} status is not PASS", file=sys.stderr)
            return 1
        want = rec["runtime"][args.patch_set]["sha256"]
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
    p_b.add_argument("--patch-set", choices=sorted(PATCH_SETS), required=True)
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
