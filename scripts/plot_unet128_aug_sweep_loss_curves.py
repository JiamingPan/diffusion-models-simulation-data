#!/usr/bin/env python
"""Training-loss curves for the UNet-128 augmentation sweep and its no-aug baseline.

Reads only existing metrics files (no training, no sampling):

* augmentation arms (d4shift, d4equiv, warp; N = 64..2048):
  ``<output_dir>/metrics.json`` for every row of ``local/unet128_aug_sweep/runs.json``;
* no-aug baseline (N = 64..32768): ``local/nf_generalize_fig2/manifest.json`` rows with
  ``arch == 'u128'``. Their checkpoint dirs have no ``metrics.json``; the training script
  wrote ``metrics_epoch_<E>.json`` instead. If exactly one such file exists it is used and
  its name is recorded. Files whose first epoch_lr is far below the base LR were written
  by a continued run (local/nf_generalize_fig2_continue400k) and do not start at update 0;
  the offset is not stored, so the per-update record is not used. For those runs the
  original 0-200k loss is taken from the archived training log (``TRAIN_LOG_FALLBACK``),
  one line per epoch "Epoch E - avg loss: X" (X printed to 4 decimals = round(epoch_loss, 4),
  verified on N = 2048). The epoch average is placed at update (E+1)*steps_per_epoch and
  smoothed over round(2000/steps_per_epoch) epochs; window means use epochs lying wholly
  inside the update window. ``loss_source`` in the CSV says which record was used. Files
  slightly longer than 200000 updates (partial last epoch) use the first 200000 updates
  (``truncated_to_200k``).

Also writes crosscheck_log_vs_metrics.json (log method vs metrics on a run that has both,
noaug N = 2048) and rounding_check.csv (effect of the 4-decimal log rounding on window
means, measured on exact epoch_loss records at the same loss levels).

The loss is the min-SNR-weighted v-prediction loss at random timesteps, one value per
optimizer update. Absolute loss levels are not comparable across arms (augmented /
equivariant runs fit a different target distribution).

Writes a NEW folder ``results/unet128_aug_sweep/loss_curves_<YYYYmmdd_HHMMSS>/``
(mkdir exist_ok=False) with loss_curves.{png,pdf}, loss_curves_baseline_all_N.{png,pdf},
convergence.csv, inputs.json and a copy of this script.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS_JSON = ROOT / "local" / "unet128_aug_sweep" / "runs.json"
MANIFEST_JSON = ROOT / "local" / "nf_generalize_fig2" / "manifest.json"
OUT_PARENT = ROOT / "results" / "unet128_aug_sweep"

TARGET_UPDATES = 200_000
SMOOTH_WINDOW = 2_000
PANEL_N = (64, 128, 256, 512, 1024, 2048)
ARM_ORDER = ("noaug", "d4shift", "d4equiv", "warp")
ARM_STYLE = {
    "noaug": dict(color="#333333", ls="-"),
    "d4shift": dict(color="#2a78d6", ls="-"),
    "d4equiv": dict(color="#1baf7a", ls="-"),
    "warp": dict(color="#eb6834", ls="--"),
}
LOG_DIR = "/scratch/huterer_root/huterer0/jiamingp/logs_archive/nf_generalize_fig2"
# run_name -> archived training log of the ORIGINAL 0-200k run (verified inside the script).
TRAIN_LOG_FALLBACK = {
    "nf_fig2_u128_d2p06_noaug_200k": f"{LOG_DIR}/train_50456581_10.out",
    "nf_fig2_u128_d2p07_noaug_200k": f"{LOG_DIR}/train_50456581_11.out",
    "nf_fig2_u128_d2p08_noaug_200k": f"{LOG_DIR}/train_50456581_12.out",
    "nf_fig2_u128_d2p09_noaug_200k": f"{LOG_DIR}/train_50456581_13.out",
    "nf_fig2_u128_d2p10_noaug_200k": f"{LOG_DIR}/train_50456581_14.out",
}
# Run with both a from-update-0 metrics file and a training log: used to cross-check the method.
CROSSCHECK_RUN = ("nf_fig2_u128_d2p11_noaug_200k", f"{LOG_DIR}/train_50562272_15.out")
LOG_DECIMALS = 4

WINDOWS = {
    "last": (180_000, 200_000),
    "prev": (160_000, 180_000),
    "first": (0, 10_000),
}


def find_metrics_file(ckpt_dir: str) -> tuple[str | None, str]:
    """Return (path, note). Prefer metrics.json; else a single metrics_epoch_*.json."""
    p = os.path.join(ckpt_dir, "metrics.json")
    if os.path.isfile(p):
        return p, "metrics.json"
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "metrics_epoch_*.json")))
    if len(cands) == 1:
        return cands[0], f"no metrics.json; used {os.path.basename(cands[0])}"
    if len(cands) > 1:
        return None, f"no metrics.json; {len(cands)} metrics_epoch_*.json files (ambiguous, not used)"
    return None, "no metrics file found"


def running_mean(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing running mean; returns (update_count, value) for updates w..len(x)."""
    c = np.cumsum(np.concatenate([[0.0], x.astype(np.float64)]))
    vals = (c[w:] - c[:-w]) / w
    upd = np.arange(w, len(x) + 1)
    return upd, vals


def restart_updates(epoch_lr: np.ndarray, steps_per_epoch: int, max_update: int) -> list[int]:
    """Updates at which the LR jumps up (warm restart), from the per-epoch LR record."""
    lr = np.asarray(epoch_lr, dtype=np.float64)
    jumps = np.where(lr[1:] > 10.0 * lr[:-1])[0] + 1  # epoch index of the restarted LR
    ups = [int(e * steps_per_epoch) for e in jumps]
    return [u for u in ups if 0 < u <= max_update]


def parse_train_log(log_path: str, run_name: str, ckpt_dir: str, n_epochs: int) -> np.ndarray:
    """Per-epoch avg loss from a training log; raises ValueError unless the log is verifiably
    the original run of ``run_name`` writing into ``ckpt_dir`` with epochs 0..n_epochs-1."""
    import re
    pat = re.compile(r"^Epoch (\d+) \S+ avg loss: ([0-9.eE+-]+)\s*$")
    ep, val, trained, written = [], [], [], []
    with open(log_path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            m = pat.match(line)
            if m:
                ep.append(int(m.group(1)))
                val.append(float(m.group(2)))
            elif line.startswith("Training ") and line != "Training complete.":
                trained.append(line.split(None, 1)[1])
            elif line.startswith("Metrics written to "):
                written.append(line.split("Metrics written to ", 1)[1])
    if trained != [run_name]:
        raise ValueError(f"{log_path}: 'Training' lines {trained} != [{run_name}]")
    exp = os.path.join(ckpt_dir, f"metrics_epoch_{n_epochs - 1}.json")
    if written != [exp]:
        raise ValueError(f"{log_path}: 'Metrics written' {written} != [{exp}]")
    ep = np.asarray(ep)
    if not np.array_equal(ep, np.arange(n_epochs)):
        raise ValueError(f"{log_path}: epochs not exactly 0..{n_epochs - 1} (got {len(ep)})")
    return np.asarray(val, dtype=np.float64)


def epoch_window(spe: int, a: int, b: int) -> tuple[int, int]:
    """Epoch index range [e0, e1) of epochs lying wholly inside updates [a, b)."""
    return -(-a // spe), b // spe


def stats_from_epochs(ev: np.ndarray, spe: int) -> dict:
    """Window means and smoothed minimum from per-epoch average losses."""
    out = {}
    for k, (a, b) in WINDOWS.items():
        e0, e1 = epoch_window(spe, a, b)
        e1 = min(e1, len(ev))
        out[f"mean_loss_{a // 1000}k_{b // 1000}k"] = float(ev[e0:e1].mean()) if e1 > e0 else float("nan")
    last, prev = out["mean_loss_180k_200k"], out["mean_loss_160k_180k"]
    out["rel_change_last_vs_prev"] = (last - prev) / prev
    w = max(1, int(round(SMOOTH_WINDOW / spe)))
    c = np.cumsum(np.concatenate([[0.0], ev]))
    sm = (c[w:] - c[:-w]) / w
    upd = (np.arange(w, len(ev) + 1)) * spe  # update count at the end of the window
    i = int(np.argmin(sm))
    out["min_smoothed_loss"] = float(sm[i])
    out["update_of_min_smoothed"] = int(upd[i])
    out["smoothing_window_updates"] = int(w * spe)
    e0, e1 = epoch_window(spe, *WINDOWS["last"])
    out["epoch_loss_mean_180k_200k"] = out["mean_loss_180k_200k"]
    out["epoch_loss_epochs_used"] = f"{e0}-{min(e1, len(ev)) - 1}"
    return out, (upd, sm)


def load_run(arm: str, n: int, run_name: str, ckpt_dir: str) -> dict:
    rec = dict(arm=arm, N=n, run_name=run_name, checkpoint_dir=ckpt_dir)
    path, note = find_metrics_file(ckpt_dir)
    rec["metrics_path"] = path
    rec["metrics_note"] = note
    if path is None:
        rec["status"] = "missing"
        return rec
    with open(path) as fh:
        m = json.load(fh)
    for key in ("loss", "epoch_loss", "epoch_lr"):
        if key not in m:
            rec["status"] = f"missing key {key}"
            return rec
    loss = np.asarray(m["loss"], dtype=np.float64)
    eloss = np.asarray(m["epoch_loss"], dtype=np.float64)
    elr = np.asarray(m["epoch_lr"], dtype=np.float64)
    rec["n_updates_in_file"] = int(len(loss))
    rec["n_epochs_in_file"] = int(len(eloss))
    if len(loss) % len(eloss) != 0:
        rec["status"] = "loss length not a multiple of epoch_loss length"
        return rec
    spe = len(loss) // len(eloss)
    rec["steps_per_epoch"] = int(spe)
    rec["epoch_lr_first"] = float(elr[0])
    rec["loss_mean_first100_in_file"] = float(loss[:100].mean())
    # A fresh run starts at the base LR (max of the record). A file whose first epoch LR
    # is far below that was written by a resumed/continued run and does not begin at
    # update 0; its update offset is not stored in the file, so it is not used.
    if elr[0] < 0.5 * elr.max():
        rec["metrics_note"] += ("; metrics not_from_update_0: first epoch_lr %.3g vs max %.3g "
                                "(continuation record; offset unknown)" % (elr[0], elr.max()))
        log = TRAIN_LOG_FALLBACK.get(run_name)
        if log is None:
            rec["status"] = "not_from_update_0 and no training-log fallback"
            return rec
        n_ep = TARGET_UPDATES // spe
        try:
            ev = parse_train_log(log, run_name, ckpt_dir, n_ep)
        except (OSError, ValueError) as exc:
            rec["status"] = f"not_from_update_0; log fallback failed: {exc}"
            return rec
        rec.update(status="ok", loss_source="train_log_epoch_avg", log_path=log,
                   epoch_avg=ev, restarts=[], truncated_to_200k=False)
        return rec
    if len(loss) < TARGET_UPDATES:
        rec["status"] = f"short: {len(loss)} < {TARGET_UPDATES} updates"
        return rec
    rec["truncated_to_200k"] = bool(len(loss) > TARGET_UPDATES)
    rec["status"] = "ok"
    rec["loss_source"] = "metrics"
    rec["loss"] = loss[:TARGET_UPDATES]
    n_ep = TARGET_UPDATES // spe  # whole epochs inside the first 200k updates
    rec["epoch_loss"] = eloss[: max(n_ep, 1)]
    rec["epoch_lr"] = elr
    rec["restarts"] = restart_updates(elr, spe, TARGET_UPDATES)
    return rec


def curve(rec: dict):
    if rec["loss_source"] == "metrics":
        return running_mean(rec["loss"], SMOOTH_WINDOW)
    return stats_from_epochs(rec["epoch_avg"], rec["steps_per_epoch"])[1]


def stats(rec: dict) -> dict:
    if rec["loss_source"] == "train_log_epoch_avg":
        out = stats_from_epochs(rec["epoch_avg"], rec["steps_per_epoch"])[0]
        out["n_lr_restarts_le_200k"] = ""
        return out
    loss = rec["loss"]
    spe = rec["steps_per_epoch"]
    out = {}
    for k, (a, b) in WINDOWS.items():
        out[f"mean_loss_{a // 1000}k_{b // 1000}k"] = float(loss[a:b].mean())
    last = out["mean_loss_180k_200k"]
    prev = out["mean_loss_160k_180k"]
    out["rel_change_last_vs_prev"] = (last - prev) / prev
    upd, sm = running_mean(loss, SMOOTH_WINDOW)
    i = int(np.argmin(sm))
    out["min_smoothed_loss"] = float(sm[i])
    out["update_of_min_smoothed"] = int(upd[i])
    out["smoothing_window_updates"] = SMOOTH_WINDOW
    # Cross-check on epoch_loss: epochs lying wholly inside updates [180k, 200k).
    e0 = -(-WINDOWS["last"][0] // spe)  # ceil
    e1 = WINDOWS["last"][1] // spe  # exclusive
    el = rec["epoch_loss"]
    e1 = min(e1, len(el))
    if e1 > e0:
        out["epoch_loss_mean_180k_200k"] = float(el[e0:e1].mean())
        out["epoch_loss_epochs_used"] = f"{e0}-{e1 - 1}"
    else:
        out["epoch_loss_mean_180k_200k"] = float("nan")
        out["epoch_loss_epochs_used"] = "none (epoch longer than window)"
    out["n_lr_restarts_le_200k"] = len(rec["restarts"])
    return out


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 14,
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "legend.fontsize": 12,
    })
    return plt


def plot_panels(plt, runs: dict, out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), sharex=True)
    for ax, n in zip(axes.ravel(), PANEL_N):
        restarts_drawn = False
        for arm in ARM_ORDER:
            rec = runs.get((arm, n))
            if rec is None or rec.get("status") != "ok":
                continue
            if not restarts_drawn and rec["restarts"]:
                for u in rec["restarts"]:
                    ax.axvline(u, color="0.6", lw=0.5, alpha=0.35, zorder=0)
                restarts_drawn = True
            upd, sm = curve(rec)
            st = ARM_STYLE[arm]
            ax.plot(upd, sm, color=st["color"], ls=st["ls"], lw=1.6, label=arm,
                    zorder=3 if arm == "noaug" else 2)
        ax.set_yscale("log")
        ax.set_title(f"N = {n}")
        ax.set_xlim(0, TARGET_UPDATES)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    for ax in axes[-1]:
        ax.set_xlabel("training update")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"training loss ({SMOOTH_WINDOW}-update running mean)")
    from matplotlib.lines import Line2D
    present = [a for a in ARM_ORDER if any(a == k[0] and k[1] in PANEL_N and r.get("status") == "ok"
                                           for k, r in runs.items())]
    handles = [Line2D([], [], color=ARM_STYLE[a]["color"], ls=ARM_STYLE[a]["ls"], lw=1.6) for a in present]
    fig.legend(handles, present, loc="upper center", ncol=len(present), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    missing = [f"{a} N={n}" for (a, n), r in sorted(runs.items())
               if n in PANEL_N and r.get("status") != "ok"]
    note = ("Grey vertical lines: LR warm restarts detected in epoch_lr. "
            "Absolute loss is not comparable across arms (different training targets).")
    from_log = [f"N={n}" for (a, n), r in sorted(runs.items())
                if a == "noaug" and n in PANEL_N and r.get("loss_source") == "train_log_epoch_avg"]
    if from_log:
        note += ("\nnoaug " + ", ".join(from_log) + ": per-epoch average loss from the training log "
                 "(4 decimals), smoothed over ~2000 updates.")
    if missing:
        note += "\nNot shown (no 0-200k loss record from update 0): " + ", ".join(missing)
    fig.text(0.5, 0.005, note, ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"loss_curves.{ext}", dpi=150)
    plt.close(fig)


def plot_baseline(plt, runs: dict, out_dir: Path):
    import matplotlib as mpl
    base = sorted((n, rec) for (arm, n), rec in runs.items() if arm == "noaug" and rec.get("status") == "ok")
    if not base:
        return
    l2 = np.array([np.log2(n) for n, _ in base])
    norm = mpl.colors.Normalize(vmin=l2.min(), vmax=l2.max())
    cmap = mpl.colormaps["viridis"]
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for (n, rec), v in zip(base, l2):
        upd, sm = curve(rec)
        src = "" if rec["loss_source"] == "metrics" else " (log)"
        ax.plot(upd, sm, color=cmap(norm(v)), lw=1.6, label=f"N = {n}{src}")
    ax.set_yscale("log")
    ax.set_xlim(0, TARGET_UPDATES)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    ax.set_xlabel("training update")
    ax.set_ylabel(f"training loss ({SMOOTH_WINDOW}-update running mean)")
    shown = ", ".join(str(n) for n, _ in base)
    ax.set_title(f"UNet-128 no augmentation, updates 0-200k (N = {shown})", fontsize=13)
    sm_ = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm_, ax=ax, pad=0.27)
    cb.set_label(r"$\log_2 N$")
    ax.legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"loss_curves_baseline_all_N.{ext}", dpi=150)
    plt.close(fig)


def crosscheck_log_vs_metrics(runs: dict) -> dict:
    """Apply the log method to a run that also has a from-update-0 metrics file."""
    run_name, log = CROSSCHECK_RUN
    rec = next((r for r in runs.values() if r["run_name"] == run_name), None)
    out = {"run_name": run_name, "log_path": log}
    if rec is None or rec.get("loss_source") != "metrics":
        out["status"] = "crosscheck run has no metrics-sourced record"
        return out
    spe = rec["steps_per_epoch"]
    ev = parse_train_log(log, run_name, rec["checkpoint_dir"], TARGET_UPDATES // spe)
    el = rec["epoch_loss"][: len(ev)]
    out["max_abs_log_minus_round4_epoch_loss"] = float(np.abs(ev - np.round(el, LOG_DECIMALS)).max())
    out["max_abs_log_minus_epoch_loss"] = float(np.abs(ev - el).max())
    s_log = stats_from_epochs(ev, spe)[0]
    s_met = stats(rec)
    for k in ("mean_loss_180k_200k", "mean_loss_160k_180k", "rel_change_last_vs_prev",
              "mean_loss_0k_10k", "min_smoothed_loss"):
        out[k] = {"metrics_per_update": s_met[k], "log_epoch_avg": s_log[k],
                  "diff_log_minus_metrics": s_log[k] - s_met[k],
                  "rel_diff": (s_log[k] - s_met[k]) / abs(s_met[k])}
    out["status"] = "ok"
    return out


def rounding_check(runs: dict) -> list[dict]:
    """Effect of 4-decimal rounding on a 20k-update window mean of epoch_loss, measured on
    exact epoch_loss records at the loss level of each log-sourced run (the continuation
    metrics file of the same run, last 20k updates; plus the cross-check run)."""
    rows = []
    for rec in sorted(runs.values(), key=lambda r: r["N"]):
        if rec["arm"] != "noaug" or rec.get("status") != "ok":
            continue
        if rec["loss_source"] == "train_log_epoch_avg":
            with open(rec["metrics_path"]) as fh:
                el = np.asarray(json.load(fh)["epoch_loss"], dtype=np.float64)
            src = "continuation metrics epoch_loss, last 20k updates"
        elif rec["run_name"] == CROSSCHECK_RUN[0]:
            el = rec["epoch_loss"]
            src = "metrics epoch_loss, last 20k of the first 200k updates"
        else:
            continue
        spe = rec["steps_per_epoch"]
        k = 20_000 // spe
        win = el[-k:]
        exact, rounded = float(win.mean()), float(np.round(win, LOG_DECIMALS).mean())
        rows.append({"N": rec["N"], "run_name": rec["run_name"], "record": src, "n_epochs": int(k),
                     "mean_exact": exact, "mean_rounded_4dp": rounded,
                     "rel_diff": (rounded - exact) / exact})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-parent", default=str(OUT_PARENT))
    args = ap.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_parent) / f"loss_curves_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    runs: dict[tuple[str, int], dict] = {}
    with open(RUNS_JSON) as fh:
        aug_rows = json.load(fh)
    for r in aug_rows:
        rec = load_run(r["arm_tag"], int(r["dataset_size"]), r["run_name"], r["output_dir"])
        runs[(rec["arm"], rec["N"])] = rec
    with open(MANIFEST_JSON) as fh:
        man = json.load(fh)
    for r in man:
        if r.get("arch") != "u128":
            continue
        rec = load_run("noaug", int(r["dataset_size"]), r["run_name"], r["checkpoint_dir"])
        runs[(rec["arm"], rec["N"])] = rec

    # Consistency of restart positions within a panel.
    restart_note = {}
    for n in PANEL_N:
        sets = {arm: tuple(runs[(arm, n)]["restarts"]) for arm in ARM_ORDER
                if (arm, n) in runs and runs[(arm, n)].get("status") == "ok"}
        restart_note[n] = "consistent" if len(set(sets.values())) <= 1 else f"differ: { {a: len(v) for a, v in sets.items()} }"

    cols = ["arm", "N", "run_name", "status", "loss_source", "log_path", "metrics_path", "metrics_note",
            "n_updates_in_file", "n_epochs_in_file", "steps_per_epoch", "epoch_lr_first",
            "loss_mean_first100_in_file", "truncated_to_200k",
            "mean_loss_180k_200k", "mean_loss_160k_180k", "rel_change_last_vs_prev",
            "mean_loss_0k_10k", "min_smoothed_loss", "update_of_min_smoothed",
            "smoothing_window_updates",
            "epoch_loss_mean_180k_200k", "epoch_loss_epochs_used", "n_lr_restarts_le_200k"]
    arm_rank = {a: i for i, a in enumerate(ARM_ORDER)}
    rows = []
    for key in sorted(runs, key=lambda k: (arm_rank[k[0]], k[1])):
        rec = runs[key]
        row = {c: rec.get(c, "") for c in cols}
        if rec.get("status") == "ok":
            row.update(stats(rec))
        rows.append(row)
    with open(out_dir / "convergence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    crosscheck = crosscheck_log_vs_metrics(runs)
    with open(out_dir / "crosscheck_log_vs_metrics.json", "w") as fh:
        json.dump(crosscheck, fh, indent=2)
    rounding = rounding_check(runs)
    with open(out_dir / "rounding_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rounding[0].keys()))
        w.writeheader()
        w.writerows(rounding)

    plt = setup_mpl()
    plot_panels(plt, runs, out_dir)
    plot_baseline(plt, runs, out_dir)

    inputs = {
        "created": stamp,
        "script": str(Path(__file__).resolve()),
        "runs_json": str(RUNS_JSON),
        "manifest_json": str(MANIFEST_JSON),
        "target_updates": TARGET_UPDATES,
        "smooth_window_updates": SMOOTH_WINDOW,
        "train_log_fallback": TRAIN_LOG_FALLBACK,
        "crosscheck_run": list(CROSSCHECK_RUN),
        "restart_detection": "epoch_lr[e] > 10 * epoch_lr[e-1]; update = e * steps_per_epoch",
        "restart_consistency_per_panel": {str(k): v for k, v in restart_note.items()},
        "files": [
            {k: runs[key].get(k) for k in ("arm", "N", "run_name", "status", "loss_source", "log_path",
                                            "metrics_path", "metrics_note",
                                            "n_updates_in_file", "n_epochs_in_file", "steps_per_epoch",
                                            "truncated_to_200k")}
            | ({"metrics_mtime": datetime.fromtimestamp(os.path.getmtime(runs[key]["metrics_path"])).isoformat(),
                "metrics_bytes": os.path.getsize(runs[key]["metrics_path"])}
               if runs[key].get("metrics_path") else {})
            for key in sorted(runs, key=lambda k: (arm_rank[k[0]], k[1]))
        ],
    }
    with open(out_dir / "inputs.json", "w") as fh:
        json.dump(inputs, fh, indent=2)
    shutil.copy2(Path(__file__).resolve(), out_dir / Path(__file__).name)

    print(out_dir)
    bad = [r for r in rows if r["status"] != "ok"]
    for r in bad:
        print("NOT USED:", r["arm"], r["N"], r["status"], r["metrics_note"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
