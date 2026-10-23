#!/usr/bin/env python
"""RMS bias against RMS scatter of probe Omega_m readings, per set of maps (real held-out, paper Fig. 2 sizes, and the
fine-tune controls at 300k).

Per held-out box i (sims 900-931) with readings y_ij (128 real slices or 64 generated draws):
  bias b_i = median_j y_ij - theta_i;  scatter sd_i = std_j y_ij (ddof=1);  half-width s_i = (Q84 - Q16) / 2.
Per set: RMS bias = sqrt(mean_i b_i^2), RMS scatter = sqrt(mean_i sd_i^2), their ratio, and covered68 (Q16 <= theta <= Q84).
Reads saved predictions only (same inputs as plot_probe_bias_vs_scatter.py and plot_ftctl_controls.py).
Writes a new timestamped folder; never overwrites.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from plot_ftctl_controls import ARMS, LABEL, run_info
from plot_iaifi_statement_figure import load as load_real
from plot_nick_action_items import draws_for
from plot_probe_bias_vs_scatter import generated

ROOT = Path(__file__).resolve().parents[1]


def stats(name, y, truth):
    med = np.median(y, axis=1)
    q16, q84 = np.quantile(y, [0.16, 0.84], axis=1)
    b, sd, hw = med - truth, y.std(axis=1, ddof=1), (q84 - q16) / 2
    return dict(set=name, readings_per_box=y.shape[1], covered68=int(((q16 <= truth) & (truth <= q84)).sum()),
                median_bias=float(np.median(b)), rms_bias=float(np.sqrt(np.mean(b ** 2))),
                rms_scatter_std=float(np.sqrt(np.mean(sd ** 2))), median_half_width=float(np.median(hw)),
                rms_bias_over_rms_scatter=float(np.sqrt(np.mean(b ** 2)) / np.sqrt(np.mean(sd ** 2))))


def main():
    out = ROOT / "results" / f"probe_rms_bias_scatter_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(__file__, out / Path(__file__).name)
    truth, d = load_real()
    rows = [stats("real held-out maps", d["real"], truth)]
    rows += [stats(f"paper sweep N={n}", y, truth) for n, y in generated(truth).items()]
    for n in ARMS:
        for a in ARMS[n]:
            for s in (0, 100):
                r = run_info(n, a, s)
                if s == 0 and a != "noaug":
                    continue
                name = f"(Om,s8) N={n} trained 200k" if s == 0 else f"(Om,s8) N={n} {LABEL[a]} 300k"
                rows.append(stats(name, draws_for(r["pred"], r["run"], truth), truth))
    t = pd.DataFrame(rows)
    t.to_csv(out / "rms_bias_scatter.csv", index=False)
    print(t.round(4).to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
