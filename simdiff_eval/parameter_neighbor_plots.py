"""Notebook-native views of the parameter-nearest control results."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .parameter_neighbor_control import PARAM_NAMES

STYLES = {
    "generated": ("Generated-field recovery", "#bf5538", "o"),
    "nearest_theta": ("Nearest training cosmology: true parameters", "#303030", "x"),
    "nearest_training_field": ("Nearest training fields: probe recovery", "#ad7b18", "^"),
    "heldout_real": ("Held-out real fields: probe recovery", "#30688b", "s"),
}
DISPLAY = [r"$\Omega_m$",r"$\sigma_8$",r"$A_{\rm SN1}$",r"$A_{\rm AGN1}$",r"$A_{\rm SN2}$",r"$A_{\rm AGN2}$"]


def plot_recovery(points, run_name, *, residual=False):
    selected = points[points.run_name == run_name]
    if selected.empty:
        raise ValueError("selected run has no control points")
    fig, axes = plt.subplots(2,3,figsize=(16,9),layout="constrained")
    for ax, parameter, label in zip(axes.flat,PARAM_NAMES,DISPLAY):
        sub = selected[selected.parameter == parameter]
        if sub.empty:
            ax.set_visible(False)
            continue
        for kind, (name,color,marker) in STYLES.items():
            group = sub[sub.kind == kind].sort_values("theta_in")
            if group.empty:
                continue
            x = group.theta_in.to_numpy(float)
            y = group.theta_rec_median.to_numpy(float)
            yerr = np.vstack([y-group.theta_rec_q16,group.theta_rec_q84-y])
            ax.errorbar(x,y-x if residual else y,yerr=yerr,fmt=marker,color=color,
                        label=name,ms=5,alpha=.75,capsize=2,linewidth=1)
        if residual:
            ax.axhline(0,color=".25",ls="--",lw=1)
        else:
            low = min(sub.theta_in.min(),sub.theta_rec_q16.min())
            high = max(sub.theta_in.max(),sub.theta_rec_q84.max())
            pad = .05*max(high-low,1e-6)
            ax.plot([low-pad,high+pad],[low-pad,high+pad],"--",color=".25",lw=1)
            ax.set(xlim=(low-pad,high+pad),ylim=(low-pad,high+pad))
        ax.set(title=label,xlabel="Requested parameter (physical units)",
               ylabel="Recovered − requested" if residual else "True / recovered parameter (physical units)")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
    fig.suptitle(f"{run_name}\n{'Recovery residuals' if residual else 'Recovery and nearest-parameter controls'}; bars = central 68% field/probe spread, not posterior intervals",fontsize=13)
    return fig
