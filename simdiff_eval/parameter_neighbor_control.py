"""Parameter-nearest training controls; no generator, fitting, or sampling.

Distances use frozen training-only parameter scales, not image similarity.
Multiple selected slices from a cosmology are retained as an ensemble. Their
spread describes field/probe variability, not a Bayesian posterior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAM_NAMES = ("Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2")


def parameter_matrix(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(PARAM_NAMES) or not len(array):
        raise ValueError(f"{name} must be a nonempty (N,6) parameter table")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite parameters")
    return array


def simulation_ids(value, length, name):
    array = np.asarray(value)
    if array.shape != (length,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain one integer simulation ID per row")
    if np.any(array < 0):
        raise ValueError(f"{name} contains a negative simulation ID")
    return array.astype(np.int64)


def nearest_parameter_matches(train_theta, train_sim, requested_theta, heldout_sim, scale):
    """Select the closest represented cosmology in standardized six-D space.

    Exact ties choose the lowest simulation ID and are explicitly counted.
    Held-out simulations must not occur anywhere in the selected training set.
    """
    train = parameter_matrix(train_theta, "train_theta")
    requested = parameter_matrix(requested_theta, "requested_theta")
    sims = simulation_ids(train_sim, len(train), "train_sim")
    heldout = simulation_ids(heldout_sim, len(requested), "heldout_sim")
    if len(np.unique(heldout)) != len(heldout):
        raise ValueError("heldout simulation IDs must be unique")
    if np.intersect1d(sims, heldout).size:
        raise ValueError("training and held-out cosmologies overlap")
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape != (6,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("scale must contain six finite positive training-only scales")
    unique_sims = np.unique(sims)
    unique_theta = []
    for sim in unique_sims:
        values = train[sims == sim]
        if not np.allclose(values, values[0], rtol=0, atol=1e-6):
            raise ValueError(f"training labels disagree within simulation {sim}")
        unique_theta.append(values[0])
    unique_theta = np.asarray(unique_theta)
    distance = np.linalg.norm((requested[:, None] - unique_theta[None]) / scale, axis=2)
    winner = np.argmin(distance, axis=1)
    rows = []
    for h, index in enumerate(winner):
        sim = int(unique_sims[index])
        selected = np.flatnonzero(sims == sim)
        tied = np.isclose(distance[h], distance[h, index], rtol=1e-12, atol=1e-12)
        record = {
            "heldout_sim": int(heldout[h]), "nearest_sim": sim,
            "parameter_distance": float(distance[h, index]),
            "tied_cosmologies": int(tied.sum()), "selected_training_fields": len(selected),
        }
        for p, name in enumerate(PARAM_NAMES):
            record[f"requested_{name}"] = float(requested[h, p])
            record[f"nearest_{name}"] = float(unique_theta[index, p])
        rows.append(record)
    return pd.DataFrame(rows)


def ensemble_points(prediction, requested, heldout_sim, kind):
    """One point per held-out cosmology and parameter from an explicit ensemble."""
    requested = parameter_matrix(requested, "requested")
    heldout = simulation_ids(heldout_sim, len(requested), "heldout_sim")
    if len(prediction) != len(requested):
        raise ValueError("one ensemble is required per held-out cosmology")
    rows = []
    for h, values in enumerate(prediction):
        values = parameter_matrix(values, "prediction ensemble")
        q025, q16, median, q84, q975 = np.quantile(values, [.025, .16, .5, .84, .975], axis=0)
        for p, parameter in enumerate(PARAM_NAMES):
            rows.append({
                "heldout_sim": int(heldout[h]), "parameter": parameter, "kind": kind,
                "theta_in": float(requested[h, p]), "theta_rec_median": float(median[p]),
                "theta_rec_q16": float(q16[p]), "theta_rec_q84": float(q84[p]),
                "theta_rec_q025": float(q025[p]), "theta_rec_q975": float(q975[p]),
                "n_fields": len(values),
            })
    return pd.DataFrame(rows)


def summarize_points(points):
    rows = []
    for (run, kind, parameter), sub in points.groupby(["run_name", "kind", "parameter"], sort=False):
        x = sub.theta_in.to_numpy(float)
        y = sub.theta_rec_median.to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1) if len(x) > 1 and np.var(x) > 0 else (np.nan, np.nan)
        rows.append({
            "run_name": run, "kind": kind, "parameter": parameter, "n_cosmologies": len(sub),
            "slope": slope, "intercept": intercept, "mean_residual": np.mean(y-x),
            "mae": np.mean(np.abs(y-x)),
            "recovery_interval_inclusion_68": np.mean((sub.theta_rec_q16 <= x) & (x <= sub.theta_rec_q84)),
            "recovery_interval_inclusion_95": np.mean((sub.theta_rec_q025 <= x) & (x <= sub.theta_rec_q975)),
        })
    return pd.DataFrame(rows)


def residual_decomposition(points):
    """An accounting identity, not a causal attribution of the generator bias."""
    keys = ["run_name", "heldout_sim", "parameter"]
    if points.duplicated(keys + ["kind"]).any():
        raise ValueError("duplicate control points")
    wide = points.pivot(index=keys, columns="kind", values="theta_rec_median")
    inputs = points.groupby(keys).theta_in.agg(["first", "min", "max"])
    if not np.allclose(inputs["min"], inputs["max"], rtol=0, atol=1e-6):
        raise ValueError("requested parameters disagree across controls")
    required = ["nearest_theta", "nearest_training_field", "generated"]
    if any(name not in wide for name in required):
        return pd.DataFrame()
    out = pd.DataFrame(index=wide.index)
    out["training_parameter_offset"] = wide.nearest_theta - inputs["first"]
    out["probe_on_neighbor_offset"] = wide.nearest_training_field - wide.nearest_theta
    out["generated_vs_neighbor_offset"] = wide.generated - wide.nearest_training_field
    out["generated_total_residual"] = wide.generated - inputs["first"]
    np.testing.assert_allclose(out.iloc[:, :3].sum(axis=1), out.generated_total_residual, atol=1e-10)
    return out.reset_index()
