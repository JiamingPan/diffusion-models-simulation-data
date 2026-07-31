#!/usr/bin/env python3
"""Add the reviewer-requested reference and nearest-training audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
UNET_NOTEBOOK = REPO_ROOT / "notebooks" / "nf_generalize_fig2_partial_quickcheck.ipynb"
DIT_NOTEBOOK = REPO_ROOT / "notebooks" / "nf_generalize_fig2_dit_results.ipynb"


def source(cell: dict[str, Any]) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict[str, Any], value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


def find_cell(notebook: dict[str, Any], needle: str) -> dict[str, Any]:
    matches = [cell for cell in notebook["cells"] if needle in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cell containing {needle!r}; found {len(matches)}")
    return matches[0]


def write_notebook(path: Path, notebook: dict[str, Any]) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def update_unet_notebook() -> None:
    notebook = json.loads(UNET_NOTEBOOK.read_text())

    imports = find_cell(notebook, "from simdiff_eval.io import")
    imports_source = source(imports).replace(
        "from simdiff_eval.io import as_nchw, load_real_from_config, load_real_reference_from_config",
        "from simdiff_eval.io import (\n"
        "    as_nchw, configured_training_reference_info, load_real_from_config,\n"
        "    load_real_reference_from_config,\n"
        ")",
    )
    set_source(imports, imports_source)

    load_markdown = find_cell(notebook, "## Load Available Samples")
    set_source(
        load_markdown,
        """## Load Available Samples

Each black reference curve comes from the **training subset configured for that model**, not all available CAMELS maps. The run config fixes the source files, per-source `n_samples`, selection seed, and `zthin`; the notebook verifies that this produces the manifest's stated $N_{2D}$.

Normalization statistics are computed from that complete configured subset. To keep the notebook within memory limits, the plotted reference is then evenly limited by `MAX_REAL_REFERENCE_SLICES` (default 2,048; use 0 for every configured slice). The audit table below reports both the exact configured count and the count actually used in the black histogram and $P(k)$ denominator.
""",
    )

    load_code = find_cell(notebook, "real_reference_cache: dict[str, np.ndarray]")
    old_block = """loaded = {}
load_rows = []
real_reference_cache: dict[str, np.ndarray] = {}
real_reference_kind = 'normalized from complete configured training set'
for row in rows:
    path = sample_path_for(row)
    if not path.exists():
        continue
    config_path = PROJECT_DIR / row['config']
    generated = evenly_limit(load_npz_array(path), MAX_GENERATED)
    cache_key = real_reference_cache_key(config_path)
    if cache_key not in real_reference_cache:
        real_reference_cache[cache_key] = load_real_reference_from_config(
            config_path,
            max_slices=MAX_REAL_REFERENCE_SLICES,
        )
    real = real_reference_cache[cache_key]
    loaded[row['run_name']] = {
        'spec': row,
        'real': real,
        'generated': generated,
        'sample_path': path,
        'real_reference_kind': real_reference_kind,
    }
    load_rows.append({
        'task_id': row['task_id'],
        'run_name': row['run_name'],
        'arch': row['arch'],
        'dataset_size': int(row['dataset_size']),
        'n_real_configured': int(row['dataset_size']),
        'n_real_loaded': len(real),
        'real_reference_kind': real_reference_kind,
        'n_generated': len(generated),
        'sample_path': str(path),
    })
"""
    new_block = """loaded = {}
load_rows = []
real_reference_cache: dict[str, np.ndarray] = {}
real_reference_kind = (
    'model training subset; normalized from complete configured training set; '
    'evenly capped only after normalization'
)
for row in rows:
    path = sample_path_for(row)
    if not path.exists():
        continue
    config_path = PROJECT_DIR / row['config']
    reference_info = configured_training_reference_info(config_path)
    configured_slices = int(reference_info['configured_slices'])
    expected_slices = int(row['dataset_size'])
    reference_matches_manifest = configured_slices == expected_slices
    if not reference_matches_manifest:
        raise RuntimeError(
            f\"REAL REFERENCE MISMATCH for {row['run_name']}: config selects \"
            f\"{configured_slices} slices but manifest says {expected_slices}\"
        )
    generated = evenly_limit(load_npz_array(path), MAX_GENERATED)
    cache_key = real_reference_cache_key(config_path)
    if cache_key not in real_reference_cache:
        real_reference_cache[cache_key] = load_real_reference_from_config(
            config_path,
            max_slices=MAX_REAL_REFERENCE_SLICES,
        )
    real = real_reference_cache[cache_key]
    loaded[row['run_name']] = {
        'spec': row,
        'real': real,
        'generated': generated,
        'sample_path': path,
        'reference_info': reference_info,
        'real_reference_kind': real_reference_kind,
    }
    load_rows.append({
        'task_id': row['task_id'],
        'run_name': row['run_name'],
        'arch': row['arch'],
        'dataset_size': expected_slices,
        'n_real_exact_model_subset': configured_slices,
        'n_real_sources': int(reference_info['n_sources']),
        'n_real_raw_simulations': int(reference_info['configured_raw_samples']),
        'real_zthin': int(reference_info['zthin']),
        'n_real_used_for_plot': len(real),
        'reference_complete_for_plot': len(real) == configured_slices,
        'reference_matches_manifest': reference_matches_manifest,
        'reference_selection': reference_info['selection'],
        'real_reference_kind': real_reference_kind,
        'n_generated': len(generated),
        'sample_path': str(path),
        'config_path': str(config_path),
    })
"""
    load_source = source(load_code)
    if old_block not in load_source:
        if "n_real_exact_model_subset" not in load_source:
            raise RuntimeError("Could not find the U-Net real-reference loading block")
    else:
        load_source = load_source.replace(old_block, new_block)
    load_source = load_source.replace(
        """        'n_real_exact_model_subset': configured_slices,
        'n_real_used_for_plot': len(real),""",
        """        'n_real_exact_model_subset': configured_slices,
        'n_real_sources': int(reference_info['n_sources']),
        'n_real_raw_simulations': int(reference_info['configured_raw_samples']),
        'real_zthin': int(reference_info['zthin']),
        'n_real_used_for_plot': len(real),""",
    )
    set_source(load_code, load_source)

    detail_markdown = find_cell(notebook, "## Detailed One-point and P(k) Comparisons")
    set_source(
        detail_markdown,
        """## Detailed One-point and P(k) Comparisons

The black curve in every panel is the normalized reference from **that model's configured training subset**. It is not a histogram over all available CAMELS maps. The panel subtitle states how many of the configured slices are used after the plotting cap. Use `NF_FIG2_MAX_REAL_REFERENCE_SLICES=0` to use every configured slice, or change `NF_FIG2_DETAIL_TAGS` to select the displayed training sizes.
""",
    )

    detail_code = find_cell(notebook, "DETAIL_TAGS = [x.strip()")
    detail_source = source(detail_code)
    detail_source = detail_source.replace(
        "axes[0, col].plot(centers, rh['hist'], color='black', lw=2.4, label='real')",
        "axes[0, col].plot(centers, rh['hist'], color='black', lw=2.4, label='model training subset')",
    )
    detail_source = detail_source.replace(
        """axes[0, col].set_title(rf"$N_{{2D}}={int(row['dataset_size']):,}$ one-point")""",
        """configured_slices = int(bundle['reference_info']['configured_slices'])
                axes[0, col].set_title(
                    rf"$N_{{2D}}={configured_slices:,}$ one-point" + "\\n"
                    + f"black: {len(real):,}/{configured_slices:,} training slices"
                )""",
    )
    set_source(detail_code, detail_source)
    write_notebook(UNET_NOTEBOOK, notebook)


DIT_NEAREST_MARKDOWN = """## Generated Samples Versus Nearest Training Slices

This is the requested visual copy check for the DiT runs. For each selected generated map, the notebook searches the **complete configured training subset for that exact model** in pixel mean-squared error, then shows the generated map, its closest training slice, and their absolute difference.

The summary covers DiT-L8, DiT-L12, and DiT-L16 at $N_{2D}=2^6,\\ldots,2^{10}$. A second panel shows four DiT-L16 generated maps at $N_{2D}=2^8$ so the visibly noisy sample can be checked as a possible fluke. This pixel-space plot is a visual audit; PCA and SSCD remain the primary full-sample novelty metrics.
"""


DIT_NEAREST_CODE = r"""DIT_NN_ARCHES = parse_csv_env(
    'DIT_NN_ARCHES', 'dit_l8,dit_base,dit_l16'
)
DIT_NN_TAGS = parse_csv_env('DIT_NN_TAGS', 'd2p06,d2p07,d2p08,d2p09,d2p10')
DIT_NN_MAX_GENERATED = int(os.environ.get('DIT_NN_MAX_GENERATED', '4'))
DIT_NN_SUMMARY_GENERATED_INDEX = int(os.environ.get('DIT_NN_SUMMARY_GENERATED_INDEX', '0'))
DIT_NN_AUDIT_ARCH = os.environ.get('DIT_NN_AUDIT_ARCH', 'dit_l16')
DIT_NN_AUDIT_TAG = os.environ.get('DIT_NN_AUDIT_TAG', 'd2p08')


def require_complete_training_reference(bundle: dict[str, Any]) -> int:
    configured_slices = int(bundle['reference_info']['configured_slices'])
    loaded_slices = len(bundle['real'])
    if loaded_slices != configured_slices:
        raise RuntimeError(
            'nearest-training search requires the complete configured subset: '
            f"run={bundle['spec']['run_name']} loaded={loaded_slices} "
            f'configured={configured_slices}. Increase DIT_MAX_REAL_REFERENCE_SLICES '
            'or restrict DIT_NN_TAGS to sizes at or below the cap.'
        )
    return configured_slices


def build_dit_nearest_training_audit(
    arches: list[str] = DIT_NN_ARCHES,
    tags: list[str] = DIT_NN_TAGS,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    records = []
    by_run = {}
    for arch in arches:
        for bundle in choose_bundles(tags, arch=arch):
            row = bundle['spec']
            configured_slices = require_complete_training_reference(bundle)
            matches = nearest_training_matches(
                bundle['generated'],
                bundle['real'],
                max_generated=DIT_NN_MAX_GENERATED,
                max_training=None,
                training_chunk=256,
            )
            by_run[str(row['run_name'])] = {'bundle': bundle, 'matches': matches}
            for i, generated_index in enumerate(matches['generated_index']):
                records.append({
                    'arch': str(row['arch']),
                    'run_name': str(row['run_name']),
                    'dataset_tag': str(row['dataset_tag']),
                    'dataset_size': int(row['dataset_size']),
                    'generated_index': int(generated_index),
                    'nearest_training_index': int(matches['nearest_training_index'][i]),
                    'nearest_mse': float(matches['nearest_mse'][i]),
                    'nearest_rmse': float(matches['nearest_rmse'][i]),
                    'nearest_cosine': float(matches['nearest_cosine'][i]),
                    'n_training_searched': configured_slices,
                    'reference_is_complete': True,
                    'sample_path': rel(bundle['sample_path']),
                    'config_path': rel(bundle['config_path']),
                })
    audit = pd.DataFrame(records)
    if not audit.empty:
        audit = audit.sort_values(['arch', 'dataset_size', 'generated_index'])
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        out = TABLE_DIR / 'nf_generalize_fig2_dit_pixel_nearest_training.csv'
        audit.to_csv(out, index=False)
        print('wrote', out)
        display(audit)
    return audit, by_run


def nearest_plot_limits(items: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, float]:
    fields = np.concatenate([image.ravel() for pair in items for image in pair])
    differences = np.concatenate([np.abs(first - second).ravel() for first, second in items])
    return (
        float(np.nanquantile(fields, 0.005)),
        float(np.nanquantile(fields, 0.995)),
        float(np.nanquantile(differences, 0.995)),
    )


def plot_dit_nearest_training_summary(
    audit: pd.DataFrame,
    by_run: dict[str, dict[str, Any]],
    arch: str,
) -> Path | None:
    selected_arch = audit[audit['arch'] == arch]
    if selected_arch.empty:
        display(Markdown(f'No `{arch}` samples are available for the nearest-training summary.'))
        return None
    selected = selected_arch[
        selected_arch['generated_index'] == DIT_NN_SUMMARY_GENERATED_INDEX
    ].copy()
    if selected.empty:
        selected = selected_arch.groupby('run_name', as_index=False).head(1)
    selected = selected.sort_values('dataset_size')
    items = []
    for _, record in selected.iterrows():
        bundle = by_run[str(record['run_name'])]['bundle']
        generated = bundle['generated'][int(record['generated_index']), 0]
        training = bundle['real'][int(record['nearest_training_index']), 0]
        items.append((record, generated, training))
    vmin, vmax, diff_vmax = nearest_plot_limits([(gen, train) for _, gen, train in items])

    fig, axes = plt.subplots(
        3, len(items), figsize=(3.05 * len(items), 8.9),
        squeeze=False, constrained_layout=True,
    )
    for col, (record, generated, training) in enumerate(items):
        axes[0, col].imshow(generated, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, col].imshow(training, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[2, col].imshow(np.abs(generated - training), cmap='magma', vmin=0, vmax=diff_vmax)
        axes[0, col].set_title(dataset_size_label(int(record['dataset_size'])), fontweight='bold')
        axes[2, col].text(
            0.5, 0.02,
            f"MSE={record['nearest_mse']:.3g}; cos={record['nearest_cosine']:.3f}",
            transform=axes[2, col].transAxes, ha='center', va='bottom', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.82, pad=2),
        )
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
    for row_index, label in enumerate(('generated', 'nearest training', 'absolute difference')):
        axes[row_index, 0].set_ylabel(label, fontsize=14, fontweight='bold')
    fig.suptitle(
        f'{arch_label(arch)} generated samples versus nearest training slices',
        fontsize=21, fontweight='bold',
    )
    out = QUICKCHECK_DIR / f'nf_generalize_fig2_{arch}_generated_vs_nearest_training.png'
    QUICKCHECK_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.show()
    print('wrote', out)
    return out


def plot_dit_nearest_training_fluke_audit(
    audit: pd.DataFrame,
    by_run: dict[str, dict[str, Any]],
    arch: str = DIT_NN_AUDIT_ARCH,
    audit_tag: str = DIT_NN_AUDIT_TAG,
) -> Path | None:
    selected = audit[
        (audit['arch'] == arch) & (audit['dataset_tag'] == audit_tag)
    ].sort_values('generated_index')
    if selected.empty:
        display(Markdown(f'No DiT nearest-training records are available for `{audit_tag}`.'))
        return None
    run_name = str(selected.iloc[0]['run_name'])
    bundle = by_run[run_name]['bundle']
    items = []
    for _, record in selected.iterrows():
        generated = bundle['generated'][int(record['generated_index']), 0]
        training = bundle['real'][int(record['nearest_training_index']), 0]
        items.append((record, generated, training))
    vmin, vmax, diff_vmax = nearest_plot_limits([(gen, train) for _, gen, train in items])

    fig, axes = plt.subplots(
        len(items), 3, figsize=(9.6, 3.0 * len(items)),
        squeeze=False, constrained_layout=True,
    )
    for row_index, (record, generated, training) in enumerate(items):
        axes[row_index, 0].imshow(generated, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[row_index, 1].imshow(training, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[row_index, 2].imshow(np.abs(generated - training), cmap='magma', vmin=0, vmax=diff_vmax)
        axes[row_index, 0].set_ylabel(
            f"generated {int(record['generated_index'])}", fontsize=12, fontweight='bold'
        )
        axes[row_index, 2].text(
            0.5, 0.02,
            f"MSE={record['nearest_mse']:.3g}; cos={record['nearest_cosine']:.3f}",
            transform=axes[row_index, 2].transAxes, ha='center', va='bottom', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.82, pad=2),
        )
        for ax in axes[row_index]:
            ax.set_xticks([])
            ax.set_yticks([])
    for col, title in enumerate(('generated', 'nearest training', 'absolute difference')):
        axes[0, col].set_title(title, fontsize=14, fontweight='bold')
    n_value = int(selected.iloc[0]['dataset_size'])
    fig.suptitle(
        f'{arch_label(arch)} {dataset_size_label(n_value)}: '
        f'{len(items)} generated-sample nearest-training audit',
        fontsize=19, fontweight='bold',
    )
    out = QUICKCHECK_DIR / (
        f'nf_generalize_fig2_{arch}_{audit_tag}_nearest_training_audit.png'
    )
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.show()
    print('wrote', out)
    return out


dit_pixel_nn_df, dit_pixel_nn_by_run = build_dit_nearest_training_audit()
dit_pixel_nn_summary_paths = {
    arch: plot_dit_nearest_training_summary(dit_pixel_nn_df, dit_pixel_nn_by_run, arch)
    for arch in DIT_NN_ARCHES
}
dit_pixel_nn_fluke_path = plot_dit_nearest_training_fluke_audit(dit_pixel_nn_df, dit_pixel_nn_by_run)
"""


DIT_REVIEW_CHECKLIST = """## Nick review checklist

This notebook now answers the three review questions directly:

1. **Nearest training examples:** generated DiT maps are shown beside their nearest slice from the **exact training subset used by that model**, with a difference map and pixel-space similarity values.
2. **Black physical-statistics reference:** every one-point and $P(k)$ black curve is rebuilt from that model's configured training subset. The notebook checks the configured slice count against $N_{2D}$ and reports any plotting cap.
3. **In-distribution check:** an **SSCD Fréchet distance** compares generated maps with heldout real maps and is divided by a **real-vs-real split baseline**. A ratio near one means the generated set is no farther from heldout data than two finite real subsets are from each other. A much larger ratio flags novel but potentially invalid samples.

The third check is FID-style but uses SSCD features instead of literal ImageNet Inception features. Literal ImageNet Inception is a weak representation for single-channel HI fields; the calculation and finite-sample baseline are otherwise the same Fréchet distribution comparison.
"""


DIT_DISTRIBUTION_MARKDOWN = """## SSCD Distribution Distance: Are Novel Samples In Distribution?

Nearest-neighbor novelty can assign a high score to a visibly bad sample simply because it is far from every training image. This section adds the requested distribution-level check.

For each run, it loads the SSCD embeddings already cached by the full nearest-neighbor analysis, compares generated embeddings with **heldout real** embeddings, and reports a Fréchet feature distance. Because finite samples have a nonzero Fréchet distance even when both sets are real, the generated-to-heldout value is normalized by a same-size **real-vs-real split baseline**.

Read the ratio as follows:

- near 1: generated-to-real separation is comparable to finite real-split variability;
- substantially above 1: generated samples are farther from the real distribution, so a high novelty score may be invalid;
- below 1: not automatically better, because copying or low diversity can also reduce a distribution distance.

This is an in-distribution diagnostic, not a replacement for nearest-training plots, the one-point PDF, or $P(k)$.
"""


DIT_DISTRIBUTION_CODE = r"""SSCD_CACHE_DIR = RESULTS_DIR / 'cache' / 'sscd_full_nn'
DIT_DISTRIBUTION_ARCHES = parse_csv_env(
    'DIT_DISTRIBUTION_ARCHES', 'dit_l8,dit_base,dit_l16'
)
DIT_DISTRIBUTION_TAGS = parse_csv_env(
    'DIT_DISTRIBUTION_TAGS',
    'd2p06,d2p07,d2p08,d2p09,d2p10,d2p11,d2p12,d2p13,d2p14,d2p15',
)
DIT_DISTRIBUTION_COMPONENTS = int(os.environ.get('DIT_DISTRIBUTION_COMPONENTS', '64'))
DIT_DISTRIBUTION_SEED = int(os.environ.get('DIT_DISTRIBUTION_SEED', str(SEED)))


def find_sscd_embedding_cache(run_name: str, kind: str) -> Path | None:
    pattern = f'{run_name}_{kind}_{SAMPLE_LABEL}_seed{SEED}_*.pt'
    matches = sorted(
        SSCD_CACHE_DIR.glob(pattern),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    return matches[-1] if matches else None


def load_sscd_embedding_cache(path: Path) -> np.ndarray:
    import torch

    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location='cpu')
    tensor = payload['embeddings'] if isinstance(payload, dict) else payload
    features = np.asarray(tensor.detach().cpu(), dtype=np.float64)
    if features.ndim != 2 or len(features) < 4:
        raise ValueError(f'invalid SSCD embedding cache {path}: shape={features.shape}')
    return features


def project_to_real_pca(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    max_components: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    center = real_features.mean(axis=0, keepdims=True)
    centered_real = real_features - center
    centered_generated = generated_features - center
    max_rank = min(
        int(max_components),
        centered_real.shape[0] - 2,
        centered_real.shape[1],
    )
    if max_rank < 1:
        raise ValueError('not enough heldout real embeddings for PCA projection')
    _, _, right_vectors = np.linalg.svd(centered_real, full_matrices=False)
    basis = right_vectors[:max_rank].T
    return centered_real @ basis, centered_generated @ basis, max_rank


def evaluate_sscd_distribution_distance() -> pd.DataFrame:
    rows = []
    if manifest_df.empty:
        display(Markdown('No manifest is available for the SSCD distribution-distance audit.'))
        return pd.DataFrame()

    selected = manifest_df[
        manifest_df['arch'].isin(DIT_DISTRIBUTION_ARCHES)
        & manifest_df['dataset_tag'].isin(DIT_DISTRIBUTION_TAGS)
    ].sort_values(['arch', 'dataset_size'])
    for _, run in selected.iterrows():
        run_name = str(run['run_name'])
        heldout_path = find_sscd_embedding_cache(run_name, 'heldout')
        generated_path = find_sscd_embedding_cache(run_name, 'generated')
        if heldout_path is None or generated_path is None:
            rows.append({
                'arch': str(run['arch']),
                'arch_label': arch_label(run['arch']),
                'run_name': run_name,
                'dataset_tag': str(run['dataset_tag']),
                'dataset_size': int(run['dataset_size']),
                'status': 'missing SSCD cache',
                'heldout_cache': rel(heldout_path),
                'generated_cache': rel(generated_path),
            })
            continue

        heldout = load_sscd_embedding_cache(heldout_path)
        generated = load_sscd_embedding_cache(generated_path)
        heldout_projected, generated_projected, rank = project_to_real_pca(
            heldout,
            generated,
            DIT_DISTRIBUTION_COMPONENTS,
        )
        n_eval = min(len(generated_projected), len(heldout_projected) // 2)
        if n_eval < 2:
            raise ValueError(f'not enough equal-size samples for {run_name}')

        rng = np.random.default_rng(DIT_DISTRIBUTION_SEED)
        real_indices = rng.permutation(len(heldout_projected))[: 2 * n_eval]
        generated_indices = rng.permutation(len(generated_projected))[:n_eval]
        real_first = heldout_projected[real_indices[:n_eval]]
        real_second = heldout_projected[real_indices[n_eval:]]
        generated_eval = generated_projected[generated_indices]

        real_baseline = real_split_frechet_baseline(
            np.concatenate([real_first, real_second], axis=0),
            seed=DIT_DISTRIBUTION_SEED,
        )
        generated_distance = frechet_feature_distance(generated_eval, real_first)
        baseline_distance = float(real_baseline['distance'])
        ratio = generated_distance / max(baseline_distance, 1e-12)
        rows.append({
            'arch': str(run['arch']),
            'arch_label': arch_label(run['arch']),
            'run_name': run_name,
            'dataset_tag': str(run['dataset_tag']),
            'dataset_size': int(run['dataset_size']),
            'status': 'ok',
            'feature_space': 'SSCD projected on heldout-real PCA',
            'pca_rank': int(rank),
            'n_generated_eval': int(n_eval),
            'n_heldout_eval': int(n_eval),
            'generated_to_heldout_frechet': float(generated_distance),
            'real_split_frechet': baseline_distance,
            'generated_to_heldout_over_real_split': float(ratio),
            'heldout_cache': rel(heldout_path),
            'generated_cache': rel(generated_path),
        })

    frame = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / 'nf_generalize_fig2_dit_sscd_frechet_distribution_distance.csv'
    frame.to_csv(out, index=False)
    print('wrote', out)
    display(frame)
    return frame


def plot_sscd_distribution_distance(frame: pd.DataFrame) -> Path | None:
    required_columns = {
        'status',
        'generated_to_heldout_over_real_split',
        'dataset_size',
        'arch',
    }
    if frame.empty or not required_columns.issubset(frame.columns):
        display(Markdown(
            '**SSCD distribution-distance plot skipped:** cached heldout and generated '
            'SSCD embeddings were not found. Run the SSCD full-nearest-neighbor analysis first.'
        ))
        return None
    valid = frame[
        (frame['status'] == 'ok')
        & pd.to_numeric(
            frame['generated_to_heldout_over_real_split'], errors='coerce'
        ).notna()
    ].copy()
    if valid.empty:
        display(Markdown(
            '**SSCD distribution-distance plot skipped:** cached heldout and generated '
            'SSCD embeddings were not found. Run the SSCD full-nearest-neighbor analysis first.'
        ))
        return None

    fig, ax = plt.subplots(figsize=(11.8, 6.8), constrained_layout=True)
    for arch in DIT_ARCH_ORDER:
        sub = valid[valid['arch'] == arch].sort_values('dataset_size')
        if sub.empty:
            continue
        ax.plot(
            np.log2(sub['dataset_size'].astype(float)),
            sub['generated_to_heldout_over_real_split'].astype(float),
            color=DIT_ARCH_COLORS[arch],
            marker=DIT_ARCH_MARKERS[arch],
            lw=2.8,
            ms=8,
            label=arch_label(arch),
        )
    exponents = sorted(np.log2(valid['dataset_size'].astype(float)).astype(int).unique())
    ax.set_xticks(exponents)
    ax.set_xticklabels([f'$2^{{{exponent}}}$' for exponent in exponents])
    ax.axhline(1.0, color='0.25', lw=1.8, ls='--', label='real-vs-real baseline')
    ax.set_yscale('log')
    ax.set_xlabel(r'Training images $N_{2D}$')
    ax.set_ylabel('Generated-to-heldout / real-split Fréchet distance')
    ax.set_title('SSCD distribution distance: novelty does not imply validity', pad=14)
    ax.grid(alpha=0.18, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    out = QUICKCHECK_DIR / 'nf_generalize_fig2_dit_sscd_frechet_distribution_distance.png'
    QUICKCHECK_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.show()
    print('wrote', out)
    return out


dit_sscd_distribution_df = evaluate_sscd_distribution_distance()
dit_sscd_distribution_plot = plot_sscd_distribution_distance(dit_sscd_distribution_df)
"""


def update_dit_notebook() -> None:
    notebook = json.loads(DIT_NOTEBOOK.read_text())

    imports = find_cell(notebook, "from simdiff_eval.io import")
    imports_source = source(imports).replace(
        "from simdiff_eval.io import as_nchw, load_real_from_config, load_real_reference_from_config",
        "from simdiff_eval.io import (\n"
        "        as_nchw, configured_training_reference_info, load_real_from_config,\n"
        "        load_real_reference_from_config,\n"
        "    )",
    )
    canonical_metric_import = (
        "from simdiff_eval.metrics import "
        "batch_power_spectra, field_histogram, frechet_feature_distance, "
        "nearest_training_matches, real_split_frechet_baseline"
    )
    import_lines = []
    for line in imports_source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(
            "from simdiff_eval.metrics import batch_power_spectra, field_histogram"
        ):
            indent = line[: len(line) - len(stripped)]
            newline = "\n" if line.endswith("\n") else ""
            line = indent + canonical_metric_import + newline
        import_lines.append(line)
    imports_source = "".join(import_lines)
    set_source(imports, imports_source)

    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("codex_section")
        not in {"nick-review-checklist", "dit-distribution-distance"}
    ]
    tldr = find_cell(notebook, "## tl;dr")
    checklist_at = notebook["cells"].index(tldr) + 1
    notebook["cells"][checklist_at:checklist_at] = [
        {
            "cell_type": "markdown",
            "id": "nick-review-checklist",
            "metadata": {"codex_section": "nick-review-checklist"},
            "source": DIT_REVIEW_CHECKLIST.splitlines(keepends=True),
        }
    ]

    load_markdown = find_cell(notebook, "## Load Generated and Real Reference Slices")
    set_source(
        load_markdown,
        """## Load Generated and Real Reference Slices

Each reference is loaded from the **training subset configured for that exact DiT run**, not from all available CAMELS maps. The config's source paths, `n_samples`, seed, and `zthin` define the subset. The notebook verifies that its exact slice count equals the manifest's $N_{2D}$.

Normalization uses the complete configured subset. For memory safety, general plotting may then take an even cap controlled by `DIT_MAX_REAL_REFERENCE_SLICES` (default 2,048; 0 means all). The nearest-training audit below refuses to run unless the complete configured subset is loaded.
""",
    )

    load_code = find_cell(notebook, "real_reference_cache: dict[str, np.ndarray]")
    load_source = source(load_code)
    load_source = load_source.replace(
        "real_reference_kind = 'normalized from complete configured training set'",
        "real_reference_kind = (\n"
        "    'model training subset; normalized from complete configured training set; '\n"
        "    'evenly capped only after normalization'\n"
        ")",
    )
    load_source = load_source.replace(
        """        try:
            generated = evenly_limit(load_npz_array(sample_path), MAX_GENERATED)
            cache_key = real_reference_cache_key(cfg_path)""",
        """        try:
            reference_info = configured_training_reference_info(cfg_path)
            configured_slices = int(reference_info['configured_slices'])
            expected_slices = int(row.get('dataset_size'))
            reference_matches_manifest = configured_slices == expected_slices
            if not reference_matches_manifest:
                raise RuntimeError(
                    f\"REAL REFERENCE MISMATCH for {row.get('run_name')}: config selects \"
                    f\"{configured_slices} slices but manifest says {expected_slices}\"
                )
            generated = evenly_limit(load_npz_array(sample_path), MAX_GENERATED)
            cache_key = real_reference_cache_key(cfg_path)""",
    )
    load_source = load_source.replace(
        """                'config_path': cfg_path,
                'real_reference_kind': real_reference_kind,""",
        """                'config_path': cfg_path,
                'reference_info': reference_info,
                'real_reference_kind': real_reference_kind,""",
    )
    load_source = load_source.replace(
        """                'n_real_configured': int(row.get('dataset_size')),
                'n_real_loaded': len(real),
                'real_reference_kind': real_reference_kind,""",
        """                'n_real_exact_model_subset': configured_slices,
                'n_real_sources': int(reference_info['n_sources']),
                'n_real_raw_simulations': int(reference_info['configured_raw_samples']),
                'real_zthin': int(reference_info['zthin']),
                'n_real_used_for_plot': len(real),
                'reference_complete_for_plot': len(real) == configured_slices,
                'reference_matches_manifest': reference_matches_manifest,
                'reference_selection': reference_info['selection'],
                'real_reference_kind': real_reference_kind,""",
    )
    load_source = load_source.replace(
        """                'n_real_exact_model_subset': configured_slices,
                'n_real_used_for_plot': len(real),""",
        """                'n_real_exact_model_subset': configured_slices,
                'n_real_sources': int(reference_info['n_sources']),
                'n_real_raw_simulations': int(reference_info['configured_raw_samples']),
                'real_zthin': int(reference_info['zthin']),
                'n_real_used_for_plot': len(real),""",
    )
    set_source(load_code, load_source)

    onepoint_matches = [
        cell
        for cell in notebook["cells"]
        if (
            "## One-point and P(k) Fidelity Across Data Size" in source(cell)
            or "## One-point and P(k) Agreement Across Data Size" in source(cell)
        )
    ]
    if len(onepoint_matches) != 1:
        raise RuntimeError(
            "Expected one DiT one-point/P(k) section; "
            f"found {len(onepoint_matches)}"
        )
    onepoint_markdown = onepoint_matches[0]
    set_source(
        onepoint_markdown,
        """## One-point and P(k) Agreement Across Data Size

These panels ask a different question from memorization: do generated samples match real CAMELS summary statistics? The black curve is the reference from **that model's configured training subset**, not all CAMELS maps. The panel subtitle reports how many configured slices are used after the memory cap. Small-data runs can look good here by copying training slices, so read this together with the PCA/SSCD novelty curves and nearest-training plots.
""",
    )
    onepoint_code = find_cell(notebook, "def plot_dit_onepoint_pk")
    onepoint_source = source(onepoint_code).replace(
        "axes[0, col].plot(centers, rh['hist'], color='black', lw=2.2, label='real')",
        "axes[0, col].plot(centers, rh['hist'], color='black', lw=2.2, label='model training subset')",
    ).replace(
        """axes[0, col].set_title(f"{dataset_size_label(int(row['dataset_size']))} one-point")""",
        """configured_slices = int(bundle['reference_info']['configured_slices'])
        axes[0, col].set_title(
            f"{dataset_size_label(configured_slices)} one-point\\n"
            f"black: {len(real):,}/{configured_slices:,} training slices"
        )""",
    )
    set_source(onepoint_code, onepoint_source)

    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("codex_section") != "dit-nearest-training"
    ]
    image_grid = find_cell(notebook, "image_grid_path = plot_dit_image_grid")
    insert_at = notebook["cells"].index(image_grid) + 1
    new_cells = [
        {
            "cell_type": "markdown",
            "id": "dit-nn-md",
            "metadata": {"codex_section": "dit-nearest-training"},
            "source": DIT_NEAREST_MARKDOWN.splitlines(keepends=True),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "dit-nn-code",
            "metadata": {"codex_section": "dit-nearest-training"},
            "outputs": [],
            "source": DIT_NEAREST_CODE.splitlines(keepends=True),
        },
    ]
    notebook["cells"][insert_at:insert_at] = new_cells

    nearest_code = find_cell(notebook, "dit_pixel_nn_df, dit_pixel_nn_by_run")
    distribution_at = notebook["cells"].index(nearest_code) + 1
    notebook["cells"][distribution_at:distribution_at] = [
        {
            "cell_type": "markdown",
            "id": "dit-distribution-md",
            "metadata": {"codex_section": "dit-distribution-distance"},
            "source": DIT_DISTRIBUTION_MARKDOWN.splitlines(keepends=True),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "dit-distribution-code",
            "metadata": {"codex_section": "dit-distribution-distance"},
            "outputs": [],
            "source": DIT_DISTRIBUTION_CODE.splitlines(keepends=True),
        },
    ]
    write_notebook(DIT_NOTEBOOK, notebook)


def main() -> None:
    update_unet_notebook()
    update_dit_notebook()
    print(f"updated {UNET_NOTEBOOK}")
    print(f"updated {DIT_NOTEBOOK}")


if __name__ == "__main__":
    main()
