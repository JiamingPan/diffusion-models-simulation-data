#!/usr/bin/env python3
"""Build a reader-facing DiT results notebook with clearer figures and captions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "notebooks" / "nf_generalize_fig2_dit_results.ipynb"
DEFAULT_OUTPUT = DEFAULT_INPUT


def cell_source(cell: dict[str, Any]) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict[str, Any], value: str, *, clear_output: bool = True) -> None:
    cell["source"] = value.splitlines(keepends=True)
    if clear_output and cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


def find_cell(notebook: dict[str, Any], needle: str) -> dict[str, Any]:
    matches = [cell for cell in notebook["cells"] if needle in cell_source(cell)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one cell containing {needle!r}; found {len(matches)}"
        )
    return matches[0]


def find_cell_any(notebook: dict[str, Any], needles: tuple[str, ...]) -> dict[str, Any]:
    matches = [
        cell
        for cell in notebook["cells"]
        if any(needle in cell_source(cell) for needle in needles)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one cell containing one of {needles!r}; found {len(matches)}"
        )
    return matches[0]


def markdown_cell(text: str, section: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": section.replace("_", "-")[:64],
        "metadata": {"reader_section": section},
        "source": text.strip().splitlines(keepends=True),
    }


def code_cell(text: str, section: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": section.replace("_", "-")[:64],
        "metadata": {"reader_section": section},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def replace_reader_cell(
    notebook: dict[str, Any],
    *,
    section: str,
    text: str,
    after_needle: str,
) -> None:
    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("reader_section") != section
    ]
    anchor = find_cell(notebook, after_needle)
    index = notebook["cells"].index(anchor) + 1
    notebook["cells"].insert(index, markdown_cell(text, section))


def replace_between(source: str, start: str, end: str, replacement: str) -> str:
    start_index = source.find(start)
    end_index = source.find(end, start_index)
    if start_index < 0 or end_index < 0:
        raise RuntimeError(f"Could not replace block between {start!r} and {end!r}")
    return source[:start_index] + replacement + source[end_index:]


def ensure_unique_cell_ids(notebook: dict[str, Any]) -> None:
    """Normalize legacy notebooks to the cell-ID requirement in nbformat 4.5."""
    seen: set[str] = set()
    for index, cell in enumerate(notebook["cells"]):
        if isinstance(cell.get("source"), str):
            cell["source"] = cell["source"].splitlines(keepends=True)
        cell_id = cell.get("id")
        if not cell_id or cell_id in seen:
            digest = hashlib.sha1(
                (
                    f"{index}\0{cell.get('cell_type', '')}\0{cell_source(cell)}"
                ).encode("utf-8")
            ).hexdigest()[:12]
            cell_id = f"dit-{index:03d}-{digest}"
            cell["id"] = cell_id
        seen.add(cell_id)


ONEPOINT_CODE = r'''
def plot_dit_onepoint_pk(
    tags: list[str] = DETAIL_TAGS,
    arch: str = DETAIL_ARCH,
) -> dict[str, Path] | None:
    """Plot readable one-point and power-spectrum panels on separate figures."""
    if not loaded:
        display(Markdown('No loaded DiT samples available for one-point/P(k) plots.'))
        return None
    if not SIMDIFF_EVAL_AVAILABLE:
        display(Markdown('`simdiff_eval` unavailable; cannot compute one-point/P(k) diagnostics.'))
        return None

    bundles = choose_bundles(tags, max_count=5, arch=arch)
    if not bundles:
        display(Markdown(
            f'No selected `{arch_label(arch)}` bundles available for one-point/P(k) plots.'
        ))
        return None

    curves = []
    rows = []
    for bundle in bundles:
        row = bundle['spec']
        real = bundle['real']
        generated = bundle['generated']
        reference_info = bundle.get('reference_info', {})
        configured_slices = int(
            reference_info.get('configured_slices', row.get('dataset_size', len(real)))
        )

        real_hist = field_histogram(real, bins=140)
        edges = np.asarray(real_hist['bin_edges'])
        centers = 0.5 * (edges[:-1] + edges[1:])
        generated_density, _ = np.histogram(
            generated.ravel(), bins=edges, density=True
        )

        pk_real, kbins = batch_power_spectra(real, nbins=PK_NBINS)
        pk_generated, _ = batch_power_spectra(generated, nbins=PK_NBINS)
        mean_real = np.clip(np.nanmean(pk_real, axis=0), 1e-30, None)
        mean_generated = np.nanmean(pk_generated, axis=0)
        ratio = mean_generated / mean_real
        finite = np.isfinite(ratio)

        curves.append({
            'dataset_size': int(row['dataset_size']),
            'configured_slices': configured_slices,
            'n_real_used': len(real),
            'centers': centers,
            'real_hist': real_hist['hist'],
            'generated_hist': generated_density,
            'kbins': kbins,
            'pk_ratio': ratio,
        })
        rows.append({
            'arch': row.get('arch'),
            'arch_label': row.get('arch_label'),
            'run_name': row.get('run_name'),
            'dataset_tag': row.get('dataset_tag'),
            'dataset_size': int(row['dataset_size']),
            'n_real_exact_model_subset': configured_slices,
            'n_real_used_for_plot': len(real),
            'sample_path': rel(bundle.get('sample_path')),
            'config_path': rel(bundle.get('config_path')),
            'generated_shape': tuple(generated.shape),
            'real_shape': tuple(real.shape),
            'generated_mean': float(np.nanmean(generated)),
            'real_mean': float(np.nanmean(real)),
            'generated_std': float(np.nanstd(generated)),
            'real_std': float(np.nanstd(real)),
            'generated_min': float(np.nanmin(generated)),
            'real_min': float(np.nanmin(real)),
            'generated_max': float(np.nanmax(generated)),
            'real_max': float(np.nanmax(real)),
            'pk_ratio_median': float(np.nanmedian(ratio[finite])) if finite.any() else np.nan,
            'pk_ratio_min': float(np.nanmin(ratio[finite])) if finite.any() else np.nan,
            'pk_ratio_max': float(np.nanmax(ratio[finite])) if finite.any() else np.nan,
            'max_abs_pk_ratio_minus_1': (
                float(np.nanmax(np.abs(ratio[finite] - 1.0))) if finite.any() else np.nan
            ),
        })

    ncols = min(3, len(curves))
    nrows = int(np.ceil(len(curves) / ncols))
    generated_color = DIT_ARCH_COLORS.get(arch, '#0072B2')

    fig_pdf, pdf_axes = plt.subplots(
        nrows, ncols, figsize=(5.1 * ncols, 4.3 * nrows), squeeze=False
    )
    for axis, curve in zip(pdf_axes.ravel(), curves):
        axis.plot(
            curve['centers'], curve['real_hist'],
            color='black', lw=2.5, label='model training subset',
        )
        axis.plot(
            curve['centers'], curve['generated_hist'],
            color=generated_color, lw=2.3, label=f'{arch_label(arch)} generated',
        )
        axis.set_yscale('log')
        axis.set_title(
            f"{dataset_size_label(curve['dataset_size'])} training images\n"
            f"black: {curve['n_real_used']:,}/{curve['configured_slices']:,} slices",
            fontsize=15,
            pad=9,
        )
        axis.set_xlabel('Normalized field value')
        axis.set_ylabel('Pixel PDF')
        axis.grid(axis='y', alpha=0.14)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    for axis in pdf_axes.ravel()[len(curves):]:
        axis.set_visible(False)
    pdf_handles = [
        Line2D([0], [0], color='black', lw=2.5, label='model training subset'),
        Line2D([0], [0], color=generated_color, lw=2.3, label=f'{arch_label(arch)} generated'),
    ]
    fig_pdf.suptitle(
        f'{arch_label(arch)} one-point distributions',
        fontsize=21,
        fontweight='semibold',
        y=0.985,
    )
    fig_pdf.legend(
        handles=pdf_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    fig_pdf.subplots_adjust(
        left=0.075, right=0.985, bottom=0.09, top=0.82, hspace=0.38, wspace=0.26
    )
    pdf_out = QUICKCHECK_DIR / f'nf_generalize_fig2_{arch}_onepoint.png'
    QUICKCHECK_DIR.mkdir(parents=True, exist_ok=True)
    fig_pdf.savefig(pdf_out, bbox_inches='tight', dpi=300)
    plt.show()
    print('wrote', pdf_out)

    finite_ratio_chunks = [
        np.asarray(curve['pk_ratio'])[np.isfinite(curve['pk_ratio'])]
        for curve in curves
        if np.isfinite(curve['pk_ratio']).any()
    ]
    ratio_values = (
        np.concatenate(finite_ratio_chunks)
        if finite_ratio_chunks
        else np.asarray([1.0])
    )
    shared_upper = max(2.0, float(np.nanquantile(ratio_values, 0.99)) * 1.10)
    fig_pk, pk_axes = plt.subplots(
        nrows, ncols, figsize=(5.1 * ncols, 4.2 * nrows), squeeze=False
    )
    for axis, curve in zip(pk_axes.ravel(), curves):
        axis.plot(
            curve['kbins'], curve['pk_ratio'],
            marker='o', ms=4.7, lw=2.2, color=generated_color,
        )
        axis.axhline(1.0, color='black', ls='--', lw=1.5)
        axis.set_ylim(0, shared_upper)
        axis.set_title(
            f"{dataset_size_label(curve['dataset_size'])} training images",
            fontsize=15,
            pad=9,
        )
        axis.set_xlabel(r'$k$ bin')
        axis.set_ylabel(r'$P_{\mathrm{generated}}(k)/P_{\mathrm{real}}(k)$')
        axis.grid(axis='y', alpha=0.14)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    for axis in pk_axes.ravel()[len(curves):]:
        axis.set_visible(False)
    fig_pk.suptitle(
        f'{arch_label(arch)} power-spectrum ratios',
        fontsize=21,
        fontweight='semibold',
        y=0.985,
    )
    fig_pk.text(
        0.5,
        0.925,
        'The dashed line is exact agreement; all panels share the same vertical scale.',
        ha='center',
        fontsize=13,
        color='0.32',
    )
    fig_pk.subplots_adjust(
        left=0.075, right=0.985, bottom=0.09, top=0.82, hspace=0.36, wspace=0.26
    )
    pk_out = QUICKCHECK_DIR / f'nf_generalize_fig2_{arch}_pk_ratio.png'
    fig_pk.savefig(pk_out, bbox_inches='tight', dpi=300)
    plt.show()
    print('wrote', pk_out)

    fidelity_summary = pd.DataFrame(rows).sort_values(['arch', 'dataset_size'])
    display(fidelity_summary)
    table_out = TABLE_DIR / f'nf_generalize_fig2_{arch}_fidelity_summary.csv'
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fidelity_summary.to_csv(table_out, index=False)
    print('wrote', table_out)
    return {'onepoint': pdf_out, 'pk_ratio': pk_out}


fidelity_plot_path = plot_dit_onepoint_pk()
'''.strip()


EXPANDED_IMAGE_CODE = r'''
def choose_bundles(tags: list[str], arch: str | None = None) -> list[dict[str, Any]]:
    """Return exact requested bundles in tag order, without silent fallback."""
    requested = list(dict.fromkeys(str(tag) for tag in tags))
    by_tag = {
        str(bundle['spec'].get('dataset_tag')): bundle
        for bundle in loaded.values()
        if arch is None or str(bundle['spec'].get('arch')) == arch
    }
    missing = [tag for tag in requested if tag not in by_tag]
    if missing:
        display(Markdown(
            f"`{arch_label(arch) if arch else 'all architectures'}` requested tags are missing: "
            + ', '.join(f'`{tag}`' for tag in missing)
        ))
    return [by_tag[tag] for tag in requested if tag in by_tag]


def plot_dit_image_grid(
    sample_index: int = 0,
    tags: list[str] = IMAGE_TAGS,
    arch: str = IMAGE_ARCH,
    block_name: str = 'selected',
) -> Path | None:
    if not loaded:
        display(Markdown('No loaded DiT samples available for image grid.'))
        return None
    bundles = choose_bundles(tags, arch=arch)
    if not bundles:
        display(Markdown(f'No exact bundles available for `{arch_label(arch)}` image grid.'))
        return None

    values = []
    for bundle in bundles:
        gen_idx = min(sample_index, len(bundle['generated']) - 1)
        values.extend([
            bundle['generated'][gen_idx, 0].ravel(),
            bundle['real'][0, 0].ravel(),
        ])
    flat = np.concatenate(values)
    vmin, vmax = np.nanquantile(flat, [0.005, 0.995])

    fig, axes = plt.subplots(
        2,
        len(bundles),
        figsize=(3.05 * len(bundles), 6.2),
        squeeze=False,
        constrained_layout=True,
    )
    for col, bundle in enumerate(bundles):
        spec = bundle['spec']
        gen_idx = min(sample_index, len(bundle['generated']) - 1)
        axes[0, col].imshow(
            bundle['generated'][gen_idx, 0], cmap='viridis', vmin=vmin, vmax=vmax
        )
        axes[1, col].imshow(
            bundle['real'][0, 0], cmap='viridis', vmin=vmin, vmax=vmax
        )
        axes[0, col].set_title(dataset_size_label(int(spec['dataset_size'])), pad=8)
        for axis in axes[:, col]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_ylabel('generated', fontsize=15, fontweight='bold')
    axes[1, 0].set_ylabel('training subset', fontsize=15, fontweight='bold')
    fig.suptitle(
        f'{arch_label(arch)} generated maps: {block_name.replace("_", " ")}',
        fontsize=21,
        fontweight='semibold',
    )
    QUICKCHECK_DIR.mkdir(parents=True, exist_ok=True)
    out = QUICKCHECK_DIR / (
        f'nf_generalize_fig2_{arch}_{block_name}_generated_image_grid.png'
    )
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.show()
    print('wrote', out)
    return out


image_grid_paths = {}
for image_arch in DIT_ARCH_ORDER:
    for image_block_name, image_block_tags in DATA_TAG_BLOCKS.items():
        image_grid_paths[(image_arch, image_block_name)] = plot_dit_image_grid(
            sample_index=int(os.environ.get('DIT_IMAGE_SAMPLE_INDEX', '0')),
            tags=image_block_tags,
            arch=image_arch,
            block_name=image_block_name,
        )
image_grid_path = image_grid_paths
'''.strip()


EXPANDED_PHYSICAL_CODE = r'''
PHYSICAL_HIST_EDGES = np.linspace(-1.0, 1.0, 141, dtype=np.float64)
REAL_REFERENCE_RAW_BATCH_SIZE = int(os.environ.get('DIT_REAL_REFERENCE_RAW_BATCH_SIZE', '4'))
physical_curve_cache: dict[tuple[str, str], dict[str, Any]] = {}
real_physical_cache: dict[str, dict[str, Any]] = {}


def _radial_power_geometry(shape: tuple[int, int], nbins: int) -> tuple[np.ndarray, list[np.ndarray]]:
    height, width = shape
    ky = np.fft.fftfreq(height) * height
    kx = np.fft.fftfreq(width) * width
    kkx, kky = np.meshgrid(kx, ky)
    kvals = np.sqrt(kkx**2 + kky**2)
    valid = kvals > 0
    edges = np.linspace(kvals[valid].min(), kvals[valid].max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    masks = [
        (kvals >= edges[index]) & (kvals < edges[index + 1])
        for index in range(nbins)
    ]
    return centers, masks


def _aggregate_physical_batches(batches, *, nbins: int) -> dict[str, Any]:
    """Aggregate a PDF and mean P(k) without materializing every image."""
    histogram_counts = np.zeros(len(PHYSICAL_HIST_EDGES) - 1, dtype=np.int64)
    power_sum = np.zeros(nbins, dtype=np.float64)
    image_count = 0
    pixel_count = 0
    kbins = None
    masks = None
    for batch in batches:
        batch = as_nchw(np.asarray(batch, dtype=np.float32))
        if not len(batch):
            continue
        fields = np.asarray(batch[:, 0], dtype=np.float64)
        histogram_counts += np.histogram(fields.ravel(), bins=PHYSICAL_HIST_EDGES)[0]
        pixel_count += int(fields.size)
        if masks is None:
            kbins, masks = _radial_power_geometry(tuple(fields.shape[-2:]), nbins)
        centered = fields - fields.mean(axis=(-2, -1), keepdims=True)
        fft = np.fft.fftn(centered, axes=(-2, -1))
        power = (fft * fft.conj()).real / (fields.shape[-2] * fields.shape[-1])
        for index, mask in enumerate(masks):
            if mask.any():
                power_sum[index] += float(np.sum(np.mean(power[:, mask], axis=1)))
        image_count += len(fields)
    if image_count == 0 or kbins is None:
        raise RuntimeError('No images were available for physical-statistics aggregation.')
    widths = np.diff(PHYSICAL_HIST_EDGES)
    in_range = int(histogram_counts.sum())
    density = histogram_counts / np.clip(in_range * widths, 1, None)
    return {
        'hist': density,
        'hist_counts': histogram_counts,
        'hist_edges': PHYSICAL_HIST_EDGES.copy(),
        'kbins': kbins,
        'mean_pk': power_sum / image_count,
        'n_images': int(image_count),
        'pixel_coverage': float(in_range / pixel_count),
    }


def _band_log_error(ratio: np.ndarray, start: float, stop: float) -> float:
    ratio = np.asarray(ratio, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(ratio) & (ratio > 0))
    if not len(finite_indices):
        return np.nan
    lo = int(np.floor(start * len(finite_indices)))
    hi = int(np.floor(stop * len(finite_indices)))
    if stop >= 1.0:
        hi = len(finite_indices)
    selected = finite_indices[lo:max(lo + 1, hi)]
    return float(np.mean(np.abs(np.log10(ratio[selected]))))


def _physical_curve_for_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    spec = bundle['spec']
    arch = str(spec['arch'])
    tag = str(spec['dataset_tag'])
    key = (arch, tag)
    if key in physical_curve_cache:
        return physical_curve_cache[key]

    reference_key = real_reference_cache_key(bundle['config_path'])
    if reference_key not in real_physical_cache:
        real_physical_cache[reference_key] = _aggregate_physical_batches(
            iter_real_reference_batches_from_config(
                bundle['config_path'], raw_batch_size=REAL_REFERENCE_RAW_BATCH_SIZE
            ),
            nbins=PK_NBINS,
        )
    real_stats = real_physical_cache[reference_key]
    expected_real = int(bundle['reference_info']['configured_slices'])
    if int(real_stats['n_images']) != expected_real:
        raise RuntimeError(
            f"Exact real-reference count mismatch for {spec['run_name']}: "
            f"aggregated {real_stats['n_images']} but config selects {expected_real}."
        )
    generated_stats = _aggregate_physical_batches(
        [bundle['generated']], nbins=PK_NBINS
    )
    ratio = generated_stats['mean_pk'] / np.clip(real_stats['mean_pk'], 1e-30, None)
    widths = np.diff(real_stats['hist_edges'])
    curve = {
        'arch': arch,
        'arch_label': arch_label(arch),
        'dataset_tag': tag,
        'dataset_size': int(spec['dataset_size']),
        'run_name': str(spec['run_name']),
        'real_hist': real_stats['hist'],
        'generated_hist': generated_stats['hist'],
        'hist_edges': real_stats['hist_edges'],
        'kbins': real_stats['kbins'],
        'pk_ratio': ratio,
        'n_real_exact_model_subset': int(real_stats['n_images']),
        'n_generated': int(generated_stats['n_images']),
        'real_pixel_coverage': float(real_stats['pixel_coverage']),
        'generated_pixel_coverage': float(generated_stats['pixel_coverage']),
        'onepoint_hist_l1': float(
            np.sum(np.abs(generated_stats['hist'] - real_stats['hist']) * widths)
        ),
        'pk_log_ratio_mae': _band_log_error(ratio, 0.0, 1.0),
        'pk_low_log_ratio_mae': _band_log_error(ratio, 0.0, 1.0 / 3.0),
        'pk_mid_log_ratio_mae': _band_log_error(ratio, 1.0 / 3.0, 2.0 / 3.0),
        'pk_high_log_ratio_mae': _band_log_error(ratio, 2.0 / 3.0, 1.0),
        'pk_ratio_median': float(np.nanmedian(ratio)),
        'pk_ratio_min': float(np.nanmin(ratio)),
        'pk_ratio_max': float(np.nanmax(ratio)),
        'max_abs_pk_ratio_minus_1': float(np.nanmax(np.abs(ratio - 1.0))),
        'sample_path': rel(bundle['sample_path']),
        'config_path': rel(bundle['config_path']),
    }
    physical_curve_cache[key] = curve
    return curve


def build_dit_physical_summary() -> pd.DataFrame:
    rows = []
    for arch in DIT_ARCH_ORDER:
        for bundle in choose_bundles(ALL_DATA_TAGS, arch=arch):
            curve = _physical_curve_for_bundle(bundle)
            rows.append({
                name: value for name, value in curve.items()
                if name not in {
                    'real_hist', 'generated_hist', 'hist_edges', 'kbins', 'pk_ratio'
                }
            })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = summary.sort_values(['arch', 'dataset_size']).reset_index(drop=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = TABLE_DIR / 'nf_generalize_fig2_dit_physical_summary.csv'
    summary.to_csv(summary_path, index=False)
    for arch, sub in summary.groupby('arch', sort=False):
        sub.to_csv(TABLE_DIR / f'nf_generalize_fig2_{arch}_fidelity_summary.csv', index=False)
    print('wrote', summary_path)
    return summary


dit_physical_summary_df = build_dit_physical_summary()
display(dit_physical_summary_df)


def plot_dit_onepoint_pk(
    tags: list[str] = DETAIL_TAGS,
    arch: str = DETAIL_ARCH,
    block_name: str = 'selected',
) -> dict[str, Path] | None:
    bundles = choose_bundles(tags, arch=arch)
    if not bundles:
        return None
    curves = [_physical_curve_for_bundle(bundle) for bundle in bundles]
    color = DIT_ARCH_COLORS[arch]
    centers = 0.5 * (
        curves[0]['hist_edges'][:-1] + curves[0]['hist_edges'][1:]
    )

    fig_pdf, pdf_axes = plt.subplots(
        1, len(curves), figsize=(3.65 * len(curves), 4.2), squeeze=False
    )
    for axis, curve in zip(pdf_axes.ravel(), curves):
        axis.plot(centers, curve['real_hist'], color='black', lw=2.4)
        axis.plot(centers, curve['generated_hist'], color=color, lw=2.2)
        axis.set_yscale('log')
        axis.set_title(dataset_size_label(curve['dataset_size']), pad=8)
        axis.set_xlabel('Normalized field value')
        axis.grid(axis='y', alpha=0.14)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    pdf_axes[0, 0].set_ylabel('Pixel PDF')
    fig_pdf.suptitle(
        f'{arch_label(arch)} one-point distributions: {block_name.replace("_", " ")}',
        fontsize=20,
        fontweight='semibold',
        y=0.99,
    )
    fig_pdf.legend(
        handles=[
            Line2D([0], [0], color='black', lw=2.4, label='exact model training subset'),
            Line2D([0], [0], color=color, lw=2.2, label='generated'),
        ],
        loc='upper center', bbox_to_anchor=(0.5, 0.90), ncol=2, frameon=False,
    )
    fig_pdf.subplots_adjust(left=0.07, right=0.99, bottom=0.15, top=0.76, wspace=0.28)
    QUICKCHECK_DIR.mkdir(parents=True, exist_ok=True)
    pdf_out = QUICKCHECK_DIR / f'nf_generalize_fig2_{arch}_{block_name}_onepoint.png'
    fig_pdf.savefig(pdf_out, bbox_inches='tight', dpi=300)
    plt.show()
    print('wrote', pdf_out)

    ratios = np.concatenate([
        curve['pk_ratio'][np.isfinite(curve['pk_ratio'])] for curve in curves
    ])
    shared_upper = max(2.0, float(np.nanquantile(ratios, 0.99)) * 1.08)
    fig_pk, pk_axes = plt.subplots(
        1, len(curves), figsize=(3.65 * len(curves), 4.2), squeeze=False
    )
    for axis, curve in zip(pk_axes.ravel(), curves):
        axis.plot(curve['kbins'], curve['pk_ratio'], color=color, marker='o', ms=4, lw=2)
        axis.axhline(1.0, color='black', ls='--', lw=1.4)
        axis.set_ylim(0, shared_upper)
        axis.set_title(dataset_size_label(curve['dataset_size']), pad=8)
        axis.set_xlabel(r'$k$ bin')
        axis.grid(axis='y', alpha=0.14)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    pk_axes[0, 0].set_ylabel(r'$P_{\rm generated}(k)/P_{\rm real}(k)$')
    fig_pk.suptitle(
        f'{arch_label(arch)} power-spectrum ratios: {block_name.replace("_", " ")}',
        fontsize=20,
        fontweight='semibold',
        y=0.99,
    )
    fig_pk.text(
        0.5, 0.89, 'One is exact agreement; every panel uses the same vertical scale.',
        ha='center', fontsize=12.5, color='0.3',
    )
    fig_pk.subplots_adjust(left=0.07, right=0.99, bottom=0.15, top=0.75, wspace=0.28)
    pk_out = QUICKCHECK_DIR / f'nf_generalize_fig2_{arch}_{block_name}_pk_ratio.png'
    fig_pk.savefig(pk_out, bbox_inches='tight', dpi=300)
    plt.show()
    print('wrote', pk_out)

    outputs = {'onepoint': pdf_out, 'pk_ratio': pk_out}
    if block_name == 'high_data':
        fig_zoom, zoom_axes = plt.subplots(
            1, len(curves), figsize=(3.65 * len(curves), 4.0), squeeze=False
        )
        for axis, curve in zip(zoom_axes.ravel(), curves):
            axis.plot(curve['kbins'], curve['pk_ratio'], color=color, marker='o', ms=4, lw=2)
            axis.axhline(1.0, color='black', ls='--', lw=1.4)
            axis.set_ylim(0.75, 1.25)
            axis.set_title(dataset_size_label(curve['dataset_size']), pad=8)
            axis.set_xlabel(r'$k$ bin')
            axis.grid(axis='y', alpha=0.14)
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
        zoom_axes[0, 0].set_ylabel(r'$P_{\rm generated}(k)/P_{\rm real}(k)$')
        fig_zoom.suptitle(
            f'{arch_label(arch)} high-data power-spectrum ratio (zoom)',
            fontsize=20, fontweight='semibold', y=0.98,
        )
        fig_zoom.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.78, wspace=0.28)
        zoom_out = QUICKCHECK_DIR / f'nf_generalize_fig2_{arch}_high_data_pk_ratio_zoom.png'
        fig_zoom.savefig(zoom_out, bbox_inches='tight', dpi=300)
        plt.show()
        print('wrote', zoom_out)
        outputs['pk_ratio_zoom'] = zoom_out
    return outputs


fidelity_plot_paths = {}
for fidelity_arch in DIT_ARCH_ORDER:
    for fidelity_block_name, fidelity_block_tags in DATA_TAG_BLOCKS.items():
        fidelity_plot_paths[(fidelity_arch, fidelity_block_name)] = plot_dit_onepoint_pk(
            tags=fidelity_block_tags,
            arch=fidelity_arch,
            block_name=fidelity_block_name,
        )
fidelity_plot_path = fidelity_plot_paths


def plot_physical_error_summaries(summary: pd.DataFrame) -> dict[str, Path] | None:
    if summary.empty:
        return None
    outputs = {}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.3))
    for arch in DIT_ARCH_ORDER:
        sub = summary[summary['arch'] == arch].sort_values('dataset_size')
        x = np.log2(sub['dataset_size']).astype(int)
        style = dict(
            color=DIT_ARCH_COLORS[arch], marker=DIT_ARCH_MARKERS[arch], lw=2.5,
            ms=7, label=arch_label(arch),
        )
        axes[0].plot(x, sub['onepoint_hist_l1'], **style)
        axes[1].plot(x, sub['pk_log_ratio_mae'], **style)
    for axis, title, ylabel in zip(
        axes,
        ['One-point distribution error', 'Power-spectrum error'],
        [r'$L_1$ distance', r'mean $|\log_{10}(P_{gen}/P_{real})|$'],
    ):
        axis.set_xticks(range(6, 16), [rf'$2^{{{i}}}$' for i in range(6, 16)])
        axis.set_xlabel(r'Training images $N_{2D}$')
        axis.set_ylabel(ylabel)
        axis.set_title(title, pad=9)
        axis.grid(axis='y', alpha=0.16)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    fig.suptitle('Physical-statistics error across all DiT training sizes', fontsize=21, fontweight='semibold')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.90), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.75, wspace=0.23)
    out = QUICKCHECK_DIR / 'nf_generalize_fig2_dit_physical_error_by_data_size.png'
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.show()
    outputs['total_error'] = out

    fig_band, band_axes = plt.subplots(1, 3, figsize=(17.2, 5.0), sharey=True)
    columns = [
        ('pk_low_log_ratio_mae', 'Low $k$'),
        ('pk_mid_log_ratio_mae', 'Middle $k$'),
        ('pk_high_log_ratio_mae', 'High $k$'),
    ]
    for axis, (column, title) in zip(band_axes, columns):
        for arch in DIT_ARCH_ORDER:
            sub = summary[summary['arch'] == arch].sort_values('dataset_size')
            axis.plot(
                np.log2(sub['dataset_size']).astype(int), sub[column],
                color=DIT_ARCH_COLORS[arch], marker=DIT_ARCH_MARKERS[arch],
                lw=2.3, ms=7, label=arch_label(arch),
            )
        axis.set_xticks(range(6, 16), [rf'$2^{{{i}}}$' for i in range(6, 16)])
        axis.set_xlabel(r'Training images $N_{2D}$')
        axis.set_title(title, pad=9)
        axis.grid(axis='y', alpha=0.16)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    band_axes[0].set_ylabel(r'mean $|\log_{10}(P_{gen}/P_{real})|$')
    fig_band.suptitle('Power-spectrum error by scale', fontsize=21, fontweight='semibold')
    fig_band.legend(loc='upper center', bbox_to_anchor=(0.5, 0.89), ncol=3, frameon=False)
    fig_band.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.73, wspace=0.14)
    band_out = QUICKCHECK_DIR / 'nf_generalize_fig2_dit_pk_error_by_scale.png'
    fig_band.savefig(band_out, bbox_inches='tight', dpi=300)
    plt.show()
    outputs['scale_error'] = band_out
    return outputs


physical_error_plot_paths = plot_physical_error_summaries(dit_physical_summary_df)


def build_novelty_physical_table(summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for feature, metrics in [('PCA', pca_metrics), ('SSCD', sscd_metrics)]:
        if metrics.empty or 'gen_gl_q95' not in metrics.columns:
            continue
        metric_rows = ensure_arch_columns(metrics)[
            ['arch', 'dataset_tag', 'dataset_size', 'gen_gl_q95']
        ].drop_duplicates(['arch', 'dataset_tag'])
        joined = summary.merge(
            metric_rows, on=['arch', 'dataset_tag', 'dataset_size'], how='inner'
        )
        joined['feature'] = feature
        frames.append(joined)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


novelty_physical_df = build_novelty_physical_table(dit_physical_summary_df)
if not novelty_physical_df.empty:
    fig_joint, joint_axes = plt.subplots(1, 2, figsize=(14.5, 5.5), sharey=True)
    for axis, feature in zip(joint_axes, ['PCA', 'SSCD']):
        sub_feature = novelty_physical_df[novelty_physical_df['feature'] == feature]
        for arch in DIT_ARCH_ORDER:
            sub = sub_feature[sub_feature['arch'] == arch]
            axis.scatter(
                sub['gen_gl_q95'], sub['pk_log_ratio_mae'],
                color=DIT_ARCH_COLORS[arch], marker=DIT_ARCH_MARKERS[arch],
                s=75, label=arch_label(arch), alpha=0.9,
            )
        axis.axvline(0.5, color='0.4', ls=':', lw=1.4)
        axis.set_xlabel(f'{feature} q95 novelty score')
        axis.set_title(f'{feature} embedding', pad=9)
        axis.grid(alpha=0.15)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    joint_axes[0].set_ylabel(r'mean $|\log_{10}(P_{gen}/P_{real})|$')
    fig_joint.suptitle('DiT novelty versus physical-statistics error', fontsize=21, fontweight='semibold')
    fig_joint.text(
        0.5, 0.89,
        'Useful samples lie toward high novelty and low physical error; novelty alone is insufficient.',
        ha='center', fontsize=12.5, color='0.3',
    )
    handles, labels = joint_axes[0].get_legend_handles_labels()
    fig_joint.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.82), ncol=3, frameon=False)
    fig_joint.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.70, wspace=0.16)
    novelty_physical_out = QUICKCHECK_DIR / 'nf_generalize_fig2_dit_novelty_vs_physical_error.png'
    fig_joint.savefig(novelty_physical_out, bbox_inches='tight', dpi=300)
    plt.show()
    print('wrote', novelty_physical_out)
'''.strip()


VALIDITY_SCATTER = r'''

if not l16_validity_audit.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), sharey=True)
    for ax, (column, title) in zip(
        axes,
        [
            ('pca_gen_gl_q95', 'PCA novelty versus physical error'),
            ('sscd_gen_gl_q95', 'SSCD novelty versus physical error'),
        ],
    ):
        plot_data = l16_validity_audit.dropna(
            subset=[column, 'max_abs_pk_ratio_minus_1']
        )
        ax.scatter(
            plot_data[column],
            plot_data['max_abs_pk_ratio_minus_1'],
            s=95,
            c=np.log2(plot_data['dataset_size']),
            cmap='viridis',
            edgecolor='white',
            linewidth=0.9,
            zorder=3,
        )
        for _, point in plot_data.iterrows():
            exponent = int(round(np.log2(point['dataset_size'])))
            ax.annotate(
                rf'$2^{{{exponent}}}$',
                (point[column], point['max_abs_pk_ratio_minus_1']),
                xytext=(7, 6),
                textcoords='offset points',
                fontsize=11,
            )
        ax.axvline(0.5, color='0.35', ls=':', lw=1.4)
        ax.axhline(0.5, color='0.55', ls='--', lw=1.2)
        ax.set_xlabel('q95 novelty score')
        ax.set_title(title, fontsize=16, pad=10)
        ax.grid(alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].set_ylabel(r'Maximum $|P_{\mathrm{gen}}/P_{\mathrm{real}} - 1|$')
    fig.suptitle(
        'DiT-L16 novelty must be checked against physical agreement',
        fontsize=20,
        fontweight='semibold',
        y=0.98,
    )
    fig.text(
        0.5,
        0.91,
        'Upper-right points are far from training neighbors but also have a large power-spectrum error.',
        ha='center',
        fontsize=12.5,
        color='0.32',
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.79, wspace=0.16)
    l16_joint_out = QUICKCHECK_DIR / 'nf_generalize_fig2_dit_l16_novelty_vs_pk_error.png'
    fig.savefig(l16_joint_out, bbox_inches='tight', dpi=300)
    plt.show()
    print('wrote', l16_joint_out)
'''.rstrip()


CAPACITY_PLOT = r'''
fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6), sharey=True)
label_offsets = {
    'UNet-64': (8, 7),
    'UNet-128': (8, -17),
    'UNet-256': (-58, 8),
    'DiT-L8': (8, 8),
    'DiT-L12 / base': (-86, 10),
    'DiT-L16': (-58, -18),
}
family_style = {
    'UNet': {'color': '#6E6E6E', 'marker': 'o', 'label': 'UNet width'},
    'DiT': {'color': '#0072B2', 'marker': 'D', 'label': 'DiT depth'},
}
for ax, feature_name in zip(axes, ['PCA', 'SSCD']):
    sub = capacity_n50[
        (capacity_n50['feature'] == feature_name)
        & capacity_n50['n_cross'].notna()
    ].copy()
    if sub.empty:
        ax.set_visible(False)
        continue
    for family, style in family_style.items():
        family_data = sub[sub['family'] == family]
        ax.scatter(
            family_data['model_params'],
            family_data['n_cross'],
            color=style['color'],
            marker=style['marker'],
            s=90,
            edgecolor='white',
            linewidth=0.9,
            label=style['label'],
            zorder=3,
        )
        for _, row in family_data.iterrows():
            offset = label_offsets.get(row['model'], (7, 7))
            annotation_color = '#8B1A1A' if row['model'] == 'DiT-L16' else '0.18'
            ax.annotate(
                row['model'],
                (row['model_params'], row['n_cross']),
                xytext=offset,
                textcoords='offset points',
                fontsize=10.5,
                color=annotation_color,
            )
    ax.set_xscale('log')
    ax.set_yscale('log', base=2)
    ax.set_xlabel('Trainable parameters')
    ax.set_title(f'{feature_name} q95 crossing', fontsize=17, pad=10)
    ax.grid(axis='y', alpha=0.16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[0].set_ylabel(r'$N_{50}$ training images')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.88),
    ncol=2,
    frameon=False,
)
fig.suptitle(
    'Exploratory capacity diagnostic at fixed 200k updates',
    fontsize=21,
    fontweight='semibold',
    y=0.985,
)
fig.text(
    0.5,
    0.92,
    'Points are shown without a fitted or connecting line. DiT-L16 is not valid evidence for a scaling law.',
    ha='center',
    fontsize=12.5,
    color='0.32',
)
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.74, wspace=0.16)

out = QUICKCHECK_DIR / 'nf_generalize_fig2_dit_depth_capacity_n50_q95.png'
fig.savefig(out, bbox_inches='tight', dpi=300)
plt.show()
print('wrote', out)

'''.lstrip()


FIGURE_INVENTORY_CODE = r'''
saved_diagnostic_specs = [
    ('Training-loss comparison', QUICKCHECK_DIR / 'nf_generalize_fig2_dit_training_curves.png'),
    ('PCA/SSCD depth curves', QUICKCHECK_DIR / 'nf_generalize_fig2_dit_depth_gl_curves_q95.png'),
    ('DiT depth versus UNet', QUICKCHECK_DIR / 'nf_generalize_fig2_dit_depth_vs_unet_pca_sscd_q95.png'),
    ('Exploratory capacity diagnostic', QUICKCHECK_DIR / 'nf_generalize_fig2_dit_depth_capacity_n50_q95.png'),
    ('PCA paper-style curve', QUICKCHECK_DIR / 'nf_generalize_fig2_dit_pca_full_nn_paper_style_gl_curves.png'),
    ('SSCD paper-style curve', QUICKCHECK_DIR / 'nf_generalize_fig2_dit_sscd_full_nn_paper_style_gl_curves.png'),
]
for arch in DIT_ARCH_ORDER:
    saved_diagnostic_specs.extend([
        (
            f'{arch_label(arch)} generated image grid',
            QUICKCHECK_DIR / f'nf_generalize_fig2_{arch}_generated_image_grid.png',
        ),
        (
            f'{arch_label(arch)} one-point distributions',
            QUICKCHECK_DIR / f'nf_generalize_fig2_{arch}_onepoint.png',
        ),
        (
            f'{arch_label(arch)} power-spectrum ratios',
            QUICKCHECK_DIR / f'nf_generalize_fig2_{arch}_pk_ratio.png',
        ),
    ])

saved_diagnostic_inventory = pd.DataFrame([
    {
        'diagnostic': title,
        'path': rel(path),
        'exists': path.exists(),
        'size_mb': path.stat().st_size / 1024**2 if path.exists() else np.nan,
    }
    for title, path in saved_diagnostic_specs
])
display(saved_diagnostic_inventory)

if os.environ.get('DIT_SHOW_SAVED_DIAGNOSTICS', '0') == '1':
    for title, path in saved_diagnostic_specs:
        display(Markdown(f'### {title}'))
        if path.exists():
            display(Image(filename=str(path), width=950))
        else:
            display(Markdown(f'Missing: `{rel(path)}`'))
else:
    display(Markdown(
        'Duplicate figures are hidden by default. Set '
        '`DIT_SHOW_SAVED_DIAGNOSTICS=1` before running the notebook to display them.'
    ))
'''.strip()


CONDITIONAL_AUDIT_MARKDOWN = r'''
## Appendix: Conditional Calibration Input Audit

The DiT depth sweep above is **unconditional**: it measures memorization,
novelty, and physical statistics while changing architecture and training-set
size. This appendix audits a separate **conditional UNet** experiment used for
the cosmological-parameter recovery figure.

The conditional generator is given the complete six-dimensional CAMELS
parameter vector
$(\Omega_m,\sigma_8,A_{\rm SN1},A_{\rm AGN1},A_{\rm SN2},A_{\rm AGN2})$.
The code below verifies that both normalized and raw vectors have all six
columns and that every held-out condition has the requested number of generated
seeds. The poster's $\Omega_m$ panel is one projection of this full-vector
experiment; it is not evidence that the model was conditioned on $\Omega_m$
alone.

The 16th-to-84th percentile bars summarize variation across generated seeds at
a fixed requested condition. Their inclusion fraction is labeled
**seed-interval inclusion; not posterior coverage** because the seeds are not
samples from a Bayesian posterior over cosmological parameters.
'''.strip()


CONDITIONAL_AUDIT_CODE = r'''
expected_parameter_count = 6
conditional_parameter_names = [
    'Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_SN2', 'A_AGN2'
]
conditional_root = PROJECT_DIR / 'results' / 'nf_conditional_bias_probe'
conditional_sample_root = conditional_root / 'samples'
conditional_manifest_path = PROJECT_DIR / 'local' / 'nf_conditional_bias_probe' / 'manifest.json'

conditional_manifest_rows = []
if conditional_manifest_path.exists():
    conditional_manifest_rows = json.loads(conditional_manifest_path.read_text())
manifest_by_run = {
    str(row.get('run_name')): row for row in conditional_manifest_rows
}

conditional_audit_rows = []
conditional_sample_paths = sorted(conditional_sample_root.glob('*.npz'))
for sample_path in conditional_sample_paths:
    try:
        with np.load(sample_path, allow_pickle=True) as payload:
            files = set(payload.files)
            required = {'samples', 'theta_norm_repeated', 'theta_raw', 'heldout_indices', 'samples_per_cosmology'}
            missing_keys = sorted(required - files)
            if missing_keys:
                raise KeyError('missing arrays: ' + ', '.join(missing_keys))
            samples = np.asarray(payload['samples'])
            theta_norm_repeated = np.asarray(payload['theta_norm_repeated'])
            theta_raw = np.asarray(payload['theta_raw'])
            heldout_indices = np.atleast_1d(np.asarray(payload['heldout_indices']))
            samples_per_cosmology = int(np.asarray(payload['samples_per_cosmology']).item())
            run_name_value = payload['run_name'].item() if 'run_name' in files else sample_path.stem
            run_name = str(run_name_value)
        manifest_row = manifest_by_run.get(run_name, {})
        heldout_manifest_ok = False
        training_heldout_disjoint = False
        if manifest_row:
            heldout_path = Path(str(manifest_row.get('heldout_indices_path', '')))
            pairs_path = Path(str(manifest_row.get('selected_pairs_path', '')))
            if not heldout_path.is_absolute():
                heldout_path = PROJECT_DIR / heldout_path
            if not pairs_path.is_absolute():
                pairs_path = PROJECT_DIR / pairs_path
            if heldout_path.exists() and pairs_path.exists():
                manifest_heldout = np.atleast_1d(np.loadtxt(heldout_path, dtype=np.int64))
                training_pairs = pd.read_csv(pairs_path)
                training_simulations = set(training_pairs['simulation_index'].astype(int))
                heldout_manifest_ok = bool(np.array_equal(heldout_indices, manifest_heldout))
                training_heldout_disjoint = bool(
                    training_simulations.isdisjoint(set(manifest_heldout.astype(int)))
                )
        norm_shape_ok = bool(
            theta_norm_repeated.ndim == 2
            and theta_norm_repeated.shape[1] == expected_parameter_count
        )
        raw_shape_ok = bool(
            theta_raw.ndim == 2
            and theta_raw.shape[1] == expected_parameter_count
        )
        condition_count_ok = bool(
            len(theta_norm_repeated) == len(theta_raw) == len(heldout_indices)
        )
        repetition_ok = bool(
            len(samples) == len(theta_raw) * samples_per_cosmology
        )
        manifest_order = manifest_row.get('param_names', [])
        order_ok = bool(
            not manifest_order or list(manifest_order) == conditional_parameter_names
        )
        conditional_audit_rows.append({
            'run_name': run_name,
            'sample_path': rel(sample_path),
            'theta_norm_shape': tuple(theta_norm_repeated.shape),
            'theta_raw_shape': tuple(theta_raw.shape),
            'n_heldout_conditions': len(theta_raw),
            'samples_per_condition': samples_per_cosmology,
            'n_generated': len(samples),
            'six_normalized_parameters': norm_shape_ok,
            'six_raw_parameters': raw_shape_ok,
            'heldout_alignment_ok': condition_count_ok,
            'sample_repetition_ok': repetition_ok,
            'parameter_order_ok': order_ok,
            'heldout_indices_match_manifest': heldout_manifest_ok,
            'training_and_heldout_simulations_disjoint': training_heldout_disjoint,
            'full_vector_audit_pass': bool(
                norm_shape_ok and raw_shape_ok and condition_count_ok
                and repetition_ok and order_ok and heldout_manifest_ok
                and training_heldout_disjoint
            ),
        })
    except Exception as exc:
        conditional_audit_rows.append({
            'run_name': sample_path.stem,
            'sample_path': rel(sample_path),
            'full_vector_audit_pass': False,
            'error': repr(exc),
        })

conditional_input_audit_df = pd.DataFrame(conditional_audit_rows)
if conditional_input_audit_df.empty:
    display(Markdown(
        'Conditional sample files are not present in this checkout. '
        'Run the conditional pipeline on Great Lakes, then rerun this appendix.'
    ))
else:
    display(conditional_input_audit_df)
    if not conditional_input_audit_df['full_vector_audit_pass'].fillna(False).all():
        display(Markdown('**Conditional input audit failed for at least one file; do not use its calibration result.**'))

calibration_candidates = sorted(
    conditional_root.glob('**/bias_probe_per_cosmology_points.csv'),
    key=lambda path: path.stat().st_mtime if path.exists() else 0,
    reverse=True,
)
conditional_calibration_points = pd.DataFrame()
conditional_calibration_path = None
for candidate in calibration_candidates:
    candidate_df = pd.read_csv(candidate)
    required_columns = {
        'parameter', 'theta_in', 'theta_rec_median', 'theta_rec_q16', 'theta_rec_q84'
    }
    if required_columns.issubset(candidate_df.columns):
        available_parameters = set(candidate_df['parameter'].astype(str))
        if set(conditional_parameter_names).issubset(available_parameters):
            conditional_calibration_points = candidate_df
            conditional_calibration_path = candidate
            break

if conditional_calibration_points.empty:
    display(Markdown(
        'No six-parameter calibration table is available yet. The provenance '
        'audit above remains valid, but no recovery panel is drawn.'
    ))
else:
    display(Markdown(f'Calibration source: `{rel(conditional_calibration_path)}`'))
    fig_conditional, conditional_axes = plt.subplots(2, 3, figsize=(15.6, 9.3))
    inclusion_rows = []
    regime_values = list(dict.fromkeys(
        conditional_calibration_points.get(
            'regime', pd.Series(['all'] * len(conditional_calibration_points))
        ).astype(str)
    ))
    regime_colors = {
        regime: color for regime, color in zip(
            regime_values, ['#D55E00', '#0072B2', '#009E73', '#CC79A7']
        )
    }
    for axis, parameter in zip(conditional_axes.ravel(), conditional_parameter_names):
        parameter_rows = conditional_calibration_points[
            conditional_calibration_points['parameter'].astype(str) == parameter
        ].copy()
        for regime, regime_rows in parameter_rows.groupby(
            parameter_rows.get('regime', pd.Series('all', index=parameter_rows.index)).astype(str),
            sort=False,
        ):
            x = regime_rows['theta_in'].to_numpy(dtype=float)
            median = regime_rows['theta_rec_median'].to_numpy(dtype=float)
            q16 = regime_rows['theta_rec_q16'].to_numpy(dtype=float)
            q84 = regime_rows['theta_rec_q84'].to_numpy(dtype=float)
            order = np.argsort(x)
            axis.errorbar(
                x[order], median[order],
                yerr=np.vstack([median[order] - q16[order], q84[order] - median[order]]),
                fmt='o', ms=4.5, capsize=2.5, color=regime_colors.get(regime, '0.3'),
                alpha=0.82, label=regime,
            )
            included = (x >= q16) & (x <= q84)
            inclusion_rows.append({
                'parameter': parameter,
                'regime': regime,
                'seed_interval_inclusion_fraction': float(np.mean(included)),
                'n_conditions': int(len(included)),
                'diagnostic': 'seed-interval inclusion; not posterior coverage',
            })
        if not parameter_rows.empty:
            limits = np.nanmin(parameter_rows[['theta_in', 'theta_rec_q16']].to_numpy()), np.nanmax(
                parameter_rows[['theta_in', 'theta_rec_q84']].to_numpy()
            )
            axis.plot(limits, limits, color='0.25', ls='--', lw=1.4)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
        axis.set_title(parameter, pad=8)
        axis.set_xlabel('requested value')
        axis.set_ylabel('recovered value')
        axis.grid(alpha=0.15)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    handles, labels = conditional_axes[0, 0].get_legend_handles_labels()
    if handles:
        fig_conditional.legend(
            handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93),
            ncol=max(1, len(labels)), frameon=False,
        )
    fig_conditional.suptitle(
        'Conditional recovery uses the full six-parameter input vector',
        fontsize=21, fontweight='semibold', y=0.985,
    )
    fig_conditional.subplots_adjust(
        left=0.07, right=0.98, bottom=0.08, top=0.84, hspace=0.38, wspace=0.28
    )
    conditional_plot_path = QUICKCHECK_DIR / 'nf_conditional_bias_probe_six_parameter_calibration.png'
    fig_conditional.savefig(conditional_plot_path, bbox_inches='tight', dpi=300)
    plt.show()
    print('wrote', conditional_plot_path)
    seed_interval_inclusion_df = pd.DataFrame(inclusion_rows)
    display(Markdown('### Seed-interval inclusion; not posterior coverage'))
    display(seed_interval_inclusion_df)
'''.strip()


def update_notebook(input_path: Path, output_path: Path) -> None:
    notebook = json.loads(input_path.read_text())

    title = find_cell_any(
        notebook,
        (
            "# DiT Fig.2 Generalization Sweep",
            "# DiT Generalization and Physical-Validity Diagnostics",
        ),
    )
    set_source(
        title,
        """# DiT Generalization and Physical-Validity Diagnostics

Reader-facing analysis of the Fig.2-style DiT depth sweep. The notebook keeps three questions separate:

1. **Optimization:** did the denoising objective decrease?
2. **Novelty:** are generated maps unlike their nearest training examples?
3. **Physical agreement:** do generated maps reproduce the one-point distribution and power spectrum?

A model is useful only when all three checks are interpreted together.
""",
        clear_output=False,
    )

    tldr = find_cell(notebook, "## tl;dr")
    set_source(
        tldr,
        """## tl;dr

- DiT-L8 and DiT-L12 show a recognizable shift from training-neighbor behavior toward novel samples as the training set grows.
- The original DiT-L16 curve is anomalous at small data sizes. Some samples receive high PCA or SSCD novelty scores while visibly failing the power-spectrum check. Those points are **novel but not physically valid**.
- Training loss alone does not resolve this: DiT-L16 can fit the denoising objective well while generating statistically incorrect fields.
- The black one-point and power-spectrum references come from the **exact training subset configured for each model**, not the full CAMELS collection.
- The clean DiT-L16 300k sweep is the appropriate replacement experiment. Until it is complete, the notebook does not claim a depth-capacity scaling law.
""",
        clear_output=False,
    )

    nick_checklist = find_cell(notebook, "## Nick review checklist")
    set_source(
        nick_checklist,
        """## Nick review checklist

This notebook answers the requested review questions directly:

1. **Nearest training examples:** generated maps are compared with nearest slices from the complete training subset configured for each model, in both pixel and SSCD spaces.
2. **Black physical-statistics reference:** every one-point and $P(k)$ black curve is rebuilt from every slice in that model's exact training subset. It is not the full CAMELS collection and is not capped for plotting.
3. **In-distribution check:** SSCD Fréchet distance is normalized by a real-vs-real split baseline, so novel but out-of-distribution samples can be separated from useful generalization. This is an FID-style distribution check in domain-relevant SSCD features, not literal ImageNet Inception FID.
4. **Full data-size coverage:** generated maps, one-point PDFs, and $P(k)$ errors are shown for $2^6$ through $2^{15}$ for DiT-L8, DiT-L12, and DiT-L16.
5. **Conditional-input provenance:** the calibration appendix verifies all six CAMELS parameters, held-out manifest alignment, and disjoint training/test simulation indices before displaying recovery results.
""",
        clear_output=False,
    )

    context = find_cell(notebook, "## Context & Methods")
    context_index = notebook["cells"].index(context) + 1
    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("reader_section") != "reader_map"
    ]
    context = find_cell(notebook, "## Context & Methods")
    context_index = notebook["cells"].index(context) + 1
    notebook["cells"].insert(
        context_index,
        markdown_cell(
            """### How to read the figures

| Diagnostic | Question | Good result | Important limitation |
|---|---|---|---|
| Training loss | Is the optimizer fitting the denoising objective? | Loss decreases and remains finite | Low loss can coexist with memorization or bad physical statistics |
| Nearest training image | Is a generated field a near-copy? | Difference image is structured rather than nearly blank | Pixel distance is sensitive to shifts and does not measure distributional validity |
| PCA / SSCD q95 novelty | Is the generated set unusually close to training examples? | Score increases away from zero | A high score can also come from out-of-distribution noise |
| SSCD Fréchet ratio | Is the generated distribution close to heldout real maps? | Ratio near the real-vs-real baseline of one | Representation-dependent; not a replacement for physical statistics |
| One-point PDF | Are pixel values distributed correctly? | Generated and black curves overlap | Ignores spatial arrangement |
| Power spectrum | Is spatial structure correct across scale? | Generated/real ratio stays near one | Does not test every higher-order statistic |

The notebook therefore reads from **optimization**, to **visual copy checks**, to **distribution and physical-statistics checks**, and only then to the generalization curves.
""",
            "reader_map",
        ),
    )

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell_source(cell)
        source = source.replace(
            "'font.family': 'serif'", "'font.family': 'sans-serif'"
        )
        source = source.replace(
            "'font.serif': ['DejaVu Serif']",
            "'font.sans-serif': ['DejaVu Sans']",
        )
        source = source.replace(
            "'mathtext.fontset': 'dejavuserif'",
            "'mathtext.fontset': 'dejavusans'",
        )
        set_source(cell, source, clear_output=False)

    setup = find_cell(notebook, "plt.rcParams.update")
    setup_source = cell_source(setup)
    tag_block = """ALL_DATA_TAGS = [f'd2p{i:02d}' for i in range(6, 16)]
LOW_DATA_TAGS = ALL_DATA_TAGS[:5]
HIGH_DATA_TAGS = ALL_DATA_TAGS[5:]
DATA_TAG_BLOCKS = {
    'low_transition': LOW_DATA_TAGS,
    'high_data': HIGH_DATA_TAGS,
}
"""
    if "ALL_DATA_TAGS = [f'd2p{i:02d}' for i in range(6, 16)]" not in setup_source:
        setup_source = setup_source.replace(
            "SEED = int(os.environ.get('SEED', '123'))\n",
            "SEED = int(os.environ.get('SEED', '123'))\n\n" + tag_block,
        )
    setup_source = setup_source.replace(
        "as_nchw, configured_training_reference_info, load_real_from_config,\n"
        "        load_real_reference_from_config,",
        "as_nchw, configured_training_reference_info, iter_real_reference_batches_from_config,\n"
        "        load_real_from_config, load_real_reference_from_config,",
    )
    if "'figure.facecolor': 'white'" not in setup_source:
        setup_source = setup_source.replace(
            "'figure.dpi': 130,",
            "'figure.dpi': 130,\n"
            "    'figure.facecolor': 'white',\n"
            "    'axes.facecolor': 'white',\n"
            "    'font.family': 'sans-serif',\n"
            "    'font.sans-serif': ['DejaVu Sans'],\n"
            "    'mathtext.fontset': 'dejavusans',",
        )
    set_source(setup, setup_source)

    image_code = find_cell(notebook, "def choose_bundles")
    set_source(image_code, EXPANDED_IMAGE_CODE)

    onepoint_markdown = find_cell_any(
        notebook,
        (
            "## One-point and P(k)",
            "## One-point Distribution and Power-Spectrum Agreement",
        ),
    )
    set_source(
        onepoint_markdown,
        """## One-point Distribution and Power-Spectrum Agreement

These figures test physical agreement, not memorization.

- The **black curve is computed from every slice in the exact training subset used by that model**. It is not built from the complete CAMELS dataset and is not a capped plotting sample. The calculation streams batches from disk so the $2^{15}$ reference remains exact without exhausting notebook memory.
- The one-point distribution compares pixel values but ignores where those values occur.
- The power-spectrum ratio tests spatial structure. A ratio of one is exact agreement; values above or below one mean too much or too little power at that scale.

The PDF and $P(k)$ panels are separated into larger figures so their labels and scale-dependent failures remain readable.
""",
        clear_output=False,
    )
    onepoint_code = find_cell(notebook, "def plot_dit_onepoint_pk")
    set_source(onepoint_code, EXPANDED_PHYSICAL_CODE)

    validity_code = find_cell(notebook, "l16_validity_audit = build_l16_validity_audit()")
    validity_source = cell_source(validity_code)
    if "nf_generalize_fig2_dit_l16_novelty_vs_pk_error.png" not in validity_source:
        validity_source = validity_source.rstrip() + "\n" + VALIDITY_SCATTER + "\n"
    set_source(validity_code, validity_source)

    capacity_markdown = find_cell_any(
        notebook,
        ("## Fixed-Budget Capacity Check", "## Appendix: Exploratory Capacity Check"),
    )
    set_source(
        capacity_markdown,
        """## Appendix: Exploratory Capacity Check

This is a compact summary of the original fixed-200k sweep. The x-axis is parameter count and the y-axis is the interpolated data size where q95 novelty crosses 0.5.

The points are deliberately **not connected or fit with a line**. DiT-L16 has physically invalid small-data samples, so its apparent crossing cannot support a capacity-scaling claim. Use this figure only to locate hypotheses for a clean rerun.
""",
        clear_output=False,
    )
    capacity_code = find_cell(notebook, "# Parameter counts.")
    capacity_source = cell_source(capacity_code)
    capacity_source = replace_between(
        capacity_source,
        "fig, axes = plt.subplots(1, 2",
        "if not capacity_ratios.empty:",
        CAPACITY_PLOT,
    )
    set_source(capacity_code, capacity_source)

    quickcheck_markdown = find_cell_any(
        notebook,
        ("## Existing Quickcheck Figures", "## Appendix: Saved Diagnostic Inventory"),
    )
    set_source(
        quickcheck_markdown,
        """## Appendix: Saved Diagnostic Inventory

The notebook already creates the main figures above. Re-displaying every saved PNG made the notebook long and repeated the same evidence. This section now lists saved files and only displays them when explicitly requested.
""",
        clear_output=False,
    )
    quickcheck_code = find_cell_any(
        notebook,
        ("def show_existing_figure", "saved_diagnostic_specs = ["),
    )
    set_source(quickcheck_code, FIGURE_INVENTORY_CODE)

    legacy_continuation_needle = (
        "fig.suptitle('DiT depth comparison with continued L16 checkpoints'"
    )
    legacy_continuation_cells = [
        cell
        for cell in notebook["cells"]
        if legacy_continuation_needle in cell_source(cell)
    ]
    if len(legacy_continuation_cells) > 1:
        raise RuntimeError(
            "Expected at most one legacy continued-L16 comparison cell; "
            f"found {len(legacy_continuation_cells)}"
        )
    if legacy_continuation_cells:
        continuation_code = legacy_continuation_cells[0]
        continuation_source = cell_source(continuation_code)
        continuation_source = continuation_source.replace(
            "fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.0), sharey=True)",
            "fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.6), sharey=True)",
        )
        continuation_source = continuation_source.replace(
            "left=0.075, right=0.985, bottom=0.16, top=0.72, wspace=0.09",
            "left=0.075, right=0.985, bottom=0.14, top=0.69, wspace=0.09",
        )
        continuation_source = continuation_source.replace(
            "fig.suptitle('DiT depth comparison with continued L16 checkpoints', fontsize=25, y=0.97)",
            "fig.suptitle('DiT depth comparison with continued L16 checkpoints', fontsize=24, y=0.975)",
        )
        continuation_source = continuation_source.replace(
            "ha='center', fontsize=15, color='0.28',",
            "ha='center', fontsize=14, color='0.28',",
        )
        continuation_source = continuation_source.replace(
            "bbox_to_anchor=(0.5, 0.825),",
            "bbox_to_anchor=(0.5, 0.805),",
        )
        set_source(continuation_code, continuation_source)

    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("reader_section")
        not in {"conditional_audit_intro", "conditional_audit_code"}
    ]
    rerun_cell = find_cell(notebook, "## Great Lakes Rerun Command")
    rerun_index = notebook["cells"].index(rerun_cell)
    notebook["cells"].insert(
        rerun_index,
        markdown_cell(CONDITIONAL_AUDIT_MARKDOWN, "conditional_audit_intro"),
    )
    notebook["cells"].insert(
        rerun_index + 1,
        code_cell(CONDITIONAL_AUDIT_CODE, "conditional_audit_code"),
    )

    explanations = [
        (
            "loss_reading",
            "loss_df = pd.DataFrame(loss_rows)",
            """### Reading the optimization plots

- Each panel fixes the training-set size and compares model depth. The horizontal axis counts optimizer updates, not epochs.
- The vertical axis is the denoising objective on a logarithmic scale. The smoothed curves remove the 4,000-update restart oscillation so the long-run trend is visible.
- DiT-L16 can reach a low training loss at small and intermediate $N_{2D}$. That confirms optimization is occurring, but it does **not** establish good samples: the same checkpoints can still fail $P(k)$.
- The $2^{15}$ inset uses a linear vertical scale because the three curves are nearly coincident on the main logarithmic panel.
""",
        ),
        (
            "image_grid_reading",
            "image_grid_path = image_grid_paths",
            """### Reading the generated-map grid

Read columns as increasing training-set size. The top row is one generated example and the bottom row is one real reference displayed with the same color limits.

This is a qualitative failure check only. A plausible-looking field can still copy a training map, and a visibly different field can be out of distribution. The nearest-training, Fréchet, PDF, and $P(k)$ sections make those distinctions quantitative.
""",
        ),
        (
            "nearest_reading",
            "dit_pixel_nn_fluke_path = plot_dit_nearest_training_fluke_audit",
            """### Reading the nearest-training panels

- **Generated:** the model output.
- **Nearest training:** the closest slice from that model's complete configured training subset.
- **Absolute difference:** what cannot be explained by copying that nearest slice.

A nearly blank difference panel, very small MSE, and cosine similarity near one are evidence of copying. A large difference only establishes novelty; it must still be paired with the in-distribution and physical-statistics checks.
""",
        ),
        (
            "frechet_reading",
            "dit_sscd_distribution_plot = plot_sscd_distribution_distance",
            """### Reading the SSCD Fréchet-distance plot

The plotted ratio compares generated-to-heldout distance with a real-vs-real finite-sample baseline. A ratio near one means the generated distribution is about as close to heldout data as two real subsets are to each other. A much larger ratio flags outputs that may be novel simply because they are outside the real distribution.

This is the FID-style check requested in the project discussion, using SSCD features rather than ImageNet Inception features because the data are single-channel HI maps.
""",
        ),
        (
            "onepoint_reading",
            "fidelity_plot_path = fidelity_plot_paths",
            """### Reading the physical-statistics figures

For the one-point PDF, overlap with the black curve means the marginal pixel distribution is correct. For $P(k)$, agreement means the ratio stays close to one across all $k$ bins. Departures at high $k$ indicate incorrect small-scale structure even when the one-point distribution looks convincing.

The key DiT-L16 failure is that some low-data samples look novel in PCA/SSCD while their power-spectrum ratio differs from one by factors of several.
""",
        ),
        (
            "small_novelty_reading",
            "nf_generalize_fig2_dit_small_data_pca_sscd_q95_by_depth.png",
            """### Reading the small-data novelty comparison

Each panel fixes one DiT depth. Blue and orange show the same generated samples in two representation spaces. Agreement between PCA and SSCD makes the novelty conclusion more robust; disagreement means the result depends on the embedding.

L8 and L12 move from low to high novelty as data increase. L16 is already assigned moderate novelty at several small $N_{2D}$ values, which is suspicious because the corresponding image and $P(k)$ checks are poor.
""",
        ),
        (
            "small_images_reading",
            "small_data_image_paths = {}",
            """### What to look for across depths

Compare the same training-set-size column across L8, L12, and L16. The DiT-L16 $2^8$ example has visibly noisy or patch-like structure that does not resemble a successful cosmological field. The multi-sample nearest-training audit checks whether that image is a single fluke or a repeated failure mode.
""",
        ),
        (
            "small_fidelity_reading",
            "small_data_fidelity_paths = {}",
            """### Physical interpretation of the depth comparison

The one-point PDF can remain deceptively close even when spatial structure is wrong. The lower $P(k)$ panels are therefore the stronger warning here. A curve well above one has excess power; a curve below one is missing power. L16 at small and intermediate data sizes does not show a clean monotonic improvement.
""",
        ),
        (
            "small_loss_reading",
            "small_data_loss_paths = {}",
            """### Why the loss curves do not clear L16

All three depths reduce the denoising objective. L16 often reaches the smallest loss because the deeper network can fit the finite training set more aggressively. Since its generated fields can still have incorrect $P(k)$, this is evidence that loss alone is an insufficient checkpoint-selection criterion.
""",
        ),
        (
            "validity_reading",
            "l16_validity_audit = build_l16_validity_audit()",
            """### Reading the joint novelty-versus-error plot

Moving right means the samples are farther from their nearest training neighbors. Moving down means the power spectrum is closer to the training reference. The desired region is therefore **right and low**.

The problematic L16 points are right and high: they are novel according to PCA or SSCD but physically inconsistent. This is why they must not be counted as evidence that the deeper model generalizes with less data.
""",
        ),
        (
            "generalization_reading",
            "combined_curve = plot_dit_generalization_curves",
            """### Reading the DiT generalization curves

The horizontal 0.5 line is a descriptive midpoint, not a universal physical threshold. Curves that move from near zero to near one show a transition away from training-neighbor behavior as data increase.

L8 and L12 show the expected qualitative transition. The nonmonotonic L16 curve should be read as a failed validity check, not as a surprising reversal of the capacity relationship.
""",
        ),
        (
            "fixed_budget_reading",
            "combined_dit_unet_comparison = plot_dit_vs_unet_combined",
            """### Main fixed-budget comparison

This is the cleanest architecture comparison because every displayed DiT depth uses the same 200k-update budget. The L8-to-L12 shift is qualitatively consistent with a deeper model needing more data before leaving the copy-like regime.

The L16 points do not extend that trend cleanly because several small-data outputs are physically invalid. The clean 300k L16 rerun is evaluated separately rather than silently replacing points in this fixed-budget figure.
""",
        ),
        (
            "capacity_reading",
            "capacity_ratios = pd.DataFrame",
            """### Why this is only exploratory

$N_{50}$ compresses each full transition curve into one interpolated number. That is convenient, but it hides nonmonotonicity, slope, and physical failures. The absence of connecting lines is intentional: these six points do not yet establish a fitted scaling relation.
""",
        ),
        (
            "continuation_reading",
            "continuation_table_audit_df = pd.DataFrame",
            """### Reading the legacy continuation experiment

Rows and curves compare 200k, 225k, 250k, 275k, and 300k checkpoints for the low-data L16 runs. This experiment is useful for locating when failures appear, but the legacy loader reset optimizer and scheduler state between stages. It is therefore a checkpoint diagnostic, not a controlled statement about the effect of additional training.

Warnings are kept in this markdown rather than placed over the figure.
""",
        ),
    ]
    for section, needle, text in explanations:
        replace_reader_cell(
            notebook,
            section=section,
            text=text,
            after_needle=needle,
        )

    notebook.setdefault("metadata", {}).setdefault("reader_update", {})
    notebook["metadata"]["reader_update"] = {
        "audience": "research collaborators familiar with diffusion models",
        "purpose": "plain-language plot interpretation and layout cleanup",
        "source_notebook": str(input_path),
        "requires_rerun": True,
    }
    ensure_unique_cell_ids(notebook)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_notebook(args.input.resolve(), args.output.resolve())
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
