# U-Net VGG Patch-Level Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, VGG16-first analysis that tests whether local CAMELS feature families cross novelty-and-fidelity thresholds at different training-set sizes in the completed U-Net sweep.

**Architecture:** Extract fixed 16 by 16 patches from real and generated 128 by 128 fields, encode morphology with frozen VGG16 `relu3_3`, and fit one held-out-real StandardScaler, PCA64 compressor, and eight-cluster basis. Evaluate every U-Net width and data size against deterministic training-reference reservoirs with cluster-conditioned novelty, occupancy, feature fidelity, amplitude fidelity, map-level bootstrap intervals, and strict provenance audits; raw-patch PCA32 is a separately labeled control.

**Tech Stack:** Python 3, NumPy, pandas, SciPy, scikit-learn, PyTorch, TorchVision VGG16, Matplotlib, Jupyter, pytest, Slurm.

## Global Constraints

- Reuse only the completed unconditional U-Net-64, U-Net-128, and U-Net-256 runs at 200k optimizer updates and data sizes `2^6` through `2^15`; do not train or sample models.
- The primary encoder is frozen TorchVision VGG16 `relu3_3`; PCA64 only compresses its 512-dimensional pooled activations.
- The control encoder is raw standardized 16 by 16 patches compressed independently to PCA32 and labeled `raw-patch PCA` everywhere.
- Extract a fixed non-overlapping 8 by 8 grid of 16 by 16 patches from each 128 by 128 field.
- Standardize each patch independently, retain its original mean and standard deviation, clip standardized pixels to `[-4, 4]`, map to `[0, 1]`, and audit clipped-pixel fractions.
- Fit the StandardScaler, VGG-PCA64, raw PCA32, and MiniBatchKMeans with eight clusters exactly once on a held-out-real basis pool; never refit them per model, data size, or generated sample set.
- The basis pool, held-out-real evaluation pool, and all evaluated training subsets must be disjoint at simulation and slice level.
- Use every training image when the subset has at most 2,048 images; otherwise select a deterministic image-stratified 2,048-image reservoir and include all 64 patches.
- Bootstrap whole images, never individual patches, and report 95 percent intervals.
- Never call novelty alone generalization; joint acceptance requires novelty consistent with held-out real data and occupancy plus feature fidelity consistent with matched real-versus-real baselines.
- Store large feature caches under `/scratch/huterer_root/huterer0/$USER/nf_generalize_fig2_vgg_patch_features`; keep compact manifests, tables, figures, and audit files in the repository result tree.
- Great Lakes GPU arrays must use one GPU per task and at most two simultaneous tasks, use `afterok`, create log directories before submission, and never download VGG weights on compute nodes.
- Do not overwrite existing full-image PCA, SSCD, U-Net, or DiT artifacts.

---

## File Map

**Core modules**

- Create `simdiff_eval/patch_features.py`: deterministic patch extraction, preprocessing, frozen VGG encoding, raw descriptors, fitted-basis serialization, hashing, and assignment.
- Create `simdiff_eval/patch_generalization.py`: reservoirs, same-cluster nearest neighbors, real-only calibration, bootstrap summaries, distribution metrics, transition extraction, and joint acceptance.

**Pipeline scripts**

- Create `scripts/prepare_nf_generalize_fig2_vgg_patch_features.py`: validate the base sweep and write the analysis manifest.
- Create `scripts/fit_nf_generalize_fig2_vgg_patch_basis.py`: construct disjoint real pools and fit the frozen VGG and raw-PCA bases.
- Create `scripts/extract_nf_generalize_fig2_vgg_patch_references.py`: cache training reservoirs and held-out-real descriptors by data size.
- Create `scripts/extract_nf_generalize_fig2_vgg_patch_generated.py`: cache generated descriptors for all 30 U-Net runs.
- Create `scripts/analyze_nf_generalize_fig2_vgg_patch_run.py`: compute one architecture/data-size result with image bootstrap.
- Create `scripts/aggregate_nf_generalize_fig2_vgg_patch_results.py`: combine run tables, estimate thresholds, and render final figures.
- Create `scripts/audit_nf_generalize_fig2_vgg_patch_results.py`: fail closed on missing, inconsistent, or non-finite artifacts.
- Create `scripts/update_nf_generalize_fig2_unet_vgg_patch_notebook.py`: generate a compact results-only notebook.

**Great Lakes orchestration**

- Create `scripts/slurm/precheck_nf_generalize_fig2_vgg_patch.sbatch`.
- Create `scripts/slurm/fit_nf_generalize_fig2_vgg_patch_basis.sbatch`.
- Create `scripts/slurm/extract_nf_generalize_fig2_vgg_patch_references_array.sbatch`.
- Create `scripts/slurm/extract_nf_generalize_fig2_vgg_patch_generated_array.sbatch`.
- Create `scripts/slurm/analyze_nf_generalize_fig2_vgg_patch_array.sbatch`.
- Create `scripts/slurm/aggregate_nf_generalize_fig2_vgg_patch.sbatch`.
- Create `scripts/slurm/audit_nf_generalize_fig2_vgg_patch.sbatch`.
- Create `scripts/slurm/submit_nf_generalize_fig2_vgg_patch.sh`.
- Modify `scripts/slurm/README.md`: document scope, artifacts, resources, and restart commands.

**Notebook and tests**

- Create `notebooks/nf_generalize_fig2_unet_vgg_patch_generalization.ipynb` through the updater script.
- Create `tests/test_patch_features.py`.
- Create `tests/test_patch_generalization.py`.
- Create `tests/test_nf_fig2_vgg_patch_sweep.py`.
- Create `tests/test_vgg_patch_notebook.py`.
- Modify `requirements.txt`: add an explicit compatible `scikit-learn` dependency if it is not already declared.

---

### Task 1: Deterministic patch extraction and preprocessing

**Files:**
- Create: `simdiff_eval/patch_features.py`
- Test: `tests/test_patch_features.py`

**Interfaces:**
- Produces: `extract_patch_grid(images: np.ndarray, patch_size: int = 16) -> PatchBatch`
- Produces: `standardize_patches(patches: np.ndarray, clip: float = 4.0) -> StandardizedPatches`
- Defines: `PatchBatch(values, image_index, row, col)` and `StandardizedPatches(values, mean, std, clipped_fraction)` frozen dataclasses.

- [ ] **Step 1: Write failing extraction and preprocessing tests**

```python
def test_extract_patch_grid_preserves_order_and_provenance():
    images = np.arange(2 * 128 * 128, dtype=np.float32).reshape(2, 128, 128)
    batch = extract_patch_grid(images)
    assert batch.values.shape == (128, 16, 16)
    assert (batch.image_index[:64] == 0).all()
    assert (batch.image_index[64:] == 1).all()
    np.testing.assert_array_equal(batch.values[0], images[0, :16, :16])
    np.testing.assert_array_equal(batch.values[63], images[0, 112:128, 112:128])

def test_standardize_patches_retains_amplitude_and_reports_clipping():
    patches = np.stack([np.zeros((16, 16)), np.arange(256).reshape(16, 16)]).astype(np.float32)
    result = standardize_patches(patches, clip=4.0)
    assert result.values.shape == (2, 16, 16)
    assert np.isfinite(result.values).all()
    assert ((0.0 <= result.values) & (result.values <= 1.0)).all()
    assert result.mean.shape == result.std.shape == (2,)
    assert 0.0 <= result.clipped_fraction <= 1.0
```

- [ ] **Step 2: Run the tests and confirm the missing API failure**

Run: `pytest -q tests/test_patch_features.py -k 'extract_patch_grid or standardize_patches'`

Expected: collection or import failure because `patch_features.py` and its interfaces do not exist.

- [ ] **Step 3: Implement the dataclasses and deterministic transformations**

```python
@dataclass(frozen=True)
class PatchBatch:
    values: np.ndarray
    image_index: np.ndarray
    row: np.ndarray
    col: np.ndarray

@dataclass(frozen=True)
class StandardizedPatches:
    values: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    clipped_fraction: float

def extract_patch_grid(images: np.ndarray, patch_size: int = 16) -> PatchBatch:
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]
    if images.ndim != 3 or images.shape[1:] != (128, 128):
        raise ValueError(f"expected [N,128,128] or [N,1,128,128], got {images.shape}")
    if 128 % patch_size:
        raise ValueError("patch_size must divide 128")
    grid = 128 // patch_size
    patches = images.reshape(len(images), grid, patch_size, grid, patch_size).transpose(0, 1, 3, 2, 4)
    return PatchBatch(
        values=patches.reshape(-1, patch_size, patch_size),
        image_index=np.repeat(np.arange(len(images)), grid * grid),
        row=np.tile(np.repeat(np.arange(grid), grid), len(images)),
        col=np.tile(np.arange(grid), len(images) * grid),
    )
```

Use `std_floor=1e-6`; zero-variance patches become zero before clipping and map to `0.5`. Count clipping before mapping with `(x < -clip) | (x > clip)`.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_patch_features.py -k 'extract_patch_grid or standardize_patches'`

Expected: PASS.

- [ ] **Step 5: Commit the patch extraction unit**

```bash
git add simdiff_eval/patch_features.py tests/test_patch_features.py
git commit -m "Add deterministic CAMELS patch extraction"
```

### Task 2: Frozen intermediate VGG16 encoder and fitted basis

**Files:**
- Modify: `simdiff_eval/patch_features.py`
- Modify: `tests/test_patch_features.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `StandardizedPatches.values` in `[0,1]`.
- Produces: `resolve_vgg16_relu3_3(model: torch.nn.Module) -> torch.nn.Sequential`
- Produces: `encode_vgg_patches(values: np.ndarray, encoder: torch.nn.Module, device: torch.device, batch_size: int) -> np.ndarray`
- Produces: `fit_patch_basis(vgg_features: np.ndarray, raw_patches: np.ndarray, n_clusters: int, random_state: int) -> PatchFeatureBasis`
- Produces: `save_patch_basis(...)`, `load_patch_basis(...)`, and `basis_sha256(...)`.

- [ ] **Step 1: Add failing tests for the exact layer, descriptor width, and frozen serialization**

```python
def test_relu3_3_mapping_and_descriptor_width():
    model = torchvision.models.vgg16(weights=None)
    encoder = resolve_vgg16_relu3_3(model)
    assert len(encoder) == 16
    x = np.full((3, 16, 16), 0.5, dtype=np.float32)
    features = encode_vgg_patches(x, encoder, torch.device("cpu"), batch_size=2)
    assert features.shape == (3, 512)

def test_patch_basis_round_trip_has_stable_hash(tmp_path):
    basis = fit_patch_basis(vgg_features, raw_patches, n_clusters=8, random_state=123)
    path = tmp_path / "basis.joblib"
    save_patch_basis(path, basis)
    digest_before = basis_sha256(path)
    loaded = load_patch_basis(path)
    assert basis_sha256(path) == digest_before
    np.testing.assert_allclose(loaded.kmeans.cluster_centers_, basis.kmeans.cluster_centers_)
```

- [ ] **Step 2: Run the VGG tests and confirm failure**

Run: `pytest -q tests/test_patch_features.py -k 'relu3_3 or patch_basis'`

Expected: FAIL because the VGG and basis functions are missing.

- [ ] **Step 3: Implement VGG preprocessing, pooling, fitting, and stable persistence**

Resolve `relu3_3` by checking that `model.features[14]` is `Conv2d(256,256,3,1,1)` and `model.features[15]` is `ReLU`, then return `model.features[:16].eval()` with every parameter set to `requires_grad_(False)`. Resize to 64 by 64, repeat RGB channels, apply ImageNet mean/std, run under `torch.inference_mode()`, and concatenate adaptive average and maximum pooling to produce 512 values.

```python
@dataclass
class PatchFeatureBasis:
    vgg_scaler: StandardScaler
    vgg_pca: PCA
    raw_scaler: StandardScaler
    raw_pca: PCA
    kmeans: MiniBatchKMeans
    metadata: dict[str, Any]

def fit_patch_basis(vgg_features, raw_patches, n_clusters=8, random_state=123):
    vgg_scaler = StandardScaler().fit(vgg_features)
    vgg_pca = PCA(n_components=64, whiten=False, random_state=random_state).fit(
        vgg_scaler.transform(vgg_features)
    )
    raw = raw_patches.reshape(len(raw_patches), -1)
    raw_scaler = StandardScaler().fit(raw)
    raw_pca = PCA(n_components=32, whiten=False, random_state=random_state).fit(raw_scaler.transform(raw))
    compressed = vgg_pca.transform(vgg_scaler.transform(vgg_features))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=20).fit(compressed)
    return PatchFeatureBasis(vgg_scaler, vgg_pca, raw_scaler, raw_pca, kmeans, metadata={})
```

Persist with `joblib.dump(..., compress=3)` and hash the file bytes with SHA-256. Add an explicit scikit-learn version floor compatible with the Great Lakes environment only after checking `python -c 'import sklearn; print(sklearn.__version__)` locally and in the precheck.

- [ ] **Step 4: Run all patch-feature tests**

Run: `pytest -q tests/test_patch_features.py`

Expected: PASS on CPU without downloading pretrained weights because tests use `weights=None`.

- [ ] **Step 5: Commit the VGG feature basis**

```bash
git add simdiff_eval/patch_features.py tests/test_patch_features.py requirements.txt
git commit -m "Add frozen VGG patch feature basis"
```

### Task 3: Analysis manifest and disjoint real pools

**Files:**
- Create: `scripts/prepare_nf_generalize_fig2_vgg_patch_features.py`
- Create: `scripts/fit_nf_generalize_fig2_vgg_patch_basis.py`
- Create: `tests/test_nf_fig2_vgg_patch_sweep.py`

**Interfaces:**
- Consumes: `local/nf_generalize_fig2/manifest.json`, each run YAML, and existing sample archives.
- Produces: `prepare_manifest(base_manifest_path: Path, output_dir: Path) -> list[dict[str, Any]]`.
- Produces: `audit_split_manifest(split_manifest: dict[str, Any], training_identities: set[str]) -> dict[str, Any]`.
- Produces: `local/nf_generalize_fig2_vgg_patch/manifest.json` with exactly 30 run rows.
- Produces: `local/nf_generalize_fig2_vgg_patch/split_manifest.json` with basis/evaluation simulation and slice identities.
- Produces: `$FEATURE_CACHE_ROOT/basis/vgg_patch_basis.joblib` and `basis_metadata.json`.

- [ ] **Step 1: Write failing manifest and disjointness tests**

```python
def test_analysis_manifest_has_fixed_unet_grid(tmp_path):
    manifest = prepare_manifest(base_manifest_path, tmp_path)
    assert len(manifest) == 30
    assert {row["arch_label"] for row in manifest} == {"U-Net-64", "U-Net-128", "U-Net-256"}
    assert {row["dataset_tag"] for row in manifest} == {f"d2p{x:02d}" for x in range(6, 16)}
    assert {row["target_updates"] for row in manifest} == {200_000}

def test_split_manifest_is_disjoint_from_all_training_and_itself():
    audit = audit_split_manifest(split_manifest, training_identities)
    assert audit["training_overlap"] == []
    assert audit["basis_evaluation_overlap"] == []
```

- [ ] **Step 2: Run sweep tests and confirm failure**

Run: `pytest -q tests/test_nf_fig2_vgg_patch_sweep.py -k 'manifest or disjoint'`

Expected: FAIL because the preparation APIs are missing.

- [ ] **Step 3: Implement manifest preparation and explicit split auditing**

Parse every source YAML through structured YAML APIs. Resolve sample paths, run names, architecture labels, data tags, configured training identities, and normalization metadata. Assert the three architectures use identical data selections at each `dataset_tag` before setting `reference_cache_key=dataset_tag`.

Build the held-out complement of the union of every evaluated training identity. Stratify by source where possible, split by raw simulation before selecting slices, and deterministically assign 128 simulations to the basis pool and 128 to evaluation using seed 123. If the complement contains fewer than 256 simulations, fail with a count report instead of reducing the split silently. Record why any source, such as an exhausted CV component, is absent.

- [ ] **Step 4: Implement basis fitting with pretrained-weight preflight**

The script must accept:

```text
--manifest PATH --split-manifest PATH --cache-root PATH
--vgg-weights PATH --basis-images 2048 --batch-size 512 --seed 123
```

Load VGG16 weights only from `--vgg-weights`; fail before data loading if missing. Materialize at most 2,048 basis images using the split manifest, extract 131,072 patches, fit the frozen transforms, save representative real patches nearest each cluster center, and write metadata containing layer name, preprocessing constants, explained variance, inertia, cluster counts, source identities, and basis SHA-256.

- [ ] **Step 5: Run focused tests and syntax checks**

Run: `pytest -q tests/test_nf_fig2_vgg_patch_sweep.py -k 'manifest or disjoint'`

Run: `python -m py_compile scripts/prepare_nf_generalize_fig2_vgg_patch_features.py scripts/fit_nf_generalize_fig2_vgg_patch_basis.py`

Expected: PASS.

- [ ] **Step 6: Commit preparation and basis scripts**

```bash
git add scripts/prepare_nf_generalize_fig2_vgg_patch_features.py scripts/fit_nf_generalize_fig2_vgg_patch_basis.py tests/test_nf_fig2_vgg_patch_sweep.py
git commit -m "Prepare frozen VGG patch analysis basis"
```

### Task 4: Reference and generated feature caches

**Files:**
- Create: `scripts/extract_nf_generalize_fig2_vgg_patch_references.py`
- Create: `scripts/extract_nf_generalize_fig2_vgg_patch_generated.py`
- Modify: `tests/test_nf_fig2_vgg_patch_sweep.py`

**Interfaces:**
- Consumes: analysis/split manifests and `PatchFeatureBasis`.
- Produces: `select_reference_images(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame`.
- Produces: `write_feature_cache(path: Path, arrays: dict[str, np.ndarray], provenance: dict[str, Any]) -> None`.
- Produces: `load_feature_cache(path: Path, expected_basis_sha256: str) -> FeatureCache`.
- Produces: one reference cache per data size under `$FEATURE_CACHE_ROOT/references/{dataset_tag}.npz`.
- Produces: one generated cache per run under `$FEATURE_CACHE_ROOT/generated/{run_name}.npz`.
- Cache keys: `vgg_pca`, `raw_pca`, `cluster`, `patch_mean`, `patch_std`, `image_id`, `row`, `col`, plus provenance JSON.

- [ ] **Step 1: Add failing tests for reservoir determinism and cache provenance**

```python
def test_reservoir_is_image_stratified_and_deterministic():
    first = select_reference_images(frame, limit=2048, seed=123)
    second = select_reference_images(frame, limit=2048, seed=123)
    pd.testing.assert_frame_equal(first, second)
    assert first["image_id"].nunique() == min(2048, frame["image_id"].nunique())

def test_cache_rejects_moving_basis(tmp_path):
    write_feature_cache(path, arrays, provenance={"basis_sha256": "abc"})
    with pytest.raises(ValueError, match="basis hash"):
        load_feature_cache(path, expected_basis_sha256="def")
```

- [ ] **Step 2: Run cache tests and confirm failure**

Run: `pytest -q tests/test_nf_fig2_vgg_patch_sweep.py -k 'reservoir or cache'`

Expected: FAIL because cache selection and validation are absent.

- [ ] **Step 3: Implement reference extraction**

For each data tag, reconstruct the exact training subset from its YAML with `simdiff_eval.io` helpers. Select all images up to 2,048, otherwise stratify the deterministic reservoir by source and simulation, then include all 64 patches from each selected image. Encode the fixed held-out evaluation pool once under the same configured normalization contract and include it in each data-size cache with distinct `split=train_reservoir|heldout_evaluation` labels.

Accept `--reservoir-images` so the transition sensitivity jobs can create separately named 1,024-, 2,048-, and 4,096-image caches without overwriting the default.

- [ ] **Step 4: Implement generated extraction**

Read each existing DPM-Solver 50-step archive from the prepared manifest, require exactly the recorded sample label and source run, normalize the array shape, encode every generated field, and store the generated seed and sample index in image-level provenance. Never substitute a continuation, DDPM, DiT, or differently labeled sample path.

- [ ] **Step 5: Run cache tests and static checks**

Run: `pytest -q tests/test_nf_fig2_vgg_patch_sweep.py -k 'reservoir or cache'`

Run: `python -m py_compile scripts/extract_nf_generalize_fig2_vgg_patch_references.py scripts/extract_nf_generalize_fig2_vgg_patch_generated.py`

Expected: PASS.

- [ ] **Step 6: Commit cache extraction**

```bash
git add scripts/extract_nf_generalize_fig2_vgg_patch_references.py scripts/extract_nf_generalize_fig2_vgg_patch_generated.py tests/test_nf_fig2_vgg_patch_sweep.py
git commit -m "Cache VGG patch references and generations"
```

### Task 5: Cluster-conditioned novelty, fidelity, and image bootstrap

**Files:**
- Create: `simdiff_eval/patch_generalization.py`
- Create: `tests/test_patch_generalization.py`

**Interfaces:**
- Produces: `cosine_nearest_within_cluster(query, reference, query_cluster, reference_cluster, query_image_id=None, reference_image_id=None) -> NearestResult`.
- Produces: `calibrate_copy_threshold(training_cache, cluster: int, quantile: float = 0.95) -> float` with same-image exclusion.
- Produces: `cluster_metrics(generated_cache, reference_cache, cluster: int, rng: np.random.Generator) -> dict[str, float]`.
- Produces: `bootstrap_cluster_metrics(..., n_bootstrap: int = 500) -> pd.DataFrame` resampling image IDs.
- Produces: `bootstrap_image_ids(image_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray`.
- Test helper: `synthetic_cluster_metrics(novelty_matches_real: bool, fidelity_matches_real: bool) -> dict[str, bool]` remains local to `tests/test_patch_generalization.py`.

- [ ] **Step 1: Write failing same-image exclusion and metric tests**

```python
def test_copy_threshold_excludes_patches_from_same_image():
    result = cosine_nearest_within_cluster(
        x, x, clusters, clusters, query_image_id=image_ids, reference_image_id=image_ids
    )
    assert np.all(result.reference_image_id != image_ids)

def test_novel_but_wrong_generation_fails_joint_acceptance():
    metrics = synthetic_cluster_metrics(novelty_matches_real=True, fidelity_matches_real=False)
    assert metrics["novelty_accept"]
    assert not metrics["fidelity_accept"]
    assert not metrics["joint_accept"]

def test_bootstrap_samples_whole_images():
    draws = bootstrap_image_ids(np.array([0, 0, 1, 1]), np.random.default_rng(7))
    for image_id in np.unique(draws):
        assert (draws == image_id).sum() % 2 == 0
```

- [ ] **Step 2: Run metric tests and confirm failure**

Run: `pytest -q tests/test_patch_generalization.py`

Expected: FAIL because `patch_generalization.py` is missing.

- [ ] **Step 3: Implement chunked same-cluster cosine search and real-only calibration**

L2-normalize compressed VGG descriptors and compute cosine products in bounded blocks. Require at least 100 query and 100 reference patches in a cluster. For train-versus-train calibration, mask all references sharing the query image ID, not only the identical patch. Define copied as nearest cosine at or above the cluster/data q95 threshold; define novelty fraction as one minus copied fraction. Compute the same quantity for held-out-real patches against the training reservoir.

- [ ] **Step 4: Implement occupancy, feature-distribution, amplitude, and joint metrics**

Use `simdiff_eval.metrics.frechet_feature_distance` with matched deterministic subsampling and `real_split_frechet_baseline`. Add mean-vector L2 error, covariance Frobenius error, occupancy absolute error, global Jensen-Shannon divergence, and one-dimensional Wasserstein distances for patch mean and standard deviation.

Derive real-only acceptance bands from repeated matched held-out-real splits. Set:

```python
novelty_accept = abs(generated_novelty - heldout_novelty) <= real_novelty_tolerance
occupancy_accept = generated_occupancy_error <= real_occupancy_q95
fidelity_accept = generated_frechet <= real_split_frechet_q95
amplitude_accept = max(mean_wasserstein, std_wasserstein) <= amplitude_real_q95
joint_accept = novelty_accept and occupancy_accept and fidelity_accept and amplitude_accept
```

Never tune these bands from generated samples.

- [ ] **Step 5: Implement image-level bootstrap**

Resample generated and held-out-real image IDs independently with replacement, carry all 64 patches for a selected image, reuse bootstrap draws across clusters, and report median, 2.5th, and 97.5th percentiles for every scalar metric. Record insufficient clusters instead of emitting zero or NaN as a valid score.

- [ ] **Step 6: Run metric tests**

Run: `pytest -q tests/test_patch_generalization.py`

Expected: PASS.

- [ ] **Step 7: Commit the metric engine**

```bash
git add simdiff_eval/patch_generalization.py tests/test_patch_generalization.py
git commit -m "Add feature-conditioned generalization metrics"
```

### Task 6: Per-run analysis, aggregation, and falsification figures

**Files:**
- Create: `scripts/analyze_nf_generalize_fig2_vgg_patch_run.py`
- Create: `scripts/aggregate_nf_generalize_fig2_vgg_patch_results.py`
- Modify: `tests/test_nf_fig2_vgg_patch_sweep.py`

**Interfaces:**
- Per-run output: `results/nf_generalize_fig2_vgg_patch/tables/runs/{run_name}_cluster_metrics.csv` plus JSON provenance.
- Aggregate outputs: `cluster_metrics_long.csv`, `thresholds.csv`, `reservoir_sensitivity.csv`, and publication-ready PNG/PDF figures.
- Produces: `find_boolean_crossings(exponents: Sequence[int], accepted: Sequence[bool]) -> list[tuple[int, int, str]]`.
- Produces: `aggregate_run_tables(paths: Sequence[Path], expected_manifest: Sequence[dict[str, Any]]) -> pd.DataFrame`.

- [ ] **Step 1: Add failing aggregation tests for all 30 runs and non-monotonic crossings**

```python
def test_find_crossings_preserves_nonmonotonic_behavior():
    crossings = find_boolean_crossings([6, 7, 8, 9, 10], [False, True, False, True, True])
    assert crossings == [(6, 7, "enter"), (7, 8, "exit"), (8, 9, "enter")]

def test_aggregate_requires_complete_run_grid(tmp_path):
    with pytest.raises(ValueError, match="30"):
        aggregate_run_tables(incomplete_paths, expected_manifest)
```

- [ ] **Step 2: Run aggregation tests and confirm failure**

Run: `pytest -q tests/test_nf_fig2_vgg_patch_sweep.py -k 'crossings or complete_run_grid'`

Expected: FAIL because aggregation functions are missing.

- [ ] **Step 3: Implement per-run analysis**

Load one generated cache and its data-size reference cache, verify basis and source hashes, compute all eight VGG clusters with 500 image bootstrap replicates, compute the raw-patch PCA32 control without reusing VGG cluster labels, and write atomic CSV/JSON outputs. Include counts and insufficiency reasons for every cluster.

- [ ] **Step 4: Implement aggregation and transition diagnostics**

Require all architecture/data-size combinations. For each architecture and cluster, report every joint-acceptance crossing and the first sustained crossing that remains accepted at all larger sampled data sizes. Compute cluster prevalence on held-out real data, participation-ratio effective dimension from cluster covariance eigenvalues, and threshold separations in powers of two.

Set the preregistered result flag only when two populated clusters differ by at least two data-size doublings and their bootstrap intervals do not erase the separation. Otherwise write `feature_threshold_separation_supported=false` and lead the notebook with the null result.

- [ ] **Step 5: Render the fixed figure suite**

Create separate, readable figures for:

1. representative held-out-real patches nearest each VGG cluster center;
2. cluster occupancy versus data size for each U-Net width;
3. cluster-conditioned novelty plus held-out-real baselines;
4. within-cluster VGG Fr\'echet fidelity plus real-split baselines;
5. amplitude fidelity;
6. joint-acceptance heatmaps;
7. threshold versus prevalence and effective dimension;
8. VGG-primary versus raw-patch-PCA control summary; and
9. reservoir sensitivity at 1,024, 2,048, and 4,096 images near observed crossings.

Do not connect missing/insufficient values, do not name clusters automatically, and place legends outside dense plotting areas.

- [ ] **Step 6: Run aggregation tests and CLI smoke tests**

Run: `pytest -q tests/test_nf_fig2_vgg_patch_sweep.py -k 'crossings or complete_run_grid'`

Run: `python scripts/analyze_nf_generalize_fig2_vgg_patch_run.py --help`

Run: `python scripts/aggregate_nf_generalize_fig2_vgg_patch_results.py --help`

Expected: PASS and both help commands exit 0.

- [ ] **Step 7: Commit analysis and figures**

```bash
git add scripts/analyze_nf_generalize_fig2_vgg_patch_run.py scripts/aggregate_nf_generalize_fig2_vgg_patch_results.py tests/test_nf_fig2_vgg_patch_sweep.py
git commit -m "Analyze VGG patch generalization thresholds"
```

### Task 7: Results-only notebook

**Files:**
- Create: `scripts/update_nf_generalize_fig2_unet_vgg_patch_notebook.py`
- Create: `notebooks/nf_generalize_fig2_unet_vgg_patch_generalization.ipynb`
- Create: `tests/test_vgg_patch_notebook.py`

**Interfaces:**
- Consumes only audited compact CSV, JSON, and figure outputs.
- Produces one executable notebook that does not fit VGG, PCA, or KMeans.
- Test helper: `notebook_source(path: Path) -> str` remains local to `tests/test_vgg_patch_notebook.py`.

- [ ] **Step 1: Write failing notebook-structure tests**

```python
def test_notebook_is_vgg_primary_and_results_only():
    source = notebook_source(NOTEBOOK)
    assert "VGG16 relu3_3" in source
    assert "PCA64 is compression" in source
    assert "raw-patch PCA control" in source
    assert "feature_threshold_separation_supported" in source
    assert "MiniBatchKMeans(" not in source
    assert "torchvision.models.vgg16(" not in source
```

- [ ] **Step 2: Run notebook tests and confirm failure**

Run: `pytest -q tests/test_vgg_patch_notebook.py`

Expected: FAIL because the updater and notebook do not exist.

- [ ] **Step 3: Build the notebook generator and notebook**

Use `nbformat` to create sections in this order: audit and question, fixed feature basis, cluster gallery, occupancy, novelty, within-feature fidelity, amplitude, joint thresholds, predictor diagnostics, raw-PCA control, reservoir sensitivity, limitations, and conclusion. The first interpretation cell must state whether the preregistered threshold-separation criterion passed; it must not imply success from a visually interesting gallery.

All plot cells read finalized files and display standalone figures with short collaborator-level explanations. Hide repeated file-writing output and avoid tables wider than the notebook viewport.

- [ ] **Step 4: Run notebook tests and compile every code cell**

Run: `python scripts/update_nf_generalize_fig2_unet_vgg_patch_notebook.py`

Run: `pytest -q tests/test_vgg_patch_notebook.py`

Expected: PASS; notebook is structurally valid and all code cells compile without executing Great Lakes-only data reads.

- [ ] **Step 5: Commit the notebook**

```bash
git add scripts/update_nf_generalize_fig2_unet_vgg_patch_notebook.py notebooks/nf_generalize_fig2_unet_vgg_patch_generalization.ipynb tests/test_vgg_patch_notebook.py
git commit -m "Add VGG patch generalization notebook"
```

### Task 8: Fail-closed audit and dependency-safe Slurm pipeline

**Files:**
- Create: `scripts/audit_nf_generalize_fig2_vgg_patch_results.py`
- Create: `scripts/slurm/precheck_nf_generalize_fig2_vgg_patch.sbatch`
- Create: `scripts/slurm/fit_nf_generalize_fig2_vgg_patch_basis.sbatch`
- Create: `scripts/slurm/extract_nf_generalize_fig2_vgg_patch_references_array.sbatch`
- Create: `scripts/slurm/extract_nf_generalize_fig2_vgg_patch_generated_array.sbatch`
- Create: `scripts/slurm/analyze_nf_generalize_fig2_vgg_patch_array.sbatch`
- Create: `scripts/slurm/aggregate_nf_generalize_fig2_vgg_patch.sbatch`
- Create: `scripts/slurm/audit_nf_generalize_fig2_vgg_patch.sbatch`
- Create: `scripts/slurm/submit_nf_generalize_fig2_vgg_patch.sh`
- Modify: `scripts/slurm/README.md`
- Modify: `tests/test_nf_fig2_vgg_patch_sweep.py`

**Interfaces:**
- Produces: `local/nf_generalize_fig2_vgg_patch/final_audit.json` with `status=PASS|FAIL`, counts, issues, missing paths, hashes, and provenance mismatches.
- Produces: `audit_results(manifest: Sequence[dict[str, Any]], artifact_root: Path) -> dict[str, Any]`.
- Submitter accepts: `START_STAGE=1..4`, `REUSE_EXISTING_MANIFEST=0|1`, `PROJECT_DIR`, `FEATURE_CACHE_ROOT`, `VGG_WEIGHTS`, and `ACCOUNT`.

- [ ] **Step 1: Add failing audit and submitter tests**

```python
def test_audit_fails_on_basis_hash_mismatch(tmp_path):
    report = audit_results(manifest, artifacts_with_one_wrong_basis_hash)
    assert report["status"] == "FAIL"
    assert report["basis_hash_mismatches"]

def test_submitter_throttles_gpu_arrays_to_two():
    text = Path("scripts/slurm/submit_nf_generalize_fig2_vgg_patch.sh").read_text()
    assert "--array=0-9%2" in text
    assert "--array=0-29%2" in text
    assert "afterok" in text
```

- [ ] **Step 2: Run pipeline tests and confirm failure**

Run: `pytest -q tests/test_nf_fig2_vgg_patch_sweep.py -k 'audit or submitter'`

Expected: FAIL because the audit and Slurm files are absent.

- [ ] **Step 3: Implement the final audit**

Require one basis, ten default reference caches, 30 generated caches, 30 run metric tables, aggregate tables, the fixed figure suite, and the notebook. Verify all basis hashes, exact source paths and sample labels, 30-run completeness, finite metrics for sufficiently populated clusters, explicit insufficiency records otherwise, pool disjointness, bootstrap count 500, and no generated-dependent calibration artifacts. Write JSON atomically and exit 1 on `FAIL`.

- [ ] **Step 4: Implement Slurm stages and resources**

Use these exact defaults:

| Stage | Partition | Time | CPU | Memory | GPU/concurrency |
|---|---:|---:|---:|---:|---:|
| precheck | spgpu | 00:30:00 | 4 | 16G | 1 / 1 |
| fit basis | spgpu | 04:00:00 | 8 | 80G | 1 / 1 |
| references `0-9%2` | spgpu | 06:00:00 | 8 | 80G | 1 / 2 |
| generated `0-29%2` | spgpu | 02:00:00 | 8 | 40G | 1 / 2 |
| metrics `0-29%8` | standard | 04:00:00 | 8 | 64G | 0 / 8 |
| aggregate | standard | 02:00:00 | 8 | 64G | 0 / 1 |
| audit | standard | 00:30:00 | 4 | 16G | 0 / 1 |

Stage 1 runs precheck, basis, references, generated, metrics, aggregate, audit. Stage 2 requires `REUSE_EXISTING_MANIFEST=1`, validates the basis hash, and starts references/generated. Stage 3 validates caches and starts metrics. Stage 4 validates metric tables and starts aggregate/audit. Every downstream submission uses `afterok`, log directories are created before `sbatch`, and the submitter prints all job IDs and restart instructions.

The precheck must import NumPy, pandas, SciPy, scikit-learn, torch, torchvision, and `simdiff_eval`; verify the local pretrained VGG file; verify all 30 sample archives; perform the split-overlap audit; and execute a three-patch CPU VGG smoke test.

- [ ] **Step 5: Document execution and restart behavior**

Add a `VGG patch-level U-Net analysis` section to `scripts/slurm/README.md` stating that the pipeline uses existing samples only, its scratch/home output locations, the two-GPU ceiling, VGG weight requirement, stage meanings, and final audit path.

- [ ] **Step 6: Run pipeline tests and shell checks**

Run: `pytest -q tests/test_nf_fig2_vgg_patch_sweep.py -k 'audit or submitter'`

Run: `bash -n scripts/slurm/*nf_generalize_fig2_vgg_patch*`

Run: `python -m py_compile scripts/audit_nf_generalize_fig2_vgg_patch_results.py`

Expected: PASS.

- [ ] **Step 7: Commit orchestration and audit**

```bash
git add scripts/audit_nf_generalize_fig2_vgg_patch_results.py scripts/slurm/*nf_generalize_fig2_vgg_patch* scripts/slurm/README.md tests/test_nf_fig2_vgg_patch_sweep.py
git commit -m "Add audited VGG patch analysis pipeline"
```

### Task 9: Full local verification and Great Lakes action preview

**Files:**
- Verify all files listed above.
- Do not modify or stage unrelated dirty files.

**Interfaces:**
- Produces a locally verified branch ready for review.
- Does not push or submit compute without separate protected-action approval.

- [ ] **Step 1: Run the complete focused test suite**

Run:

```bash
pytest -q \
  tests/test_patch_features.py \
  tests/test_patch_generalization.py \
  tests/test_nf_fig2_vgg_patch_sweep.py \
  tests/test_vgg_patch_notebook.py
```

Expected: PASS.

- [ ] **Step 2: Run repository-level static verification**

Run: `git diff --check`

Run: `python -m compileall -q simdiff_eval scripts`

Run: `bash -n scripts/slurm/*nf_generalize_fig2_vgg_patch*`

Expected: all commands exit 0.

- [ ] **Step 3: Review scientific invariants from generated fixtures**

Run: `pytest -q tests/test_nf_fig2_vgg_patch_sweep.py -k 'disjoint or moving_basis or generated_dependent or incomplete or insufficient'`

Expected: PASS, proving overlap, basis drift, generated-tuned thresholds, incomplete grids, and low-count clusters fail closed.

- [ ] **Step 4: Commit any verification-only corrections**

```bash
git add simdiff_eval/patch_features.py simdiff_eval/patch_generalization.py \
  scripts/prepare_nf_generalize_fig2_vgg_patch_features.py \
  scripts/fit_nf_generalize_fig2_vgg_patch_basis.py \
  scripts/extract_nf_generalize_fig2_vgg_patch_references.py \
  scripts/extract_nf_generalize_fig2_vgg_patch_generated.py \
  scripts/analyze_nf_generalize_fig2_vgg_patch_run.py \
  scripts/aggregate_nf_generalize_fig2_vgg_patch_results.py \
  scripts/audit_nf_generalize_fig2_vgg_patch_results.py \
  scripts/update_nf_generalize_fig2_unet_vgg_patch_notebook.py \
  scripts/slurm/*nf_generalize_fig2_vgg_patch* \
  notebooks/nf_generalize_fig2_unet_vgg_patch_generalization.ipynb \
  tests/test_patch_features.py tests/test_patch_generalization.py \
  tests/test_nf_fig2_vgg_patch_sweep.py tests/test_vgg_patch_notebook.py \
  requirements.txt scripts/slurm/README.md
git commit -m "Verify VGG patch generalization pipeline"
```

- [ ] **Step 5: Preview, but do not execute, the protected Great Lakes submission**

Exact target after a reviewed push/merge:

```bash
cd /home/jiamingp/diffusion_models_repo
bash scripts/gl_safe_pull.sh main
PROJECT_DIR=/home/jiamingp/diffusion_models_repo \
FEATURE_CACHE_ROOT=/scratch/huterer_root/huterer0/$USER/nf_generalize_fig2_vgg_patch_features \
VGG_WEIGHTS=/nfs/turbo/lsa-huterer/$USER/models/vgg/vgg16-397923af.pth \
ACCOUNT=huterer2 \
bash scripts/slurm/submit_nf_generalize_fig2_vgg_patch.sh
```

Expected compute effect after a later explicit `APPROVE RUN`: one precheck GPU job, one basis GPU job, ten reference tasks and 30 generated-feature tasks throttled to two simultaneous GPUs, 30 CPU metric tasks, one CPU aggregate job, and one CPU audit job. It launches no diffusion training and no sampling.

---

## Definition of Done

- VGG16 `relu3_3` is the tested primary encoder; VGG-PCA64 is compression only and raw-patch PCA32 is visibly separated as a control.
- One frozen, hashed eight-cluster basis is fit on an audited held-out-real pool and reused for all 30 U-Net runs.
- Training, basis, and evaluation identities are proven disjoint.
- Every architecture/data-size result reports cluster occupancy, novelty relative to training and held-out-real baselines, VGG feature fidelity, amplitude fidelity, counts, and image-bootstrap intervals.
- Joint thresholds are calibrated from real data only, preserve non-monotonic behavior, and lead to an explicit supported/not-supported falsification result.
- The notebook is results-only, readable, and does not refit any representation.
- The final audit fails closed and reports `PASS` only when all expected artifacts and hashes are valid.
- Local tests and static checks pass; pushing and Great Lakes submission remain separately approved external actions.
