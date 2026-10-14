"""CPU tests for the inference timestep_spacing override (ledger dpm50_trailing_grid_test).

scripts/patch_cosmodiff_timestep_spacing.py is applied to PRIVATE copies of cosmodiff built in a temporary directory
with scripts/cosmodiff_private_runtime.py, with the two patch sets the run uses:
  sample      = stock sample patch set + timestep_spacing             (arms a and c, non-CFG checkpoints)
  sample_cfg  = stock sample patch set + cfg_null_flag + timestep_spacing  (arm b, CFG null-flag checkpoint)
Torch-side checks run in a child process that imports cosmodiff from that copy, so the shared
cosmo_diffusion_main checkout and this pytest process are never touched.

Checked, for the project scheduler config (DDPMScheduler, 500 steps, squaredcos_cap_v2, zero-SNR, v_prediction,
timestep_spacing leading; byte-identical in the three checkpoints of this run):
  * default (no override) DPMSolverMultistep 50-step grid is the stock one, [450, 441, ..., 9];
  * --timestep_spacing trailing gives [499, 489, ..., 9];
  * the override reaches sched_cls.from_config (spy), generate() runs on that grid, the npz records it, the log
    line prints first/last timestep, and a trailing grid that does not start at 499 exits;
  * scripts/sample_nf_conditional_bias_probe.py --timestep-spacing reaches cosmodiff_sample.py (end to end) and
    passes nothing by default.
Skipped when the shared cosmodiff checkout is not present (e.g. on the Mac).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# scheduler_config.json of the three checkpoints (sha256 a7a9b48d6d56fd1b5f35497189876cd78632ba4dca3198a38094fd2b526dbbd4)
SCHED_CONFIG = {
    "_class_name": "DDPMScheduler", "_diffusers_version": "0.38.0", "beta_end": 0.02,
    "beta_schedule": "squaredcos_cap_v2", "beta_start": 0.0001, "clip_sample": False, "clip_sample_range": 1.0,
    "dynamic_thresholding_ratio": 0.995, "num_train_timesteps": 500, "prediction_type": "v_prediction",
    "rescale_betas_zero_snr": True, "sample_max_value": 2.0, "steps_offset": 0, "thresholding": False,
    "timestep_spacing": "leading", "trained_betas": None, "variance_type": "fixed_small",
}
REAL_CKPTS = [
    "/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_conditional_omsig_continuous_200k/"
    "nf_cond_omsigc_hi_u128_d2p15_n32768_fresh200k_checkpoints/checkpoint-epoch-0195",
    "/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_cond_omsigc_cfg_200k/"
    "nf_cond_omsigc_hi_u128_d2p15_n32768_cfg10flag_fresh200k_checkpoints/checkpoint-epoch-0195",
    "/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_conditional_omsig_continuous_200k/"
    "nf_cond_omsigc_hi_u128_d2p12_n4096_fresh200k_checkpoints/checkpoint-epoch-1562",
]
LEADING_50 = [450 - 9 * i for i in range(50)]          # 450, 441, ..., 9
TRAILING_50 = [499 - 10 * i for i in range(50)]        # 499, 489, ..., 9
PATCH_SETS = {"sample": ["timestep_spacing"], "sample_cfg": ["cfg_null_flag", "timestep_spacing"]}


# ----------------------------------------------------------------------------- child (runs in the runtime)
def _child(out: Path, mode: str, work: Path) -> None:  # noqa: C901
    import importlib.util
    import os

    from simdiff_eval.torch_compat import install_torch_backend_compat

    install_torch_backend_compat(entry_point="tests.test_dpm50_trailing_grid.child")
    import diffusers
    import numpy as np
    import torch
    import yaml
    from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel

    import cosmodiff
    from cosmodiff import optim, utils

    torch.set_num_threads(4)
    rt = Path(os.environ["COSMODIFF_DIR"])
    script = rt / "scripts/cosmodiff_sample.py"
    res: dict = {"mode": mode, "cosmodiff_file": cosmodiff.__file__,
                 "from_runtime": cosmodiff.__file__.startswith(str(rt)),
                 "script_text_has_cfg_null": "cfg_null=_cfg_null" in script.read_text()}

    # 1. grids for the project scheduler config ----------------------------------------------------------
    sched_dir = work / "sched"
    sched_dir.mkdir()
    (sched_dir / "scheduler_config.json").write_text(json.dumps(SCHED_CONFIG, indent=2))
    real = {}
    for ck in REAL_CKPTS:
        p = Path(ck) / "scheduler_config.json"
        real[ck] = (json.loads(p.read_text()) == SCHED_CONFIG) if p.exists() else None
    res["real_ckpt_sched_config_equal"] = real
    ddpm = DDPMScheduler.from_pretrained(str(sched_dir))
    spec = importlib.util.spec_from_file_location("patched_cosmodiff_sample", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    stock = DPMSolverMultistepScheduler.from_config(ddpm.config)
    stock.set_timesteps(50)
    res["grid_stock"] = [int(t) for t in stock.timesteps]
    grids = {}
    for name, ov in (("default", None), ("trailing", "trailing"), ("leading", "leading")):
        s = DPMSolverMultistepScheduler.from_config(ddpm.config, **mod.timestep_spacing_kwargs(ov))
        grids[name] = {"grid": mod.inference_timestep_grid(s, 50), "spacing": s.config.timestep_spacing}
    res["grids"] = grids
    res["kwargs_none"] = mod.timestep_spacing_kwargs(None)
    try:
        mod.timestep_spacing_kwargs("bogus")
        res["bogus_spacing_raises"] = False
    except ValueError:
        res["bogus_spacing_raises"] = True
    try:
        mod.check_timestep_grid(LEADING_50, "trailing", 500)
        res["trailing_check_rejects_450"] = False
    except SystemExit:
        res["trailing_check_rejects_450"] = True
    mod.check_timestep_grid(TRAILING_50, "trailing", 500)  # must not raise
    mod.check_timestep_grid(LEADING_50, None, 500)  # default: no check

    # 2. tiny checkpoint with the same scheduler config --------------------------------------------------
    dim = 3 if mode == "sample_cfg" else 2
    torch.manual_seed(0)
    model = UNet2DConditionModel(sample_size=32, in_channels=1, out_channels=1, layers_per_block=1,
                                 block_out_channels=(32, 64), down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                                 up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), norm_num_groups=32,
                                 cross_attention_dim=32, encoder_hid_dim=dim)
    imgs = torch.rand(16, 1, 32, 32) * 2 - 1
    labs = torch.randn(16, dim)
    if dim == 3:
        labs[:, 2] = 0
    optim.train(utils.ArrayDataset(imgs, labels=labs), model, noise_scheduler=DDPMScheduler.from_pretrained(str(sched_dir)),
                optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4), output_dir=str(work / "ckpt"),
                num_epochs=1, batch_size=8, checkpoint_every_n_epochs=1, mixed_precision="no", force_cpu=True,
                conditioning="continuous", cfg_dropout=0.0, min_snr_gamma=5.0, verbose=False)
    ck = utils.find_latest_checkpoint(str(work / "ckpt"))
    saved = json.loads((Path(ck) / "scheduler_config.json").read_text())
    res["tiny_ckpt_sched_config_equal"] = ({k: v for k, v in saved.items() if k != "_diffusers_version"}
                                           == {k: v for k, v in SCHED_CONFIG.items() if k != "_diffusers_version"})

    lab_file = work / "heldout_norm.npy"
    hl = np.random.default_rng(1).standard_normal((2, dim)).astype(np.float32)
    if dim == 3:
        hl[:, 2] = 0
    np.save(lab_file, hl)
    train_cfg = {"cfg_dropout": 0.0}
    if mode == "sample_cfg":
        train_cfg["cfg_null"] = "flag"
    cfg_path = work / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "io": {"output_dir": str(work / "ckpt")}, "train": train_cfg,
        "data": {"norm_kwargs": {"center": 0.0, "xmax": 1.0}},
        "generate": {"conditioning": "continuous", "continuous_labels": str(lab_file), "n_samples": 2,
                     "batch_size": 2, "seed": 123}}))

    # 3. in-process spy: the override reaches sched_cls.from_config --------------------------------------
    calls = []
    orig = DPMSolverMultistepScheduler.from_config.__func__

    def spy(cls, config=None, **kwargs):
        calls.append({"timestep_spacing": kwargs.get("timestep_spacing", "<not passed>")})
        return orig(cls, config, **kwargs)

    DPMSolverMultistepScheduler.from_config = classmethod(spy)
    spy_runs = {}
    try:
        for name, extra in (("none", []), ("trailing", ["--timestep_spacing", "trailing"])):
            calls.clear()
            npz = work / f"spy_{name}.npz"
            argv = ["cosmodiff_sample.py", "--config", str(cfg_path), "--filepath", str(npz), "--scheduler",
                    "DPMSolverMultistepScheduler", "--num_steps", "50", "--device", "cpu", *extra]
            old_argv, sys.argv = sys.argv, argv
            try:
                mod.main()
            finally:
                sys.argv = old_argv
            with np.load(npz, allow_pickle=True) as z:
                spy_runs[name] = {"from_config_calls": list(calls),
                                  "inference_timesteps": [int(t) for t in z["inference_timesteps"]],
                                  "timestep_spacing": str(z["timestep_spacing"]),
                                  "override": str(z["timestep_spacing_override"]),
                                  "samples_shape": list(z["samples"].shape),
                                  "samples_finite": bool(np.isfinite(z["samples"]).all())}
    finally:
        DPMSolverMultistepScheduler.from_config = classmethod(orig)
    res["spy"] = spy_runs
    res["diffusers_version"] = diffusers.__version__

    # 4. CLI: log line, --timestep_spacing without --scheduler is refused --------------------------------
    r = subprocess.run([sys.executable, str(script), "--config", str(cfg_path), "--filepath", str(work / "cli.npz"),
                        "--scheduler", "DPMSolverMultistepScheduler", "--num_steps", "50", "--device", "cpu",
                        "--timestep_spacing", "trailing"], capture_output=True, text=True, env=os.environ.copy())
    res["cli_trailing"] = {"rc": r.returncode,
                           "log": [ln for ln in r.stdout.splitlines() if ln.startswith("[timestep_spacing]")],
                           "stderr_tail": r.stderr[-600:]}
    r = subprocess.run([sys.executable, str(script), "--config", str(cfg_path), "--filepath", str(work / "cli2.npz"),
                        "--num_steps", "50", "--device", "cpu", "--timestep_spacing", "trailing"],
                       capture_output=True, text=True, env=os.environ.copy())
    res["cli_no_scheduler"] = {"rc": r.returncode, "npz": (work / "cli2.npz").exists(), "stderr_tail": r.stderr[-400:]}

    # 5. end to end through scripts/sample_nf_conditional_bias_probe.py ----------------------------------
    hi = work / "heldout_indices.txt"
    np.savetxt(hi, np.array([900, 901]), fmt="%d")
    raw = work / "heldout_raw.npy"
    np.save(raw, np.zeros((2, 6), np.float32))
    lab_k1 = work / "heldout_norm_k1.npy"
    np.save(lab_k1, hl)
    wrapper = {}
    for name, extra in (("none", []), ("trailing", ["--timestep-spacing", "trailing"])):
        row = {"run_name": f"tiny_{name}", "regime": "test", "dataset_size": 16, "config": str(cfg_path),
               "sample_path": str(work / f"wrap_{name}_seed{{seed}}_dpm50_trailing_heldout_k{{k}}.npz"),
               "heldout_indices_path": str(hi), "heldout_sample_params_norm_path": str(lab_k1),
               "heldout_raw_params_path": str(raw)}
        man = work / f"manifest_{name}.json"
        man.write_text(json.dumps([row]))
        r = subprocess.run([sys.executable, str(ROOT / "scripts/sample_nf_conditional_bias_probe.py"),
                            "--project-dir", str(ROOT), "--manifest", str(man), "--cosmodiff-sample", str(script),
                            "--samples-per-cosmology", "1", "--batch-size", "2", "--num-steps", "50",
                            "--device", "cpu", *extra], capture_output=True, text=True, env=os.environ.copy())
        out_npz = work / f"wrap_{name}_seed123_dpm50_trailing_heldout_k1.npz"
        rec = {"rc": r.returncode, "stderr_tail": r.stderr[-600:],
               "running_line": [ln for ln in r.stdout.splitlines() if ln.startswith("Running:")],
               "npz": out_npz.exists()}
        if out_npz.exists():
            with np.load(out_npz, allow_pickle=True) as z:
                rec["inference_timesteps"] = [int(t) for t in z["inference_timesteps"]]
                rec["annotated"] = "theta_norm_repeated" in z.files
        wrapper[name] = rec
    res["wrapper"] = wrapper
    out.write_text(json.dumps(res, indent=2))


# ----------------------------------------------------------------------------- pytest side
@pytest.fixture(scope="module")
def child_results(tmp_path_factory):
    import cosmodiff_private_runtime as cpr

    if not (cpr.COSMODIFF_SRC / "cosmodiff/optim.py").exists():
        pytest.skip(f"shared cosmodiff checkout not found at {cpr.COSMODIFF_SRC}")
    base = tmp_path_factory.mktemp("timestep_spacing")
    results = {}
    for mode, extra in PATCH_SETS.items():
        rt = base / f"runtime_{mode}"
        cpr.build_runtime(rt, cpr.BASE_PATCH_SETS["sample"] + extra)
        work = base / f"work_{mode}"
        work.mkdir()
        out = base / f"child_{mode}.json"
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child", str(out), mode, str(work)],
                           env=cpr.runtime_env(rt), capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not out.exists():
            raise AssertionError(f"child {mode} failed rc={r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        results[mode] = json.loads(out.read_text())
    return results


MODES = tuple(PATCH_SETS)


def test_wrapper_cli_args_pure():
    import sample_nf_conditional_bias_probe as w

    assert w.timestep_spacing_cli_args(None) == []
    assert w.timestep_spacing_cli_args("trailing") == ["--timestep_spacing", "trailing"]
    with pytest.raises(ValueError):
        w.timestep_spacing_cli_args("bogus")


def test_patch_refuses_shared_checkout_and_double_apply(tmp_path):
    import cosmodiff_private_runtime as cpr

    patch = ROOT / "scripts/patch_cosmodiff_timestep_spacing.py"
    r = subprocess.run([sys.executable, str(patch), str(cpr.COSMODIFF_SRC)], capture_output=True, text=True)
    assert r.returncode != 0 and "shared" in (r.stderr + r.stdout)
    src = cpr.COSMODIFF_SRC / "scripts/cosmodiff_sample.py"
    if not src.exists():
        pytest.skip("shared cosmodiff checkout not found")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/cosmodiff_sample.py").write_text(src.read_text())
    assert subprocess.run([sys.executable, str(patch), str(tmp_path)]).returncode == 0
    r = subprocess.run([sys.executable, str(patch), str(tmp_path)], capture_output=True, text=True)
    assert r.returncode != 0 and "already" in (r.stderr + r.stdout)
    assert src.read_text().count("timestep_spacing patch v1") == 0  # shared file untouched


@pytest.mark.parametrize("mode", MODES)
def test_imports_private_runtime(child_results, mode):
    r = child_results[mode]
    assert r["from_runtime"], r["cosmodiff_file"]
    assert r["script_text_has_cfg_null"] == (mode == "sample_cfg")


@pytest.mark.parametrize("mode", MODES)
def test_real_checkpoints_share_the_scheduler_config(child_results, mode):
    vals = child_results[mode]["real_ckpt_sched_config_equal"]
    present = [v for v in vals.values() if v is not None]
    if not present:
        pytest.skip("real checkpoints not visible")
    assert all(present), vals


@pytest.mark.parametrize("mode", MODES)
def test_default_grid_unchanged_and_trailing_grid(child_results, mode):
    r = child_results[mode]
    assert r["grid_stock"] == LEADING_50
    assert r["kwargs_none"] == {}
    assert r["grids"]["default"] == {"grid": LEADING_50, "spacing": "leading"}
    assert r["grids"]["leading"]["grid"] == LEADING_50
    assert r["grids"]["trailing"] == {"grid": TRAILING_50, "spacing": "trailing"}
    assert r["bogus_spacing_raises"] and r["trailing_check_rejects_450"]


@pytest.mark.parametrize("mode", MODES)
def test_override_reaches_from_config_and_generate(child_results, mode):
    r = child_results[mode]
    assert r["tiny_ckpt_sched_config_equal"]
    none, trail = r["spy"]["none"], r["spy"]["trailing"]
    assert none["from_config_calls"] == [{"timestep_spacing": "<not passed>"}]
    assert trail["from_config_calls"] == [{"timestep_spacing": "trailing"}]
    assert none["inference_timesteps"] == LEADING_50 and none["timestep_spacing"] == "leading"
    assert none["override"] == "none"
    assert trail["inference_timesteps"] == TRAILING_50 and trail["timestep_spacing"] == "trailing"
    assert trail["override"] == "trailing"
    for s in (none, trail):
        assert s["samples_shape"] == [2, 1, 32, 32] and s["samples_finite"]


@pytest.mark.parametrize("mode", MODES)
def test_cli_logs_first_last_and_refuses_without_scheduler(child_results, mode):
    r = child_results[mode]
    assert r["cli_trailing"]["rc"] == 0, r["cli_trailing"]["stderr_tail"]
    assert r["cli_trailing"]["log"] == [
        "[timestep_spacing] override=trailing spacing=trailing n=50 first_timestep=499 last_timestep=9"]
    assert r["cli_no_scheduler"]["rc"] != 0 and not r["cli_no_scheduler"]["npz"]
    assert "--timestep_spacing needs --scheduler" in r["cli_no_scheduler"]["stderr_tail"]


@pytest.mark.parametrize("mode", MODES)
def test_wrapper_passes_spacing_end_to_end(child_results, mode):
    w = child_results[mode]["wrapper"]
    for name in ("none", "trailing"):
        assert w[name]["rc"] == 0, w[name]["stderr_tail"]
        assert w[name]["npz"] and w[name]["annotated"]
    assert "--timestep_spacing" not in w["none"]["running_line"][0]
    assert w["trailing"]["running_line"][0].endswith("--timestep_spacing trailing")
    assert w["none"]["inference_timesteps"] == LEADING_50
    assert w["trailing"]["inference_timesteps"] == TRAILING_50


if __name__ == "__main__" and len(sys.argv) == 5 and sys.argv[1] == "child":
    _child(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
