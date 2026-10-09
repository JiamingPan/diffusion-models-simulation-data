"""CPU tests for the flagged CFG null (ledger nf_cond_omsigc_cfg; scripts/patch_cosmodiff_cfg_null_flag.py).

The patch is applied to PRIVATE copies of cosmodiff built in a temporary directory with
scripts/cosmodiff_private_runtime.py (train patch set + cfg_null_flag, and sample patch set + cfg_null_flag).
The torch-side checks run in a child process that imports cosmodiff from that copy, so the shared
cosmo_diffusion_main checkout and this pytest process are never touched. Small model (sample_size 32,
encoder_hid_dim 3). Skipped when the shared cosmodiff checkout is not present (e.g. on the Mac).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


# ----------------------------------------------------------------------------- child (runs in the runtime)
def _child(out: Path, mode: str, work: Path) -> None:  # noqa: C901
    import os

    from simdiff_eval.torch_compat import install_torch_backend_compat

    install_torch_backend_compat(entry_point="tests.test_cfg_null_flag.child")
    import numpy as np
    import torch
    import yaml
    from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel

    import cosmodiff
    from cosmodiff import optim, utils

    torch.set_num_threads(4)
    res: dict = {"mode": mode, "cosmodiff_file": cosmodiff.__file__,
                 "from_runtime": cosmodiff.__file__.startswith(os.environ["COSMODIFF_DIR"])}

    def sched_cfg():
        return DDPMScheduler(num_train_timesteps=500, beta_schedule="squaredcos_cap_v2",
                             rescale_betas_zero_snr=True, prediction_type="v_prediction", clip_sample=False)

    def dpm():
        return DPMSolverMultistepScheduler.from_config(sched_cfg().config)

    def tiny_model(seed=0):
        torch.manual_seed(seed)
        m = UNet2DConditionModel(sample_size=32, in_channels=1, out_channels=1, layers_per_block=1,
                                 block_out_channels=(32, 64), down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                                 up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), norm_num_groups=32,
                                 cross_attention_dim=32, encoder_hid_dim=3)
        with torch.no_grad():  # move off the default init so label effects are not tiny
            for p in m.parameters():
                p.add_(0.05 * torch.randn_like(p))
        return m.eval()

    def flagged(n, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.cat([torch.randn(n, 2, generator=g), torch.zeros(n, 1)], dim=1)

    # 1. label dropout ------------------------------------------------------------------
    lab = flagged(10000, 1)
    out_l = optim.apply_cfg_label_dropout(lab, 0.1, "continuous", cfg_null="flag",
                                          generator=torch.Generator().manual_seed(2))
    null = torch.tensor([0.0, 0.0, 1.0])
    is_null = (out_l == null).all(1)
    unchanged = (out_l == lab).all(1)
    res["drop_rate"] = float(is_null.float().mean())
    res["every_row_null_xor_unchanged"] = bool(((is_null ^ unchanged)).all())
    res["dropped_rows_exact_flag_null"] = bool((out_l[is_null] == null).all())
    try:
        optim.apply_cfg_label_dropout(lab, 0.1, "continuous", cfg_null=None)
        res["dropout_without_flag_raises"] = False
    except ValueError:
        res["dropout_without_flag_raises"] = True
    try:
        optim.apply_cfg_label_dropout(torch.ones(4, 3), 0.1, "continuous", cfg_null="flag")
        res["nonzero_flag_column_raises"] = False
    except ValueError:
        res["nonzero_flag_column_raises"] = True

    # 2. projected null vs projected zeros -----------------------------------------------
    model = tiny_model()
    with torch.no_grad():
        pn = model.encoder_hid_proj(null[None])
        pz = model.encoder_hid_proj(torch.zeros(1, 3))
    res["encoder_hid_proj_in_features"] = int(model.encoder_hid_proj.in_features)
    res["proj_null_minus_proj_zero_maxabs"] = float((pn - pz).abs().max())

    # 3. guided prediction is uncond + w (cond - uncond) ---------------------------------
    torch.manual_seed(3)
    x = torch.randn(4, 1, 32, 32)
    t = torch.full((4,), 250, dtype=torch.long)
    lab4 = flagged(4, 4)[:, None, :]
    nl4 = optim.cfg_null_labels(lab4)
    with torch.no_grad():
        preds = {w: optim.cfg_guided_prediction(model, x, t, lab4, nl4, w, "encoder_hidden_states")
                 for w in (0.0, 0.5, 1.0, 2.0)}
        cond = model(x, timestep=t, return_dict=False, encoder_hidden_states=lab4)[0]
        uncond = model(x, timestep=t, return_dict=False, encoder_hidden_states=nl4)[0]
    scale = float((preds[1.0] - preds[0.0]).abs().max())
    res["pred_cond_minus_uncond_maxabs"] = scale
    res["pred_linearity_max_err"] = {str(w): float((preds[w] - preds[0.0] - w * (preds[1.0] - preds[0.0])).abs().max())
                                     for w in (0.0, 0.5, 2.0)}
    res["pred_w0_vs_uncond_maxabs"] = float((preds[0.0] - uncond).abs().max())
    res["pred_w1_vs_cond_maxabs"] = float((preds[1.0] - cond).abs().max())

    # 4. generate() endpoints and linearity -----------------------------------------------
    lab2 = flagged(4, 5)

    def gen(labels, w, cfg_null="flag", steps=4):
        return optim.generate(model, dpm(), batch_size=4, image_shape=(1, 32, 32), labels=labels,
                              guidance_scale=w, conditioning="continuous", num_steps=steps,
                              device=torch.device("cpu"), generator=torch.Generator().manual_seed(123),
                              cfg_null=cfg_null)

    g_none, g1, g0 = gen(lab2, None), gen(lab2, 1.0), gen(lab2, 0.0)
    g_null = gen(optim.cfg_null_labels(lab2), None, cfg_null=None)
    res["gen_w1_vs_none_maxabs"] = float((g1 - g_none).abs().max())
    res["gen_w0_vs_nulllabels_maxabs"] = float((g0 - g_null).abs().max())
    res["gen_none_vs_nulllabels_maxabs"] = float((g_none - g_null).abs().max())
    one = {w: gen(lab2, w, steps=1) for w in (0.0, 0.5, 1.0, 2.0)}
    res["gen1step_linearity_max_err"] = {str(w): float((one[w] - one[0.0] - w * (one[1.0] - one[0.0])).abs().max())
                                         for w in (0.5, 2.0)}
    res["gen1step_scale"] = float((one[1.0] - one[0.0]).abs().max())
    try:
        gen(lab2, 1.5, cfg_null=None)
        res["generate_guidance_without_flag_raises"] = False
    except ValueError:
        res["generate_guidance_without_flag_raises"] = True

    # 5. train(): refuses continuous CFG without the flag; uses the flag null; checkpoint round trip
    imgs = torch.rand(32, 1, 32, 32) * 2 - 1
    labs = flagged(32, 6)
    try:
        optim.train(utils.ArrayDataset(imgs, labels=labs), tiny_model(), output_dir=str(work / "must_not_exist"),
                    num_epochs=1, batch_size=8, mixed_precision="no", force_cpu=True, conditioning="continuous",
                    cfg_dropout=0.1, verbose=False)
        res["train_without_cfg_null_raises"] = False
    except ValueError:
        res["train_without_cfg_null_raises"] = True
    res["train_without_cfg_null_created_output"] = (work / "must_not_exist").exists()

    seen = []
    spy_model = tiny_model(7)
    spy_model.register_forward_pre_hook(
        lambda mod, args, kwargs: seen.append(kwargs["encoder_hidden_states"].detach().reshape(-1, 3).clone()),
        with_kwargs=True)
    optim.train(utils.ArrayDataset(imgs, labels=labs), spy_model, noise_scheduler=sched_cfg(),
                optimizer=torch.optim.AdamW(spy_model.parameters(), lr=1e-4), output_dir=str(work / "spy"),
                num_epochs=1, batch_size=8, checkpoint_every_n_epochs=1, mixed_precision="no", force_cpu=True,
                conditioning="continuous", cfg_dropout=0.5, cfg_null="flag", min_snr_gamma=5.0, verbose=False)
    rows = torch.cat(seen)
    data_rows = {tuple(r.tolist()) for r in labs}
    n_null = int((rows == null).all(1).sum())
    n_data = sum(tuple(r.tolist()) in data_rows for r in rows)
    res["train_rows_seen"], res["train_rows_null"], res["train_rows_data"] = int(len(rows)), n_null, int(n_data)
    res["train_rows_zero_vector"] = int((rows == 0).all(1).sum())

    model_t = tiny_model(8)
    optim.train(utils.ArrayDataset(imgs, labels=labs), model_t, noise_scheduler=sched_cfg(),
                optimizer=torch.optim.AdamW(model_t.parameters(), lr=1e-4), output_dir=str(work / "ckpt"),
                num_epochs=1, batch_size=8, checkpoint_every_n_epochs=1, mixed_precision="no", force_cpu=True,
                conditioning="continuous", cfg_dropout=0.1, cfg_null="flag", min_snr_gamma=5.0, verbose=False)
    ck = utils.find_latest_checkpoint(str(work / "ckpt"))
    loaded, *_ = utils.load_checkpoint(ck)
    sd_t, sd_l = model_t.state_dict(), loaded.state_dict()
    res["ckpt_path"] = ck
    res["ckpt_encoder_hid_dim"] = int(loaded.config.encoder_hid_dim)
    res["ckpt_state_dict_max_diff"] = float(max((sd_t[k].float() - sd_l[k].float()).abs().max() for k in sd_t))
    res["ckpt_same_keys"] = sorted(sd_t) == sorted(sd_l)

    # 6. sample script passes train.cfg_null through (sample runtime only) ---------------
    if mode == "sample":
        lab_file = work / "labels_flag.npy"
        np.save(lab_file, flagged(4, 9).numpy().astype(np.float32))
        base = {"io": {"output_dir": str(work / "ckpt")}, "train": {"cfg_dropout": 0.1, "cfg_null": "flag"},
                "generate": {"conditioning": "continuous", "continuous_labels": str(lab_file), "n_samples": 4,
                             "batch_size": 2, "seed": 123}}
        no_flag = json.loads(json.dumps(base))
        no_flag["train"].pop("cfg_null")
        script = Path(os.environ["COSMODIFF_DIR"]) / "scripts/cosmodiff_sample.py"
        runs = {}
        for name, cfg, w in (("flag_w0p5", base, "0.5"), ("noflag_w0p5", no_flag, "0.5"), ("noflag_none", no_flag, None)):
            cp = work / f"{name}.yaml"
            cp.write_text(yaml.safe_dump(cfg))
            cmd = [sys.executable, str(script), "--config", str(cp), "--filepath", str(work / f"{name}.npz"),
                   "--scheduler", "DPMSolverMultistepScheduler", "--num_steps", "2", "--device", "cpu"]
            if w is not None:
                cmd += ["--guidance_scale", w]
            r = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
            runs[name] = {"rc": r.returncode, "npz": (work / f"{name}.npz").exists(),
                          "stderr_tail": r.stderr[-400:]}
        res["sample_script"] = runs
    out.write_text(json.dumps(res, indent=2))


# ----------------------------------------------------------------------------- pytest side
@pytest.fixture(scope="module")
def child_results(tmp_path_factory):
    import cosmodiff_private_runtime as cpr

    if not (cpr.COSMODIFF_SRC / "cosmodiff/optim.py").exists():
        pytest.skip(f"shared cosmodiff checkout not found at {cpr.COSMODIFF_SRC}")
    base = tmp_path_factory.mktemp("cfg_null_flag")
    results = {}
    for mode in ("train", "sample"):
        rt = base / f"runtime_{mode}"
        cpr.build_runtime(rt, cpr.BASE_PATCH_SETS[mode] + ["cfg_null_flag"])
        work = base / f"work_{mode}"
        work.mkdir()
        out = base / f"child_{mode}.json"
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child", str(out), mode, str(work)],
                           env=cpr.runtime_env(rt), capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not out.exists():
            raise AssertionError(f"child {mode} failed rc={r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        results[mode] = json.loads(out.read_text())
    return results


MODES = ("train", "sample")


@pytest.mark.parametrize("mode", MODES)
def test_imports_private_runtime(child_results, mode):
    assert child_results[mode]["from_runtime"], child_results[mode]["cosmodiff_file"]


@pytest.mark.parametrize("mode", MODES)
def test_dropout_rows_exact_null_and_rate(child_results, mode):
    r = child_results[mode]
    assert r["every_row_null_xor_unchanged"] and r["dropped_rows_exact_flag_null"]
    assert abs(r["drop_rate"] - 0.1) < 0.012  # 4 sigma for n = 1e4
    assert r["dropout_without_flag_raises"] and r["nonzero_flag_column_raises"]


@pytest.mark.parametrize("mode", MODES)
def test_projected_null_differs_from_projected_zero(child_results, mode):
    r = child_results[mode]
    assert r["encoder_hid_proj_in_features"] == 3
    assert r["proj_null_minus_proj_zero_maxabs"] > 1e-3


@pytest.mark.parametrize("mode", MODES)
def test_guided_prediction_linear_in_w(child_results, mode):
    r = child_results[mode]
    assert r["pred_cond_minus_uncond_maxabs"] > 1e-4
    tol = 1e-5 * max(1.0, r["pred_cond_minus_uncond_maxabs"])
    assert all(v <= tol for v in r["pred_linearity_max_err"].values()), r["pred_linearity_max_err"]
    assert r["pred_w0_vs_uncond_maxabs"] <= 1e-6 and r["pred_w1_vs_cond_maxabs"] <= 1e-5


@pytest.mark.parametrize("mode", MODES)
def test_generate_endpoints(child_results, mode):
    r = child_results[mode]
    assert r["gen_w1_vs_none_maxabs"] < 1e-4
    assert r["gen_w0_vs_nulllabels_maxabs"] < 1e-4
    assert r["gen_none_vs_nulllabels_maxabs"] > 1e-3  # the label matters, so the endpoint checks are not vacuous
    assert r["gen1step_scale"] > 1e-4
    assert all(v <= 1e-4 * max(1.0, r["gen1step_scale"]) for v in r["gen1step_linearity_max_err"].values())
    assert r["generate_guidance_without_flag_raises"]


@pytest.mark.parametrize("mode", MODES)
def test_train_uses_flag_null_and_refuses_without(child_results, mode):
    r = child_results[mode]
    assert r["train_without_cfg_null_raises"] and not r["train_without_cfg_null_created_output"]
    assert r["train_rows_null"] >= 1 and r["train_rows_zero_vector"] == 0
    assert r["train_rows_null"] + r["train_rows_data"] == r["train_rows_seen"]


@pytest.mark.parametrize("mode", MODES)
def test_checkpoint_roundtrip_encoder_hid_dim_3(child_results, mode):
    r = child_results[mode]
    assert r["ckpt_encoder_hid_dim"] == 3 and r["ckpt_same_keys"] and r["ckpt_state_dict_max_diff"] == 0.0


def test_sample_script_passes_cfg_null(child_results):
    s = child_results["sample"]["sample_script"]
    assert s["flag_w0p5"]["rc"] == 0 and s["flag_w0p5"]["npz"], s["flag_w0p5"]
    assert s["noflag_w0p5"]["rc"] != 0 and "cfg_null" in s["noflag_w0p5"]["stderr_tail"]
    assert s["noflag_none"]["rc"] == 0  # unguided sampling does not need the null


if __name__ == "__main__" and len(sys.argv) == 5 and sys.argv[1] == "child":
    _child(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
