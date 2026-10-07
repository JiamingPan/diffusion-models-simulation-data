"""Mandatory preflight on the existing GL torch/diffusers environment."""
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('diffusers')
from simdiff_eval.d4_unet64 import D4ScalarUNet2DModel,d4
from simdiff_eval.unet64_sweep_transforms import SymmetryPreserving,SmoothPeriodicWarp


def model():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    return D4ScalarUNet2DModel(sample_size=16,layers_per_block=1,
        block_out_channels=(4,8,8),norm_num_groups=4).eval()


def test_equivariance_all_eight_elements():
    m=model(); x=torch.randn(1,1,16,16)
    with torch.no_grad():
        y=m(x,torch.tensor([25]),return_dict=False)[0]
        for g in range(8):
            actual=m(d4(x,g),torch.tensor([25]),return_dict=False)[0]
            torch.testing.assert_close(actual,d4(y,g),atol=3e-5,rtol=3e-4)


def test_checkpoint_roundtrip_and_gradient(tmp_path):
    m=model(); x=torch.randn(1,1,16,16)
    y=m(x,torch.tensor([25]),return_dict=False)[0]
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    m.save_pretrained(tmp_path)
    loaded=D4ScalarUNet2DModel.from_pretrained(tmp_path,local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(x,torch.tensor([25]),return_dict=False)[0],y)


def test_torch_transforms_match_numpy():
    import numpy as np
    a=np.random.default_rng(9).normal(size=(1,16,16)).astype('float32')
    for cls in (SymmetryPreserving,SmoothPeriodicWarp):
        expected=cls(seed=1)(a)
        actual=cls(seed=1)(torch.from_numpy(a)).numpy()
        np.testing.assert_allclose(actual,expected,atol=5e-7,rtol=2e-6)
