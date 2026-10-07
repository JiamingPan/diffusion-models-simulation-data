import numpy as np
from simdiff_eval.unet64_sweep_transforms import SymmetryPreserving, SmoothPeriodicWarp

def radial_sums(a):
    a=a.squeeze(); a=a-a.mean()
    k=np.fft.fftfreq(a.shape[0])*a.shape[0]
    rings=np.rint(k[:,None]**2+k[None,:]**2).astype(int)
    return np.bincount(rings.ravel(),weights=(abs(np.fft.fft2(a))**2).ravel())

def test_symmetry_preserves_pixels_and_radial_power():
    a=np.random.default_rng(11).normal(size=(1,32,32))
    transform=SymmetryPreserving(7)
    for _ in range(10):
        b=transform(a)
        assert np.array_equal(np.sort(a.ravel()),np.sort(b.ravel()))
        np.testing.assert_allclose(radial_sums(a),radial_sums(b),atol=1e-8)

def test_warp_zero_identity_and_constant():
    a=np.random.default_rng(11).normal(size=(1,32,32)).astype('float32')
    np.testing.assert_array_equal(SmoothPeriodicWarp(amplitude=0)(a),a)
    np.testing.assert_allclose(SmoothPeriodicWarp()(np.ones_like(a)),1)

def test_warp_finite_reproducible_and_changes_power():
    a=np.random.default_rng(1).normal(size=(1,32,32)).astype('float32')
    b=SmoothPeriodicWarp(seed=3)(a)
    np.testing.assert_array_equal(b,SmoothPeriodicWarp(seed=3)(a))
    assert b.shape==a.shape and b.dtype==a.dtype and np.isfinite(b).all()
    assert not np.allclose(radial_sums(a),radial_sums(b))

def test_all_thirty_planned_tasks():
    import importlib.util
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('plan',root/'scripts/prepare_unet64_three_method_plan.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    records=module.build(root)
    assert len(records)==30
    assert len({(r['arm'],r['N']) for r in records})==30
    assert all('NOT RUNNABLE' in r['status'] for r in records)
