"""Reference transforms for the proposed UNet-64 sweeps; no training side effects.

Applied to already-normalized maps. Private RNG state can be serialized.
Warp: composition of two periodic sinusoidal shears; not physics-preserving.
"""
import numpy as np


class SymmetryPreserving:
    def __init__(self, seed=123):
        self.rng = np.random.default_rng(seed)

    def __call__(self, image):
        if image.shape[-2] != image.shape[-1]:
            raise ValueError('Square maps required')
        element = int(self.rng.integers(8))
        shifts = tuple(map(int, self.rng.integers(image.shape[-1], size=2)))
        if isinstance(image, np.ndarray):
            out = np.rot90(image, element % 4, axes=(-2,-1))
            if element >= 4: out = np.flip(out, axis=-1)
            return np.roll(out, shifts, axis=(-2,-1)).copy()
        import torch
        out = torch.rot90(image, element % 4, dims=(-2,-1))
        if element >= 4: out = out.flip(-1)
        return torch.roll(out, shifts, dims=(-2,-1)).contiguous()


class SmoothPeriodicWarp:
    """Fixed proposed recipe: amplitude 2 pixels; wavelength 32 pixels at H=128.

    Random phases/directions per fetch, no amplitude sweep. Composed continuous
    shears are invertible; bilinear resampling changes statistics. This is a
    deliberate distorted-data control, not extra independent cosmologies.
    """
    def __init__(self, seed=123, amplitude=2.0, waves=4):
        if amplitude < 0 or not isinstance(waves, int) or waves < 1:
            raise ValueError('Invalid warp parameters')
        self.rng = np.random.default_rng(seed)
        self.amplitude, self.waves = float(amplitude), waves

    def __call__(self, image):
        h,w = image.shape[-2:]
        if h != w: raise ValueError('Square maps required')
        phase = self.rng.uniform(0,2*np.pi,2)
        amp = self.amplitude * self.rng.choice([-1,1],2)
        y,x = np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
        sx = x + amp[0]*np.sin(2*np.pi*self.waves*y/h + phase[0])
        sy = y + amp[1]*np.sin(2*np.pi*self.waves*sx/w + phase[1])
        x0,y0 = np.floor(sx).astype(int),np.floor(sy).astype(int)
        fx,fy = sx-x0,sy-y0
        indices = [(y0%h,x0%w), (y0%h,(x0+1)%w), ((y0+1)%h,x0%w), ((y0+1)%h,(x0+1)%w)]
        weights = [(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy]
        if isinstance(image,np.ndarray):
            return sum(image[...,yy,xx]*weight for (yy,xx),weight in zip(indices,weights)).astype(image.dtype)
        import torch
        out = torch.zeros_like(image)
        for (yy,xx),weight in zip(indices,weights):
            iy = torch.as_tensor(yy,device=image.device)
            ix = torch.as_tensor(xx,device=image.device)
            wt = torch.as_tensor(weight,device=image.device,dtype=image.dtype)
            out = out + image[...,iy,ix]*wt
        return out
