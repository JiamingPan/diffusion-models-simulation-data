# Debug Note: RandomFlip and Quantile Min-Max

This note explains two code changes:

1. Fixed `RandomFlip` in `cosmo_diffusion/cosmodiff/augment.py`.
2. Added `normalization: quantile_minmax` in `cosmo_diffusion/cosmodiff/utils.py`.

The big picture:

- The sampler test suggests the DDPM/DDIM sampler is probably not the main cause of the bad `P(k)`.
- The next suspects are upstream: data augmentation, data normalization, and how the model sees the training field.
- These changes make the next runs cleaner and easier to interpret.

---

## 1. What Shape Does the Model See?

The dataset gives the augmentation code one image slice at a time:

```text
x.shape = (C, H, W)
        = (1, 128, 128)
```

So the tensor dimensions are:

| Dimension | Meaning | Positive index | Negative index |
|---|---:|---:|---:|
| Channel | one HI field channel | `0` | `-3` |
| Height | vertical spatial axis | `1` | `-2` |
| Width | horizontal spatial axis | `2` | `-1` |

Your config says:

```yaml
RandomFlip:
  dims: [-1, -2]
  p: 0.5
```

This means:

```text
Flip width with probability 0.5.
Flip height with probability 0.5.
```

So the intended augmentation is spatial left-right and up-down flipping.

---

## 2. The RandomFlip Bug

### Old code

The old code was:

```python
flip = torch.rand(len(self.dims), device='cpu')
flip = torch.where(flip < self.p)[0].tolist()
return torch.flip(x, flip)
```

The confusing line is:

```python
torch.where(flip < self.p)[0].tolist()
```

This returns the positions where the boolean mask is true. It does **not** return the values from `self.dims`.

### Concrete example

Suppose:

```python
self.dims = (-1, -2)
self.p = 0.5
```

Then maybe the random numbers are:

```python
flip = torch.tensor([0.20, 0.80])
```

The mask is:

```python
flip < self.p
# tensor([ True, False ])
```

Now this line:

```python
torch.where(flip < self.p)[0].tolist()
```

returns:

```python
[0]
```

Why `[0]`?

Because the boolean mask is:

```text
position:   0      1
mask:     True   False
```

The true value is at position `0`, so `torch.where(...)` returns `[0]`.

But the actual requested dimension at position `0` in `self.dims` is:

```python
self.dims[0] == -1
```

So the old code confused two different things:

| Concept | Value in this example |
|---|---:|
| Position inside `self.dims` | `0` |
| Actual tensor dim requested by config | `-1` |

The old code used the position `0` as the tensor dimension.

That means it called:

```python
torch.flip(x, [0])
```

But `dim=0` is the channel dimension. Since `C=1`, flipping the channel dimension does nothing.

### What the old code actually did

For `self.dims = (-1, -2)`, these are the possible outcomes:

| Random mask | Intended tensor dims | Old tensor dims | Actual old effect |
|---|---:|---:|---|
| `[True, False]` | `[-1]` | `[0]` | Flip channel, mostly no-op |
| `[False, True]` | `[-2]` | `[1]` | Flip height |
| `[True, True]` | `[-1, -2]` | `[0, 1]` | Flip channel and height, effectively just height |
| `[False, False]` | `[]` | `[]` | No flip |

So the old code never correctly flipped width.

This means the augmentation was weaker than intended. It was not doing the full spatial flip symmetry.

---

## 3. The Fixed RandomFlip Code

The fixed code is:

```python
flip_mask = torch.rand(len(self.dims), device='cpu') < self.p
flip_dims = [dim for dim, do_flip in zip(self.dims, flip_mask.tolist()) if do_flip]
return torch.flip(x, flip_dims) if flip_dims else x
```

Now the random boolean mask is matched back to the actual requested dimensions.

Example:

```python
self.dims = (-1, -2)
flip_mask = [True, False]

flip_dims = [
    dim
    for dim, do_flip in zip(self.dims, flip_mask)
    if do_flip
]

# flip_dims = [-1]
```

So the fixed code calls:

```python
torch.flip(x, [-1])
```

That correctly flips width.

The fixed behavior is:

| Random mask | Fixed tensor dims | Actual fixed effect |
|---|---:|---|
| `[True, False]` | `[-1]` | Flip width |
| `[False, True]` | `[-2]` | Flip height |
| `[True, True]` | `[-1, -2]` | Flip width and height |
| `[False, False]` | `[]` | No flip |

This matches the config.

---

## 4. Why This Matters for the Diffusion Model

For cosmological fields, flips and rolls are natural symmetries:

- A field shifted left/right should still be physically valid.
- A field flipped left/right or up/down should still be physically valid.
- The power spectrum should not depend on absolute position or orientation.

So augmentation helps the model learn the distribution instead of memorizing one preferred orientation.

But if the flip code is wrong, then the augmented run is not actually testing the augmentation you think it is testing. Fixing `RandomFlip` makes the next runs cleaner.

---

## 5. What Global Min-Max Does

The original preprocessing was:

```python
if log:
    images = images.log()

if minmax:
    images = images - images.min()
    images = images / images.max() * 2 - 1.0
```

Mathematically, after the log transform:

```text
y = 2 * (x - global_min) / (global_max - global_min) - 1
```

So:

```text
global_min -> -1
global_max -> +1
```

This is called global min-max because the min and max are computed over the whole loaded dataset, not separately for each image.

That part is important:

- It is not per-image normalization.
- A bright halo in one slice can affect the scale used for all slices.
- A tiny number of extreme voxels can define the positive endpoint.

For this HI field, the distribution is very skewed. After `log + global minmax`, I checked the local data and found approximately:

| Quantity | Normalized value |
|---|---:|
| Median pixel | `-0.61` |
| 99.9 percentile pixel | `0.22` |
| Maximum pixel | `1.00` |

So most pixels live on the negative side of `[-1, 1]`. The upper part of the range is mostly reserved for rare bright structures.

That can make the learning problem awkward:

- The model has little dynamic range for the common field values.
- Rare bright structures dominate the scaling.
- The generated images may get the wrong small-scale texture.
- This can show up as excess high-`k` power in `P(k)`.

---

## 6. What `quantile_minmax` Means

`quantile_minmax` changes how the endpoints are chosen.

Instead of:

```text
absolute minimum -> -1
absolute maximum -> +1
```

it uses percentiles:

```text
q_low percentile -> -1
q_high percentile -> +1
```

For run7, the config is:

```yaml
data:
  log: true
  minmax: false
  normalization: quantile_minmax
  norm_kwargs:
    q_low: 0.001
    q_high: 99.99
```

The code is:

```python
q_low = float(norm_kwargs.get("q_low", 0.001))
q_high = float(norm_kwargs.get("q_high", 99.99))

quantiles = torch.tensor([q_low / 100.0, q_high / 100.0], device=images.device)
lo, hi = torch.quantile(images.flatten(), quantiles)

images = (images - lo) / (hi - lo) * 2 - 1.0
images = images.clamp(-1.0, 1.0)
```

Mathematically:

```text
lo = percentile(x, q_low)
hi = percentile(x, q_high)

y = 2 * (x - lo) / (hi - lo) - 1
y = clamp(y, -1, 1)
```

So values below `lo` become `-1`, and values above `hi` become `+1`.

This is still a global normalization over the training data. It is not per-image normalization.

The difference is that the absolute most extreme values no longer set the whole scale.

---

## 7. Simple Toy Example

Suppose the data are:

```python
x = [0, 1, 2, 3, 1000]
```

Global min-max uses:

```text
min = 0
max = 1000
```

Then values `0, 1, 2, 3` all get squeezed near `-1`, because the huge value `1000` controls the scale.

Quantile min-max tries to avoid letting only the most extreme values define the whole range.

The goal is not to delete bright halos. The goal is to give the bulk of the field more usable numerical range while still clipping only the most extreme tail.

---

## 8. Why We Created Run6 and Run7

The next two runs isolate one question:

```text
Is global min-max normalization causing the P(k) mismatch?
```

The runs are:

| Run | Augmentation | Flip code | Normalization | Purpose |
|---|---|---|---|---|
| `run6` | On | Fixed | Global min-max | New controlled baseline |
| `run7` | On | Fixed | Quantile min-max | Test normalization hypothesis |

Everything else is intended to stay matched:

- same model
- same LR
- same scheduler
- same epochs
- same `gradient_accumulation_steps`
- same augmentation choices

The interpretation is:

| Result | Meaning |
|---|---|
| `run7` improves high-`k` `P(k)` | Global min-max was likely hurting the learned distribution |
| `run7` looks similar to `run6` | Normalization is not the main cause |
| Both runs remain bad | Look next at objective, model capacity, 2D slicing, or physical-space evaluation |

---

## 9. Short Version

`RandomFlip` was wrong because `torch.where(mask)[0]` returns positions inside the mask, not the configured tensor dimensions. With `dims=(-1, -2)`, the old code used `[0, 1]` instead of `[-1, -2]`, so it flipped channel/height instead of width/height.

`quantile_minmax` is a different global normalization. It maps selected percentiles to `[-1, 1]` instead of mapping the absolute minimum and maximum to `[-1, 1]`. This tests whether rare extreme halos are distorting the scale and causing the model to learn too much small-scale power.
