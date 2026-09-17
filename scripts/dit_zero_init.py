"""Fresh-only zero-modulation/output ablation, not the full Facebook DiT init."""
from __future__ import annotations


def zero_modulation_and_output(model, *, fresh: bool) -> dict:
    import torch
    from diffusers import DiTTransformer2DModel
    from diffusers.models.normalization import AdaLayerNormZero

    if not fresh or getattr(model, "_zero_init_applied", False):
        raise RuntimeError("zero initialization is allowed exactly once on a fresh model")
    if type(model) is not DiTTransformer2DModel:
        raise TypeError("expected the native DiTTransformer2DModel")
    width = model.config.num_attention_heads * model.config.attention_head_dim
    if len(model.transformer_blocks) != model.config.num_layers:
        raise ValueError("DiT layer count differs from config")
    targets = []
    for i, block in enumerate(model.transformer_blocks):
        if type(block.norm1) is not AdaLayerNormZero:
            raise TypeError(f"block {i} does not expose AdaLayerNormZero")
        targets.append((f"transformer_blocks.{i}.norm1.linear", block.norm1.linear, 6 * width))
    targets.extend([
        ("proj_out_1", model.proj_out_1, 2 * width),
        ("proj_out_2", model.proj_out_2, model.config.patch_size**2 * model.out_channels),
    ])
    # Validate the entire API before changing any parameter.
    for name, layer, outputs in targets:
        if (type(layer) is not torch.nn.Linear or layer.bias is None
                or tuple(layer.weight.shape) != (outputs, width)
                or tuple(layer.bias.shape) != (outputs,)):
            raise ValueError(f"unexpected projection API: {name}")
    with torch.no_grad():
        for _, layer, _ in targets:
            layer.weight.zero_()
            layer.bias.zero_()
    model._zero_init_applied = True
    return {"kind": "zero_modulation_and_output", "fresh_only": True,
            "linear_modules": [name for name, _, _ in targets],
            "zeroed_tensors": 2 * len(targets),
            "note": "Other weights, embedders and initialization remain native diffusers."}
