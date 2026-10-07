"""Route continuous labels into the timestep embedding of diffusers' UNet2DConditionModel.

cosmodiff's continuous conditioning always calls ``model(x, timestep=t, encoder_hidden_states=labels)``
with labels of shape (B, 1, D) (training) or (B, D) (sampling). For a UNet2DConditionModel built with
``class_embed_type="projection"`` and no cross-attention blocks, this patch passes those labels on as
``class_labels`` (B, D). diffusers then adds MLP(labels) to the timestep embedding, which reaches every
ResNet block (additively, or as scale-shift with ``resnet_time_scale_shift="scale_shift"``), as in
Diffusion-HMC (Mudur et al.) and ADM/EDM.

Middle block: inside UNet2DConditionModel, diffusers builds ``mid_block_type="UNetMidBlock2D"`` as a single
ResBlock without attention (num_layers=0, add_attention=False). For these projection-conditioned models the
patch rebuilds it as ResBlock + self-attention + ResBlock, the same middle block as the cross-attention baseline
(minus the cross-attention) and the unconditional UNet2DModel. The patch runs in training and sampling, so
checkpoints and models stay consistent.

Models without ``class_embed_type="projection"`` are untouched, so the existing cross-attention runs
behave exactly as before. Install from sitecustomize (``install()``); the patch is applied when the
diffusers module is imported, so importing this module does not import diffusers.
"""
from __future__ import annotations

import functools
import importlib.abc
import sys

TARGET = "diffusers.models.unets.unet_2d_condition"
_MARKER = "_timeemb_label_routing"


def _patch(module) -> None:
    cls = module.UNet2DConditionModel
    if getattr(cls.forward, _MARKER, False):
        return
    original = cls.forward

    @functools.wraps(original)
    def forward(self, sample, timestep, encoder_hidden_states=None, class_labels=None, *args, **kwargs):
        if self.config.class_embed_type == "projection" and class_labels is None and encoder_hidden_states is not None:
            class_labels = encoder_hidden_states.reshape(encoder_hidden_states.shape[0], -1)
            expected = self.config.projection_class_embeddings_input_dim
            if class_labels.shape[1] != expected:
                raise ValueError(f"label dim {class_labels.shape[1]} != projection_class_embeddings_input_dim {expected}")
        return original(self, sample, timestep, encoder_hidden_states, class_labels, *args, **kwargs)

    setattr(forward, _MARKER, True)
    cls.forward = forward

    original_init = cls.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        cfg = self.config
        if cfg.class_embed_type == "projection" and cfg.mid_block_type == "UNetMidBlock2D":
            from simdiff_eval.torch_compat import install_torch_backend_compat

            install_torch_backend_compat(entry_point="timeemb_conditioning")  # idempotent; diffusers is already loaded here
            from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D

            head_dim = cfg.attention_head_dim
            head_dim = head_dim[-1] if isinstance(head_dim, (list, tuple)) else head_dim
            self.mid_block = UNetMidBlock2D(
                in_channels=cfg.block_out_channels[-1],
                temb_channels=self.time_embedding.linear_2.out_features,
                dropout=cfg.dropout,
                num_layers=1,
                resnet_eps=cfg.norm_eps,
                resnet_act_fn=cfg.act_fn,
                output_scale_factor=cfg.mid_block_scale_factor,
                resnet_groups=cfg.norm_num_groups,
                attn_groups=cfg.norm_num_groups,
                resnet_time_scale_shift=cfg.resnet_time_scale_shift,
                attention_head_dim=head_dim,
                add_attention=True,
            )

    cls.__init__ = __init__


class _PostImportPatcher(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    if TARGET in sys.modules:
        _patch(sys.modules[TARGET])
    elif not any(isinstance(f, _PostImportPatcher) for f in sys.meta_path):
        sys.meta_path.insert(0, _PostImportPatcher())
