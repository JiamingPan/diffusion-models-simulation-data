#!/usr/bin/env python
"""One real-data forward/backward check through the zero-init factory."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    row = json.loads(args.plan.read_text())["runs"][0]
    cfg = yaml.safe_load(Path(row["config"]).read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("A40 smoke requires CUDA")
    cfg = copy.deepcopy(cfg)
    cfg["global"]["device"] = "cuda"
    cfg["data"]["n_samples"] = [min(1, int(n)) for n in cfg["data"]["n_samples"]]
    from cosmodiff import utils
    data = utils.parse_config_data(cfg)["data"]
    built = utils.parse_config_model(cfg)
    model, optimizer, scheduler = built["model"], built["optimizer"], built["noise_scheduler"]
    if not getattr(model, "_cosmodiff_adaln_zero_initialized", False):
        raise RuntimeError("model did not pass through the zero-init factory")
    zero_layers = [block.norm1.linear for block in model.transformer_blocks]
    zero_layers += [model.proj_out_1, model.proj_out_2]
    if any(torch.count_nonzero(layer.weight).item() or torch.count_nonzero(layer.bias).item()
           for layer in zero_layers):
        raise RuntimeError("zero-init targets are nonzero before the first step")
    images = data.arrays[:1].float().cuda()
    labels = data.labels[:1].long().cuda()
    timestep = torch.tensor([249], device="cuda", dtype=torch.long)
    noise = torch.randn_like(images)
    noisy = scheduler.add_noise(images, noise, timestep)
    target = scheduler.get_velocity(images, noise, timestep)
    pred = model(noisy, timestep=timestep, class_labels=labels, return_dict=False)[0]
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    optimizer.step()
    if not torch.isfinite(loss) or torch.count_nonzero(model.proj_out_2.weight).item() == 0:
        raise RuntimeError("first zero-init optimizer step did not update the output head")
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"A40 ZERO-INIT SMOKE PASSED: loss={loss.item():.8f} peak_GiB={peak:.3f}")


if __name__ == "__main__":
    main()
