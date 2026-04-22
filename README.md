# Diffusion Models on Structured Simulation Data

This repository contains training and evaluation code for diffusion-based generative modeling on structured simulation data. The project focuses on building reproducible pipelines for model training, sampling, and robustness evaluation on non-natural-image domains.

## Current goals
- Train UNet-based diffusion models on structured simulation data
- Build reproducible config-driven training workflows
- Study generative behavior and robustness across data regimes
- Support both notebook-based exploration and script-based training

## Planned structure
- `configs/`: experiment configurations
- `src/`: reusable code for data, models, training, and evaluation
- `scripts/`: command-line training / evaluation entry points
- `notebooks/`: exploratory analysis and visualization
