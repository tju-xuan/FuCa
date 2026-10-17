# FuCa-Image

This repository contains the official **FuCa** inference implementation for FLUX.1-dev and Qwen-Image.

This repository provides inference code only.

## Requirements

Install the required packages for the corresponding backbone:

```bash
# FLUX.1-dev
pip install -r requirements-flux.txt

# Qwen-Image
pip install -r requirements-qwen.txt
```

## Training

FuCa is training-free. No training is required.

## Inference

Run FLUX.1-dev and Qwen-Image:

```bash
cd FuCa-Image
bash scripts/run_fuca.sh
```

Run one backbone only:

```bash
TARGET=flux bash scripts/run_fuca.sh
TARGET=qwen bash scripts/run_fuca.sh
```

Outputs are saved to:

```text
results/fuca_<timestamp>/
```

The script generates images with dense 50-step sampling and the slow and fast FuCa schedules. Each run saves generated images and inference metadata in `run_info.json`.

## Pre-trained Models

Place checkpoints under:

```text
models/FLUX.1-dev
models/Qwen-Image
```

or set:

```bash
export FLUX_MODEL_PATH=./models/FLUX.1-dev
export QWEN_IMAGE_MODEL_PATH=./models/Qwen-Image
```

## Reproduction Settings

Default settings:

```text
FLUX.1-dev: 1024x1024, seed=42, guidance_scale=3.5
Qwen-Image: 1664x928, seed=0, QWEN_OFFLOAD_MODE=none
```

## License

Please follow the licenses of the original model checkpoints and dependencies.
