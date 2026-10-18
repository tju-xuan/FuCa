<div align="center">

# [NeurIPS 2026] *FuCa*: Cache the Future: Training-Free Self-Revision for Diffusion Transformer Acceleration

<p>
  <a href="https://neurips.cc/virtual/2026/poster/149314"><img src="https://img.shields.io/badge/Paper-NeurIPS%202026-blue" alt="NeurIPS 2026 paper"></a>
</p>

</div>

## 🔥 News

- Our paper has been accepted at **NeurIPS 2026**!

## 📖 Overview

This repository is the official implementation of **Cache the Future: Training-Free Self-Revision for Diffusion Transformer Acceleration**.

This repository provides inference code only.

The code release contains **FuCa**, a training-free caching method for accelerated diffusion generation:

- `FuCa-Image/`: FuCa for image generation backbones, including FLUX.1-dev and Qwen-Image.
- FuCa-Video: Not uploaded yet.

## 🛠 Requirements

Image-generation dependency files are provided in `FuCa-Image/`.

For image generation:

```setup
cd FuCa-Image
pip install -r requirements.txt
```

For FLUX.1-dev or Qwen-Image only:

```setup
cd FuCa-Image
pip install -r requirements-flux.txt
pip install -r requirements-qwen.txt
```

## 🧠 Training

FuCa is training-free. No training step is required.

## 🚀 Inference

Run image generation:

```bash
cd FuCa-Image
bash scripts/run_fuca.sh
```

Run one image backbone only:

```bash
cd FuCa-Image
TARGET=flux bash scripts/run_fuca.sh
TARGET=qwen bash scripts/run_fuca.sh
```

## 📦 Pre-trained Models

FuCa does not introduce trained weights. Use the original pretrained model checkpoints for each backbone.

Pretrained models on Hugging Face: [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image).

For `FuCa-Image/`, place or configure the image backbones as described in `FuCa-Image/README.md`:

```text
FuCa-Image/models/FLUX.1-dev
FuCa-Image/models/Qwen-Image
```

or set:

```bash
export FLUX_MODEL_PATH=<flux_model_path>
export QWEN_IMAGE_MODEL_PATH=<qwen_image_model_path>
```

## 👍 Acknowledgements

We thank the teams behind [FLUX](https://github.com/black-forest-labs/flux), [Qwen-Image](https://github.com/QwenLM/Qwen-Image), [Wan2.1](https://github.com/Wan-Video/Wan2.1), and [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) for sharing their models and code with the community.

## 🤝 Contributing

Please keep changes scoped to the relevant subproject:

- image-generation code under `FuCa-Image/`
- shared public documentation in this top-level `README.md`

Do not commit model checkpoints, upstream cloned repositories, generated media, metrics outputs, or temporary experiment folders.

