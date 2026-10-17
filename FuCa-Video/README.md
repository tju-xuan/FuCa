# FuCa-Video

This repository contains the training-free FuCa inference code for video generation on Wan2.1 and HunyuanVideo. It does not include upstream model source trees, checkpoints, or generated outputs.

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The requirements file keeps the packages needed by the FuCa inference scripts and upstream model code.

## Upstream Code

The following directories are intentionally ignored and must be prepared by users:

| Directory | What to put there |
| --- | --- |
| `WAN2.1/wan/` | Wan2.1 upstream Python package |
| `WAN2.1/models/` | Wan2.1 model assets/checkpoints |
| `HunyuanVideo/hyvideo/` | HunyuanVideo upstream Python package |
| `HunyuanVideo/ckpts/` | HunyuanVideo model assets/checkpoints |

Example setup:

```bash
cp -r <wan_upstream_repo>/wan WAN2.1/wan

cp -r <hunyuanvideo_upstream_repo>/hyvideo HunyuanVideo/hyvideo
```

This code only provides FuCa-Alpha. There are no FuCa-trained weights because the method is training-free.

## Single Prompt

Wan2.1:

```bash
cd WAN2.1
bash scripts/run_single.sh
```

HunyuanVideo:

```bash
cd HunyuanVideo
bash scripts/run_single.sh
```

The scripts expose `DEVICE`, `CKPT_DIR` or `MODEL_BASE`, `PROMPT`, and `OUTPUT_DIR` as environment variables.

## VBench-350 Generation

Wan2.1:

```bash
cd WAN2.1
bash scripts/run_vbench_350.sh
```

HunyuanVideo:

```bash
cd HunyuanVideo
bash scripts/run_vbench_350.sh
```

Both scripts use `../prompts/VBench_full_info.json`, select the uniform 350-prompt subset, and write a manifest under the output directory. Only `--run-type fuca-alpha` is supported.

