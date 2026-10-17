from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline


DEFAULT_MODEL_PATH = Path(
    os.environ.get("FLUX_MODEL_PATH", str(Path(__file__).resolve().parents[1] / "models" / "FLUX.1-dev"))
)


def reset_transformer_state(transformer, num_steps: int) -> None:
    transformer.cnt = 0
    transformer.num_steps = int(num_steps)


def build_full_sigmas(num_steps: int) -> np.ndarray:
    return np.linspace(1.0, 1.0 / num_steps, num_steps, dtype=np.float32)


def canonicalize_anchor_indices(anchor_indices: Sequence[int], num_steps: int) -> np.ndarray:
    ordered: list[int] = []
    seen = set()
    for raw_index in anchor_indices:
        index = int(raw_index)
        if index < 0 or index >= num_steps:
            raise ValueError(f"Anchor index {index} is outside [0, {num_steps - 1}].")
        if index not in seen:
            ordered.append(index)
            seen.add(index)
    if not ordered:
        raise ValueError("At least one anchor index is required.")
    return np.array(sorted(ordered), dtype=np.int32)


def load_flux_pipeline(model_path: Path = DEFAULT_MODEL_PATH) -> DiffusionPipeline:
    pipeline = DiffusionPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipeline.to("cuda")
    return pipeline


def decode_packed_latents_to_image(
    pipeline: DiffusionPipeline,
    packed_latents: torch.Tensor,
    height: int,
    width: int,
) -> Image.Image:
    vae_device = next(pipeline.vae.parameters()).device
    vae_dtype = next(pipeline.vae.parameters()).dtype
    packed_latents = packed_latents.to(device=vae_device, dtype=vae_dtype)
    unpacked_latents = pipeline._unpack_latents(packed_latents, height, width, pipeline.vae_scale_factor)
    unpacked_latents = (unpacked_latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    decoded = pipeline.vae.decode(unpacked_latents, return_dict=False)[0].detach()
    return pipeline.image_processor.postprocess(decoded, output_type="pil")[0]
