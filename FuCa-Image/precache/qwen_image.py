from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift


DEFAULT_MODEL_PATH = Path(
    os.environ.get("QWEN_IMAGE_MODEL_PATH", str(Path(__file__).resolve().parents[1] / "models" / "Qwen-Image"))
)


@dataclass
class GenerationResult:
    image: Image.Image
    elapsed: float


def reset_transformer_state(
    transformer,
    num_steps: int,
) -> None:
    transformer.cnt = 0
    transformer.num_steps = int(num_steps)


def build_full_sigmas(num_steps: int) -> np.ndarray:
    return np.linspace(1.0, 1.0 / num_steps, num_steps, dtype=np.float32)


def normalize_scheduler_coordinates(timestep_values: np.ndarray) -> np.ndarray:
    raw = np.array(timestep_values, dtype=np.float64)
    if raw.size <= 1:
        return np.zeros_like(raw)
    denom = raw[0] - raw[-1]
    if abs(denom) < 1e-12:
        return np.linspace(0.0, 1.0, raw.size, dtype=np.float64)
    coords = (raw[0] - raw) / denom
    if np.any(np.diff(coords) < -1e-8):
        return np.linspace(0.0, 1.0, raw.size, dtype=np.float64)
    return coords


def estimate_anchor_slopes(anchor_times: np.ndarray, anchor_values: np.ndarray) -> np.ndarray:
    slopes = np.zeros_like(anchor_values, dtype=np.float32)
    if anchor_values.shape[0] <= 1:
        return slopes
    intervals = np.diff(anchor_times).astype(np.float32)
    secants = np.diff(anchor_values, axis=0) / intervals[:, None]
    if anchor_values.shape[0] == 2:
        slopes[0] = secants[0]
        slopes[1] = secants[0]
        return slopes
    slopes[0] = secants[0]
    slopes[-1] = secants[-1]
    for index in range(1, anchor_values.shape[0] - 1):
        left_h = intervals[index - 1]
        right_h = intervals[index]
        slopes[index] = (right_h * secants[index - 1] + left_h * secants[index]) / (left_h + right_h)
    return slopes


def reconstruct_dense_hermite(anchor_times: np.ndarray, anchor_values: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    slopes = estimate_anchor_slopes(anchor_times, anchor_values)
    reconstructed = np.empty((len(query_times), anchor_values.shape[1]), dtype=np.float32)
    for query_index, query_time in enumerate(query_times):
        if query_time <= anchor_times[0]:
            reconstructed[query_index] = anchor_values[0]
            continue
        if query_time >= anchor_times[-1]:
            reconstructed[query_index] = anchor_values[-1]
            continue
        interval_index = int(np.searchsorted(anchor_times, query_time, side="right") - 1)
        interval_index = max(0, min(interval_index, len(anchor_times) - 2))
        t0 = anchor_times[interval_index]
        t1 = anchor_times[interval_index + 1]
        y0 = anchor_values[interval_index]
        y1 = anchor_values[interval_index + 1]
        m0 = slopes[interval_index]
        m1 = slopes[interval_index + 1]
        interval = float(t1 - t0)
        if interval <= 1e-12:
            reconstructed[query_index] = y0
            continue
        s = float((query_time - t0) / interval)
        s2 = s * s
        s3 = s2 * s
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2
        reconstructed[query_index] = h00 * y0 + h10 * interval * m0 + h01 * y1 + h11 * interval * m1
    return reconstructed


def load_qwen_pipeline(model_path: Path) -> DiffusionPipeline:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    offload_mode = os.environ.get("QWEN_OFFLOAD_MODE", "model").strip().lower()
    pipeline = DiffusionPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if device == "cuda":
        pipeline.enable_attention_slicing("auto")
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
        if offload_mode == "sequential":
            pipeline.enable_sequential_cpu_offload(gpu_id=0, device=device)
        elif offload_mode == "model":
            pipeline.enable_model_cpu_offload(gpu_id=0, device=device)
        elif offload_mode == "none":
            pipeline.to(device)
        else:
            raise ValueError("QWEN_OFFLOAD_MODE must be one of: sequential, model, none.")
    else:
        pipeline.to(device)
        pipeline.vae.enable_tiling()
    return pipeline


def timed_generate_image(
    pipeline: DiffusionPipeline,
    prompt: str,
    negative_prompt: Optional[str],
    true_cfg_scale: float,
    num_inference_steps: int,
    seed: int,
    height: int,
    width: int,
    max_sequence_length: int,
    sigmas: Optional[list[float]] = None,
    **_unused,
) -> GenerationResult:
    effective_steps = len(sigmas) if sigmas is not None else num_inference_steps
    reset_transformer_state(
        pipeline.transformer,
        num_steps=effective_steps,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    call_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "true_cfg_scale": true_cfg_scale,
        "num_inference_steps": effective_steps,
        "generator": generator,
        "height": height,
        "width": width,
        "max_sequence_length": max_sequence_length,
        "output_type": "pil",
    }
    if sigmas is not None:
        call_kwargs["sigmas"] = [float(x) for x in sigmas]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    image = pipeline(**call_kwargs).images[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return GenerationResult(image=image, elapsed=time.perf_counter() - start)


def decode_packed_latents_to_image(
    pipeline: DiffusionPipeline,
    packed_latents: torch.Tensor,
    height: int,
    width: int,
) -> Image.Image:
    vae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_dtype = next(pipeline.vae.parameters()).dtype
    latents = pipeline._unpack_latents(
        packed_latents.to(device=vae_device, dtype=vae_dtype),
        height,
        width,
        pipeline.vae_scale_factor,
    )
    latents_mean = torch.tensor(pipeline.vae.config.latents_mean, device=vae_device, dtype=vae_dtype).view(
        1, pipeline.vae.config.z_dim, 1, 1, 1
    )
    latents_std = 1.0 / torch.tensor(pipeline.vae.config.latents_std, device=vae_device, dtype=vae_dtype).view(
        1, pipeline.vae.config.z_dim, 1, 1, 1
    )
    latents = latents / latents_std + latents_mean
    decoded = pipeline.vae.decode(latents, return_dict=False)[0][:, :, 0].detach()
    return pipeline.image_processor.postprocess(decoded, output_type="pil")[0]


def replay_euler_denoising_from_velocity(
    pipeline: DiffusionPipeline,
    replay_velocity: np.ndarray,
    target_num_steps: int,
    seed: int,
    height: int,
    width: int,
) -> tuple[Image.Image, float]:
    if replay_velocity.shape[0] != target_num_steps:
        raise ValueError(f"Expected {target_num_steps} replay steps, got {replay_velocity.shape[0]}.")
    device = torch.device("cuda" if torch.cuda.is_available() else next(pipeline.transformer.parameters()).device)
    dtype = next(pipeline.transformer.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = pipeline.prepare_latents(
        1,
        pipeline.transformer.config.in_channels // 4,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    )
    sigmas = build_full_sigmas(target_num_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 8192),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 0.9),
    )
    pipeline.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device, mu=mu)
    pipeline.scheduler.set_begin_index(0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for step_index, timestep in enumerate(pipeline.scheduler.timesteps):
            model_output = torch.from_numpy(replay_velocity[step_index]).to(device=device, dtype=dtype).view_as(latents)
            latents = pipeline.scheduler.step(model_output, timestep, latents, return_dict=False)[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    image = decode_packed_latents_to_image(pipeline, latents, height, width)
    return image, time.perf_counter() - start


def prepare_online_qwen_context(
    pipeline: DiffusionPipeline,
    prompt: str,
    negative_prompt: Optional[str],
    true_cfg_scale: float,
    num_inference_steps: int,
    seed: int,
    height: int,
    width: int,
    max_sequence_length: int,
) -> dict[str, Any]:
    device = pipeline._execution_device
    do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
    prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
        prompt=prompt,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )
    if do_true_cfg:
        negative_prompt_embeds, negative_prompt_embeds_mask = pipeline.encode_prompt(
            prompt=negative_prompt,
            prompt_embeds=None,
            prompt_embeds_mask=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
    else:
        negative_prompt_embeds = None
        negative_prompt_embeds_mask = None

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = pipeline.prepare_latents(
        1,
        pipeline.transformer.config.in_channels // 4,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents=None,
    )
    img_shapes = [[(1, height // pipeline.vae_scale_factor // 2, width // pipeline.vae_scale_factor // 2)]]

    sigmas = build_full_sigmas(num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    pipeline.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device, mu=mu)
    if pipeline.attention_kwargs is None:
        pipeline._attention_kwargs = {}

    return {
        "latents": latents,
        "timesteps": pipeline.scheduler.timesteps,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
        "do_true_cfg": do_true_cfg,
        "img_shapes": img_shapes,
        "guidance": None,
    }


def qwen_predict_velocity(
    pipeline: DiffusionPipeline,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    img_shapes: list[list[tuple[int, int, int]]],
    guidance: Optional[torch.Tensor],
    do_true_cfg: bool,
    true_cfg_scale: float,
    negative_prompt_embeds: Optional[torch.Tensor],
    negative_prompt_embeds_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    timestep_batch = timestep.expand(latents.shape[0]).to(latents.dtype)
    with pipeline.transformer.cache_context("cond"):
        noise_pred = pipeline.transformer(
            hidden_states=latents,
            timestep=timestep_batch / 1000,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            attention_kwargs=pipeline.attention_kwargs,
            return_dict=False,
        )[0]

    if do_true_cfg:
        with pipeline.transformer.cache_context("uncond"):
            neg_noise_pred = pipeline.transformer(
                hidden_states=latents,
                timestep=timestep_batch / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=negative_prompt_embeds_mask,
                encoder_hidden_states=negative_prompt_embeds,
                img_shapes=img_shapes,
                attention_kwargs=pipeline.attention_kwargs,
                return_dict=False,
            )[0]
        combined = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(combined, dim=-1, keepdim=True)
        noise_pred = combined * (cond_norm / noise_norm.clamp_min(1e-8))

    return noise_pred
