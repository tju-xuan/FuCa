from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from diffusers.pipelines.flux.pipeline_flux import calculate_shift

from precache.flux import build_full_sigmas


def apply_groupwise_secant_vcorr(
    next_velocity: torch.Tensor,
    velocity: torch.Tensor,
    secant_dx: torch.Tensor,
    correction_dx: torch.Tensor,
    alpha_eps: float,
    alpha_clip: float,
    channel_groups: int,
) -> tuple[torch.Tensor, np.ndarray]:
    next_velocity_f32 = next_velocity.to(torch.float32)
    velocity_f32 = velocity.to(torch.float32)
    delta_v = next_velocity_f32 - velocity_f32

    if channel_groups <= 1:
        secant_norm_sq = torch.sum(secant_dx * secant_dx)
        alpha = torch.sum(secant_dx * correction_dx) / (secant_norm_sq + float(alpha_eps))
        if alpha_clip > 0.0:
            alpha = torch.clamp(alpha, min=-float(alpha_clip), max=float(alpha_clip))
        corrected = next_velocity_f32 + alpha * delta_v
        return corrected.to(next_velocity.dtype), np.array([float(alpha.detach().cpu().item())], dtype=np.float32)

    last_dim = next_velocity_f32.shape[-1]
    group_bounds = np.linspace(0, last_dim, num=channel_groups + 1, dtype=np.int32)
    corrected = next_velocity_f32.clone()
    alpha_values: list[float] = []
    for group_idx in range(channel_groups):
        start = int(group_bounds[group_idx])
        end = int(group_bounds[group_idx + 1])
        if end <= start:
            continue
        group_slice = slice(start, end)
        secant_group = secant_dx[..., group_slice]
        correction_group = correction_dx[..., group_slice]
        delta_v_group = delta_v[..., group_slice]
        secant_norm_sq = torch.sum(secant_group * secant_group)
        alpha = torch.sum(secant_group * correction_group) / (secant_norm_sq + float(alpha_eps))
        if alpha_clip > 0.0:
            alpha = torch.clamp(alpha, min=-float(alpha_clip), max=float(alpha_clip))
        corrected[..., group_slice] = next_velocity_f32[..., group_slice] + alpha * delta_v_group
        alpha_values.append(float(alpha.detach().cpu().item()))
    return corrected.to(next_velocity.dtype), np.asarray(alpha_values, dtype=np.float32)


def prepare_online_flux_context(
    pipeline,
    prompt: str,
    prompt_2: Optional[str],
    negative_prompt: Optional[str],
    negative_prompt_2: Optional[str],
    true_cfg_scale: float,
    num_inference_steps: int,
    seed: int,
    height: int,
    width: int,
    max_sequence_length: int,
    guidance_scale: float,
) -> dict[str, Any]:
    device = pipeline._execution_device
    do_true_cfg = true_cfg_scale > 1.0 and negative_prompt is not None

    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    if do_true_cfg:
        negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = pipeline.encode_prompt(
            prompt=negative_prompt,
            prompt_2=negative_prompt_2,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
            lora_scale=None,
        )
    else:
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        negative_text_ids = None

    generator = torch.Generator("cpu").manual_seed(seed)
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    latents, latent_image_ids = pipeline.prepare_latents(
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents=None,
    )

    full_sigmas = build_full_sigmas(num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    pipeline.scheduler.set_timesteps(sigmas=full_sigmas.tolist(), device=device, mu=mu)

    if pipeline.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])
    else:
        guidance = None

    if getattr(pipeline, "_joint_attention_kwargs", None) is None:
        pipeline._joint_attention_kwargs = {}

    return {
        "latents": latents,
        "timesteps": pipeline.scheduler.timesteps,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        "negative_text_ids": negative_text_ids,
        "do_true_cfg": do_true_cfg,
        "true_cfg_scale": true_cfg_scale,
        "guidance": guidance,
        "latent_image_ids": latent_image_ids,
    }


def flux_predict_velocity(
    pipeline,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    guidance: Optional[torch.Tensor],
    do_true_cfg: bool,
    true_cfg_scale: float,
    negative_prompt_embeds: Optional[torch.Tensor],
    negative_pooled_prompt_embeds: Optional[torch.Tensor],
    negative_text_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    joint_attention_kwargs = pipeline.joint_attention_kwargs
    timestep_batch = timestep.expand(latents.shape[0]).to(latents.dtype)

    noise_pred = pipeline.transformer(
        hidden_states=latents,
        timestep=timestep_batch / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        joint_attention_kwargs=joint_attention_kwargs,
        return_dict=False,
    )[0]

    if do_true_cfg:
        neg_noise_pred = pipeline.transformer(
            hidden_states=latents,
            timestep=timestep_batch / 1000,
            guidance=guidance,
            pooled_projections=negative_pooled_prompt_embeds,
            encoder_hidden_states=negative_prompt_embeds,
            txt_ids=negative_text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

    return noise_pred
