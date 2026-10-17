from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift

from precache.qwen_image import (
    build_full_sigmas,
    normalize_scheduler_coordinates,
    prepare_online_qwen_context,
    qwen_predict_velocity,
    reconstruct_dense_hermite,
    replay_euler_denoising_from_velocity,
    reset_transformer_state,
)


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


def run_online_anchor_segment_secant_vcorr(
    pipeline,
    prompt: str,
    negative_prompt: Optional[str],
    true_cfg_scale: float,
    num_inference_steps: int,
    seed: int,
    height: int,
    width: int,
    max_sequence_length: int,
    anchor_indices: Sequence[int],
    correction_start_anchor: int,
    alpha_eps: float,
    alpha_clip: float,
    channel_groups: int,
) -> dict:
    run_start = time.perf_counter()
    reset_transformer_state(
        pipeline.transformer,
        num_steps=len(anchor_indices),
    )

    context = prepare_online_qwen_context(
        pipeline=pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
    )
    latents = context["latents"]
    prompt_embeds = context["prompt_embeds"]
    prompt_embeds_mask = context["prompt_embeds_mask"]
    negative_prompt_embeds = context["negative_prompt_embeds"]
    negative_prompt_embeds_mask = context["negative_prompt_embeds_mask"]
    do_true_cfg = context["do_true_cfg"]
    img_shapes = context["img_shapes"]
    guidance = context["guidance"]

    full_sigmas = build_full_sigmas(num_inference_steps)
    sparse_sigmas = full_sigmas[np.array(anchor_indices, dtype=np.int32)]
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    pipeline.scheduler.set_timesteps(sigmas=sparse_sigmas.tolist(), device=latents.device, mu=mu)
    timesteps = pipeline.scheduler.timesteps
    pipeline.scheduler.set_begin_index(0)

    anchor_velocities = []
    cached_velocity = None

    with torch.no_grad():
        for sparse_idx, timestep in enumerate(timesteps):
            if cached_velocity is None:
                velocity = qwen_predict_velocity(
                    pipeline=pipeline,
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    guidance=guidance,
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=true_cfg_scale,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                )
            else:
                velocity = cached_velocity
                cached_velocity = None

            anchor_velocities.append(velocity.detach().float().reshape(-1).cpu().numpy().copy())
            prev_latents = latents
            raw_next_latents = pipeline.scheduler.step(velocity, timestep, latents, return_dict=False)[0]
            latents = raw_next_latents

            is_last_sparse_step = sparse_idx == len(anchor_indices) - 1
            current_anchor_dense_idx = int(anchor_indices[sparse_idx])
            if not is_last_sparse_step and current_anchor_dense_idx >= correction_start_anchor:
                next_timestep = timesteps[sparse_idx + 1]
                next_velocity = qwen_predict_velocity(
                    pipeline=pipeline,
                    latents=latents,
                    timestep=next_timestep,
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    guidance=guidance,
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=true_cfg_scale,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                )
                sigma_curr = pipeline.scheduler.sigmas[pipeline.scheduler.step_index - 1]
                sigma_next = pipeline.scheduler.sigmas[pipeline.scheduler.step_index]
                dt = sigma_next - sigma_curr
                corrected_next_latents = (
                    prev_latents.to(torch.float32) + dt * 0.5 * (velocity.to(torch.float32) + next_velocity.to(torch.float32))
                )

                secant_dx = raw_next_latents.to(torch.float32) - prev_latents.to(torch.float32)
                correction_dx = corrected_next_latents - raw_next_latents.to(torch.float32)
                corrected_next_velocity, _ = apply_groupwise_secant_vcorr(
                    next_velocity=next_velocity,
                    velocity=velocity,
                    secant_dx=secant_dx,
                    correction_dx=correction_dx,
                    alpha_eps=alpha_eps,
                    alpha_clip=alpha_clip,
                    channel_groups=channel_groups,
                )
                latents = corrected_next_latents.to(prev_latents.dtype)
                cached_velocity = corrected_next_velocity

    anchor_sparse_elapsed = time.perf_counter() - run_start
    anchor_velocities_np = np.stack(anchor_velocities, axis=0).astype(np.float32, copy=False)
    anchor_timesteps_np = timesteps.detach().float().cpu().numpy().astype(np.float64, copy=False)
    anchor_time_coords = normalize_scheduler_coordinates(anchor_timesteps_np)

    full_context = prepare_online_qwen_context(
        pipeline=pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
    )
    full_time_coords = normalize_scheduler_coordinates(full_context["timesteps"].detach().float().cpu().numpy())
    corrected_dense_velocity = reconstruct_dense_hermite(
        anchor_times=anchor_time_coords,
        anchor_values=anchor_velocities_np,
        query_times=full_time_coords,
    )
    image, dense_replay_elapsed = replay_euler_denoising_from_velocity(
        pipeline=pipeline,
        replay_velocity=corrected_dense_velocity,
        target_num_steps=num_inference_steps,
        seed=seed,
        height=height,
        width=width,
    )

    return {
        "image": image,
        "anchor_sparse_elapsed": float(anchor_sparse_elapsed),
        "dense_replay_elapsed": float(dense_replay_elapsed),
    }
