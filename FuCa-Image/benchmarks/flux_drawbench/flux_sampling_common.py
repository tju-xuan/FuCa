from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from precache.flux import decode_packed_latents_to_image, reset_transformer_state  # noqa: E402
from run_flux_online_anchor_segment_secant_vcorr import (  # noqa: E402
    flux_predict_velocity,
    prepare_online_flux_context,
)

LATENCY_SCOPE = 'sampling_only_no_encoder_decoder'


def append_sampling_latency_fields(record: dict[str, Any], elapsed: float) -> dict[str, Any]:
    sampling_latency_seconds = float(elapsed)
    record['latency_seconds'] = sampling_latency_seconds
    record['sampling_latency_seconds'] = sampling_latency_seconds
    record['latency_scope'] = LATENCY_SCOPE
    return record


def summarize_sampling_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    avg_sampling_latency_seconds = sum(float(item['sampling_latency_seconds']) for item in results) / max(len(results), 1)
    return {
        'latency_scope': LATENCY_SCOPE,
        'avg_latency_seconds': avg_sampling_latency_seconds,
        'avg_sampling_latency_seconds': avg_sampling_latency_seconds,
    }


def manual_flux_sampling_only(
    pipeline: DiffusionPipeline,
    prompt: str,
    *,
    num_steps: int,
    seed: int,
    height: int,
    width: int,
    guidance_scale: float,
    max_sequence_length: int = 512,
    prompt_2: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    negative_prompt_2: Optional[str] = None,
    true_cfg_scale: float = 1.0,
    reset_state: bool = True,
) -> tuple[Any, float]:
    if reset_state:
        reset_transformer_state(
            pipeline.transformer,
            num_steps=num_steps,
        )

    pipeline._joint_attention_kwargs = {}
    context = prepare_online_flux_context(
        pipeline=pipeline,
        prompt=prompt,
        prompt_2=prompt_2,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=num_steps,
        seed=seed,
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
        guidance_scale=guidance_scale,
    )
    latents = context['latents']

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for timestep in context['timesteps']:
            velocity = flux_predict_velocity(
                pipeline=pipeline,
                latents=latents,
                timestep=timestep,
                prompt_embeds=context['prompt_embeds'],
                pooled_prompt_embeds=context['pooled_prompt_embeds'],
                text_ids=context['text_ids'],
                latent_image_ids=context['latent_image_ids'],
                guidance=context['guidance'],
                do_true_cfg=context['do_true_cfg'],
                true_cfg_scale=context['true_cfg_scale'],
                negative_prompt_embeds=context['negative_prompt_embeds'],
                negative_pooled_prompt_embeds=context['negative_pooled_prompt_embeds'],
                negative_text_ids=context['negative_text_ids'],
            )
            latents = pipeline.scheduler.step(velocity, timestep, latents, return_dict=False)[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sampling_elapsed = time.perf_counter() - start

    image = decode_packed_latents_to_image(
        pipeline=pipeline,
        packed_latents=latents,
        height=height,
        width=width,
    )
    return image, sampling_elapsed
