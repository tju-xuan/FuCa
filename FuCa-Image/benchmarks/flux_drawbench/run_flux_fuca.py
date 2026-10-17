from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from diffusers.pipelines.flux.pipeline_flux import calculate_shift

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from precache.flux import (  # noqa: E402
    build_full_sigmas,
    canonicalize_anchor_indices,
    decode_packed_latents_to_image,
    load_flux_pipeline,
    reset_transformer_state,
)
from run_flux_online_anchor_segment_secant_vcorr import (  # noqa: E402
    apply_groupwise_secant_vcorr,
    flux_predict_velocity,
    prepare_online_flux_context,
)


def read_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_anchor_indices(value: str) -> list[int]:
    anchors = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not anchors:
        raise ValueError("Empty anchor indices")
    return anchors


def run_flux_fuca(
    *,
    pipeline,
    prompt: str,
    true_cfg_scale: float,
    num_inference_steps: int,
    seed: int,
    height: int,
    width: int,
    max_sequence_length: int,
    guidance_scale: float,
    anchor_indices: list[int],
    correction_start_anchor: int,
    alpha_eps: float,
    alpha_clip: float,
    channel_groups: int,
) -> dict[str, Any]:
    reset_transformer_state(
        pipeline.transformer,
        num_steps=len(anchor_indices),
    )

    context = prepare_online_flux_context(
        pipeline=pipeline,
        prompt=prompt,
        prompt_2=None,
        negative_prompt=None,
        negative_prompt_2=None,
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
        guidance_scale=guidance_scale,
    )
    latents = context["latents"]
    prompt_embeds = context["prompt_embeds"]
    pooled_prompt_embeds = context["pooled_prompt_embeds"]
    text_ids = context["text_ids"]
    negative_prompt_embeds = context["negative_prompt_embeds"]
    negative_pooled_prompt_embeds = context["negative_pooled_prompt_embeds"]
    negative_text_ids = context["negative_text_ids"]
    do_true_cfg = context["do_true_cfg"]
    guidance = context["guidance"]
    latent_image_ids = context["latent_image_ids"]

    full_sigmas = build_full_sigmas(num_inference_steps)
    sparse_sigmas = full_sigmas[np.asarray(anchor_indices, dtype=np.int32)]
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

    cached_velocity: Optional[torch.Tensor] = None
    run_start = time.perf_counter()

    with torch.no_grad():
        for sparse_idx, timestep in enumerate(timesteps):
            if cached_velocity is None:
                velocity = flux_predict_velocity(
                    pipeline=pipeline,
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_image_ids,
                    guidance=guidance,
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=true_cfg_scale,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                )
            else:
                velocity = cached_velocity
                cached_velocity = None

            prev_latents = latents
            raw_next_latents = pipeline.scheduler.step(velocity, timestep, latents, return_dict=False)[0]
            latents = raw_next_latents

            is_last_sparse_step = sparse_idx == len(anchor_indices) - 1
            current_anchor_dense_idx = int(anchor_indices[sparse_idx])
            next_anchor_dense_idx = int(anchor_indices[sparse_idx + 1]) if not is_last_sparse_step else None
            should_apply_x = not is_last_sparse_step and current_anchor_dense_idx >= correction_start_anchor
            if not should_apply_x:
                continue

            next_timestep = timesteps[sparse_idx + 1]
            next_velocity = flux_predict_velocity(
                pipeline=pipeline,
                latents=latents,
                timestep=next_timestep,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_ids=text_ids,
                latent_image_ids=latent_image_ids,
                guidance=guidance,
                do_true_cfg=do_true_cfg,
                true_cfg_scale=true_cfg_scale,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                negative_text_ids=negative_text_ids,
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
    image = decode_packed_latents_to_image(
        pipeline=pipeline,
        packed_latents=latents.detach().cpu().to(torch.float16),
        height=height,
        width=width,
    )
    return {
        "image": image,
        "anchor_sparse_elapsed": float(anchor_sparse_elapsed),
        "total_elapsed": float(anchor_sparse_elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch FLUX FuCa generation.")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--anchor-indices", type=str, default="0,1,2,3,4,6,11,19,30,44")
    parser.add_argument("--correction-start-anchor", type=int, default=4)
    parser.add_argument("--alpha-eps", type=float, default=1e-8)
    parser.add_argument("--alpha-clip", type=float, default=1.0)
    parser.add_argument("--channel-groups", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    prompts = read_prompts(args.prompt_file)
    if args.limit > 0:
        prompts = prompts[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchor_indices = canonicalize_anchor_indices(parse_anchor_indices(args.anchor_indices), args.num_steps).tolist()

    pipeline = load_flux_pipeline(args.model_path)
    results = []
    for index, prompt in enumerate(prompts):
        sample_seed = int(args.seed + index)
        run = run_flux_fuca(
            pipeline=pipeline,
            prompt=prompt,
            true_cfg_scale=args.true_cfg_scale,
            num_inference_steps=args.num_steps,
            seed=sample_seed,
            height=args.height,
            width=args.width,
            max_sequence_length=args.max_sequence_length,
            guidance_scale=args.guidance_scale,
            anchor_indices=anchor_indices,
            correction_start_anchor=args.correction_start_anchor,
            alpha_eps=args.alpha_eps,
            alpha_clip=args.alpha_clip,
            channel_groups=args.channel_groups,
        )
        out_path = args.output_dir / f"output_{index:02d}.png"
        run["image"].save(out_path)
        results.append(
            {
                "index": index,
                "prompt": prompt,
                "image_path": str(out_path),
                "latency_seconds": float(run["total_elapsed"]),
                "anchor_sparse_latency_seconds": float(run["anchor_sparse_elapsed"]),
            }
        )
        print(
            f"[FLUX FuCa] index={index:03d} "
            f"seed={sample_seed} latency={float(run['total_elapsed']):.3f}s",
            flush=True,
        )
        del run

    summary = {
        "method": "fuca",
        "prompt_file": str(args.prompt_file),
        "model_path": str(args.model_path),
        "num_steps": int(args.num_steps),
        "width": int(args.width),
        "height": int(args.height),
        "seed": int(args.seed),
        "guidance_scale": float(args.guidance_scale),
        "max_sequence_length": int(args.max_sequence_length),
        "true_cfg_scale": float(args.true_cfg_scale),
        "anchor_indices": [int(x) for x in anchor_indices],
        "correction_start_anchor": int(args.correction_start_anchor),
        "alpha_eps": float(args.alpha_eps),
        "alpha_clip": float(args.alpha_clip),
        "channel_groups": int(args.channel_groups),
        "results": results,
        "avg_latency_seconds": sum(x["latency_seconds"] for x in results) / max(len(results), 1),
        "avg_anchor_sparse_latency_seconds": sum(x["anchor_sparse_latency_seconds"] for x in results) / max(len(results), 1),
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
