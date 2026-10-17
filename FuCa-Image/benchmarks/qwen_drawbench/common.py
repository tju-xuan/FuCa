from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from diffusers import DiffusionPipeline

from precache.qwen_image import (
    DEFAULT_MODEL_PATH,
    load_qwen_pipeline,
    timed_generate_image,
)
from run_qwen_online_anchor_segment_secant_vcorr import run_online_anchor_segment_secant_vcorr


DEFAULT_ANCHOR_INDICES = [0, 1, 2, 3, 4, 6, 11, 19, 30, 44]


def read_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_anchor_indices(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def full_token_hw(height: int, width: int, pipeline: DiffusionPipeline) -> tuple[int, int]:
    return height // pipeline.vae_scale_factor // 2, width // pipeline.vae_scale_factor // 2


def build_step_summary(
    step_kinds: Sequence[str],
    token_ratios: Sequence[float],
    actual_token_counts: Sequence[int],
    full_token_count: int,
) -> dict[str, Any]:
    return {
        "step_count": len(step_kinds),
        "full_steps": int(sum(kind == "full" for kind in step_kinds)),
        "predicted_steps": int(sum(kind == "predict" for kind in step_kinds)),
        "hybrid_steps": 0,
        "avg_token_ratio": float(sum(token_ratios) / max(len(token_ratios), 1)),
        "min_token_ratio": float(min(token_ratios)) if token_ratios else 0.0,
        "max_token_ratio": float(max(token_ratios)) if token_ratios else 0.0,
        "actual_token_count_sum": int(sum(actual_token_counts)),
        "full_token_count": int(full_token_count),
    }


def ensure_pipeline_attention_kwargs(pipeline: DiffusionPipeline) -> None:
    if not hasattr(pipeline, "_attention_kwargs") or getattr(pipeline, "_attention_kwargs", None) is None:
        object.__setattr__(pipeline, "_attention_kwargs", {})


def build_result(
    image,
    elapsed: float,
    step_kinds: Sequence[str],
    token_ratios: Sequence[float],
    actual_token_counts: Sequence[int],
    metadata: dict[str, Any],
    full_token_count: int,
) -> dict[str, Any]:
    return {
        "image": image,
        "latency_seconds": float(elapsed),
        "step_summary": build_step_summary(step_kinds, token_ratios, actual_token_counts, full_token_count),
        "metadata": metadata,
    }


def run_dense_prompt(
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
    generation = timed_generate_image(
        pipeline=pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
        sigmas=None,
    )
    full_h, full_w = full_token_hw(height, width, pipeline)
    full_tokens = full_h * full_w
    return build_result(
        image=generation.image,
        elapsed=generation.elapsed,
        step_kinds=["full"] * num_inference_steps,
        token_ratios=[1.0] * num_inference_steps,
        actual_token_counts=[full_tokens] * num_inference_steps,
        metadata={},
        full_token_count=full_tokens,
    )


def run_ours_prompt(
    pipeline: DiffusionPipeline,
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
) -> dict[str, Any]:
    result = run_online_anchor_segment_secant_vcorr(
        pipeline=pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
        anchor_indices=anchor_indices,
        correction_start_anchor=correction_start_anchor,
        alpha_eps=alpha_eps,
        alpha_clip=alpha_clip,
        channel_groups=channel_groups,
    )
    full_h, full_w = full_token_hw(height, width, pipeline)
    full_tokens = full_h * full_w
    anchor_set = set(int(x) for x in anchor_indices)
    step_kinds = ["full" if step in anchor_set else "predict" for step in range(num_inference_steps)]
    token_ratios = [1.0 if kind == "full" else 0.0 for kind in step_kinds]
    token_counts = [full_tokens if kind == "full" else 0 for kind in step_kinds]
    total_elapsed = float(result["anchor_sparse_elapsed"]) + float(result["dense_replay_elapsed"])
    return build_result(
        image=result["image"],
        elapsed=total_elapsed,
        step_kinds=step_kinds,
        token_ratios=token_ratios,
        actual_token_counts=token_counts,
        metadata={
            "anchor_indices": [int(x) for x in anchor_indices],
            "anchor_sparse_elapsed": float(result["anchor_sparse_elapsed"]),
            "dense_replay_elapsed": float(result["dense_replay_elapsed"]),
        },
        full_token_count=full_tokens,
    )


METHOD_RUNNERS = {
    "dense50": run_dense_prompt,
    "ours": run_ours_prompt,
}


def method_names() -> list[str]:
    return sorted(METHOD_RUNNERS.keys())


def default_model_path() -> Path:
    return Path(DEFAULT_MODEL_PATH)


def load_pipeline(path: Path | None = None) -> DiffusionPipeline:
    pipeline = load_qwen_pipeline(path or DEFAULT_MODEL_PATH)
    ensure_pipeline_attention_kwargs(pipeline)
    torch.set_grad_enabled(False)
    return pipeline
