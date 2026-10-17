from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_ANCHOR_INDICES,
    METHOD_RUNNERS,
    default_model_path,
    load_pipeline,
    method_names,
    parse_anchor_indices,
    read_prompts,
    write_json,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen-Image dense50 or FuCa.")
    parser.add_argument("--method", type=str, required=True, choices=method_names())
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1664)
    parser.add_argument("--height", type=int, default=928)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--anchor-indices", type=str, default=",".join(str(x) for x in DEFAULT_ANCHOR_INDICES))
    parser.add_argument("--correction-start-anchor", type=int, default=4)
    parser.add_argument("--alpha-eps", type=float, default=1e-8)
    parser.add_argument("--alpha-clip", type=float, default=1.0)
    parser.add_argument("--channel-groups", type=int, default=1)
    return parser


def method_kwargs(args: argparse.Namespace) -> dict:
    if args.method != "ours":
        return {}
    return {
        "anchor_indices": parse_anchor_indices(args.anchor_indices),
        "correction_start_anchor": args.correction_start_anchor,
        "alpha_eps": args.alpha_eps,
        "alpha_clip": args.alpha_clip,
        "channel_groups": args.channel_groups,
    }


def main() -> None:
    args = build_argparser().parse_args()
    prompts = read_prompts(args.prompt_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline(args.model_path)
    runner = METHOD_RUNNERS[args.method]
    extra_kwargs = method_kwargs(args)

    results = []
    for index, prompt in enumerate(prompts):
        sample_seed = args.seed + index
        run = runner(
            pipeline=pipeline,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            true_cfg_scale=args.true_cfg_scale,
            num_inference_steps=args.num_steps,
            seed=sample_seed,
            height=args.height,
            width=args.width,
            max_sequence_length=args.max_sequence_length,
            **extra_kwargs,
        )
        image_path = args.output_dir / f"output_{index:03d}.png"
        run["image"].save(image_path)
        results.append(
            {
                "index": index,
                "prompt": prompt,
                "image_path": str(image_path),
                "latency_seconds": float(run["latency_seconds"]),
                "step_summary": run["step_summary"],
                "metadata": run["metadata"],
            }
        )
        print(
            f"[Qwen {args.method}] index={index:03d} seed={sample_seed} latency={float(run['latency_seconds']):.3f}s",
            flush=True,
        )

    summary = {
        "method": args.method,
        "prompt_file": str(args.prompt_file),
        "model_path": str(args.model_path),
        "width": args.width,
        "height": args.height,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "negative_prompt": args.negative_prompt,
        "true_cfg_scale": args.true_cfg_scale,
        "max_sequence_length": args.max_sequence_length,
        "results": results,
        "avg_latency_seconds": sum(item["latency_seconds"] for item in results) / max(len(results), 1),
        "avg_token_ratio": sum(item["step_summary"]["avg_token_ratio"] for item in results) / max(len(results), 1),
    }
    write_json(args.output_dir / "run_info.json", summary)


if __name__ == "__main__":
    main()
