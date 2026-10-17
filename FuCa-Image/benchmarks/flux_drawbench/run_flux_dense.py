from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

from flux_sampling_common import append_sampling_latency_fields, manual_flux_sampling_only, summarize_sampling_results


def read_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--guidance-scale', type=float, default=3.5)
    parser.add_argument('--method-name', type=str, default='dense50')
    args = parser.parse_args()

    prompts = read_prompts(args.prompt_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = DiffusionPipeline.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.float16,
    ).to('cuda')

    results = []
    for i, prompt in enumerate(prompts):
        image, sampling_elapsed = manual_flux_sampling_only(
            pipeline=pipeline,
            prompt=prompt,
            num_steps=args.num_steps,
            seed=args.seed + i,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            max_sequence_length=512,
        )
        out_path = args.output_dir / f'output_{i:02d}.png'
        image.save(out_path)
        record = {
            'index': i,
            'prompt': prompt,
            'image_path': str(out_path),
        }
        results.append(append_sampling_latency_fields(record, sampling_elapsed))

    summary = {
        'method': args.method_name,
        'prompt_file': str(args.prompt_file),
        'model_path': str(args.model_path),
        'num_steps': args.num_steps,
        'width': args.width,
        'height': args.height,
        'seed': args.seed,
        'results': results,
        **summarize_sampling_results(results),
    }
    (args.output_dir / 'run_info.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
