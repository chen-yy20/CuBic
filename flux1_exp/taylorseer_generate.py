"""Flux.1 TaylorSeer baseline via the diffusers TaylorSeer cache hook.

Runs the uncached pipeline and every requested cache interval in one process, so that
speedups are normalized by the uncached run on the same code path.
"""

import argparse
import json
import os
import time

import torch
from diffusers import FluxPipeline
from diffusers.hooks import TaylorSeerCacheConfig

from utils.launch_defaults import resolve_model_path
from utils.results import get_prompt_id

PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--cache_intervals", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--disable_cache_before_step", type=int, default=3)
    parser.add_argument("--max_order", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2, help="Timed runs per setting; the fastest is reported")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def generate(pipe, args):
    generator = torch.Generator("cuda").manual_seed(args.seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            max_sequence_length=512,
            generator=generator,
        ).images[0]
    torch.cuda.synchronize()
    return image, (time.perf_counter() - start) * 1000.0


def timed(pipe, args):
    runs = [generate(pipe, args) for _ in range(max(1, args.repeats))]
    return runs[-1][0], min(ms for _, ms in runs)


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model_path, "--model_path")
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    prompt_id = get_prompt_id(args.prompt)
    os.makedirs(args.output_dir, exist_ok=True)

    generate(pipe, args)
    image, ori_ms = timed(pipe, args)
    image.save(os.path.join(args.output_dir, f"ori_{prompt_id}_seed{args.seed}.png"))
    records = [{"name": "ori", "latency_ms": ori_ms}]

    for interval in args.cache_intervals:
        pipe.transformer.enable_cache(
            TaylorSeerCacheConfig(
                cache_interval=interval,
                disable_cache_before_step=args.disable_cache_before_step,
                max_order=args.max_order,
            )
        )
        image, ms = timed(pipe, args)
        pipe.transformer.disable_cache()
        name = f"taylorseer_n{interval}"
        image.save(os.path.join(args.output_dir, f"{name}_{prompt_id}_seed{args.seed}.png"))
        records.append({"name": name, "cache_interval": interval, "latency_ms": ms, "speedup": ori_ms / ms})
        print(f"{name}: {ms:.1f} ms, speedup {ori_ms / ms:.2f}x", flush=True)

    with open(os.path.join(args.output_dir, "taylorseer_records.json"), "w") as f:
        json.dump({"prompt": args.prompt, "seed": args.seed, "records": records}, f, indent=2)


if __name__ == "__main__":
    main()
