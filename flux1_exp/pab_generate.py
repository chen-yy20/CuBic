"""Flux + fixed-frequency PAB inference entry."""

from __future__ import annotations

import argparse
import os
import sys
from statistics import mean
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.pab_flux import FluxPABManager, attach_flux_pab, resolve_skip_range_for_target
from flux.pipeline import FluxPipeline
from flux.transformer import FluxTransformer2DModel
from utils.launch_defaults import resolve_model_path
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import (
    get_time_statistics_dict,
    init_timer,
    print_time_statistics,
    save_time_stats_to_file,
)


PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 3.5
NUM_IMAGES = 3
SEEDS = [42 + i for i in range(NUM_IMAGES)]
PAB_SKIP_RANGE = 3
PAB_WARMUP_STEPS = 0
PAB_COOLDOWN_STEPS = 1
PAB_LAYER_INTERVAL = 1
TAG = "pab"
REQUIRED_TIMER_NAMES = ["generate_e2e", "dit"]


def _format_pab_trace(trace: list[dict]) -> str:
    return "".join("1" if bool(row.get("should_compute")) else "0" for row in trace)


def _merge_quality_and_pab_metrics(quality_metrics, per_seed_trace: dict[str, dict]):
    merged = dict(quality_metrics) if isinstance(quality_metrics, dict) else {}
    merged["pab_compute_trace"] = {"per_seed": per_seed_trace}
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux PAB inference script")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--num_images", type=int, default=NUM_IMAGES)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--pab_skip_range", type=int, default=PAB_SKIP_RANGE)
    parser.add_argument("--pab_warmup_steps", type=int, default=PAB_WARMUP_STEPS)
    parser.add_argument("--pab_cooldown_steps", type=int, default=PAB_COOLDOWN_STEPS)
    parser.add_argument("--pab_layer_interval", type=int, default=PAB_LAYER_INTERVAL)
    parser.add_argument(
        "--target_compute_ratio",
        type=float,
        default=None,
        help="If set, choose the fixed PAB skip range closest to this step-level compute ratio.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--baseline_dir", type=str, default=None)
    parser.add_argument("--disable_quality_eval", action="store_true", default=False)
    return parser.parse_args()


def load_pipeline(model_path: str) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.transformer = FluxTransformer2DModel.from_pretrained(
        f"{model_path}/transformer",
        torch_dtype=torch.bfloat16,
    )
    return pipe.to("cuda")


def _mean_quality(per_seed_quality: dict[str, dict]) -> dict | None:
    valid_metrics = [m for m in per_seed_quality.values() if m]
    if not valid_metrics:
        return None
    metric_names = ["psnr", "ssim", "lpips", "fid", "fvd"]
    mean_metrics = {}
    for name in metric_names:
        values = [float(m[name]) for m in valid_metrics if m.get(name) is not None]
        mean_metrics[name] = mean(values) if values else None
    return mean_metrics


def main() -> None:
    args = parse_args()
    init_timer()
    args.model_path = resolve_model_path(args.model_path, "--model_path")
    if args.target_compute_ratio is not None:
        args.pab_skip_range = resolve_skip_range_for_target(
            total_steps=args.num_inference_steps,
            target_compute_ratio=args.target_compute_ratio,
            warmup_steps=args.pab_warmup_steps,
            cooldown_steps=args.pab_cooldown_steps,
        )

    seeds = args.seeds if args.seeds is not None else [42 + i for i in range(args.num_images)]
    prompt_id = get_prompt_id(args.prompt)
    model_id = os.path.basename(args.model_path.rstrip("/"))
    tag = f"{TAG}_r{args.pab_skip_range}_li{args.pab_layer_interval}"
    output_dir = args.output_dir or os.path.join("results", "flux", tag, prompt_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[pab-flux] Loading pipeline from {args.model_path}...", flush=True)
    pipe = load_pipeline(args.model_path)
    print(
        f"[pab-flux] Pipeline loaded. GPU memory: {torch.cuda.memory_allocated() / 1024**2:.0f} MB",
        flush=True,
    )

    transformer: FluxTransformer2DModel = pipe.transformer
    manager = FluxPABManager(
        total_steps=args.num_inference_steps,
        skip_range=args.pab_skip_range,
        warmup_steps=args.pab_warmup_steps,
        cooldown_steps=args.pab_cooldown_steps,
        layer_interval=args.pab_layer_interval,
    )
    attach_flux_pab(transformer, manager)
    print(
        f"[pab-flux] skip_range={args.pab_skip_range} warmup={args.pab_warmup_steps} "
        f"cooldown={args.pab_cooldown_steps} layer_interval={args.pab_layer_interval} "
        f"target_layers={len(manager.target_layers)}/{manager.total_layers}",
        flush=True,
    )

    per_seed_quality: dict[str, dict] = {}
    per_seed_trace: dict[str, dict] = {}
    per_seed_ledgers: dict[str, dict] = {}

    for seed in seeds:
        manager.reset()
        generator = torch.Generator("cuda").manual_seed(seed)
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

        ledger = manager.budget_ledger()
        trace = ledger["trace"]
        per_seed_ledgers[f"seed{seed}"] = ledger
        per_seed_trace[f"seed{seed}"] = {
            "text": _format_pab_trace(trace),
            "steps": trace,
            "budget": {key: value for key, value in ledger.items() if key != "trace"},
        }
        print(
            f"[pab-flux] seed={seed} compute/reuse sequence "
            f"(1=compute target attn,0=reuse): {_format_pab_trace(trace)} "
            f"attention_compute_ratio={ledger['effective_attention_compute_ratio']:.4f}",
            flush=True,
        )

        output_path = os.path.join(output_dir, f"{tag}_{prompt_id}_seed{seed}.png")
        image.save(output_path)
        print(f"[pab-flux] Saved {output_path}", flush=True)

        if not args.disable_quality_eval:
            baseline_dir = args.baseline_dir or os.path.join("results", "flux", "ori", prompt_id)
            baseline_path = os.path.join(baseline_dir, f"ori_{prompt_id}_seed{seed}.png")
            quality_result = evaluate_quality_with_origin(
                image_path=output_path,
                tag=tag,
                save_metrics=False,
                baseline_path=baseline_path,
                output_dir=output_dir,
            )
            per_seed_quality[f"seed{seed}"] = (
                quality_result["metrics"] if quality_result is not None else {}
            )

    quality_metrics = None
    if not args.disable_quality_eval:
        quality_metrics = {
            "per_seed": per_seed_quality,
            "mean": _mean_quality(per_seed_quality),
        }

    representative_ledger = next(iter(per_seed_ledgers.values()), None) or manager.budget_ledger()
    print_time_statistics(timer_names=REQUIRED_TIMER_NAMES)
    save_time_stats_to_file(
        os.path.join(output_dir, "timing_stats.txt"),
        timer_names=REQUIRED_TIMER_NAMES,
    )
    timing = get_time_statistics_dict(timer_names=REQUIRED_TIMER_NAMES)
    e2e = timing.get("generate_e2e", {})
    if e2e.get("avg_ms") is not None and len(seeds) > 0:
        time_estimate_seconds = float(e2e["avg_ms"]) * len(seeds) / 1000.0
        print(f"TIME_ESTIMATE: {time_estimate_seconds:.2f}s", flush=True)

    params = {
        "model": "flux",
        "method": tag,
        "model_id": model_id,
        "model_path": args.model_path,
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_images": len(seeds),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seeds": seeds,
        "pab_skip_range": args.pab_skip_range,
        "pab_warmup_steps": args.pab_warmup_steps,
        "pab_cooldown_steps": args.pab_cooldown_steps,
        "pab_layer_interval": args.pab_layer_interval,
        "target_compute_ratio": args.target_compute_ratio,
        "quality_eval": not args.disable_quality_eval,
        "baseline_dir": args.baseline_dir or os.path.join("results", "flux", "ori", prompt_id),
        "cache_budget_ledger": {
            **{key: value for key, value in representative_ledger.items() if key != "trace"},
            "per_seed": {
                seed_key: {key: value for key, value in ledger.items() if key != "trace"}
                for seed_key, ledger in per_seed_ledgers.items()
            },
        },
    }
    params_path = save_params_and_metrics(
        output_dir,
        tag,
        params,
        timing,
        quality_metrics=_merge_quality_and_pab_metrics(quality_metrics, per_seed_trace),
    )
    print(f"[pab-flux] Params & metrics saved to {params_path}", flush=True)


if __name__ == "__main__":
    main()
