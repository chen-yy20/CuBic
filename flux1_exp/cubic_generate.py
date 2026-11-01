import argparse
import os
import sys
import time
from pathlib import Path
from statistics import mean

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flux.transformer import FluxTransformer2DModel
from flux_cubic.pipeline import CubicFluxPipeline
from utils.launch_defaults import resolve_model_path
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import get_time_statistics_dict, init_timer, print_time_statistics, save_time_stats_to_file

HEIGHT = 1024
WIDTH = 1024
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 3.5
NUM_IMAGES = 3
BASE_SEED = 42
WARMUP_RUNS = 0
CUBIC_WARMUP_STEPS = 1
CUBIC_COOLDOWN_STEPS = 1

CURVATURE_ERROR_TOLERANCE = 2e-4
CUBIC_ALPHA = 3.0
CUBIC_BETA = 4.0


curvature_visualize = False # Set to True to enable curvature visualization, which will store curvature maps and token selection maps for each step in the specified results directory. This is useful for analyzing the behavior of the cubic scheduling, but will consume more storage and slightly more time due to the additional I/O operations.
quality_eval = True

TAG = "cubic_curvonly_balanced"
REQUIRED_TIMER_NAMES = [
    "generate_e2e",
    "anchor_step_monitor",
    "anchor_step_partitioner",
    "anchor_step_optimizer",
    "anchor_step_total",
    "warmup_cooldown_step",
    "selective_forward",
]

def _build_step_latency_metrics(step_trace, per_seed_trace=None):
    trace = [dict(item) for item in step_trace]
    payload = {
        "steps": trace,
        "text_lines": [item["text"] for item in trace],
    }
    if per_seed_trace is not None:
        payload["per_seed"] = {
            seed_key: {
                "steps": [dict(item) for item in seed_trace],
                "text_lines": [item["text"] for item in seed_trace],
            }
            for seed_key, seed_trace in per_seed_trace.items()
        }
    return {"cubic_step_trace": payload}


def _build_cache_budget_ledger(per_seed_trace: dict[str, list[dict]], fallback_trace: list[dict]) -> dict:
    traces = per_seed_trace or {"trace": fallback_trace}
    per_seed = {}
    ratios = []
    total_steps_values = []
    full_compute_values = []
    reuse_values = []
    for seed_key, trace in traces.items():
        rows = [dict(item) for item in trace]
        total_steps = len(rows)
        active_ratios = [float(row.get("active_ratio", 0.0)) for row in rows]
        ratio = mean(active_ratios) if active_ratios else None
        full_compute_steps = sum(1 for row in rows if bool(row.get("full_compute")))
        reuse_steps = max(0, total_steps - full_compute_steps)
        total_steps_values.append(total_steps)
        full_compute_values.append(full_compute_steps)
        reuse_values.append(reuse_steps)
        if ratio is not None:
            ratios.append(float(ratio))
        per_seed[seed_key] = {
            "total_steps": int(total_steps),
            "full_compute_steps": int(full_compute_steps),
            "reuse_steps": int(reuse_steps),
            "mean_step_compute_ratio": float(ratio) if ratio is not None else None,
            "mean_token_compute_ratio": float(ratio) if ratio is not None else None,
            "effective_dit_compute_ratio": float(ratio) if ratio is not None else None,
        }
    mean_ratio = mean(ratios) if ratios else None
    return {
        "method": "cubic",
        "budget_kind": "active_token_proxy",
        "budget_precision": "token_active_ratio_proxy_not_exact_flops",
        "total_steps": int(round(mean(total_steps_values))) if total_steps_values else 0,
        "full_compute_steps": float(mean(full_compute_values)) if full_compute_values else 0.0,
        "reuse_steps": float(mean(reuse_values)) if reuse_values else 0.0,
        "mean_step_compute_ratio": float(mean_ratio) if mean_ratio is not None else None,
        "mean_layer_compute_ratio": float(mean_ratio) if mean_ratio is not None else None,
        "mean_token_compute_ratio": float(mean_ratio) if mean_ratio is not None else None,
        "effective_dit_compute_ratio": float(mean_ratio) if mean_ratio is not None else None,
        "effective_dit_compute_ratio_note": (
            "Cubic uses sparse active-token DiT compute; this is an active-token proxy, "
            "not an exact FLOPs measurement."
        ),
        "per_seed": per_seed,
    }


def _merge_quality_and_step_metrics(quality_metrics, step_trace, per_seed_trace=None):
    merged = dict(quality_metrics) if isinstance(quality_metrics, dict) else {}
    merged.update(_build_step_latency_metrics(step_trace, per_seed_trace=per_seed_trace))
    return merged


def _print_step_latency_trace(step_trace, seed=None):
    prefix = "Cubic per-step latency / active tokens"
    if seed is not None:
        prefix += f" (seed={seed})"
    print(f"{prefix}:", flush=True)
    for item in step_trace:
        print(item["text"], flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux cubic inference script")
    parser.add_argument("--model_path", type=str, default=None, help="Model path or HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Generation prompt")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Output image height")
    parser.add_argument("--width", type=int, default=WIDTH, help="Output image width")
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS, help="Inference steps")
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base random seed")
    parser.add_argument("--num_images", type=int, default=NUM_IMAGES, help="Number of images/seeds to generate")
    parser.add_argument("--warmup_runs", type=int, default=WARMUP_RUNS, help="Warmup runs before timed generation")
    parser.add_argument("--cubic_warmup_steps", type=int, default=CUBIC_WARMUP_STEPS)
    parser.add_argument("--cubic_cooldown_steps", type=int, default=CUBIC_COOLDOWN_STEPS)
    parser.add_argument("--cubic_anchor_policy", choices=["dynamic", "fixed"], default="dynamic")
    parser.add_argument("--cubic_anchor_gap_multiplier", type=float, default=2.0)
    parser.add_argument(
        "--cubic_anchor_interval",
        type=int,
        default=0,
        help="Fixed anchor interval; a positive value selects the fixed policy",
    )
    parser.add_argument("--cubic_min_anchor_gap", type=int, default=8)
    parser.add_argument("--cubic_max_anchor_gap", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory override")
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default=None,
        help="Optional Flux origin baseline directory for quality evaluation",
    )
    parser.add_argument(
        "--cubic_curvature_error_tolerance",
        type=float,
        default=CURVATURE_ERROR_TOLERANCE,
        help="Curvature-only reuse tolerance; lower values recompute more often",
    )
    parser.add_argument(
        "--cubic_alpha",
        type=float,
        default=CUBIC_ALPHA,
        help="Curvature exponent alpha used by the Cubic interval optimizer",
    )
    parser.add_argument(
        "--cubic_beta",
        type=float,
        default=CUBIC_BETA,
        help="Interval exponent beta used by the Cubic interval optimizer",
    )
    parser.add_argument("--cubic_partition_mode", type=str, default="uniform", choices=["uniform", "uniform_grid", "morphology"])
    parser.add_argument("--cubic_uniform_grid_h", type=int, default=0)
    parser.add_argument("--cubic_uniform_grid_w", type=int, default=0)
    parser.add_argument("--cubic_uniform_common_factor_blocks", type=str, default="true")
    parser.add_argument("--cubic_min_block_size", type=int, default=None)
    parser.add_argument("--cubic_max_block_size", type=int, default=None)
    parser.add_argument("--cubic_cv_threshold", type=float, default=None)
    parser.add_argument(
        "--time_budget_seconds",
        type=float,
        default=0.0,
        help="Optional wall-clock budget; stops at seed boundaries after 80%% of this budget",
    )
    parser.add_argument(
        "--cubic_error_probe",
        action="store_true",
        default=False,
        help="Enable diagnostic full-forward oracle probes on selective steps",
    )
    parser.add_argument(
        "--cubic_error_probe_max_steps",
        type=int,
        default=0,
        help="Maximum selective steps to probe per seed; 0 means no explicit limit",
    )
    parser.add_argument(
        "--cubic_error_probe_output_path",
        type=str,
        default=None,
        help="Optional JSONL path for block-level error probe records",
    )
    parser.add_argument(
        "--disable_visualization",
        action="store_true",
        default=not curvature_visualize,
        help="Disable curvature visualization export",
    )
    parser.add_argument(
        "--disable_quality_eval",
        action="store_true",
        default=not quality_eval,
        help="Disable quality evaluation against ori baseline",
    )
    return parser.parse_args()


def load_pipeline(model_path: str) -> CubicFluxPipeline:
    pipe = CubicFluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.transformer = FluxTransformer2DModel.from_pretrained(
        f"{model_path}/transformer",
        torch_dtype=torch.bfloat16,
    )
    return pipe.to("cuda")


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path, "--model_path")
    model_id = os.path.basename(model_path.rstrip("/"))
    tag = TAG
    prompt_id = get_prompt_id(args.prompt)
    output_dir = args.output_dir or f"./results/flux/{tag}/{prompt_id}"
    seeds = [args.seed + i for i in range(max(1, int(args.num_images)))]

    init_timer()
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_pipeline(model_path)
    print(f"Model loaded, GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB", flush=True)

    per_seed_quality: dict[str, dict] = {}
    per_seed_step_trace: dict[str, list[dict]] = {}
    per_seed_anchor_drift: dict[str, list[dict]] = {}
    completed_seeds: list[int] = []
    run_start_time = time.perf_counter()
    stopped_by_time_guard = False
    time_estimate_seconds = None
    error_probe_paths: list[str] = []

    for warmup_idx in range(max(0, int(args.warmup_runs))):
        warmup_generator = torch.Generator("cuda").manual_seed(seeds[0] + warmup_idx)
        with torch.inference_mode():
            pipe(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=512,
                generator=warmup_generator,
                cubic_warmup_steps=args.cubic_warmup_steps,
                cubic_cooldown_steps=args.cubic_cooldown_steps,
                cubic_anchor_policy=args.cubic_anchor_policy,
                cubic_anchor_gap_multiplier=args.cubic_anchor_gap_multiplier,
                cubic_anchor_interval=args.cubic_anchor_interval or None,
                cubic_min_anchor_gap=args.cubic_min_anchor_gap,
                cubic_max_anchor_gap=args.cubic_max_anchor_gap,
                cubic_curvature_error_tolerance=args.cubic_curvature_error_tolerance,
                cubic_alpha=args.cubic_alpha,
                cubic_beta=args.cubic_beta,
                cubic_partition_mode=args.cubic_partition_mode,
                cubic_uniform_grid_h=args.cubic_uniform_grid_h,
                cubic_uniform_grid_w=args.cubic_uniform_grid_w,
                cubic_uniform_common_factor_blocks=args.cubic_uniform_common_factor_blocks.lower() in {"1", "true", "yes", "on"},
                cubic_min_block_size=args.cubic_min_block_size,
                cubic_max_block_size=args.cubic_max_block_size,
                cubic_cv_threshold=args.cubic_cv_threshold,
                cubic_enable_visualization=not args.disable_visualization,
                cubic_results_dir=output_dir,
                cubic_visual_tag=f"warmup{warmup_idx}",
            )

    for seed in seeds:
        if args.time_budget_seconds > 0:
            elapsed = time.perf_counter() - run_start_time
            if elapsed >= 0.8 * float(args.time_budget_seconds):
                stopped_by_time_guard = True
                print(
                    f"TIME_GUARD: stopping before seed={seed} "
                    f"elapsed={elapsed:.2f}s budget={float(args.time_budget_seconds):.2f}s",
                    flush=True,
                )
                break

        seed_start_time = time.perf_counter()
        generator = torch.Generator("cuda").manual_seed(seed)
        with torch.inference_mode():
            probe_path = None
            if args.cubic_error_probe:
                if args.cubic_error_probe_output_path:
                    base, ext = os.path.splitext(args.cubic_error_probe_output_path)
                    probe_path = f"{base}_seed{seed}{ext or '.jsonl'}"
                else:
                    probe_path = os.path.join(output_dir, f"{tag}_error_probe_seed{seed}.jsonl")
                error_probe_paths.append(probe_path)
            image = pipe(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=512,
                generator=generator,
                cubic_warmup_steps=args.cubic_warmup_steps,
                cubic_cooldown_steps=args.cubic_cooldown_steps,
                cubic_anchor_policy=args.cubic_anchor_policy,
                cubic_anchor_gap_multiplier=args.cubic_anchor_gap_multiplier,
                cubic_anchor_interval=args.cubic_anchor_interval or None,
                cubic_min_anchor_gap=args.cubic_min_anchor_gap,
                cubic_max_anchor_gap=args.cubic_max_anchor_gap,
                cubic_curvature_error_tolerance=args.cubic_curvature_error_tolerance,
                cubic_alpha=args.cubic_alpha,
                cubic_beta=args.cubic_beta,
                cubic_partition_mode=args.cubic_partition_mode,
                cubic_uniform_grid_h=args.cubic_uniform_grid_h,
                cubic_uniform_grid_w=args.cubic_uniform_grid_w,
                cubic_uniform_common_factor_blocks=args.cubic_uniform_common_factor_blocks.lower() in {"1", "true", "yes", "on"},
                cubic_min_block_size=args.cubic_min_block_size,
                cubic_max_block_size=args.cubic_max_block_size,
                cubic_cv_threshold=args.cubic_cv_threshold,
                cubic_error_probe=args.cubic_error_probe,
                cubic_error_probe_max_steps=args.cubic_error_probe_max_steps,
                cubic_error_probe_output_path=probe_path,
                cubic_enable_visualization=not args.disable_visualization,
                cubic_results_dir=output_dir,
                cubic_visual_tag=f"seed{seed}",
            ).images[0]
        step_trace = pipe.get_step_trace()
        per_seed_step_trace[f"seed{seed}"] = step_trace
        per_seed_anchor_drift[f"seed{seed}"] = pipe.get_anchor_curvature_drift()
        _print_step_latency_trace(step_trace, seed=seed)
        output_path = os.path.join(output_dir, f"{tag}_{prompt_id}_seed{seed}.png")
        image.save(output_path)
        print(f"Stored {output_path}!", flush=True)

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
            per_seed_quality[f"seed{seed}"] = quality_result["metrics"] if quality_result is not None else {}

        completed_seeds.append(seed)
        if len(completed_seeds) == 1:
            first_seed_seconds = time.perf_counter() - seed_start_time
            time_estimate_seconds = first_seed_seconds * len(seeds)
            print(f"TIME_ESTIMATE: {time_estimate_seconds:.2f}s", flush=True)

    quality_metrics = None
    if not args.disable_quality_eval:
        valid_metrics = [m for m in per_seed_quality.values() if m]
        if valid_metrics:
            metric_names = ["psnr", "ssim", "lpips", "fid", "fvd"]
            mean_metrics = {}
            for name in metric_names:
                values = [float(m[name]) for m in valid_metrics if m.get(name) is not None]
                mean_metrics[name] = mean(values) if values else None
            quality_metrics = {
                "per_seed": per_seed_quality,
                "mean": mean_metrics,
            }
        else:
            quality_metrics = {
                "per_seed": per_seed_quality,
                "mean": None,
            }

    print_time_statistics(timer_names=REQUIRED_TIMER_NAMES)
    save_time_stats_to_file(
        os.path.join(output_dir, "timing_stats.txt"),
        timer_names=REQUIRED_TIMER_NAMES,
    )
    first_step_trace = next(iter(per_seed_step_trace.values()), [])
    params = {
        "model": "flux",
        "method": tag,
        "model_id": model_id,
        "model_path": model_path,
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_images": len(seeds),
        "completed_num_images": len(completed_seeds),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seeds": seeds,
        "completed_seeds": completed_seeds,
        "interval_policy": "curvature_only_error_tolerance",
        "curvature_error_tolerance": args.cubic_curvature_error_tolerance,
        "alpha": args.cubic_alpha,
        "beta": args.cubic_beta,
        "warmup_steps": args.cubic_warmup_steps,
        "cooldown_steps": args.cubic_cooldown_steps,
        "anchor_policy": "fixed" if args.cubic_anchor_interval > 0 else args.cubic_anchor_policy,
        "anchor_gap_multiplier": float(args.cubic_anchor_gap_multiplier),
        "fixed_anchor_interval": int(args.cubic_anchor_interval) if args.cubic_anchor_interval > 0 else None,
        "min_anchor_gap": int(args.cubic_min_anchor_gap),
        "max_anchor_gap": int(args.cubic_max_anchor_gap),
        "curvature_contrast_gamma": 0.5,
        "partition_mode": args.cubic_partition_mode,
        "uniform_grid_h": args.cubic_uniform_grid_h,
        "uniform_grid_w": args.cubic_uniform_grid_w,
        "uniform_common_factor_blocks": args.cubic_uniform_common_factor_blocks.lower() in {"1", "true", "yes", "on"},
        "min_block_size": args.cubic_min_block_size,
        "max_block_size": args.cubic_max_block_size,
        "cv_threshold": args.cubic_cv_threshold,
        "quality_eval": not args.disable_quality_eval,
        "baseline_dir": args.baseline_dir or os.path.join("results", "flux", "ori", prompt_id),
        "visualization_enabled": not args.disable_visualization,
        "warmup_runs": max(0, int(args.warmup_runs)),
        "time_budget_seconds": float(args.time_budget_seconds),
        "time_guard_fraction": 0.8,
        "stopped_by_time_guard": stopped_by_time_guard,
        "time_estimate_seconds": time_estimate_seconds,
        "error_probe_enabled": bool(args.cubic_error_probe),
        "error_probe_max_steps": int(args.cubic_error_probe_max_steps),
        "error_probe_paths": error_probe_paths,
        "cache_budget_ledger": _build_cache_budget_ledger(per_seed_step_trace, first_step_trace),
    }
    params_path = save_params_and_metrics(
        output_dir,
        tag,
        params,
        get_time_statistics_dict(timer_names=REQUIRED_TIMER_NAMES),
        quality_metrics={
            **_merge_quality_and_step_metrics(
                quality_metrics,
                first_step_trace,
                per_seed_trace=per_seed_step_trace,
            ),
            "cubic_anchor_curvature_drift": {"per_seed": per_seed_anchor_drift},
        },
    )
    print(f"Params & metrics saved to {params_path}", flush=True)


if __name__ == "__main__":
    main()
