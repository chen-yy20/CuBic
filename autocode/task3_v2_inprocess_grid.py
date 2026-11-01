#!/usr/bin/env python3
"""Task 3 v2 matched-active-ratio in-process grid for Flux1 CuBic.

This script is intentionally launched once through ``launch.py``. It loads the
Flux1 CuBic pipeline once, then loops over alpha/beta/tolerance/seed conditions
inside the same SLURM allocation to avoid repeated queueing and weight loading.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
FLUX1_ROOT = REPO_ROOT / "flux1_exp"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FLUX1_ROOT))

import torch

from flux1_exp.cubic_generate import (  # noqa: E402
    GUIDANCE_SCALE,
    HEIGHT,
    NUM_INFERENCE_STEPS,
    PROMPT,
    REQUIRED_TIMER_NAMES,
    TAG,
    WIDTH,
    _merge_quality_and_step_metrics,
    _print_step_latency_trace,
    load_pipeline,
)
from utils.launch_defaults import resolve_model_path  # noqa: E402
from utils.quality_metric import evaluate_quality_with_origin  # noqa: E402
from utils.results import get_prompt_id, save_params_and_metrics  # noqa: E402
from utils.timer import (  # noqa: E402
    get_time_statistics_dict,
    init_timer,
    print_time_statistics,
    save_time_stats_to_file,
)


MATCHED_GRID = {
    (2, 4): [4e-5, 7e-5, 1e-4],
    (3, 4): [1e-4, 2e-4, 4e-4],
    (2, 3): [4e-4, 8e-4, 1.2e-3],
    (2, 5): [2e-5, 5e-5, 1e-4],
}
EXTENDED_GRID = {
    (1.81, 3.43): [3e-4, 6e-4, 1e-3],
    (2.5, 4): [7e-5, 1.2e-4, 2e-4],
    (3.5, 4): [2e-4, 4e-4, 8e-4],
    (4, 4): [4e-4, 8e-4, 1.6e-3],
    (3, 3): [1e-3, 2e-3, 4e-3],
    (3, 3.5): [4e-4, 8e-4, 1.6e-3],
    (3, 4.5): [5e-5, 1e-4, 2e-4],
    (3, 5): [2e-5, 5e-5, 1e-4],
}
CONFIRM_GRID = {
    (1.81, 3.43): [3e-4],
    (2.5, 4): [1.2e-4],
    (3, 3): [4e-3],
    (3, 3.5): [8e-4],
    (3, 4.5): [5e-5],
    (3, 5): [1e-5],
    (3.5, 4): [4e-4],
    (4, 4): [8e-4],
}
GAMMA_GRID = {
    (1.5, 4): [2e-4],       # gamma = 0.30
    (2, 4): [2e-4],         # theoretical gamma = 0.40
    (1.81, 3.426): [2e-4],  # fitted gamma ~= 0.409
    (2.5, 4): [2e-4],       # gamma = 0.50
    (3, 4): [2e-4],         # gamma = 0.60
}
GRID_PROFILES = {
    "matched": MATCHED_GRID,
    "extended": EXTENDED_GRID,
    "confirm": CONFIRM_GRID,
    "gamma": GAMMA_GRID,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 3 v2 matched-active-ratio Flux1 CuBic grid")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--output_root", type=str, default="results/flux/task3_v2_matched_grid")
    parser.add_argument("--baseline_dir", type=str, default=None)
    parser.add_argument("--time_budget_seconds", type=float, default=7200.0)
    parser.add_argument("--condition_limit", type=int, default=0)
    parser.add_argument("--disable_quality_eval", action="store_true", default=False)
    parser.add_argument("--grid_profile", choices=sorted(GRID_PROFILES), default="matched")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--use_target_speedup", action="store_true", default=False)
    parser.add_argument("--target_speedup", type=float, default=2.0)
    parser.add_argument("--curvature_contrast_gamma", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--cooldown_steps", type=int, default=1)
    return parser.parse_args()


def fmt_tol(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")


def condition_iter(
    grid: dict[tuple[float, float], list[float]],
    seeds: list[int],
    condition_limit: int = 0,
):
    emitted = 0
    for (alpha, beta), tolerances in grid.items():
        for tolerance in tolerances:
            for seed in seeds:
                if condition_limit > 0 and emitted >= condition_limit:
                    return
                emitted += 1
                yield alpha, beta, tolerance, seed


def summarize_trace(step_trace: list[dict]) -> dict:
    if not step_trace:
        return {}
    selective = [s for s in step_trace if s.get("phase") == "phase3_cruise_selective"]
    all_ratios = [float(s.get("active_ratio", 0.0)) for s in step_trace]
    selective_ratios = [float(s.get("active_ratio", 0.0)) for s in selective]
    return {
        "steps": len(step_trace),
        "full_compute_steps": sum(1 for s in step_trace if bool(s.get("full_compute"))),
        "mean_active_ratio_all": mean(all_ratios) if all_ratios else None,
        "mean_active_ratio_selective": mean(selective_ratios) if selective_ratios else None,
        "zero_active_selective_steps": sum(1 for s in selective if int(s.get("active_tokens", 0)) == 0),
        "selective_steps": len(selective),
    }


def summarize_frequency_plans(plans: list[dict[int, int]]) -> dict:
    flattened = [int(interval) for plan in plans for interval in plan.values()]
    histogram: dict[str, int] = {}
    for interval in flattened:
        key = str(interval)
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "plan_count": len(plans),
        "first_plan": {str(key): int(value) for key, value in plans[0].items()} if plans else {},
        "interval_histogram": histogram,
    }


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args.model_path, "--model_path")
    model_id = os.path.basename(model_path.rstrip("/"))
    prompt_id = get_prompt_id(args.prompt)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else REPO_ROOT / "results" / "flux" / "ori" / prompt_id
    grid = GRID_PROFILES[args.grid_profile]

    total_conditions = len(list(condition_iter(grid, args.seeds, args.condition_limit)))
    run_start = time.perf_counter()
    condition_summaries = []

    print(
        f"GRID_INPROCESS_START: conditions={total_conditions} model_path={model_path} "
        f"output_root={output_root}",
        flush=True,
    )
    init_timer()
    pipe = load_pipeline(model_path)
    original_optimize = pipe.frequency_optimizer.optimize
    print(f"Model loaded once, GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB", flush=True)

    for index, (alpha, beta, tolerance, seed) in enumerate(
        condition_iter(grid, args.seeds, args.condition_limit),
        start=1,
    ):
        elapsed = time.perf_counter() - run_start
        if args.time_budget_seconds > 0 and elapsed >= 0.8 * float(args.time_budget_seconds):
            print(
                f"TIME_GUARD: stopping before condition={index} elapsed={elapsed:.2f}s "
                f"budget={float(args.time_budget_seconds):.2f}s",
                flush=True,
            )
            break

        condition = f"a{alpha}_b{beta}_tol{fmt_tol(tolerance)}_seed{seed}"
        output_dir = output_root / condition
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{condition}.jsonl"
        condition_start = time.perf_counter()
        print(f"GRID_CONDITION_START: {condition} ({index}/{total_conditions})", flush=True)
        init_timer()
        captured_frequency_plans: list[dict[int, int]] = []

        def capture_optimize(*opt_args, **opt_kwargs):
            plan = original_optimize(*opt_args, **opt_kwargs)
            captured_frequency_plans.append(
                {int(block_id): int(interval) for block_id, interval in plan.block_intervals.items()}
            )
            return plan

        pipe.frequency_optimizer.optimize = capture_optimize
        pipe.use_target_speedup = bool(args.use_target_speedup)
        pipe.curvature_contrast_gamma = float(args.curvature_contrast_gamma)

        generator = torch.Generator("cuda").manual_seed(seed)
        with torch.inference_mode():
            image = pipe(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=512,
                generator=generator,
                cubic_warmup_steps=int(args.warmup_steps),
                cubic_cooldown_steps=int(args.cooldown_steps),
                cubic_target_speedup=float(args.target_speedup),
                cubic_curvature_error_tolerance=float(tolerance),
                cubic_alpha=float(alpha),
                cubic_beta=float(beta),
                cubic_enable_visualization=False,
                cubic_results_dir=str(output_dir),
                cubic_visual_tag=f"seed{seed}",
            ).images[0]

        step_trace = pipe.get_step_trace()
        _print_step_latency_trace(step_trace, seed=seed)
        output_path = output_dir / f"{TAG}_{prompt_id}_seed{seed}.png"
        image.save(output_path)
        print(f"Stored {output_path}!", flush=True)

        per_seed_quality: dict[str, dict] = {}
        if not args.disable_quality_eval:
            baseline_path = baseline_dir / f"ori_{prompt_id}_seed{seed}.png"
            quality_result = evaluate_quality_with_origin(
                image_path=str(output_path),
                tag=TAG,
                save_metrics=False,
                baseline_path=str(baseline_path),
                output_dir=str(output_dir),
            )
            per_seed_quality[f"seed{seed}"] = quality_result["metrics"] if quality_result is not None else {}

        valid_metrics = [m for m in per_seed_quality.values() if m]
        quality_metrics = None
        if not args.disable_quality_eval:
            if valid_metrics:
                metric_names = ["psnr", "ssim", "lpips", "fid", "fvd"]
                mean_metrics = {}
                for name in metric_names:
                    values = [float(m[name]) for m in valid_metrics if m.get(name) is not None]
                    mean_metrics[name] = mean(values) if values else None
                quality_metrics = {"per_seed": per_seed_quality, "mean": mean_metrics}
            else:
                quality_metrics = {"per_seed": per_seed_quality, "mean": None}

        print_time_statistics(timer_names=REQUIRED_TIMER_NAMES)
        save_time_stats_to_file(str(output_dir / "timing_stats.txt"), timer_names=REQUIRED_TIMER_NAMES)

        condition_seconds = time.perf_counter() - condition_start
        if index == 1:
            estimate = condition_seconds * total_conditions
            print(f"TIME_ESTIMATE: {estimate:.2f}s", flush=True)

        trace_summary = summarize_trace(step_trace)
        params = {
            "model": "flux",
            "method": TAG,
            "model_id": model_id,
            "model_path": model_path,
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_images": 1,
            "completed_num_images": 1,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seeds": [seed],
            "completed_seeds": [seed],
            "interval_policy": (
                "budget_constrained" if args.use_target_speedup else "curvature_only_error_tolerance"
            ),
            "curvature_error_tolerance": float(tolerance),
            "alpha": float(alpha),
            "beta": float(beta),
            "effective_gamma": float(alpha) / (float(beta) + 1.0),
            "use_target_speedup": bool(args.use_target_speedup),
            "target_speedup": float(args.target_speedup),
            "curvature_contrast_gamma": float(args.curvature_contrast_gamma),
            "warmup_steps": int(args.warmup_steps),
            "cooldown_steps": int(args.cooldown_steps),
            "dynamic_anchor": "clamped_2x_tau_max",
            "min_anchor_gap": 8,
            "max_anchor_gap": 16,
            "partition_mode": "uniform",
            "uniform_common_factor_blocks": True,
            "quality_eval": not args.disable_quality_eval,
            "baseline_dir": str(baseline_dir),
            "visualization_enabled": False,
            "warmup_runs": 0,
            "time_budget_seconds": float(args.time_budget_seconds),
            "time_guard_fraction": 0.8,
            "condition_seconds": condition_seconds,
            "trace_summary": trace_summary,
            "frequency_plan_summary": summarize_frequency_plans(captured_frequency_plans),
        }
        params_path = save_params_and_metrics(
            str(output_dir),
            TAG,
            params,
            get_time_statistics_dict(timer_names=REQUIRED_TIMER_NAMES),
            quality_metrics=_merge_quality_and_step_metrics(
                quality_metrics,
                step_trace,
                per_seed_trace={f"seed{seed}": step_trace},
            ),
        )

        summary = {
            "condition": condition,
            "index": index,
            "total_conditions": total_conditions,
            "alpha": float(alpha),
            "beta": float(beta),
            "effective_gamma": float(alpha) / (float(beta) + 1.0),
            "tolerance": float(tolerance),
            "seed": int(seed),
            "seconds": condition_seconds,
            "output_path": str(output_path),
            "params_path": params_path,
            "quality_mean": quality_metrics.get("mean") if isinstance(quality_metrics, dict) else None,
            "trace_summary": trace_summary,
            "frequency_plan_summary": summarize_frequency_plans(captured_frequency_plans),
        }
        condition_summaries.append(summary)
        with log_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(
            f"GRID_CONDITION_DONE: {condition} seconds={condition_seconds:.2f} "
            f"params={params_path}",
            flush=True,
        )

    summary_path = output_root / "grid_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "completed": len(condition_summaries),
                "planned": total_conditions,
                "grid_profile": args.grid_profile,
                "seeds": args.seeds,
                "elapsed_seconds": time.perf_counter() - run_start,
                "conditions": condition_summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"GRID_SUMMARY: completed={len(condition_summaries)}/{total_conditions} path={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
