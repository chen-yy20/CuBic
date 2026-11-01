#!/usr/bin/env python3
"""Run the Flux anchor-horizon/block-granularity ablation matrix with one model load."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FLUX_ROOT = REPO_ROOT / "flux1_exp"
if str(FLUX_ROOT) not in sys.path:
    sys.path.insert(0, str(FLUX_ROOT))

from flux1_exp.cubic_generate import load_pipeline
from utils.launch_defaults import resolve_model_path
from utils.quality_metric import QualityEvaluator
from utils.results import get_prompt_id
from utils.timer import get_time_statistics_dict, init_timer


DEFAULT_PROMPTS = [
    "A photorealistic cute cat wearing a simple blue shirt against a clear sky.",
    "A red vintage car parked beside a quiet lake at sunrise.",
    "An astronaut riding a horse through a snowy forest.",
    "A detailed watercolor painting of a busy night market.",
    "A glass teapot filled with flowers on a wooden table.",
    "A small robot reading a book in an old library.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="In-process Flux anchor/granularity ablation matrix")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--output_root", default="results/flux/anchor_granularity_matrix")
    parser.add_argument("--prompt", default=None, help="Single-prompt override used by launch.py")
    parser.add_argument("--prompt_file", default=None, help="Text or JSONL file; JSONL uses the prompt field")
    parser.add_argument("--max_prompts", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["anchor", "granularity", "frontier"],
        default=["anchor", "granularity", "frontier"],
    )
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--warmup_runs", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=12)
    parser.add_argument("--cooldown_steps", type=int, default=6)
    parser.add_argument("--curvature_tolerance", type=float, default=2e-4)
    parser.add_argument(
        "--target_speedup",
        type=float,
        default=None,
        help="Enable the paper's fixed-budget optimizer with rho=1/target_speedup.",
    )
    parser.add_argument(
        "--frontier_tolerances",
        nargs="+",
        type=float,
        default=[1e-4, 2e-4, 4e-4],
    )
    parser.add_argument(
        "--fixed_anchor_interval",
        type=int,
        default=7,
        help="Fixed-gap control; default matches the pilot's mean dynamic-2x anchor frequency",
    )
    parser.add_argument("--time_budget_seconds", type=float, default=0.0)
    parser.add_argument("--skip_quality", action="store_true")
    return parser.parse_args()


def load_prompts(path: str | None, max_prompts: int) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS[:max_prompts]
    prompts = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{"):
            prompt = json.loads(line).get("prompt")
        else:
            prompt = line
        if prompt:
            prompts.append(str(prompt))
        if len(prompts) >= max_prompts:
            break
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def build_conditions(
    stages: list[str],
    fixed_anchor_interval: int,
    frontier_tolerances: list[float],
) -> list[dict]:
    conditions = []
    if "anchor" in stages:
        for multiplier in (1.0, 2.0, 3.0):
            conditions.append(
                {
                    "name": f"anchor_dynamic_{multiplier:g}x",
                    "stage": "anchor",
                    "anchor_policy": "dynamic",
                    "anchor_gap_multiplier": multiplier,
                    "anchor_interval": None,
                    "partition_mode": "uniform_grid",
                    "grid": 3,
                    "curvature_tolerance": None,
                }
            )
        conditions.append(
            {
                "name": f"anchor_fixed_{fixed_anchor_interval}",
                "stage": "anchor",
                "anchor_policy": "fixed",
                "anchor_gap_multiplier": 2.0,
                "anchor_interval": fixed_anchor_interval,
                "partition_mode": "uniform_grid",
                "grid": 3,
                "curvature_tolerance": None,
            }
        )
    if "granularity" in stages:
        for label, grid in (("coarse", 3), ("default", 4), ("fine", 6)):
            conditions.append(
                {
                    "name": f"granularity_{label}_{grid}x{grid}",
                    "stage": "granularity",
                    "anchor_policy": "dynamic",
                    "anchor_gap_multiplier": 2.0,
                    "anchor_interval": None,
                    "partition_mode": "uniform_grid",
                    "grid": grid,
                    "curvature_tolerance": None,
                }
            )
        conditions.append(
            {
                "name": "granularity_adaptive",
                "stage": "granularity",
                "anchor_policy": "dynamic",
                "anchor_gap_multiplier": 2.0,
                "anchor_interval": None,
                "partition_mode": "morphology",
                "grid": 0,
                "curvature_tolerance": None,
            }
        )
    if "frontier" in stages:
        for tolerance in frontier_tolerances:
            conditions.append(
                {
                    "name": f"frontier_cubic_tol{tolerance:.0e}".replace("-", "m"),
                    "stage": "frontier",
                    "anchor_policy": "dynamic",
                    "anchor_gap_multiplier": 2.0,
                    "anchor_interval": None,
                    "partition_mode": "uniform_grid",
                    "grid": 3,
                    "curvature_tolerance": tolerance,
                }
            )
    return conditions


def summarize_trace(trace: list[dict], drift: list[dict]) -> dict:
    active_ratios = [float(row["active_ratio"]) for row in trace]
    anchor_steps = [row for row in trace if row.get("phase") == "phase2_cruise_anchor"]
    return {
        "mean_active_ratio": mean(active_ratios) if active_ratios else None,
        "anchor_count": len(anchor_steps),
        "full_compute_steps": sum(bool(row.get("full_compute")) for row in trace),
        "mean_anchor_gap": mean([row["observed_anchor_gap"] for row in drift]) if drift else None,
        "mean_normalized_l1_drift": mean([row["normalized_l1_drift"] for row in drift]) if drift else None,
        "mean_rank_correlation": mean([row["rank_correlation"] for row in drift]) if drift else None,
    }


def evaluate_fidelity(evaluator: QualityEvaluator, generated_path: Path, reference_path: Path) -> dict:
    generated = evaluator.preprocess_image(str(generated_path))
    reference = evaluator.preprocess_image(str(reference_path))
    generated, reference = evaluator.resize_to_match(generated, reference)
    return {
        "psnr": evaluator.compute_psnr(reference, generated),
        "ssim": evaluator.compute_ssim(reference, generated),
        "lpips": evaluator.compute_lpips(reference, generated),
    }


def generate(pipe, args: argparse.Namespace, prompt: str, seed: int, condition: dict | None):
    generator = torch.Generator("cuda").manual_seed(seed)
    common = {
        "prompt": prompt,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "max_sequence_length": 512,
        "generator": generator,
        "cubic_enable_visualization": False,
    }
    if condition is None:
        common.update(
            {
                "cubic_warmup_steps": args.num_inference_steps,
                "cubic_cooldown_steps": 0,
            }
        )
    else:
        common.update(
            {
                "cubic_warmup_steps": args.warmup_steps,
                "cubic_cooldown_steps": args.cooldown_steps,
                "cubic_anchor_policy": condition["anchor_policy"],
                "cubic_anchor_gap_multiplier": condition["anchor_gap_multiplier"],
                "cubic_anchor_interval": condition["anchor_interval"],
                "cubic_min_anchor_gap": 1,
                "cubic_max_anchor_gap": args.num_inference_steps,
                "cubic_target_speedup": args.target_speedup,
                "cubic_curvature_error_tolerance": (
                    condition["curvature_tolerance"]
                    if condition.get("curvature_tolerance") is not None
                    else args.curvature_tolerance
                ),
                "cubic_partition_mode": condition["partition_mode"],
                "cubic_uniform_grid_h": condition["grid"],
                "cubic_uniform_grid_w": condition["grid"],
                "cubic_uniform_common_factor_blocks": False,
            }
        )
    init_timer()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        image = pipe(**common).images[0]
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000.0
    return image, latency_ms, get_time_statistics_dict(), pipe.get_step_trace(), pipe.get_anchor_curvature_drift()


def main() -> int:
    args = parse_args()
    prompts = [args.prompt] if args.prompt else load_prompts(args.prompt_file, args.max_prompts)
    conditions = build_conditions(
        args.stages,
        args.fixed_anchor_interval,
        args.frontier_tolerances,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "records.jsonl"
    summary_path = output_root / "run_summary.json"
    model_path = resolve_model_path(args.model_path, "--model_path")
    pipe = load_pipeline(model_path)
    pipe.use_target_speedup = args.target_speedup is not None
    for _ in range(args.warmup_runs):
        generate(pipe, args, prompts[0], args.seeds[0], None)
    evaluator = (
        None
        if args.skip_quality
        else QualityEvaluator(device="cpu", enable_distribution_metrics=False)
    )
    records = []
    started = time.perf_counter()
    total_accelerated = len(prompts) * len(args.seeds) * len(conditions)
    first_accelerated_seconds = None

    for prompt_index, prompt in enumerate(prompts):
        prompt_id = get_prompt_id(prompt)
        for seed in args.seeds:
            if args.time_budget_seconds > 0 and time.perf_counter() - started >= 0.8 * args.time_budget_seconds:
                break
            origin_dir = output_root / "origin" / prompt_id
            origin_dir.mkdir(parents=True, exist_ok=True)
            origin_path = origin_dir / f"origin_{prompt_id}_seed{seed}.png"
            origin_image, origin_latency, _, _, _ = generate(pipe, args, prompt, seed, None)
            origin_image.save(origin_path)

            for condition in conditions:
                condition_dir = output_root / condition["stage"] / condition["name"] / prompt_id
                condition_dir.mkdir(parents=True, exist_ok=True)
                image_path = condition_dir / f"{condition['name']}_{prompt_id}_seed{seed}.png"
                image, latency_ms, timers, trace, drift = generate(pipe, args, prompt, seed, condition)
                image.save(image_path)
                metrics = evaluate_fidelity(evaluator, image_path, origin_path) if evaluator is not None else {}
                record = {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "warmup_runs": args.warmup_runs,
                    "budget_policy": (
                        "fixed_rho" if args.target_speedup is not None else "curvature_error_tolerance"
                    ),
                    "target_speedup": args.target_speedup,
                    "rho": 1.0 / args.target_speedup if args.target_speedup is not None else None,
                    "condition": condition,
                    "origin_path": str(origin_path),
                    "image_path": str(image_path),
                    "origin_latency_ms": origin_latency,
                    "latency_ms": latency_ms,
                    "normalized_speedup": origin_latency / latency_ms,
                    "quality": metrics,
                    "trace_summary": summarize_trace(trace, drift),
                    "anchor_curvature_drift": drift,
                    "timing_stats": timers,
                }
                records.append(record)
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if first_accelerated_seconds is None:
                    first_accelerated_seconds = latency_ms / 1000.0
                    print(
                        f"TIME_ESTIMATE: {first_accelerated_seconds * total_accelerated:.2f}s "
                        f"for {total_accelerated} accelerated samples",
                        flush=True,
                    )
            summary_path.write_text(
                json.dumps(
                    {
                        "model_path": model_path,
                        "prompts": prompts,
                        "seeds": args.seeds,
                        "budget_policy": (
                            "fixed_rho" if args.target_speedup is not None else "curvature_error_tolerance"
                        ),
                        "target_speedup": args.target_speedup,
                        "rho": 1.0 / args.target_speedup if args.target_speedup is not None else None,
                        "conditions": conditions,
                        "completed_records": len(records),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        else:
            continue
        break
    print(summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
