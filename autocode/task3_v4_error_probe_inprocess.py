#!/usr/bin/env python3
"""Task 3 v4 in-process Flux1 error-probe breakdown.

Loads Flux1 CuBic once, runs a small seed set, and records block-level
selective-vs-full-oracle errors for theory-facing power-law analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FLUX1_ROOT = REPO_ROOT / "flux1_exp"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FLUX1_ROOT))

from flux1_exp.cubic_generate import (  # noqa: E402
    GUIDANCE_SCALE,
    HEIGHT,
    NUM_INFERENCE_STEPS,
    PROMPT,
    TAG,
    WIDTH,
    _merge_quality_and_step_metrics,
    _print_step_latency_trace,
    load_pipeline,
)
from utils.launch_defaults import resolve_model_path  # noqa: E402
from utils.quality_metric import evaluate_quality_with_origin  # noqa: E402
from utils.results import get_prompt_id, save_params_and_metrics  # noqa: E402
from utils.timer import get_time_statistics_dict, init_timer, print_time_statistics, save_time_stats_to_file  # noqa: E402


SEEDS = [42, 43, 44]
REQUIRED_TIMER_NAMES = [
    "generate_e2e",
    "anchor_step_monitor",
    "anchor_step_partitioner",
    "anchor_step_optimizer",
    "anchor_step_total",
    "warmup_cooldown_step",
    "selective_forward",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 3 v4 Flux1 error-probe breakdown")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--output_root", type=str, default="results/flux/task3_v4_error_probe")
    parser.add_argument("--baseline_dir", type=str, default=None)
    parser.add_argument("--probe_max_steps", type=int, default=12)
    parser.add_argument("--time_budget_seconds", type=float, default=7200.0)
    parser.add_argument("--seed_limit", type=int, default=0)
    parser.add_argument("--disable_quality_eval", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args.model_path, "--model_path")
    model_id = os.path.basename(model_path.rstrip("/"))
    prompt_id = get_prompt_id(args.prompt)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else REPO_ROOT / "results" / "flux" / "ori" / prompt_id
    seeds = SEEDS[: args.seed_limit] if args.seed_limit > 0 else list(SEEDS)

    run_start = time.perf_counter()
    summaries = []
    print(
        f"ERROR_PROBE_INPROCESS_START: seeds={seeds} model_path={model_path} output_root={output_root}",
        flush=True,
    )
    init_timer()
    pipe = load_pipeline(model_path)
    print(f"Model loaded once, GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB", flush=True)

    for index, seed in enumerate(seeds, start=1):
        elapsed = time.perf_counter() - run_start
        if args.time_budget_seconds > 0 and elapsed >= 0.8 * float(args.time_budget_seconds):
            print(f"TIME_GUARD: stopping before seed={seed} elapsed={elapsed:.2f}s", flush=True)
            break

        output_dir = output_root / f"seed{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        probe_path = output_dir / f"{TAG}_error_probe_seed{seed}.jsonl"
        seed_start = time.perf_counter()
        print(f"ERROR_PROBE_SEED_START: seed={seed} ({index}/{len(seeds)})", flush=True)
        init_timer()
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
                cubic_error_probe=True,
                cubic_error_probe_max_steps=args.probe_max_steps,
                cubic_error_probe_output_path=str(probe_path),
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

        print_time_statistics(timer_names=REQUIRED_TIMER_NAMES)
        save_time_stats_to_file(str(output_dir / "timing_stats.txt"), timer_names=REQUIRED_TIMER_NAMES)
        probe_rows = sum(1 for _ in probe_path.open("r", encoding="utf-8")) if probe_path.exists() else 0
        seed_seconds = time.perf_counter() - seed_start
        if index == 1:
            print(f"TIME_ESTIMATE: {seed_seconds * len(seeds):.2f}s", flush=True)

        params = {
            "model": "flux",
            "method": TAG,
            "model_id": model_id,
            "model_path": model_path,
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": seed,
            "alpha": 3.0,
            "beta": 4.0,
            "curvature_error_tolerance": 2e-4,
            "error_probe_enabled": True,
            "error_probe_max_steps": int(args.probe_max_steps),
            "error_probe_path": str(probe_path),
            "error_probe_rows": int(probe_rows),
            "seed_seconds": seed_seconds,
            "quality_eval": not args.disable_quality_eval,
            "baseline_dir": str(baseline_dir),
        }
        params_path = save_params_and_metrics(
            str(output_dir),
            TAG,
            params,
            get_time_statistics_dict(timer_names=REQUIRED_TIMER_NAMES),
            quality_metrics=_merge_quality_and_step_metrics(
                {"per_seed": per_seed_quality, "mean": next(iter(per_seed_quality.values()), None)}
                if not args.disable_quality_eval
                else None,
                step_trace,
                per_seed_trace={f"seed{seed}": step_trace},
            ),
        )
        summary = {
            "seed": seed,
            "seconds": seed_seconds,
            "output_path": str(output_path),
            "params_path": params_path,
            "probe_path": str(probe_path),
            "probe_rows": int(probe_rows),
        }
        summaries.append(summary)
        print(f"ERROR_PROBE_SEED_DONE: seed={seed} seconds={seed_seconds:.2f} rows={probe_rows}", flush=True)

    summary_path = output_root / "error_probe_run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "completed": len(summaries),
                "planned": len(seeds),
                "elapsed_seconds": time.perf_counter() - run_start,
                "seeds": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"ERROR_PROBE_SUMMARY: completed={len(summaries)}/{len(seeds)} path={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
