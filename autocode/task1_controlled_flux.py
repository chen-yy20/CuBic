#!/usr/bin/env python3
"""Controlled Task 1 audit runner for Flux1."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux1 controlled Task 1 audit")
    parser.add_argument("--output_root", default="results/flux/task1_controlled_v3")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--baseline_dir", default=None)
    parser.add_argument("--stages", nargs="+", default=["phase", "granularity", "frontier"])
    parser.add_argument("--phase_pairs", nargs="+", default=["1,1", "5,3", "8,4", "10,5", "12,6"])
    parser.add_argument("--best_phase", default=None, help="Override warmup,cooldown for post-phase stages.")
    parser.add_argument("--grids", nargs="+", default=["3", "4", "5", "6"])
    parser.add_argument("--cubic_tolerances", nargs="+", default=["1e-4", "2e-4", "4e-4", "8e-4"])
    parser.add_argument("--teacache_thresholds", nargs="+", default=["0.15", "0.25", "0.40", "0.60"])
    parser.add_argument("--pab_ranges", nargs="+", default=["2", "3", "5", "8"])
    parser.add_argument("--quick", action="store_true", default=False)
    return parser.parse_args()


def parse_pair(text: str) -> tuple[int, int]:
    parts = text.replace("/", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid phase pair: {text}")
    return int(parts[0]), int(parts[1])


def prompt_id(prompt: str) -> str:
    words = [
        word
        for word in prompt.lower().split()
        if word not in {"a", "an", "the", "in", "on", "at", "and", "or", "of", "with", "by"}
    ][:4]
    return "_".join(words)


def base_cmd(script: str, args: argparse.Namespace, output_dir: Path) -> list[str]:
    baseline_dir = args.baseline_dir or str(REPO_ROOT / "results" / "flux" / "ori" / prompt_id(args.prompt))
    cmd = [
        sys.executable,
        script,
        "--prompt",
        args.prompt,
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--guidance_scale",
        str(args.guidance_scale),
        "--output_dir",
        str(output_dir),
        "--baseline_dir",
        baseline_dir,
    ]
    if args.model_path:
        cmd.extend(["--model_path", args.model_path])
    return cmd


def cubic_extra(warmup: int, cooldown: int, tolerance: str, extra: list[str] | None = None) -> list[str]:
    args = [
        "--seed", "42",
        "--num_images", "1",
        "--cubic_warmup_steps", str(warmup),
        "--cubic_cooldown_steps", str(cooldown),
        "--cubic_curvature_error_tolerance", str(tolerance),
        "--disable_visualization",
    ]
    if extra:
        args.extend(extra)
    return args


def build_conditions(args: argparse.Namespace) -> list[dict]:
    stages = set(args.stages)
    conditions: list[dict] = []
    phase_pairs = [parse_pair(item) for item in (args.phase_pairs[:2] if args.quick else args.phase_pairs)]
    best_phase = parse_pair(args.best_phase) if args.best_phase else phase_pairs[-1]
    grids = args.grids[:2] if args.quick else args.grids
    cubic_tolerances = args.cubic_tolerances[:2] if args.quick else args.cubic_tolerances
    teacache_thresholds = args.teacache_thresholds[:2] if args.quick else args.teacache_thresholds
    pab_ranges = args.pab_ranges[:2] if args.quick else args.pab_ranges

    if "phase" in stages:
        for warmup, cooldown in phase_pairs:
            conditions.append({
                "stage": "phase",
                "method": "cubic",
                "name": f"phase_cubic_w{warmup}_c{cooldown}",
                "script": "flux1_exp/cubic_generate.py",
                "extra": cubic_extra(warmup, cooldown, "2e-4"),
            })

    if "granularity" in stages:
        warmup, cooldown = best_phase
        for grid in grids:
            conditions.append({
                "stage": "granularity",
                "method": "cubic",
                "name": f"gran_cubic_grid{grid}x{grid}",
                "script": "flux1_exp/cubic_generate.py",
                "extra": cubic_extra(
                    warmup,
                    cooldown,
                    "2e-4",
                    [
                        "--cubic_partition_mode", "uniform_grid",
                        "--cubic_uniform_grid_h", str(grid),
                        "--cubic_uniform_grid_w", str(grid),
                        "--cubic_uniform_common_factor_blocks", "false",
                    ],
                ),
            })
        conditions.append({
            "stage": "granularity",
            "method": "cubic",
            "name": "gran_cubic_adaptive",
            "script": "flux1_exp/cubic_generate.py",
            "extra": cubic_extra(
                warmup,
                cooldown,
                "2e-4",
                ["--cubic_partition_mode", "morphology", "--cubic_uniform_common_factor_blocks", "false"],
            ),
        })

    if "frontier" in stages:
        warmup, cooldown = best_phase
        best_grid = grids[0]
        if "granularity" not in stages and args.best_phase:
            best_grid = "4"
        for threshold in teacache_thresholds:
            conditions.append({
                "stage": "frontier",
                "method": "teacache",
                "name": f"frontier_tc_{threshold.replace('.', 'p')}",
                "script": "flux1_exp/teacache_generate.py",
                "extra": [
                    "--seeds", str(args.seed),
                    "--num_images", "1",
                    "--teacache_warmup_steps", str(warmup),
                    "--teacache_cooldown_steps", str(cooldown),
                    "--rel_l1_thresh", str(threshold),
                ],
            })
        for skip_range in pab_ranges:
            conditions.append({
                "stage": "frontier",
                "method": "pab",
                "name": f"frontier_pab_r{skip_range}",
                "script": "flux1_exp/pab_generate.py",
                "extra": [
                    "--seeds", str(args.seed),
                    "--num_images", "1",
                    "--pab_warmup_steps", str(warmup),
                    "--pab_cooldown_steps", str(cooldown),
                    "--pab_skip_range", str(skip_range),
                ],
            })
        for tolerance in cubic_tolerances:
            safe_tol = tolerance.replace("-", "m").replace(".", "p")
            conditions.append({
                "stage": "frontier",
                "method": "cubic",
                "name": f"frontier_cubic_grid{best_grid}_tol{safe_tol}",
                "script": "flux1_exp/cubic_generate.py",
                "extra": cubic_extra(
                    warmup,
                    cooldown,
                    tolerance,
                    [
                        "--cubic_partition_mode", "uniform_grid",
                        "--cubic_uniform_grid_h", str(best_grid),
                        "--cubic_uniform_grid_w", str(best_grid),
                        "--cubic_uniform_common_factor_blocks", "false",
                    ],
                ),
            })

    return conditions


def run_condition(condition: dict, args: argparse.Namespace, output_root: Path) -> dict:
    condition_dir = output_root / condition["stage"] / condition["name"]
    condition_dir.mkdir(parents=True, exist_ok=True)
    cmd = base_cmd(condition["script"], args, condition_dir) + condition["extra"]
    log_path = condition_dir / "run.log"
    print(f"[task1-controlled-flux] RUN {condition['stage']}:{condition['name']} {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    seconds = time.perf_counter() - start
    result = {
        **condition,
        "output_dir": str(condition_dir),
        "log_path": str(log_path),
        "seconds": float(seconds),
        "returncode": int(proc.returncode),
        "cmd": cmd,
    }
    print(
        f"[task1-controlled-flux] DONE {condition['name']} returncode={proc.returncode} seconds={seconds:.2f}",
        flush=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Condition failed: {condition['name']} log={log_path}")
    return result


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions(args)
    summary = {
        "model": "flux",
        "output_root": str(output_root),
        "prompt": args.prompt,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "stages": args.stages,
        "conditions": conditions,
        "results": [],
    }
    summary_path = output_root / "controlled_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    start = time.perf_counter()
    for idx, condition in enumerate(conditions):
        result = run_condition(condition, args, output_root)
        summary["results"].append(result)
        summary["elapsed_seconds"] = time.perf_counter() - start
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if idx == 0:
            print(f"TIME_ESTIMATE: {result['seconds'] * len(conditions):.2f}s", flush=True)
    print(f"[task1-controlled-flux] Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
