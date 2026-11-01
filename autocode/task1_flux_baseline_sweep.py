#!/usr/bin/env python3
"""Run Flux1 Task 1 cache-baseline sweep in one allocated job."""

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
    parser = argparse.ArgumentParser(description="Flux1 Task 1 baseline sweep")
    parser.add_argument("--output_root", default="results/flux/task1_baseline_sweep_v1")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--cooldown_steps", type=int, default=1)
    parser.add_argument("--disable_quality_eval", action="store_true", default=False)
    parser.add_argument("--quick", action="store_true", default=False)
    return parser.parse_args()


def _base_cmd(script: str, args: argparse.Namespace, output_dir: Path) -> list[str]:
    prompt_id = "_".join(
        [
            word
            for word in args.prompt.lower().split()
            if word not in {"a", "an", "the", "in", "on", "at", "and", "or", "of", "with", "by"}
        ][:4]
    )
    baseline_dir = REPO_ROOT / "results" / "flux" / "ori" / prompt_id
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
        str(baseline_dir),
    ]
    if args.model_path:
        cmd.extend(["--model_path", args.model_path])
    if args.disable_quality_eval:
        cmd.append("--disable_quality_eval")
    return cmd


def build_conditions(args: argparse.Namespace) -> list[dict]:
    if args.quick:
        return [
            {"method": "teacache", "name": "tc_0p25", "script": "flux1_exp/teacache_generate.py", "extra": ["--rel_l1_thresh", "0.25"]},
            {"method": "pab", "name": "pab_r3", "script": "flux1_exp/pab_generate.py", "extra": ["--pab_skip_range", "3"]},
            {"method": "cubic", "name": "cubic_tol2e-4", "script": "flux1_exp/cubic_generate.py", "extra": ["--cubic_curvature_error_tolerance", "2e-4", "--disable_visualization"]},
        ]
    return [
        {"method": "teacache", "name": "tc_0p25", "script": "flux1_exp/teacache_generate.py", "extra": ["--rel_l1_thresh", "0.25"]},
        {"method": "teacache", "name": "tc_0p40", "script": "flux1_exp/teacache_generate.py", "extra": ["--rel_l1_thresh", "0.4"]},
        {"method": "teacache", "name": "tc_0p60", "script": "flux1_exp/teacache_generate.py", "extra": ["--rel_l1_thresh", "0.6"]},
        {"method": "pab", "name": "pab_r2", "script": "flux1_exp/pab_generate.py", "extra": ["--pab_skip_range", "2"]},
        {"method": "pab", "name": "pab_r3", "script": "flux1_exp/pab_generate.py", "extra": ["--pab_skip_range", "3"]},
        {"method": "pab", "name": "pab_r5", "script": "flux1_exp/pab_generate.py", "extra": ["--pab_skip_range", "5"]},
        {"method": "cubic", "name": "cubic_tol1e-4", "script": "flux1_exp/cubic_generate.py", "extra": ["--cubic_curvature_error_tolerance", "1e-4", "--disable_visualization"]},
        {"method": "cubic", "name": "cubic_tol2e-4", "script": "flux1_exp/cubic_generate.py", "extra": ["--cubic_curvature_error_tolerance", "2e-4", "--disable_visualization"]},
        {"method": "cubic", "name": "cubic_tol4e-4", "script": "flux1_exp/cubic_generate.py", "extra": ["--cubic_curvature_error_tolerance", "4e-4", "--disable_visualization"]},
    ]


def run_condition(condition: dict, args: argparse.Namespace, output_root: Path) -> dict:
    condition_dir = output_root / condition["name"]
    condition_dir.mkdir(parents=True, exist_ok=True)
    cmd = _base_cmd(condition["script"], args, condition_dir)
    if condition["method"] == "cubic":
        cmd.extend(["--seed", str(args.seed), "--num_images", "1"])
        cmd.extend(["--cubic_warmup_steps", str(args.warmup_steps)])
        cmd.extend(["--cubic_cooldown_steps", str(args.cooldown_steps)])
    elif condition["method"] == "teacache":
        cmd.extend(["--seeds", str(args.seed), "--num_images", "1"])
        cmd.extend(["--teacache_warmup_steps", str(args.warmup_steps)])
        cmd.extend(["--teacache_cooldown_steps", str(args.cooldown_steps)])
    else:
        cmd.extend(["--seeds", str(args.seed), "--num_images", "1"])
        cmd.extend(["--pab_warmup_steps", str(args.warmup_steps)])
        cmd.extend(["--pab_cooldown_steps", str(args.cooldown_steps)])
    cmd.extend(condition["extra"])
    if args.disable_quality_eval and "--disable_quality_eval" not in cmd:
        cmd.append("--disable_quality_eval")

    log_path = condition_dir / "run.log"
    start = time.perf_counter()
    print(f"[task1-sweep] RUN {condition['name']}: {' '.join(cmd)}", flush=True)
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
    status = {
        "name": condition["name"],
        "method": condition["method"],
        "script": condition["script"],
        "output_dir": str(condition_dir),
        "log_path": str(log_path),
        "seconds": float(seconds),
        "returncode": int(proc.returncode),
        "cmd": cmd,
    }
    print(
        f"[task1-sweep] DONE {condition['name']} returncode={proc.returncode} seconds={seconds:.2f}",
        flush=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Condition failed: {condition['name']} log={log_path}")
    return status


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions(args)
    summary = {
        "output_root": str(output_root),
        "prompt": args.prompt,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "warmup_steps": args.warmup_steps,
        "cooldown_steps": args.cooldown_steps,
        "quick": bool(args.quick),
        "conditions": conditions,
        "results": [],
    }
    summary_path = output_root / "sweep_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    start = time.perf_counter()
    first_condition_seconds = None
    for idx, condition in enumerate(conditions):
        result = run_condition(condition, args, output_root)
        summary["results"].append(result)
        if idx == 0:
            first_condition_seconds = result["seconds"]
            print(f"TIME_ESTIMATE: {first_condition_seconds * len(conditions):.2f}s", flush=True)
        summary["elapsed_seconds"] = time.perf_counter() - start
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[task1-sweep] Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
