#!/usr/bin/env python3
"""Run Wan1.3B phase-calibrated CuBic cache granularity sweep."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT = "A couple in formal evening wear going home get caught in a heavy downpour with umbrellas, zoom in"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan1.3B CuBic granularity sweep")
    parser.add_argument("--output_root", default="results/wan/1.3B/task1_granularity_sweep_v2")
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--frame_num", type=int, default=17)
    parser.add_argument("--warmup_steps", type=int, default=8)
    parser.add_argument("--cooldown_steps", type=int, default=4)
    parser.add_argument("--baseline_path", default=None)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "morph_default",
            "morph_fine",
            "morph_coarse",
            "uniform_default",
            "uniform_fine",
        ],
        choices=[
            "morph_default",
            "morph_fine",
            "morph_coarse",
            "uniform_default",
            "uniform_fine",
            "grid_7x7",
            "grid_9x9",
            "grid_14x15",
        ],
    )
    return parser.parse_args()


def condition_extra(name: str) -> list[str]:
    base = ["--enable_visualization", "False", "--cache_reuse_mode", "output"]
    if name == "morph_default":
        return base
    if name == "morph_fine":
        return base + [
            "--partition_mode", "morphology",
            "--max_block_size", "8",
            "--foreground_focus_block_size", "2",
            "--cv_threshold", "0.35",
        ]
    if name == "morph_coarse":
        return base + [
            "--partition_mode", "morphology",
            "--max_block_size", "16",
            "--foreground_focus_block_size", "8",
            "--cv_threshold", "0.80",
        ]
    if name == "uniform_default":
        return base + [
            "--partition_mode", "uniform",
            "--min_block_size", "2",
            "--max_block_size", "16",
            "--cv_threshold", "0.50",
        ]
    if name == "uniform_fine":
        return base + [
            "--partition_mode", "uniform",
            "--min_block_size", "2",
            "--max_block_size", "8",
            "--cv_threshold", "0.35",
        ]
    if name == "grid_7x7":
        return base + [
            "--partition_mode", "uniform_grid",
            "--uniform_grid_h", "7",
            "--uniform_grid_w", "7",
        ]
    if name == "grid_9x9":
        return base + [
            "--partition_mode", "uniform_grid",
            "--uniform_grid_h", "9",
            "--uniform_grid_w", "9",
        ]
    if name == "grid_14x15":
        return base + [
            "--partition_mode", "uniform_grid",
            "--uniform_grid_h", "14",
            "--uniform_grid_w", "15",
        ]
    raise ValueError(f"Unknown condition: {name}")


def base_cmd(args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "wan_exp/cubic_generate.py",
        "--prompt",
        args.prompt,
        "--sample_steps",
        str(args.sample_steps),
        "--frame_num",
        str(args.frame_num),
        "--base_seed",
        str(args.base_seed),
        "--output_dir",
        str(output_dir),
        "--warmup_steps",
        str(args.warmup_steps),
        "--cooldown_steps",
        str(args.cooldown_steps),
    ]
    if args.ckpt_dir:
        cmd.extend(["--ckpt_dir", args.ckpt_dir])
    if args.baseline_path:
        cmd.extend(["--baseline_path", args.baseline_path])
    return cmd


def condition_master_port(condition_name: str) -> str | None:
    base = os.environ.get("MASTER_PORT")
    if not base:
        return None
    try:
        base_int = int(base)
    except ValueError:
        return None
    offset = 100 + sum(ord(ch) for ch in condition_name) % 2000
    return str(10000 + ((base_int + offset - 10000) % 50000))


def run_condition(name: str, args: argparse.Namespace, output_root: Path) -> dict:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    condition_dir = output_root / name
    condition_dir.mkdir(parents=True, exist_ok=True)
    cmd = base_cmd(args, condition_dir) + condition_extra(name)
    log_path = condition_dir / (f"run_rank{rank}.log" if world_size > 1 else "run.log")
    env = os.environ.copy()
    child_port = condition_master_port(name)
    if child_port is not None:
        env["MASTER_PORT"] = child_port
    print(f"[task1-wan-granularity][rank{rank}] RUN {name}: {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    seconds = time.perf_counter() - start
    result = {
        "name": name,
        "output_dir": str(condition_dir),
        "log_path": str(log_path),
        "rank": rank,
        "seconds": float(seconds),
        "returncode": int(proc.returncode),
        "cmd": cmd,
        "master_port": child_port,
    }
    print(f"[task1-wan-granularity][rank{rank}] DONE {name} returncode={proc.returncode} seconds={seconds:.2f}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Condition failed: {name} log={log_path}")
    return result


def main() -> int:
    args = parse_args()
    rank = int(os.getenv("RANK", "0"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "output_root": str(output_root),
        "prompt": args.prompt,
        "base_seed": args.base_seed,
        "sample_steps": args.sample_steps,
        "frame_num": args.frame_num,
        "warmup_steps": args.warmup_steps,
        "cooldown_steps": args.cooldown_steps,
        "baseline_path": args.baseline_path,
        "conditions": args.conditions,
        "results": [],
    }
    summary_path = output_root / "granularity_sweep_run_summary.json"
    if rank == 0:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    start = time.perf_counter()
    for idx, name in enumerate(args.conditions):
        result = run_condition(name, args, output_root)
        if rank == 0:
            summary["results"].append(result)
            summary["elapsed_seconds"] = time.perf_counter() - start
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            if idx == 0:
                print(f"TIME_ESTIMATE: {result['seconds'] * len(args.conditions):.2f}s", flush=True)
    if rank == 0:
        print(f"[task1-wan-granularity] Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
