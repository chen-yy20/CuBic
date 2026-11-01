#!/usr/bin/env python3
"""Run Wan1.3B Task 1 matched-phase cache-baseline sweep."""

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
    parser = argparse.ArgumentParser(description="Wan1.3B Task 1 matched baseline sweep")
    parser.add_argument("--output_root", default="results/wan/1.3B/task1_matched_sweep_v1")
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--frame_num", type=int, default=17)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--cooldown_steps", type=int, default=1)
    parser.add_argument("--baseline_path", default=None)
    parser.add_argument("--only_conditions", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true", default=False)
    return parser.parse_args()


def _base_cmd(script: str, args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        script,
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
    ]
    if args.ckpt_dir:
        cmd.extend(["--ckpt_dir", args.ckpt_dir])
    if args.baseline_path:
        cmd.extend(["--baseline_path", args.baseline_path])
    return cmd


def build_conditions(args: argparse.Namespace) -> list[dict]:
    if args.quick:
        return [
            {"method": "teacache", "name": "tc_0p07", "script": "wan_exp/teacache_generate.py", "extra": ["--teacache_thresh", "0.07"]},
            {"method": "pab", "name": "pab_s2_c5", "script": "wan_exp/pab_generate.py", "extra": ["--pab_self_range", "2", "--pab_cross_range", "5"]},
            {"method": "cubic", "name": "cubic_default", "script": "wan_exp/cubic_generate.py", "extra": ["--enable_visualization", "False"]},
        ]
    conditions = [
        {"method": "teacache", "name": "tc_0p05", "script": "wan_exp/teacache_generate.py", "extra": ["--teacache_thresh", "0.05"]},
        {"method": "teacache", "name": "tc_0p07", "script": "wan_exp/teacache_generate.py", "extra": ["--teacache_thresh", "0.07"]},
        {"method": "teacache", "name": "tc_0p10", "script": "wan_exp/teacache_generate.py", "extra": ["--teacache_thresh", "0.10"]},
        {"method": "pab", "name": "pab_s2_c5", "script": "wan_exp/pab_generate.py", "extra": ["--pab_self_range", "2", "--pab_cross_range", "5"]},
        {"method": "pab", "name": "pab_s3_c5", "script": "wan_exp/pab_generate.py", "extra": ["--pab_self_range", "3", "--pab_cross_range", "5"]},
        {"method": "cubic", "name": "cubic_default", "script": "wan_exp/cubic_generate.py", "extra": ["--enable_visualization", "False"]},
    ]
    if args.only_conditions:
        wanted = set(args.only_conditions)
        conditions = [condition for condition in conditions if condition["name"] in wanted]
    return conditions


def _condition_master_port(condition_name: str) -> str | None:
    base = os.environ.get("MASTER_PORT")
    if not base:
        return None
    try:
        base_int = int(base)
    except ValueError:
        return None
    offset = 100 + sum(ord(ch) for ch in condition_name) % 2000
    return str(10000 + ((base_int + offset - 10000) % 50000))


def run_condition(condition: dict, args: argparse.Namespace, output_root: Path) -> dict:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    condition_dir = output_root / condition["name"]
    condition_dir.mkdir(parents=True, exist_ok=True)
    cmd = _base_cmd(condition["script"], args, condition_dir)
    if condition["method"] == "teacache":
        cmd.extend(["--warmup_steps", str(args.warmup_steps), "--cooldown_steps", str(args.cooldown_steps)])
    elif condition["method"] == "pab":
        cmd.extend(["--pab_warmup", str(args.warmup_steps), "--pab_cooldown", str(args.cooldown_steps)])
    elif condition["method"] == "cubic":
        cmd.extend(["--warmup_steps", str(args.warmup_steps), "--cooldown_steps", str(args.cooldown_steps)])
    cmd.extend(condition["extra"])

    log_path = condition_dir / (f"run_rank{rank}.log" if world_size > 1 else "run.log")
    start = time.perf_counter()
    print(f"[task1-wan-sweep][rank{rank}] RUN {condition['name']}: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    child_port = _condition_master_port(condition["name"])
    if child_port is not None:
        env["MASTER_PORT"] = child_port
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
    status = {
        "name": condition["name"],
        "method": condition["method"],
        "script": condition["script"],
        "output_dir": str(condition_dir),
        "log_path": str(log_path),
        "rank": rank,
        "seconds": float(seconds),
        "returncode": int(proc.returncode),
        "cmd": cmd,
        "master_port": child_port,
    }
    print(f"[task1-wan-sweep][rank{rank}] DONE {condition['name']} returncode={proc.returncode} seconds={seconds:.2f}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Condition failed: {condition['name']} log={log_path}")
    return status


def main() -> int:
    args = parse_args()
    rank = int(os.getenv("RANK", "0"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions(args)
    summary = {
        "output_root": str(output_root),
        "prompt": args.prompt,
        "base_seed": args.base_seed,
        "sample_steps": args.sample_steps,
        "frame_num": args.frame_num,
        "warmup_steps": args.warmup_steps,
        "cooldown_steps": args.cooldown_steps,
        "baseline_path": args.baseline_path,
        "quick": bool(args.quick),
        "conditions": conditions,
        "results": [],
    }
    summary_path = output_root / "sweep_run_summary.json"
    if rank == 0:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    start = time.perf_counter()
    for idx, condition in enumerate(conditions):
        result = run_condition(condition, args, output_root)
        if rank == 0:
            summary["results"].append(result)
        if idx == 0 and rank == 0:
            print(f"TIME_ESTIMATE: {result['seconds'] * len(conditions):.2f}s", flush=True)
        if rank == 0:
            summary["elapsed_seconds"] = time.perf_counter() - start
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if rank == 0:
        print(f"[task1-wan-sweep] Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
