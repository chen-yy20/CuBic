#!/usr/bin/env python3
"""Controlled Task 1 audit runner for Wan1.3B."""

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
    parser = argparse.ArgumentParser(description="Wan1.3B controlled Task 1 audit")
    parser.add_argument("--output_root", default="results/wan/1.3B/task1_controlled_v3")
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--frame_num", type=int, default=17)
    parser.add_argument("--baseline_path", default="results/wan/1.3B/task1_origin_s20_f17_seed42/ori_couple_formal_evening_wear_t2v-1.3B.mp4")
    parser.add_argument("--stages", nargs="+", default=["phase", "granularity", "frontier"])
    parser.add_argument("--phase_pairs", nargs="+", default=["1,1", "4,2", "6,2", "8,2", "8,4"])
    parser.add_argument("--best_phase", default=None)
    parser.add_argument("--best_grid", default=None)
    parser.add_argument("--grids", nargs="+", default=["3", "4", "5", "6"])
    parser.add_argument("--teacache_thresholds", nargs="+", default=["0.05", "0.07", "0.10", "0.14"])
    parser.add_argument("--pab_self_ranges", nargs="+", default=["2", "3", "4", "5"])
    parser.add_argument("--pab_cross_range", type=int, default=5)
    parser.add_argument("--quick", action="store_true", default=False)
    return parser.parse_args()


def parse_pair(text: str) -> tuple[int, int]:
    parts = text.replace("/", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid phase pair: {text}")
    return int(parts[0]), int(parts[1])


def base_cmd(script: str, args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        script,
        "--prompt", args.prompt,
        "--sample_steps", str(args.sample_steps),
        "--frame_num", str(args.frame_num),
        "--base_seed", str(args.base_seed),
        "--output_dir", str(output_dir),
    ]
    if args.ckpt_dir:
        cmd.extend(["--ckpt_dir", args.ckpt_dir])
    if args.baseline_path:
        cmd.extend(["--baseline_path", args.baseline_path])
    return cmd


def cubic_extra(warmup: int, cooldown: int, extra: list[str] | None = None) -> list[str]:
    args = [
        "--warmup_steps", str(warmup),
        "--cooldown_steps", str(cooldown),
        "--enable_visualization", "False",
        "--cache_reuse_mode", "output",
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
    teacache_thresholds = args.teacache_thresholds[:2] if args.quick else args.teacache_thresholds
    pab_self_ranges = args.pab_self_ranges[:2] if args.quick else args.pab_self_ranges

    if "phase" in stages:
        for warmup, cooldown in phase_pairs:
            conditions.append({
                "stage": "phase",
                "method": "cubic",
                "name": f"phase_cubic_w{warmup}_c{cooldown}",
                "script": "wan_exp/cubic_generate.py",
                "extra": cubic_extra(warmup, cooldown),
            })

    if "granularity" in stages:
        warmup, cooldown = best_phase
        for grid in grids:
            conditions.append({
                "stage": "granularity",
                "method": "cubic",
                "name": f"gran_cubic_grid{grid}x{grid}",
                "script": "wan_exp/cubic_generate.py",
                "extra": cubic_extra(
                    warmup,
                    cooldown,
                    ["--partition_mode", "uniform_grid", "--uniform_grid_h", str(grid), "--uniform_grid_w", str(grid)],
                ),
            })
        conditions.append({
            "stage": "granularity",
            "method": "cubic",
            "name": "gran_cubic_adaptive",
            "script": "wan_exp/cubic_generate.py",
            "extra": cubic_extra(warmup, cooldown, ["--partition_mode", "morphology"]),
        })

    if "frontier" in stages:
        warmup, cooldown = best_phase
        best_grid = args.best_grid or ("4" if "granularity" not in stages else grids[0])
        for threshold in teacache_thresholds:
            conditions.append({
                "stage": "frontier",
                "method": "teacache",
                "name": f"frontier_tc_{threshold.replace('.', 'p')}",
                "script": "wan_exp/teacache_generate.py",
                "extra": ["--warmup_steps", str(warmup), "--cooldown_steps", str(cooldown), "--teacache_thresh", str(threshold)],
            })
        for self_range in pab_self_ranges:
            conditions.append({
                "stage": "frontier",
                "method": "pab",
                "name": f"frontier_pab_s{self_range}_c{args.pab_cross_range}",
                "script": "wan_exp/pab_generate.py",
                "extra": [
                    "--pab_warmup", str(warmup),
                    "--pab_cooldown", str(cooldown),
                    "--pab_self_range", str(self_range),
                    "--pab_cross_range", str(args.pab_cross_range),
                ],
            })
        for grid in [best_grid, str(max(2, int(best_grid) + 1)), str(max(2, int(best_grid) + 2))]:
            conditions.append({
                "stage": "frontier",
                "method": "cubic",
                "name": f"frontier_cubic_grid{grid}x{grid}",
                "script": "wan_exp/cubic_generate.py",
                "extra": cubic_extra(
                    warmup,
                    cooldown,
                    ["--partition_mode", "uniform_grid", "--uniform_grid_h", str(grid), "--uniform_grid_w", str(grid)],
                ),
            })

    return conditions


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


def run_condition(condition: dict, args: argparse.Namespace, output_root: Path) -> dict:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    condition_dir = output_root / condition["stage"] / condition["name"]
    condition_dir.mkdir(parents=True, exist_ok=True)
    cmd = base_cmd(condition["script"], args, condition_dir) + condition["extra"]
    log_path = condition_dir / (f"run_rank{rank}.log" if world_size > 1 else "run.log")
    env = os.environ.copy()
    child_port = condition_master_port(condition["name"])
    if child_port is not None:
        env["MASTER_PORT"] = child_port
    print(f"[task1-controlled-wan][rank{rank}] RUN {condition['stage']}:{condition['name']} {' '.join(cmd)}", flush=True)
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
        **condition,
        "output_dir": str(condition_dir),
        "log_path": str(log_path),
        "rank": rank,
        "seconds": float(seconds),
        "returncode": int(proc.returncode),
        "cmd": cmd,
        "master_port": child_port,
    }
    print(f"[task1-controlled-wan][rank{rank}] DONE {condition['name']} returncode={proc.returncode} seconds={seconds:.2f}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Condition failed: {condition['name']} log={log_path}")
    return result


def main() -> int:
    args = parse_args()
    rank = int(os.getenv("RANK", "0"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions(args)
    summary = {
        "model": "wan",
        "output_root": str(output_root),
        "prompt": args.prompt,
        "base_seed": args.base_seed,
        "sample_steps": args.sample_steps,
        "frame_num": args.frame_num,
        "stages": args.stages,
        "conditions": conditions,
        "results": [],
    }
    summary_path = output_root / "controlled_run_summary.json"
    if rank == 0:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    start = time.perf_counter()
    for idx, condition in enumerate(conditions):
        result = run_condition(condition, args, output_root)
        if rank == 0:
            summary["results"].append(result)
            summary["elapsed_seconds"] = time.perf_counter() - start
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            if idx == 0:
                print(f"TIME_ESTIMATE: {result['seconds'] * len(conditions):.2f}s", flush=True)
    if rank == 0:
        print(f"[task1-controlled-wan] Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
