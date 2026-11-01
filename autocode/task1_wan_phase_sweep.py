#!/usr/bin/env python3
"""Run Wan1.3B Task 1 warmup/cooldown phase calibration sweep."""

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
    parser = argparse.ArgumentParser(description="Wan1.3B phase calibration sweep")
    parser.add_argument("--output_root", default="results/wan/1.3B/task1_phase_sweep_v2")
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--frame_num", type=int, default=17)
    parser.add_argument("--baseline_path", default=None)
    parser.add_argument(
        "--phase_pairs",
        nargs="+",
        default=["1,1", "4,2", "6,2", "8,2", "8,4"],
        help="Warmup,cooldown pairs such as 6,2.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["tc_0p07", "tc_0p10", "cubic_output"],
        choices=["tc_0p07", "tc_0p10", "cubic_output", "cubic_residual"],
    )
    return parser.parse_args()


def parse_phase_pair(item: str) -> tuple[int, int]:
    parts = item.replace("/", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid phase pair: {item}")
    warmup, cooldown = int(parts[0]), int(parts[1])
    if warmup < 0 or cooldown < 0:
        raise ValueError(f"Phase pair must be non-negative: {item}")
    return warmup, cooldown


def condition_spec(name: str) -> dict:
    if name == "tc_0p07":
        return {
            "name": name,
            "method": "teacache",
            "script": "wan_exp/teacache_generate.py",
            "extra": ["--teacache_thresh", "0.07"],
        }
    if name == "tc_0p10":
        return {
            "name": name,
            "method": "teacache",
            "script": "wan_exp/teacache_generate.py",
            "extra": ["--teacache_thresh", "0.10"],
        }
    if name == "cubic_output":
        return {
            "name": name,
            "method": "cubic",
            "script": "wan_exp/cubic_generate.py",
            "extra": ["--enable_visualization", "False", "--cache_reuse_mode", "output"],
        }
    if name == "cubic_residual":
        return {
            "name": name,
            "method": "cubic",
            "script": "wan_exp/cubic_generate.py",
            "extra": ["--enable_visualization", "False", "--cache_reuse_mode", "residual"],
        }
    raise ValueError(f"Unknown condition: {name}")


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


def _condition_master_port(condition_name: str, warmup: int, cooldown: int) -> str | None:
    base = os.environ.get("MASTER_PORT")
    if not base:
        return None
    try:
        base_int = int(base)
    except ValueError:
        return None
    key = f"{condition_name}_{warmup}_{cooldown}"
    offset = 100 + sum(ord(ch) for ch in key) % 2000
    return str(10000 + ((base_int + offset - 10000) % 50000))


def run_condition(spec: dict, warmup: int, cooldown: int, args: argparse.Namespace, output_root: Path) -> dict:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    run_name = f"{spec['name']}_w{warmup}_c{cooldown}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = _base_cmd(spec["script"], args, run_dir)
    if spec["method"] == "teacache":
        cmd.extend(["--warmup_steps", str(warmup), "--cooldown_steps", str(cooldown)])
    elif spec["method"] == "cubic":
        cmd.extend(["--warmup_steps", str(warmup), "--cooldown_steps", str(cooldown)])
    else:
        raise ValueError(f"Unsupported method for phase sweep: {spec['method']}")
    cmd.extend(spec["extra"])

    log_path = run_dir / (f"run_rank{rank}.log" if world_size > 1 else "run.log")
    env = os.environ.copy()
    child_port = _condition_master_port(spec["name"], warmup, cooldown)
    if child_port is not None:
        env["MASTER_PORT"] = child_port

    print(f"[task1-wan-phase][rank{rank}] RUN {run_name}: {' '.join(cmd)}", flush=True)
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
        "name": run_name,
        "condition": spec["name"],
        "method": spec["method"],
        "warmup_steps": warmup,
        "cooldown_steps": cooldown,
        "script": spec["script"],
        "output_dir": str(run_dir),
        "log_path": str(log_path),
        "rank": rank,
        "seconds": float(seconds),
        "returncode": int(proc.returncode),
        "cmd": cmd,
        "master_port": child_port,
    }
    print(f"[task1-wan-phase][rank{rank}] DONE {run_name} returncode={proc.returncode} seconds={seconds:.2f}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Condition failed: {run_name} log={log_path}")
    return result


def main() -> int:
    args = parse_args()
    rank = int(os.getenv("RANK", "0"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    phase_pairs = [parse_phase_pair(item) for item in args.phase_pairs]
    specs = [condition_spec(name) for name in args.conditions]
    runs = [
        {"condition": spec["name"], "method": spec["method"], "warmup_steps": warmup, "cooldown_steps": cooldown}
        for warmup, cooldown in phase_pairs
        for spec in specs
    ]
    summary = {
        "output_root": str(output_root),
        "prompt": args.prompt,
        "base_seed": args.base_seed,
        "sample_steps": args.sample_steps,
        "frame_num": args.frame_num,
        "baseline_path": args.baseline_path,
        "phase_pairs": [{"warmup_steps": w, "cooldown_steps": c} for w, c in phase_pairs],
        "conditions": specs,
        "planned_runs": runs,
        "results": [],
    }
    summary_path = output_root / "phase_sweep_run_summary.json"
    if rank == 0:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    start = time.perf_counter()
    completed = 0
    for warmup, cooldown in phase_pairs:
        for spec in specs:
            result = run_condition(spec, warmup, cooldown, args, output_root)
            completed += 1
            if rank == 0:
                summary["results"].append(result)
                summary["elapsed_seconds"] = time.perf_counter() - start
                summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                if completed == 1:
                    print(f"TIME_ESTIMATE: {result['seconds'] * len(runs):.2f}s", flush=True)
    if rank == 0:
        print(f"[task1-wan-phase] Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
