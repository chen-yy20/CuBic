#!/usr/bin/env python3
"""Wan1.3B Task 3 cache-mode cross-validation.

This script is intended to be launched once through launch.py with 2 GPUs. It
loads Wan once per rank, then loops over cache reuse modes in-process.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
WAN_EXP_ROOT = REPO_ROOT / "wan_exp"
for path in (str(REPO_ROOT), str(WAN_EXP_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
import torch.distributed as dist

from dist.parallel_state import init_cp_group, init_distributed_environment
from utils.launch_defaults import resolve_model_path
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import get_time_statistics_dict, init_timer, print_time_statistics, save_time_stats_to_file
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_video, str2bool
from wan_cubic.cubic_text2video import CubicWanT2V


init_timer()

PROMPT = "A couple in formal evening wear going home get caught in a heavy downpour with umbrellas, zoom in"
TAG = "cubic"
SPEEDUP = 4
WARMUP = 9
COOLDOWN = 1
ALPHA = 2.0
BETA = 4.0
REQUIRED_TIMER_NAMES = [
    "generate_e2e",
    "anchor_step_monitor",
    "anchor_step_partitioner",
    "anchor_step_optimizer",
    "anchor_step_total",
    "selective_forward",
    "warmup_cooldown_step",
]


def _resolve_model_defaults(model_path: str) -> tuple[str, str, str]:
    if "1.3B" in model_path:
        return "t2v-1.3B", "832*480", "1.3B"
    return "t2v-14B", "1280*720", "14B"


def _default_origin_reference_path(model_path: str, prompt: str, task: str) -> str:
    _, _, model_id = _resolve_model_defaults(model_path)
    prompt_id = get_prompt_id(prompt)
    return os.path.join(
        ".",
        "results",
        "wan",
        model_id,
        "ori",
        prompt_id,
        f"ori_{prompt_id}_{task}.mp4",
    )


rank = int(os.getenv("RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
local_rank = int(os.getenv("LOCAL_RANK", 0))
if world_size > 1:
    assert world_size == 2, "only support cfg parallel"
    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=local_rank)
    init_cp_group(group_ranks=[[0, 1]], local_rank=local_rank, backend="nccl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan Task 3 cache-mode comparison")
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--output_root", default="results/wan/task3_v5_wan_cache_compare")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--offload_model", type=str2bool, default=False)
    parser.add_argument("--enable_visualization", type=str2bool, default=False)
    parser.add_argument("--straightness_eval", type=str2bool, default=False)
    parser.add_argument("--probe_max_steps", type=int, default=8)
    parser.add_argument(
        "--cache_modes",
        nargs="+",
        default=["residual", "output", "kv_only"],
        choices=["residual", "output", "kv_only"],
    )
    args = parser.parse_args()
    args.ckpt_dir = resolve_model_path(args.ckpt_dir, "--ckpt_dir")
    args.task, args.size, args.model_id = _resolve_model_defaults(args.ckpt_dir)
    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"
    assert args.size in SUPPORTED_SIZES[args.task], f"Unsupported size {args.size}"
    return args


def init_logging(cur_rank: int) -> None:
    if cur_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def build_config(args: argparse.Namespace, mode: str):
    cfg = copy.deepcopy(WAN_CONFIGS[args.task])
    total_steps = max(int(args.sample_steps), 1)
    cfg.target_speedup = float(SPEEDUP)
    cfg.warmup_fraction = max(0.0, min(1.0, float(WARMUP) / float(total_steps)))
    cfg.cooldown_fraction = max(0.0, min(1.0, float(COOLDOWN) / float(total_steps)))
    cfg.alpha = float(ALPHA)
    cfg.beta = float(BETA)
    cfg.cache_reuse_mode = str(mode)
    return cfg


def build_step_metrics(step_trace):
    trace = [dict(item) for item in step_trace]
    return {
        "cubic_step_trace": {
            "steps": trace,
            "text_lines": [item["text"] for item in trace],
        }
    }


def merge_quality_and_step_metrics(quality_metrics, step_trace):
    merged = dict(quality_metrics) if isinstance(quality_metrics, dict) else {}
    merged.update(build_step_metrics(step_trace))
    return merged


def active_ratio_summary(step_trace: list[dict]) -> dict:
    if not step_trace:
        return {}
    selective = [row for row in step_trace if "selective" in str(row.get("phase", ""))]
    rows = selective or step_trace
    return {
        "mean_active_ratio": float(sum(float(row.get("active_ratio", 0.0)) for row in rows) / max(len(rows), 1)),
        "mean_latency_ms": float(sum(float(row.get("latency_ms", 0.0)) for row in rows) / max(len(rows), 1)),
        "selective_steps": len(selective),
        "total_steps": len(step_trace),
    }


def main() -> int:
    args = parse_args()
    cur_rank = int(os.getenv("RANK", 0))
    cur_local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = cur_local_rank
    init_logging(cur_rank)

    if world_size == 1:
        raise RuntimeError("This Wan comparison is intended for 2-GPU CFG parallel.")

    if dist.is_initialized():
        seed_payload = [args.base_seed] if cur_rank == 0 else [None]
        dist.broadcast_object_list(seed_payload, src=0)
        args.base_seed = int(seed_payload[0])

    output_root = Path(args.output_root)
    if cur_rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)

    initial_mode = args.cache_modes[0]
    initial_cfg = build_config(args, initial_mode)
    logging.info("Creating CubicWanT2V pipeline once; initial mode=%s", initial_mode)
    wan_t2v = CubicWanT2V(
        config=initial_cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=cur_rank,
        t5_fsdp=False,
        dit_fsdp=False,
        t5_cpu=False,
    )

    summaries = []
    for mode in args.cache_modes:
        if dist.is_initialized():
            dist.barrier()
        mode_start = time.perf_counter()
        cfg = build_config(args, mode)
        mode_dir = output_root / mode
        if cur_rank == 0:
            mode_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Running cache reuse mode=%s", mode)
        wan_t2v.config = cfg
        wan_t2v.phase_warmup_fraction = float(getattr(cfg, "warmup_fraction", 0.15))
        wan_t2v.phase_cooldown_fraction = float(getattr(cfg, "cooldown_fraction", 0.15))
        wan_t2v.frequency_optimizer.alpha = float(getattr(cfg, "alpha", ALPHA))
        wan_t2v.frequency_optimizer.beta = float(getattr(cfg, "beta", BETA))
        wan_t2v.frequency_optimizer.target_speedup = float(getattr(cfg, "target_speedup", SPEEDUP))
        wan_t2v.selective_forward.config = cfg
        wan_t2v.selective_forward.cache_reuse_mode = str(mode)
        probe_path = mode_dir / f"{TAG}_{get_prompt_id(args.prompt)}_{args.task}_error_probe_seed{args.base_seed}.jsonl"
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            results_dir=str(mode_dir),
            eval_straightness=args.straightness_eval,
            enable_visualization=args.enable_visualization,
            error_probe=True,
            error_probe_max_steps=args.probe_max_steps,
            error_probe_output_path=str(probe_path),
        )
        step_trace = wan_t2v.get_step_trace()
        elapsed = time.perf_counter() - mode_start
        if cur_rank == 0:
            video_path = mode_dir / f"{TAG}_{get_prompt_id(args.prompt)}_{args.task}.mp4"
            cache_video(
                tensor=video[None],
                save_file=str(video_path),
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            baseline_path = _default_origin_reference_path(args.ckpt_dir, args.prompt, args.task)
            quality_result = evaluate_quality_with_origin(
                image_path=str(video_path),
                tag=TAG,
                save_metrics=False,
                baseline_path=baseline_path,
                output_dir=None,
            )
            quality_metrics = quality_result["metrics"] if quality_result is not None else None
            save_time_stats_to_file(str(mode_dir / "timing_stats.txt"), timer_names=REQUIRED_TIMER_NAMES)
            params = {
                "model": "wan",
                "method": "cubic",
                "model_id": args.model_id,
                "model_path": args.ckpt_dir,
                "prompt": args.prompt,
                "task": args.task,
                "size": args.size,
                "base_seed": args.base_seed,
                "cache_reuse_mode": mode,
                "error_probe_enabled": True,
                "error_probe_max_steps": int(args.probe_max_steps),
                "error_probe_output_path": str(probe_path),
                "alpha": ALPHA,
                "beta": BETA,
                "target_speedup": SPEEDUP,
            }
            params_path = save_params_and_metrics(
                str(mode_dir),
                TAG,
                params,
                get_time_statistics_dict(timer_names=REQUIRED_TIMER_NAMES),
                quality_metrics=merge_quality_and_step_metrics(quality_metrics, step_trace),
            )
            probe_rows = 0
            if probe_path.exists():
                probe_rows = sum(1 for line in probe_path.read_text(encoding="utf-8").splitlines() if line.strip())
            summaries.append({
                "mode": mode,
                "seconds": float(elapsed),
                "output_path": str(video_path),
                "params_path": str(params_path),
                "probe_path": str(probe_path),
                "probe_rows": int(probe_rows),
                "quality_metrics": quality_metrics,
                **active_ratio_summary(step_trace),
            })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()

    del wan_t2v
    if cur_rank == 0:
        summary_path = output_root / "cache_compare_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "cache_modes": args.cache_modes,
                    "sample_steps": args.sample_steps,
                    "frame_num": args.frame_num,
                    "seed": args.base_seed,
                    "summaries": summaries,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logging.info("Wrote %s", summary_path)
        print_time_statistics(timer_names=REQUIRED_TIMER_NAMES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
