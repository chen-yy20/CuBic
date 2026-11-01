#!/usr/bin/env python3
"""Run Wan1.3B cache-mode probe in one model load."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "wan_exp"))

from dist.parallel_state import init_cp_group, init_distributed_environment  # noqa: E402
from utils.launch_defaults import resolve_model_path  # noqa: E402
from utils.results import get_prompt_id  # noqa: E402
from utils.timer import init_timer  # noqa: E402
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.utils.utils import cache_video  # noqa: E402
from wan_cubic.cubic_text2video import CubicWanT2V  # noqa: E402


PROMPT = "A couple in formal evening wear going home get caught in a heavy downpour with umbrellas, zoom in"
MODES = ("residual", "output", "kv_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan1.3B Task 3 cache-mode probe")
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--output_root", default="results/wan/task3_v5_wan_cache_mode_probe")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame_num", type=int, default=17)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--probe_max_steps", type=int, default=4)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        assert world_size == 2, "WAN cache-mode probe only supports 2-GPU CFG parallel"
        torch.cuda.set_device(local_rank)
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend="nccl",
        )
        init_cp_group(group_ranks=[[0, 1]], local_rank=local_rank, backend="nccl")
    return rank, world_size, local_rank


def setup_logging(rank: int) -> None:
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def main() -> int:
    init_timer()
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    setup_logging(rank)
    device = local_rank

    ckpt_dir = resolve_model_path(args.ckpt_dir, "--ckpt_dir")
    task = "t2v-1.3B"
    size_key = "832*480"
    cfg = WAN_CONFIGS[task]
    total_steps = max(int(args.sample_steps), 1)
    cfg.target_speedup = 4.0
    cfg.warmup_fraction = max(0.0, min(1.0, 9.0 / float(total_steps)))
    cfg.cooldown_fraction = max(0.0, min(1.0, 1.0 / float(total_steps)))
    cfg.alpha = 2.0
    cfg.beta = 4.0

    if dist.is_initialized():
        seed_holder = [args.seed] if rank == 0 else [None]
        dist.broadcast_object_list(seed_holder, src=0)
        args.seed = int(seed_holder[0])

    logging.info("Loading Wan1.3B once for cache-mode probe.")
    wan_t2v = CubicWanT2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=False,
        t5_cpu=False,
    )

    output_root = Path(args.output_root)
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": "wan",
        "model_id": "1.3B",
        "task": task,
        "prompt": args.prompt,
        "seed": args.seed,
        "frame_num": args.frame_num,
        "sample_steps": args.sample_steps,
        "probe_max_steps": args.probe_max_steps,
        "modes": [],
    }

    for mode in args.modes:
        cfg.cache_reuse_mode = mode
        mode_dir = output_root / mode
        probe_path = mode_dir / f"cubic_{get_prompt_id(args.prompt)}_{task}_{mode}_error_probe_seed{args.seed}.jsonl"
        if rank == 0:
            mode_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        logging.info("Running mode=%s", mode)
        start = time.perf_counter()
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[size_key],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver="unipc",
            sampling_steps=args.sample_steps,
            guide_scale=args.guide_scale,
            seed=args.seed,
            offload_model=False,
            results_dir=str(mode_dir),
            eval_straightness=False,
            enable_visualization=False,
            error_probe=True,
            error_probe_max_steps=args.probe_max_steps,
            error_probe_output_path=str(probe_path),
        )
        seconds = time.perf_counter() - start
        if rank == 0:
            video_path = mode_dir / f"cubic_{get_prompt_id(args.prompt)}_{task}_{mode}.mp4"
            cache_video(
                tensor=video[None],
                save_file=str(video_path),
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            summary["modes"].append({
                "mode": mode,
                "seconds": seconds,
                "video_path": str(video_path),
                "probe_path": str(probe_path),
                "probe_rows": sum(1 for _ in probe_path.open("r", encoding="utf-8")) if probe_path.exists() else 0,
                "step_trace": wan_t2v.get_step_trace(),
            })
            (output_root / "cache_mode_probe_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        logging.info("Wrote %s", output_root / "cache_mode_probe_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
