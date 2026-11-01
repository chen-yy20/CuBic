"""Flux + TeaCache 推理入口。

TeaCache 论文：https://arxiv.org/abs/2411.19108
实现：baseline/teacache_flux.py（monkey patch FluxTransformer2DModel.forward）

用法（从 Cubic 根目录）：
    python launch.py --model flux --method teacache [--launcher python]
    python launch.py --model flux --method teacache -- --rel_l1_thresh 0.5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from statistics import mean

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.teacache_flux import reset_teacache_state, teacache_forward

from flux.pipeline import FluxPipeline
from flux.transformer import FluxTransformer2DModel
from utils.launch_defaults import resolve_model_path

# teacache_flux.py 默认 patch 的是 diffusers 的 FluxTransformer2DModel，
# 但本项目 flux/transformer.py 定义了自己的同名类，需要单独 patch。
FluxTransformer2DModel.forward = teacache_forward
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import (
    get_time_statistics_dict,
    init_timer,
    print_time_statistics,
    save_time_stats_to_file,
)

# ── 默认配置（与 ori/cubic 保持一致，便于质量对比）────────────────────────────
PROMPT             = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
HEIGHT             = 1024
WIDTH              = 1024
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE     = 3.5
NUM_IMAGES         = 3
SEEDS              = [42 + i for i in range(NUM_IMAGES)]
REL_L1_THRESH      = 0.25   # 官方默认；越大越快质量损失越多（0.25/0.4/0.6/0.8 对应 ~12/10/9/8s）
TEACACHE_WARMUP_STEPS = 1
TEACACHE_COOLDOWN_STEPS = 1

TAG      = f"teacache_{REL_L1_THRESH}"
quality_eval = True
REQUIRED_TIMER_NAMES = ["generate_e2e", "dit_forward"]


def _format_teacache_sequence(sequence: list[int]) -> str:
    return "".join(str(int(value)) for value in sequence)


def _build_teacache_sequence_metrics(per_seed_sequence: dict[str, list[int]]) -> dict:
    per_seed = {}
    for seed_key, sequence in per_seed_sequence.items():
        seq = [int(value) for value in sequence]
        compute_steps = sum(seq)
        per_seed[seed_key] = {
            "sequence": seq,
            "text": _format_teacache_sequence(seq),
            "compute_steps": int(compute_steps),
            "reuse_steps": int(len(seq) - compute_steps),
            "total_steps": int(len(seq)),
        }
    return {"teacache_compute_skip": {"per_seed": per_seed}}


def _build_cache_budget_ledger(per_seed_sequence: dict[str, list[int]]) -> dict:
    per_seed = {}
    ratios = []
    total_steps_values = []
    compute_steps_values = []
    reuse_steps_values = []
    for seed_key, sequence in per_seed_sequence.items():
        seq = [int(value) for value in sequence]
        total_steps = len(seq)
        compute_steps = sum(seq)
        reuse_steps = total_steps - compute_steps
        ratio = compute_steps / max(total_steps, 1)
        ratios.append(ratio)
        total_steps_values.append(total_steps)
        compute_steps_values.append(compute_steps)
        reuse_steps_values.append(reuse_steps)
        per_seed[seed_key] = {
            "total_steps": int(total_steps),
            "full_compute_steps": int(compute_steps),
            "reuse_steps": int(reuse_steps),
            "mean_step_compute_ratio": float(ratio),
            "effective_dit_compute_ratio": float(ratio),
        }
    mean_ratio = mean(ratios) if ratios else None
    return {
        "method": "teacache",
        "budget_kind": "step_full_dit",
        "budget_precision": "exact_for_transformer_forward_skip_proxy_for_full_pipeline",
        "total_steps": int(round(mean(total_steps_values))) if total_steps_values else 0,
        "full_compute_steps": float(mean(compute_steps_values)) if compute_steps_values else 0.0,
        "reuse_steps": float(mean(reuse_steps_values)) if reuse_steps_values else 0.0,
        "mean_step_compute_ratio": float(mean_ratio) if mean_ratio is not None else None,
        "mean_layer_compute_ratio": float(mean_ratio) if mean_ratio is not None else None,
        "mean_token_compute_ratio": 1.0,
        "effective_dit_compute_ratio": float(mean_ratio) if mean_ratio is not None else None,
        "per_seed": per_seed,
    }


def _merge_quality_and_teacache_metrics(quality_metrics, per_seed_sequence: dict[str, list[int]]):
    merged = dict(quality_metrics) if isinstance(quality_metrics, dict) else {}
    merged.update(_build_teacache_sequence_metrics(per_seed_sequence))
    return merged


# ── 参数解析 ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux TeaCache inference script")
    parser.add_argument("--model_path",          type=str,   default=None)
    parser.add_argument("--prompt",              type=str,   default=PROMPT)
    parser.add_argument("--height",              type=int,   default=HEIGHT)
    parser.add_argument("--width",               type=int,   default=WIDTH)
    parser.add_argument("--num_inference_steps", type=int,   default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance_scale",      type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--num_images",          type=int,   default=NUM_IMAGES)
    parser.add_argument("--seeds",               type=int,   nargs="+", default=None,
                        help="生成用的随机种子列表，默认 42,43,44,…")
    parser.add_argument("--rel_l1_thresh",       type=float, default=REL_L1_THRESH,
                        help="TeaCache 累积阈值；越大跳步越多越快，画质损失越大"
                             "（官方推荐 0.25/0.4/0.6/0.8，对应 ~12/10/9/8s）")
    parser.add_argument("--teacache_warmup_steps", type=int, default=TEACACHE_WARMUP_STEPS)
    parser.add_argument("--teacache_cooldown_steps", type=int, default=TEACACHE_COOLDOWN_STEPS)
    parser.add_argument("--output_dir",          type=str,   default=None,
                        help="结果输出目录，默认 ./results/flux/teacache/{prompt_id}")
    parser.add_argument("--baseline_dir",        type=str,   default=None,
                        help="Optional Flux origin baseline directory for quality evaluation")
    parser.add_argument("--disable_quality_eval", action="store_true", default=not quality_eval,
                        help="关闭相对 ori 基准的质量评测（PSNR/SSIM/LPIPS 等）")
    return parser.parse_args()


# ── Pipeline 加载 ─────────────────────────────────────────────────────────────

def load_pipeline(model_path: str) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.transformer = FluxTransformer2DModel.from_pretrained(
        f"{model_path}/transformer",
        torch_dtype=torch.bfloat16,
    )
    return pipe.to("cuda")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    init_timer()
    args.model_path = resolve_model_path(args.model_path, "--model_path")

    seeds = args.seeds if args.seeds is not None else [42 + i for i in range(args.num_images)]
    prompt_id  = get_prompt_id(args.prompt)
    model_id   = os.path.basename(args.model_path.rstrip("/"))
    tag = f"teacache_{args.rel_l1_thresh:g}"
    output_dir = args.output_dir or os.path.join("results", "flux", tag, prompt_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[teacache-flux] Loading pipeline from {args.model_path}...", flush=True)
    pipe = load_pipeline(args.model_path)
    print(
        f"[teacache-flux] Pipeline loaded. "
        f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.0f} MB",
        flush=True,
    )

    # 配置 TeaCache 超参
    transformer: FluxTransformer2DModel = pipe.transformer
    transformer.enable_teacache = True
    transformer.num_steps        = args.num_inference_steps
    transformer.rel_l1_thresh    = args.rel_l1_thresh
    transformer.teacache_warmup_steps = args.teacache_warmup_steps
    transformer.teacache_cooldown_steps = args.teacache_cooldown_steps

    print(
        f"[teacache-flux] rel_l1_thresh={args.rel_l1_thresh}  "
        f"warmup={args.teacache_warmup_steps} cooldown={args.teacache_cooldown_steps} "
        f"steps={args.num_inference_steps}  seeds={seeds}",
        flush=True,
    )

    per_seed_quality: dict[str, dict] = {}
    per_seed_teacache_sequence: dict[str, list[int]] = {}
    run_start_time = time.perf_counter()
    time_estimate_seconds = None

    for seed in seeds:
        seed_start_time = time.perf_counter()
        # TeaCache 状态必须在每次生成前重置
        reset_teacache_state(transformer)

        generator = torch.Generator("cuda").manual_seed(seed)
        with torch.inference_mode():
            image = pipe(
                args.prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=512,
                generator=generator,
            ).images[0]
        sequence = [int(value) for value in getattr(transformer, "teacache_compute_skip", [])]
        per_seed_teacache_sequence[f"seed{seed}"] = sequence
        compute_steps = sum(sequence)
        print(
            f"[teacache-flux] seed={seed} compute/reuse sequence "
            f"(1=compute,0=reuse): {_format_teacache_sequence(sequence)} "
            f"compute={compute_steps} reuse={len(sequence) - compute_steps}",
            flush=True,
        )

        output_path = os.path.join(output_dir, f"{tag}_{prompt_id}_seed{seed}.png")
        image.save(output_path)
        print(f"[teacache-flux] Saved {output_path}", flush=True)

        if not args.disable_quality_eval:
            baseline_dir = args.baseline_dir or os.path.join("results", "flux", "ori", prompt_id)
            baseline_path = os.path.join(baseline_dir, f"ori_{prompt_id}_seed{seed}.png")
            quality_result = evaluate_quality_with_origin(
                image_path=output_path,
                tag=tag,
                save_metrics=False,
                baseline_path=baseline_path,
                output_dir=output_dir,
            )
            per_seed_quality[f"seed{seed}"] = (
                quality_result["metrics"] if quality_result is not None else {}
            )
        if time_estimate_seconds is None:
            time_estimate_seconds = (time.perf_counter() - seed_start_time) * len(seeds)
            print(f"TIME_ESTIMATE: {time_estimate_seconds:.2f}s", flush=True)

    quality_metrics = None
    if not args.disable_quality_eval:
        valid_metrics = [m for m in per_seed_quality.values() if m]
        if valid_metrics:
            metric_names = ["psnr", "ssim", "lpips", "fid", "fvd"]
            mean_metrics = {}
            for name in metric_names:
                values = [float(m[name]) for m in valid_metrics if m.get(name) is not None]
                mean_metrics[name] = mean(values) if values else None
            quality_metrics = {"per_seed": per_seed_quality, "mean": mean_metrics}
        else:
            quality_metrics = {"per_seed": per_seed_quality, "mean": None}

    print_time_statistics(timer_names=REQUIRED_TIMER_NAMES)
    save_time_stats_to_file(
        os.path.join(output_dir, "timing_stats.txt"),
        timer_names=REQUIRED_TIMER_NAMES,
    )

    params = {
        "model":               "flux",
        "method":              tag,
        "model_id":            model_id,
        "model_path":          args.model_path,
        "prompt":              args.prompt,
        "height":              args.height,
        "width":               args.width,
        "num_images":          len(seeds),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale":      args.guidance_scale,
        "seeds":               seeds,
        "rel_l1_thresh":       args.rel_l1_thresh,
        "warmup_steps":        args.teacache_warmup_steps,
        "cooldown_steps":      args.teacache_cooldown_steps,
        "quality_eval":        not args.disable_quality_eval,
        "baseline_dir":        args.baseline_dir or os.path.join("results", "flux", "ori", prompt_id),
        "time_estimate_seconds": time_estimate_seconds,
        "wall_time_seconds": time.perf_counter() - run_start_time,
        "cache_budget_ledger": _build_cache_budget_ledger(per_seed_teacache_sequence),
    }
    params_path = save_params_and_metrics(
        output_dir,
        tag,
        params,
        get_time_statistics_dict(timer_names=REQUIRED_TIMER_NAMES),
        quality_metrics=_merge_quality_and_teacache_metrics(quality_metrics, per_seed_teacache_sequence),
    )
    print(f"[teacache-flux] Params & metrics saved to {params_path}", flush=True)


if __name__ == "__main__":
    main()
