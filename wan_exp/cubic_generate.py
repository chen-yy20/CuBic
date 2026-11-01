# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

_WAN_EXP_ROOT = os.path.dirname(os.path.abspath(__file__))
_CUBIC_ROOT = os.path.dirname(_WAN_EXP_ROOT)
sys.path.insert(0, _WAN_EXP_ROOT)
sys.path.insert(0, _CUBIC_ROOT)

import torch
import torch.distributed as dist

from wan_cubic.cubic_text2video import CubicWanT2V
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video, str2bool

from dist.parallel_state import (
    init_cp_group,
    init_distributed_environment,
)
from utils.launch_defaults import resolve_model_path
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import get_time_statistics_dict, init_timer, print_time_statistics, save_time_stats_to_file

init_timer()

PROMPT = 'Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.'
# PROMPT = "A couple in formal evening wear going home get caught in a heavy downpour with umbrellas, zoom in"
# Cubic optimizer / scheduler knobs
SPEEDUP =4
WARMUP = 9
COOLDOWN = 1
VISUALIZE = False
alpha = 2
beta = 4


TAG = 'cubic'
REQUIRED_TIMER_NAMES = [
    'generate_e2e',
    'anchor_step_monitor',
    'anchor_step_partitioner',
    'anchor_step_optimizer',
    'anchor_step_total',
    'selective_forward',
    'warmup_cooldown_step',
]


def _build_step_latency_metrics(step_trace):
    trace = [dict(item) for item in step_trace]
    return {
        'cubic_step_trace': {
            'steps': trace,
            'text_lines': [item['text'] for item in trace],
        }
    }


def _build_partition_metrics(partition_trace):
    anchors = [dict(item) for item in partition_trace]
    block_counts = [int(item['block_count']) for item in anchors]
    return {
        'cubic_partition_trace': {
            'anchors': anchors,
            'block_counts': block_counts,
            'mean_block_count': (
                float(sum(block_counts) / len(block_counts)) if block_counts else None
            ),
            'min_block_count': min(block_counts) if block_counts else None,
            'max_block_count': max(block_counts) if block_counts else None,
        }
    }


def _build_cache_budget_ledger(step_trace, warmup_steps, cooldown_steps):
    rows = [dict(item) for item in step_trace]
    total_steps = len(rows)
    active_ratios = [float(row.get('active_ratio', 0.0)) for row in rows]
    ratio = sum(active_ratios) / max(len(active_ratios), 1) if active_ratios else None
    full_compute_steps = sum(1 for row in rows if bool(row.get('full_compute')))
    return {
        'method': 'cubic',
        'budget_kind': 'active_token_proxy',
        'budget_precision': 'token_active_ratio_proxy_not_exact_flops',
        'total_steps': int(total_steps),
        'warmup_steps': int(warmup_steps),
        'cooldown_steps': int(cooldown_steps),
        'full_compute_steps': float(full_compute_steps),
        'reuse_steps': float(max(0, total_steps - full_compute_steps)),
        'mean_step_compute_ratio': float(ratio) if ratio is not None else None,
        'mean_layer_compute_ratio': float(ratio) if ratio is not None else None,
        'mean_token_compute_ratio': float(ratio) if ratio is not None else None,
        'effective_dit_compute_ratio': float(ratio) if ratio is not None else None,
    }


def _merge_quality_and_step_metrics(quality_metrics, step_trace):
    merged = dict(quality_metrics) if isinstance(quality_metrics, dict) else {}
    merged.update(_build_step_latency_metrics(step_trace))
    return merged


def _print_step_latency_trace(step_trace):
    print('Cubic per-step latency / active tokens:', flush=True)
    for item in step_trace:
        print(item['text'], flush=True)


def _resolve_model_defaults(model_path):
    if '1.3B' in model_path:
        return 't2v-1.3B', '832*480', '1.3B'
    return 't2v-14B', '1280*720', '14B'


def _default_output_dir(model_path, prompt):
    _, _, model_id = _resolve_model_defaults(model_path)
    return (
        f'./results/wan/{model_id}/cubic/'
        f'{get_prompt_id(prompt)}_{SPEEDUP}x_a{alpha}_b{beta}_dynanchor_bcmean'
    )


def _default_origin_reference_path(model_path, prompt, task):
    _, _, model_id = _resolve_model_defaults(model_path)
    prompt_id = get_prompt_id(prompt)
    return os.path.join(
        '.',
        'results',
        'wan',
        model_id,
        'ori',
        prompt_id,
        f'ori_{prompt_id}_{task}.mp4',
    )

rank = int(os.getenv('RANK', 0))
world_size = int(os.getenv('WORLD_SIZE', 1))
local_rank = int(os.getenv('LOCAL_RANK', 0))
if world_size > 1:
    assert world_size == 2, 'only support cfg parallel'
    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )
    init_cp_group(
        group_ranks=[[0, 1]],
        local_rank=local_rank,
        backend='nccl',
    )


def _validate_args(args):
    assert args.ckpt_dir is not None, 'Please specify the checkpoint directory.'
    assert args.task in ['t2v-1.3B', 't2v-14B'], f'Unsupport task: {args.task}'
    assert args.task in WAN_CONFIGS, f'Unsupport task: {args.task}'
    assert args.size in SUPPORTED_SIZES[args.task], (
        f'Unsupport size {args.size} for task {args.task}, '
        f"supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"
    )

    if args.sample_steps is None:
        args.sample_steps = 50

    if args.sample_shift is None:
        args.sample_shift = 5.0

    if args.frame_num is None:
        args.frame_num = 81


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Generate a video from text prompt using CubicWanT2V'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='t2v-14B',
        choices=['t2v-1.3B', 't2v-14B'],
        help='The task to run.',
    )
    parser.add_argument(
        '--size',
        type=str,
        default='1280*720',
        choices=list(SIZE_CONFIGS.keys()),
        help='The area (width*height) of the generated video.',
    )
    parser.add_argument(
        '--frame_num',
        type=int,
        default=None,
        help='How many frames to sample from a video. The number should be 4n+1',
    )
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default=None,
        help='The path to the checkpoint directory.',
    )
    parser.add_argument(
        '--offload_model',
        type=str2bool,
        default=None,
        help='Whether to offload the model to CPU after each model forward.',
    )
    parser.add_argument(
        '--ulysses_size',
        type=int,
        default=1,
        help='The size of the ulysses parallelism in DiT.',
    )
    parser.add_argument(
        '--ring_size',
        type=int,
        default=1,
        help='The size of the ring attention parallelism in DiT.',
    )
    parser.add_argument(
        '--t5_fsdp',
        action='store_true',
        default=False,
        help='Whether to use FSDP for T5.',
    )
    parser.add_argument(
        '--t5_cpu',
        action='store_true',
        default=False,
        help='Whether to place T5 model on CPU.',
    )
    parser.add_argument(
        '--dit_fsdp',
        action='store_true',
        default=False,
        help='Whether to use FSDP for DiT.',
    )
    parser.add_argument(
        '--save_file',
        type=str,
        default=None,
        help='The file to save the generated video to.',
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='The prompt to generate the video from.',
    )
    parser.add_argument(
        '--base_seed',
        type=int,
        default=-1,
        help='The seed to use for generating the video.',
    )
    parser.add_argument(
        '--sample_solver',
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help='The solver used to sample.',
    )
    parser.add_argument(
        '--sample_steps',
        type=int,
        default=None,
        help='The sampling steps.',
    )
    parser.add_argument('--warmup_steps', type=int, default=WARMUP)
    parser.add_argument('--cooldown_steps', type=int, default=COOLDOWN)
    parser.add_argument(
        '--target_speedup',
        type=float,
        default=float(SPEEDUP),
        help='Target selective-phase speedup used by the Cubic interval optimizer.',
    )
    parser.add_argument(
        '--sample_shift',
        type=float,
        default=None,
        help='Sampling shift factor for flow matching schedulers.',
    )
    parser.add_argument(
        '--sample_guide_scale',
        type=float,
        default=5.0,
        help='Classifier free guidance scale.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='output_dir.',
    )
    parser.add_argument(
        '--baseline_path',
        type=str,
        default=None,
        help='Optional origin video path for quality evaluation.',
    )
    parser.add_argument(
        '--enable_visualization',
        type=str2bool,
        default=VISUALIZE,
        help='Enable visualization generation and related artifact export.',
    )
    parser.add_argument(
        '--straightness_eval',
        type=str2bool,
        default=False,
        help='Enable straightness ground-truth evaluation (expensive, default off for optimization runs).',
    )
    parser.add_argument(
        '--cubic_error_probe',
        action='store_true',
        default=False,
        help='Enable block-level selective-vs-full oracle error probe.',
    )
    parser.add_argument(
        '--cubic_error_probe_max_steps',
        type=int,
        default=0,
        help='Maximum number of selective steps to probe; 0 means no explicit limit.',
    )
    parser.add_argument(
        '--cubic_error_probe_output_path',
        type=str,
        default=None,
        help='JSONL path for block-level oracle error probe records.',
    )
    parser.add_argument(
        '--cache_reuse_mode',
        type=str,
        default='residual',
        choices=['residual', 'output', 'kv_only'],
        help='WAN frozen-token reuse mode: residual cache, final output cache, or KV-only active-token merge.',
    )
    parser.add_argument(
        '--partition_mode',
        type=str,
        default=None,
        choices=['morphology', 'uniform', 'uniform_grid'],
        help='Optional Cubic partition mode override.',
    )
    parser.add_argument('--uniform_grid_h', type=int, default=None)
    parser.add_argument('--uniform_grid_w', type=int, default=None)
    parser.add_argument(
        '--min_block_size',
        type=int,
        default=None,
        help='Optional Cubic minimum spatial block size override.',
    )
    parser.add_argument(
        '--max_block_size',
        type=int,
        default=None,
        help='Optional Cubic maximum spatial block size override.',
    )
    parser.add_argument(
        '--cv_threshold',
        type=float,
        default=None,
        help='Optional Cubic block curvature CV threshold override.',
    )
    parser.add_argument(
        '--foreground_focus_block_size',
        type=int,
        default=None,
        help='Optional morphology foreground focus block size override.',
    )

    args = parser.parse_args()
    args.ckpt_dir = resolve_model_path(args.ckpt_dir, '--ckpt_dir')
    args.task, args.size, _ = _resolve_model_defaults(args.ckpt_dir)
    if args.prompt is None:
        args.prompt = PROMPT
    if args.output_dir is None:
        args.output_dir = _default_output_dir(args.ckpt_dir, args.prompt)
    args.base_seed = 42 if args.base_seed < 0 else args.base_seed

    _validate_args(args)
    return args


def _init_logging(cur_rank):
    if cur_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    cur_rank = int(os.getenv('RANK', 0))
    cur_world_size = int(os.getenv('WORLD_SIZE', 1))
    cur_local_rank = int(os.getenv('LOCAL_RANK', 0))
    device = cur_local_rank
    _init_logging(cur_rank)

    if args.offload_model is None:
        args.offload_model = False if cur_world_size > 1 else True
        logging.info(f'offload_model is not specified, set to {args.offload_model}.')

    if cur_world_size == 1:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), 't5_fsdp and dit_fsdp are not supported in non-distributed environments.'
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), 'context parallel are not supported in non-distributed environments.'
    elif args.ulysses_size > 1 or args.ring_size > 1:
        raise NotImplementedError('ulysses/ring parallel is not supported with cfg group init mode')

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, '`num_heads` must be divisible by `ulysses_size`.'

    # Adapt cubic hyperparameters from experiment entry to runtime config.
    total_steps = max(int(args.sample_steps), 1)
    cfg.target_speedup = float(args.target_speedup)
    cfg.warmup_fraction = max(0.0, min(1.0, float(args.warmup_steps) / float(total_steps)))
    cfg.cooldown_fraction = max(0.0, min(1.0, float(args.cooldown_steps) / float(total_steps)))
    cfg.alpha = float(alpha)
    cfg.beta = float(beta)
    cfg.cache_reuse_mode = str(args.cache_reuse_mode)
    if args.partition_mode is not None:
        cfg.partition_mode = str(args.partition_mode)
    if args.uniform_grid_h is not None:
        cfg.uniform_grid_h = int(args.uniform_grid_h)
    if args.uniform_grid_w is not None:
        cfg.uniform_grid_w = int(args.uniform_grid_w)
    if args.min_block_size is not None:
        cfg.min_block_size = int(args.min_block_size)
    if args.max_block_size is not None:
        cfg.max_block_size = int(args.max_block_size)
    if args.cv_threshold is not None:
        cfg.cv_threshold = float(args.cv_threshold)
    if args.foreground_focus_block_size is not None:
        cfg.foreground_focus_block_size = int(args.foreground_focus_block_size)

    logging.info(f'Generation job args: {args}')
    logging.info(f'Generation model config: {cfg}')
    logging.info(
        'Cubic config override: target_speedup=%.3f warmup_fraction=%.4f cooldown_fraction=%.4f dynamic_anchor=2x_tau_max curvature_aggregation=mean',
        float(cfg.target_speedup),
        float(cfg.warmup_fraction),
        float(cfg.cooldown_fraction),
    )
    logging.info('WAN cache reuse mode: %s', cfg.cache_reuse_mode)
    logging.info(
        'WAN partition override: mode=%s grid=%sx%s min_block=%s max_block=%s cv_threshold=%s fg_focus=%s',
        getattr(cfg, 'partition_mode', None),
        getattr(cfg, 'uniform_grid_h', None),
        getattr(cfg, 'uniform_grid_w', None),
        getattr(cfg, 'min_block_size', None),
        getattr(cfg, 'max_block_size', None),
        getattr(cfg, 'cv_threshold', None),
        getattr(cfg, 'foreground_focus_block_size', None),
    )

    if dist.is_initialized():
        base_seed = [args.base_seed] if cur_rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info('Creating CubicWanT2V pipeline.')
    wan_t2v = CubicWanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=cur_rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        t5_cpu=args.t5_cpu,
    )

    logging.info('Generating video ...')
    error_probe_output_path = args.cubic_error_probe_output_path
    if args.cubic_error_probe and error_probe_output_path is None and cur_rank == 0:
        error_probe_output_path = os.path.join(
            args.output_dir,
            f'{TAG}_{get_prompt_id(args.prompt)}_{args.task}_error_probe_seed{args.base_seed}.jsonl',
        )
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
        results_dir=args.output_dir,
        eval_straightness=args.straightness_eval,
        enable_visualization=args.enable_visualization,
        error_probe=args.cubic_error_probe,
        error_probe_max_steps=args.cubic_error_probe_max_steps,
        error_probe_output_path=error_probe_output_path,
    )
    step_trace = wan_t2v.get_step_trace()
    step_metrics = _build_step_latency_metrics(step_trace)
    step_metrics.update(_build_partition_metrics(wan_t2v.get_partition_trace()))

    if cur_rank == 0:
        filename = f'{TAG}_{get_prompt_id(args.prompt)}_{args.task}.mp4'
        os.makedirs(args.output_dir, exist_ok=True)
        args.save_file = os.path.join(args.output_dir, filename)

        logging.info(f'Saving generated video to {args.save_file}')
        cache_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        baseline_path = args.baseline_path or _default_origin_reference_path(
            model_path=args.ckpt_dir,
            prompt=args.prompt,
            task=args.task,
        )
        quality_result = evaluate_quality_with_origin(
            image_path=args.save_file,
            tag=TAG,
            save_metrics=False,
            baseline_path=baseline_path,
            output_dir=None,
        )
        quality_metrics = quality_result['metrics'] if quality_result is not None else None

        logging.info('Finished.')
        return quality_metrics, step_metrics

    logging.info('Finished.')
    return None, step_metrics


if __name__ == '__main__':
    args = _parse_args()
    quality_metrics, step_metrics = generate(args)
    print_time_statistics(timer_names=REQUIRED_TIMER_NAMES)

    cur_rank = int(os.getenv('RANK', 0))
    if cur_rank == 0:
        step_trace_payload = step_metrics or _build_step_latency_metrics([])
        step_trace = step_trace_payload['cubic_step_trace']['steps']
        partition_trace = step_trace_payload.get(
            'cubic_partition_trace',
            _build_partition_metrics([])['cubic_partition_trace'],
        )
        _print_step_latency_trace(step_trace)
        save_time_stats_to_file(
            os.path.join(args.output_dir, 'timing_stats.txt'),
            timer_names=REQUIRED_TIMER_NAMES,
        )
        _, _, model_id = _resolve_model_defaults(args.ckpt_dir)
        params = {
            'model': 'wan',
            'method': 'cubic',
            'model_id': model_id,
            'model_path': args.ckpt_dir,
            'prompt': args.prompt,
            'task': args.task,
            'size': args.size,
            'base_seed': args.base_seed,
            'sample_steps': args.sample_steps,
            'frame_num': args.frame_num,
            'warmup_steps': args.warmup_steps,
            'cooldown_steps': args.cooldown_steps,
            'target_speedup': args.target_speedup,
            'baseline_path': args.baseline_path,
            'curvature_aggregation': 'mean',
            'dynamic_anchor': '2x_tau_max',
            'cache_reuse_mode': args.cache_reuse_mode,
            'partition_mode': getattr(WAN_CONFIGS[args.task], 'partition_mode', None),
            'uniform_grid_h': getattr(WAN_CONFIGS[args.task], 'uniform_grid_h', None),
            'uniform_grid_w': getattr(WAN_CONFIGS[args.task], 'uniform_grid_w', None),
            'min_block_size': getattr(WAN_CONFIGS[args.task], 'min_block_size', None),
            'max_block_size': getattr(WAN_CONFIGS[args.task], 'max_block_size', None),
            'cv_threshold': getattr(WAN_CONFIGS[args.task], 'cv_threshold', None),
            'foreground_focus_block_size': getattr(WAN_CONFIGS[args.task], 'foreground_focus_block_size', None),
            'block_count': {
                'per_anchor': partition_trace['block_counts'],
                'mean': partition_trace['mean_block_count'],
                'min': partition_trace['min_block_count'],
                'max': partition_trace['max_block_count'],
            },
            'error_probe_enabled': bool(args.cubic_error_probe),
            'error_probe_max_steps': int(args.cubic_error_probe_max_steps),
            'error_probe_output_path': args.cubic_error_probe_output_path
            or (
                os.path.join(
                    args.output_dir,
                    f'{TAG}_{get_prompt_id(args.prompt)}_{args.task}_error_probe_seed{args.base_seed}.jsonl',
                )
                if args.cubic_error_probe else None
            ),
            'cache_budget_ledger': _build_cache_budget_ledger(
                step_trace,
                args.warmup_steps,
                args.cooldown_steps,
            ),
        }
        params_path = save_params_and_metrics(
            args.output_dir,
            TAG,
            params,
            get_time_statistics_dict(timer_names=REQUIRED_TIMER_NAMES),
            quality_metrics=_merge_quality_and_step_metrics(
                quality_metrics,
                step_trace,
            ),
        )
        print(f'Params & metrics saved to {params_path}', flush=True)
