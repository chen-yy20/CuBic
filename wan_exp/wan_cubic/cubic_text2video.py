import gc
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import contextmanager

import torch
import torch.cuda.amp as amp

from dist.parallel_state import get_cp_group, get_cp_worldsize
from tqdm import tqdm
from wan_exp.wan_cubic.cubic_selective_forward import WanSelectiveForwardEngine

from cubic_core import (
    CurvatureMonitor,
    StraightnessEvaluator,
    AdaptivePartitioner,
    CubicVisualizer,
    UpdateFrequencyOptimizer,
    SelectiveStepScheduler,
    Phase,
)
from utils.timer import get_timer
from wan.text2video import (
    FlowDPMSolverMultistepScheduler,
    FlowUniPCMultistepScheduler,
    WanT2V,
    get_sampling_sigmas,
    retrieve_timesteps,
)


class CubicWanT2V(WanT2V):
    """Decoupled Cubic T2V pipeline that keeps origin Wan modules untouched."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_warmup_fraction = float(getattr(self.config, 'warmup_fraction', 0.15))
        self.phase_cooldown_fraction = float(getattr(self.config, 'cooldown_fraction', 0.15))
        self.monitor = CurvatureMonitor(self.config)
        self.straightness = StraightnessEvaluator(self.config)
        self.partitioner = AdaptivePartitioner(self.config)
        self.frequency_optimizer = UpdateFrequencyOptimizer(self.config)
        self.step_scheduler = SelectiveStepScheduler(self.config)
        self.visualizer = CubicVisualizer(self.config)
        self.selective_forward = WanSelectiveForwardEngine(self.config)
        self.current_partition = None
        self.current_frequency_plan = None
        self.current_step_plan = None
        self.step_trace = []
        self.partition_trace = []
        self.next_anchor_step = None
        self.current_anchor_gap = 0
        self.error_probe_enabled = False
        self.error_probe_max_steps = 0
        self.error_probe_output_path = None
        self.error_probe_records = []
        self._error_probe_steps_done = 0
        self._block_age = {}
        self._last_block_curvatures = {}
        self._current_total_steps = 0
        self._normalized_dt_by_step = []

    def _get_inference_phase(self, step, total_steps):
        if total_steps <= 0:
            return 'phase4_cooldown'

        warmup_steps = max(
            0, int(math.floor(total_steps * self.phase_warmup_fraction)))
        cooldown_steps = max(
            0, int(math.floor(total_steps * self.phase_cooldown_fraction)))
        cooldown_start = max(0, total_steps - cooldown_steps)

        if step < warmup_steps:
            return 'phase1_warmup'

        if step >= cooldown_start:
            return 'phase4_cooldown'

        if self.next_anchor_step is None:
            self.next_anchor_step = warmup_steps

        if step == self.next_anchor_step:
            return 'phase2_cruise_anchor'
        return 'phase3_cruise_selective'

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt='',
                 seed=-1,
                 offload_model=True,
                 results_dir=None,
                 eval_straightness=False,
                 enable_visualization=True,
                 error_probe=False,
                 error_probe_max_steps=0,
                 error_probe_output_path=None):
        f_num = frame_num
        target_shape = (self.vae.model.z_dim, (f_num - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == '':
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        self.monitor.reset()
        self.straightness.reset()
        self.selective_forward.reset()
        self.current_partition = None
        self.current_frequency_plan = None
        self.current_step_plan = None
        self.step_trace = []
        self.partition_trace = []
        self.next_anchor_step = None
        self.current_anchor_gap = 0
        self.error_probe_enabled = bool(error_probe)
        self.error_probe_max_steps = max(0, int(error_probe_max_steps or 0))
        self.error_probe_output_path = error_probe_output_path
        self.error_probe_records = []
        self._error_probe_steps_done = 0
        self._block_age = {}
        self._last_block_curvatures = {}
        self._current_total_steps = int(sampling_steps)
        self._normalized_dt_by_step = []

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError('Unsupported solver.')

            self._normalized_dt_by_step = self._build_normalized_dt_by_step(timesteps)
            latents = noise
            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            with get_timer('generate_e2e'):
                warmup_steps = max(
                    0,
                    int(math.floor(len(timesteps) * self.phase_warmup_fraction)),
                )
                for step, t in enumerate(tqdm(timesteps)):
                    phase = self._get_inference_phase(step, len(timesteps))
                    if self.rank == 0:
                        logging.info(
                            '[wan-phase] step=%d/%d phase=%s',
                            step,
                            len(timesteps) - 1,
                            phase,
                        )

                    phase_enum = self._phase_to_enum(phase)
                    if self.current_partition is not None and self.current_frequency_plan is not None:
                        selective_step_idx = max(0, step - warmup_steps)
                        self.current_step_plan = self.step_scheduler.build_step_plan(
                            partition=self.current_partition,
                            block_intervals=self.current_frequency_plan.block_intervals,
                            phase=phase_enum,
                            step_idx=step,
                            selective_step_idx=selective_step_idx,
                        )
                        if self.rank == 0:
                            step_debug = self.step_scheduler.describe_step_plan(
                                self.current_step_plan,
                                self.current_partition,
                                step_idx=step,
                            )
                            logging.info(
                                '[wan-step-plan] step=%d phase=%s active_blocks=%d frozen_blocks=%d active_tokens=%d frozen_tokens=%d full_compute=%s',
                                step,
                                step_debug.phase.value,
                                step_debug.active_block_count,
                                step_debug.frozen_block_count,
                                step_debug.active_token_count,
                                step_debug.frozen_token_count,
                                step_debug.is_full_compute,
                            )

                    latent_model_input = latents
                    timestep = [t]
                    timestep = torch.stack(timestep)

                    self.model.to(self.device)
                    noise_pred = None
                    step_start = self._start_step_timer()
                    # Monitor updates at warmup tail and cruise anchors.
                    is_last_warmup = warmup_steps > 0 and phase == 'phase1_warmup' and step == warmup_steps - 1
                    is_cruise_anchor = phase == 'phase2_cruise_anchor'
                    if is_last_warmup or is_cruise_anchor:
                        with get_timer('anchor_step_total'):
                            noise_pred = self._predict_noise(
                                latent_model_input=latent_model_input,
                                timestep=timestep,
                                arg_c=arg_c,
                                arg_null=arg_null,
                                guide_scale=guide_scale,
                            )

                            t_unit = self._normalize_t_to_unit_interval(t)
                            with get_timer('anchor_step_monitor'):
                                curvature_map = self.monitor.update(
                                    x=latents[0].unsqueeze(0),
                                    v=noise_pred.unsqueeze(0),
                                    t=t_unit,
                                    step_idx=step,
                                )

                            if curvature_map is not None:
                                if self.rank == 0:
                                    logging.info(
                                        '[wan-monitor] step=%d curvature_map=%s mean=%.6e',
                                        step,
                                        tuple(curvature_map.shape),
                                        float(curvature_map.float().mean().item()),
                                    )

                                # Keep cubic decisions branch-consistent in CFG parallel by computing on both ranks.
                                Ft, Ht, Wt = curvature_map.shape
                                with get_timer('anchor_step_partitioner'):
                                    self.current_partition = self.partitioner.partition(
                                        Ft=Ft, Ht=Ht, Wt=Wt, curvature_map=curvature_map
                                    )
                                    partition_debug = self.partitioner.describe_partition(
                                        self.current_partition,
                                        curvature_map=curvature_map,
                                    )
                                    block_curvatures = self.monitor.get_block_curvatures(
                                        self.current_partition,
                                        curvature_map=curvature_map,
                                    )
                                self.partition_trace.append({
                                    'step': int(step),
                                    'block_count': int(partition_debug.block_count),
                                    'chosen_block_size': [
                                        int(value) for value in partition_debug.chosen_block_size
                                    ],
                                    'partition_mode': str(
                                        getattr(self.current_partition, 'partition_mode', 'unknown')
                                    ),
                                })
                                if self.rank == 0:
                                    logging.info(
                                        '[wan-partitioner] step=%d chosen_block_size=%s blocks=%d token_grid=(%d,%d,%d) token_stats(min=%d max=%d mean=%.2f) block_cv(mean=%.4f p90=%.4f)',
                                        step,
                                        partition_debug.chosen_block_size,
                                        partition_debug.block_count,
                                        Ft, Ht, Wt,
                                        partition_debug.block_token_min,
                                        partition_debug.block_token_max,
                                        partition_debug.block_token_mean,
                                        partition_debug.mean_block_cv,
                                        partition_debug.p90_block_cv,
                                    )
                                    for block in self.current_partition:
                                        logging.debug(
                                            '[wan-partitioner-detail] block_id=%d f=[%d,%d) h=[%d,%d) w=[%d,%d) tokens=%d',
                                            block.block_id,
                                            block.f_start, block.f_end,
                                            block.h_start, block.h_end,
                                            block.w_start, block.w_end,
                                            block.token_count,
                                        )

                                with get_timer('anchor_step_optimizer'):
                                    frequency_plan = self.frequency_optimizer.optimize(
                                        partition=self.current_partition,
                                        block_curvatures=block_curvatures,
                                    )
                                    self.current_frequency_plan = frequency_plan
                                    self._last_block_curvatures = dict(block_curvatures)
                                    self._block_age = {
                                        int(block.block_id): 0
                                        for block in self.current_partition
                                    }
                                    self.current_anchor_gap = max(1, 2 * int(frequency_plan.max_interval))
                                    self.next_anchor_step = step + self.current_anchor_gap
                                    self.current_step_plan = self.step_scheduler.build_step_plan(
                                        partition=self.current_partition,
                                        block_intervals=frequency_plan.block_intervals,
                                        phase=Phase.CRUISE_ANCHOR,
                                        step_idx=step,
                                        selective_step_idx=max(0, step - warmup_steps),
                                    )
                                if self.rank == 0:
                                    logging.info(
                                        '[wan-optimizer] step=%d assigned_tau(min=%d max=%d mean=%.3f) mean_update_rate=%.6f achieved_speedup=%.3f next_anchor_gap=%d next_anchor_step=%d',
                                        step,
                                        frequency_plan.min_interval,
                                        frequency_plan.max_interval,
                                        frequency_plan.mean_interval,
                                        frequency_plan.mean_update_rate,
                                        frequency_plan.achieved_speedup,
                                        self.current_anchor_gap,
                                        self.next_anchor_step,
                                    )

                                if self.rank == 0 and enable_visualization:
                                    partition_vis_info = self.visualizer.visualize_partition_and_frequency(
                                        partition=self.current_partition,
                                        curvature_map=curvature_map,
                                        block_curvatures=block_curvatures,
                                        block_intervals=frequency_plan.block_intervals,
                                        step_idx=step,
                                        output_dir=results_dir if results_dir else './results',
                                    )
                                    logging.info(
                                        '[wan-partition-vis] step=%d frame_strip_frames=%s summary=%s',
                                        step,
                                        partition_vis_info['frame_strip_frames'],
                                        partition_vis_info['summary_path'],
                                    )
                                    logging.info(
                                        '[wan-partition-vis] step=%d visualizations exported',
                                        step,
                                    )
                            elif is_cruise_anchor:
                                self.current_anchor_gap = 1
                                self.next_anchor_step = step + self.current_anchor_gap

                            if eval_straightness:
                                velocity_fn = self._make_velocity_fn(
                                    arg_c=arg_c,
                                    arg_null=arg_null,
                                    guide_scale=guide_scale,
                                )
                                straightness_record = self.straightness.update(
                                    x=latents[0].unsqueeze(0),
                                    v=noise_pred.unsqueeze(0),
                                    t=t_unit,
                                    step_idx=step,
                                    velocity_fn=velocity_fn,
                                )
                                if self.rank == 0 and straightness_record is not None:
                                    logging.info(
                                        '[wan-straightness] step=%d corr=%.6f mean_ratio=%.6f mean_gt=%.6e mean_proxy=%.6e',
                                        step,
                                        straightness_record.corr,
                                        straightness_record.mean_ratio,
                                        straightness_record.mean_curvature_gt,
                                        straightness_record.mean_proxy_curvature,
                                    )

                            temp_x0 = sample_scheduler.step(
                                noise_pred.unsqueeze(0),
                                t,
                                latents[0].unsqueeze(0),
                                return_dict=False,
                                generator=seed_g)[0]
                            latents = [temp_x0.squeeze(0)]
                    elif phase == 'phase3_cruise_selective':
                        with get_timer('selective_forward'):
                            noise_pred = self._predict_noise(
                                latent_model_input=latent_model_input,
                                timestep=timestep,
                                arg_c=arg_c,
                                arg_null=arg_null,
                                guide_scale=guide_scale,
                            )
                        self._maybe_record_error_probe(
                            step=step,
                            t=t,
                            latent_model_input=latent_model_input,
                            timestep=timestep,
                            arg_c=arg_c,
                            arg_null=arg_null,
                            guide_scale=guide_scale,
                            noise_pred=noise_pred,
                        )
                        temp_x0 = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latents[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                        latents = [temp_x0.squeeze(0)]
                    else:
                        with get_timer('warmup_cooldown_step'):
                            noise_pred = self._predict_noise(
                                latent_model_input=latent_model_input,
                                timestep=timestep,
                                arg_c=arg_c,
                                arg_null=arg_null,
                                guide_scale=guide_scale,
                            )
                            temp_x0 = sample_scheduler.step(
                                noise_pred.unsqueeze(0),
                                t,
                                latents[0].unsqueeze(0),
                                return_dict=False,
                                generator=seed_g)[0]
                            latents = [temp_x0.squeeze(0)]

                    self._record_step_trace(
                        step=step,
                        phase=phase,
                        total_steps=len(timesteps),
                        latency_ms=self._stop_step_timer(step_start),
                        fallback_total_tokens=self._infer_total_tokens_from_shape(target_shape),
                    )
                    self._advance_block_ages()

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        if self.rank == 0 and results_dir:
            if enable_visualization:
                self.visualizer.export_curvature_history(self.monitor.curvature_history, results_dir)
                logging.info('[wan-monitor] curvature visualizations saved to %s', results_dir)

            if eval_straightness and enable_visualization:
                self.visualizer.export_straightness_records(self.straightness.records, results_dir)
                logging.info('[wan-straightness] overview saved to %s', results_dir)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        self._write_error_probe_records()

        return videos[0] if self.rank == 0 else None

    def get_step_trace(self):
        return [dict(record) for record in self.step_trace]

    def get_partition_trace(self):
        return [dict(record) for record in self.partition_trace]

    def get_error_probe_records(self):
        return [dict(record) for record in self.error_probe_records]

    @staticmethod
    def _start_step_timer():
        torch.cuda.synchronize()
        return time.perf_counter()

    @staticmethod
    def _stop_step_timer(start_time):
        torch.cuda.synchronize()
        return (time.perf_counter() - start_time) * 1000.0

    def _infer_total_tokens_from_shape(self, target_shape):
        pt, ph, pw = self.patch_size
        return int(
            (target_shape[1] // pt)
            * (target_shape[2] // ph)
            * (target_shape[3] // pw)
        )

    def _record_step_trace(self, step, phase, total_steps, latency_ms, fallback_total_tokens):
        active_tokens, total_tokens, full_compute = self._get_step_activity_counts(fallback_total_tokens)
        self.step_trace.append({
            'step': int(step),
            'step_text': f'{int(step):02d}/{int(total_steps) - 1:02d}',
            'phase': str(phase),
            'active_tokens': int(active_tokens),
            'total_tokens': int(total_tokens),
            'active_ratio': float(active_tokens / max(total_tokens, 1)),
            'full_compute': bool(full_compute),
            'anchor_gap': int(self.current_anchor_gap),
            'next_anchor_step': None if self.next_anchor_step is None else int(self.next_anchor_step),
            'latency_ms': round(float(latency_ms), 4),
            'text': (
                f'step {int(step):02d}: '
                f'active {int(active_tokens)}/{int(total_tokens)} '
                f'latency {float(latency_ms):.4f} ms '
                f'phase={phase} '
                f'anchor_gap={int(self.current_anchor_gap)} '
                f'next_anchor={self.next_anchor_step}'
            ),
        })

    def _advance_block_ages(self):
        if self.current_partition is None or self.current_step_plan is None:
            return
        active_block_ids = set(int(block_id) for block_id in self.current_step_plan.active_block_ids)
        is_full_compute = bool(self.current_step_plan.is_full_compute)
        for block in self.current_partition:
            block_id = int(block.block_id)
            if is_full_compute or block_id in active_block_ids:
                self._block_age[block_id] = 0
            else:
                self._block_age[block_id] = int(self._block_age.get(block_id, 0)) + 1

    def _maybe_record_error_probe(
        self,
        step,
        t,
        latent_model_input,
        timestep,
        arg_c,
        arg_null,
        guide_scale,
        noise_pred,
    ):
        if not self.error_probe_enabled:
            return
        if self.rank != 0:
            return
        if self.current_partition is None or self.current_frequency_plan is None or self.current_step_plan is None:
            return
        if self.current_step_plan.is_full_compute:
            return
        if self.error_probe_max_steps > 0 and self._error_probe_steps_done >= self.error_probe_max_steps:
            return

        oracle_pred = self._predict_noise_oracle(
            latent_model_input=latent_model_input,
            timestep=timestep,
            arg_c=arg_c,
            arg_null=arg_null,
            guide_scale=guide_scale,
            cache_suffix=f'wan_error_probe_{int(step)}',
        )
        self._error_probe_steps_done += 1

        records = self._build_error_probe_records(
            step=step,
            t=t,
            selective_pred=noise_pred,
            oracle_pred=oracle_pred,
        )
        self.error_probe_records.extend(records)
        logging.info(
            '[wan-error-probe] step=%d records=%d total_records=%d',
            int(step),
            len(records),
            len(self.error_probe_records),
        )

    def _predict_noise_oracle(self, latent_model_input, timestep, arg_c, arg_null, guide_scale, cache_suffix: str):
        cond_key = f'{cache_suffix}_cond'
        uncond_key = f'{cache_suffix}_uncond'
        try:
            noise_pred_cond = self._forward_velocity_with_branch(
                latent_model_input,
                timestep,
                arg_c,
                branch_key=cond_key,
                step_plan=None,
            )[0]
            noise_pred_uncond = self._forward_velocity_with_branch(
                latent_model_input,
                timestep,
                arg_null,
                branch_key=uncond_key,
                step_plan=None,
            )[0]
            return noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
        finally:
            self.selective_forward.drop_cache(cond_key)
            self.selective_forward.drop_cache(uncond_key)

    def _build_error_probe_records(self, step, t, selective_pred, oracle_pred):
        if self.current_partition is None or self.current_frequency_plan is None or self.current_step_plan is None:
            return []

        diff = (selective_pred - oracle_pred).detach().float()
        if diff.dim() == 4:
            diff = diff.unsqueeze(0)
        token_error = self._latent_error_to_token_error(diff[0])
        t_unit = self._normalize_t_to_unit_interval(t)
        dt = self._normalized_step_size(step)
        active_block_ids = set(int(block_id) for block_id in self.current_step_plan.active_block_ids)
        token_indices_by_block = self.current_partition.get_all_token_indices()
        records = []
        for block in self.current_partition:
            block_id = int(block.block_id)
            indices = token_indices_by_block.get(block_id)
            if indices is None or indices.numel() == 0:
                continue
            block_error = token_error.index_select(0, indices.to(token_error.device))
            is_active = block_id in active_block_ids or bool(self.current_step_plan.is_full_compute)
            age_before = int(self._block_age.get(block_id, 0))
            age = 0 if is_active else age_before + 1
            records.append({
                'step': int(step),
                't': float(t_unit),
                'dt': float(dt),
                'block_id': block_id,
                'token_count': int(indices.numel()),
                'is_active': bool(is_active),
                'age_before_step': age_before,
                'age': int(age),
                'age_dt': float(age * dt),
                'tau': int(self.current_frequency_plan.block_intervals.get(block_id, 1)),
                'kappa': float(self._last_block_curvatures.get(block_id, 0.0)),
                'l2_mean': float(block_error.mean().item()),
                'l2_max': float(block_error.max().item()),
                'rmse': float(torch.sqrt(torch.mean(block_error * block_error)).item()),
                'phase': str(self.current_step_plan.phase.value),
            })
        return records

    def _latent_error_to_token_error(self, diff):
        # diff: [C, F, H, W]. Aggregate latent voxels into token-space blocks
        # following the model patch size.
        if diff.dim() != 4:
            raise ValueError(f'Expected latent diff shape [C,F,H,W], got {tuple(diff.shape)}')
        _, f_size, h_size, w_size = diff.shape
        pt, ph, pw = [int(v) for v in self.patch_size]
        Ft = f_size // pt
        Ht = h_size // ph
        Wt = w_size // pw
        trimmed = diff[:, :Ft * pt, :Ht * ph, :Wt * pw]
        vox_error = torch.sqrt(torch.sum(trimmed * trimmed, dim=0))
        token_error = vox_error.reshape(Ft, pt, Ht, ph, Wt, pw).mean(dim=(1, 3, 5))
        return token_error.reshape(-1)

    def _build_normalized_dt_by_step(self, timesteps):
        values = [self._normalize_t_to_unit_interval(t) for t in timesteps]
        if not values:
            return []
        dts = []
        for idx, value in enumerate(values):
            if idx + 1 < len(values):
                dts.append(abs(float(values[idx + 1]) - float(value)))
            elif idx > 0:
                dts.append(abs(float(value) - float(values[idx - 1])))
            else:
                dts.append(1.0)
        fallback = 1.0 / max(1, len(values) - 1)
        return [float(dt if dt > 0 else fallback) for dt in dts]

    def _normalized_step_size(self, step=None):
        if step is not None and 0 <= int(step) < len(self._normalized_dt_by_step):
            return float(self._normalized_dt_by_step[int(step)])
        return 1.0 / max(1, int(getattr(self, '_current_total_steps', 0)) - 1)

    def _write_error_probe_records(self):
        if self.rank != 0:
            return
        if not self.error_probe_output_path or not self.error_probe_records:
            return
        output_path = str(self.error_probe_output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in self.error_probe_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logging.info('[wan-error-probe] wrote %d records to %s', len(self.error_probe_records), output_path)

    def _get_step_activity_counts(self, fallback_total_tokens):
        if self.current_step_plan is not None:
            active_tokens = int(self.current_step_plan.active_token_indices.numel())
            if self.current_step_plan.is_full_compute:
                total_tokens = (
                    int(self.current_partition.total_tokens)
                    if self.current_partition is not None
                    else int(max(fallback_total_tokens, active_tokens))
                )
                return total_tokens, total_tokens, True

            frozen_tokens = int(self.current_step_plan.frozen_token_indices.numel())
            total_tokens = active_tokens + frozen_tokens
            if total_tokens <= 0:
                total_tokens = int(fallback_total_tokens)
            return active_tokens, total_tokens, False

        total_tokens = int(fallback_total_tokens)
        return total_tokens, total_tokens, True

    def _normalize_t_to_unit_interval(self, t):
        """Normalize scheduler timestep to [0, 1] expected by monitor equations."""
        if isinstance(t, torch.Tensor):
            if t.numel() == 0:
                return 0.0
            t_val = float(t.detach().reshape(-1)[0].item())
        else:
            t_val = float(t)

        if t_val > 1.0:
            denom = max(1.0, float(self.num_train_timesteps - 1))
            t_val = t_val / denom
        return max(0.0, min(1.0, t_val))

    def _unit_to_model_timestep(self, t_unit):
        t_unit = max(0.0, min(1.0, float(t_unit)))
        return t_unit * float(self.num_train_timesteps - 1)

    @staticmethod
    def _phase_to_enum(phase_name: str) -> Phase:
        if phase_name == 'phase1_warmup':
            return Phase.WARMUP
        if phase_name == 'phase2_cruise_anchor':
            return Phase.CRUISE_ANCHOR
        if phase_name == 'phase3_cruise_selective':
            return Phase.CRUISE_SELECTIVE
        return Phase.COOLDOWN

    def _make_velocity_fn(self, arg_c, arg_null, guide_scale):
        def velocity_fn(x_query, t_unit_query):
            t_model = self._unit_to_model_timestep(t_unit_query)
            timestep = torch.tensor([t_model], dtype=torch.float32, device=self.device)

            latent_model_input = [x_query[0]]
            noise_pred_cond = self._forward_velocity(
                latent_model_input,
                timestep,
                arg_c,
                branch_key='cond',
            )[0]
            noise_pred_uncond = self._forward_velocity(
                latent_model_input,
                timestep,
                arg_null,
                branch_key='uncond',
            )[0]
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
            return noise_pred.unsqueeze(0)

        return velocity_fn

    def _predict_noise(self, latent_model_input, timestep, arg_c, arg_null, guide_scale):
        if get_cp_worldsize() == 2:
            cp_group = get_cp_group()
            arg = arg_c if cp_group.rank_in_group == 0 else arg_null
            noise_pred_cp = self._forward_velocity(
                latent_model_input,
                timestep,
                arg,
                branch_key='cond' if cp_group.rank_in_group == 0 else 'uncond',
            )[0]
            noise_pred_gather = cp_group.all_gather(noise_pred_cp, dim=0, split=True)
            noise_pred_cond = noise_pred_gather[0]
            noise_pred_uncond = noise_pred_gather[1]
        else:
            noise_pred_cond = self._forward_velocity(
                latent_model_input,
                timestep,
                arg_c,
                branch_key='cond',
            )[0]
            noise_pred_uncond = self._forward_velocity(
                latent_model_input,
                timestep,
                arg_null,
                branch_key='uncond',
            )[0]

        return noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

    def _forward_velocity(self, latent_model_input, timestep, arg_dict, branch_key: str):
        return self._forward_velocity_with_branch(
            latent_model_input=latent_model_input,
            timestep=timestep,
            arg_dict=arg_dict,
            branch_key=branch_key,
            step_plan=self.current_step_plan,
        )

    def _forward_velocity_with_branch(self, latent_model_input, timestep, arg_dict, branch_key: str, step_plan):
        return self.selective_forward.forward(
            model=self.model,
            x=latent_model_input,
            t=timestep,
            context=arg_dict['context'],
            seq_len=arg_dict['seq_len'],
            step_plan=step_plan,
            branch_key=branch_key,
        )
