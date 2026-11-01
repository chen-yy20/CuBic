from __future__ import annotations

import time
import json
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from flux.pipeline import FluxPipeline, calculate_shift, retrieve_timesteps
from cubic_core import (
    AdaptivePartitioner,
    CurvatureMonitor,
    CubicVisualizer,
    Phase,
    SelectiveStepScheduler,
    UpdateFrequencyOptimizer,
)
from utils.timer import get_timer

from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from flux.transformer import FluxTransformer2DModel
from flux_cubic.cubic_selective_forward import FluxSelectiveForwardEngine


class CubicFluxPipeline(FluxPipeline):
    """Flux pipeline with Cubic stage classification + anchor monitoring."""

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
        )
        self.phase_warmup_steps = 0
        self.phase_cooldown_steps = 1

        self.target_speedup = 2.0
        self.use_target_speedup = False
        self.curvature_error_tolerance = 2e-4
        self.alpha = 3.0
        self.beta = 4.0
        self.tau_min = 1
        self.tau_max = 10

        # Curvature-only scheduling should avoid collapsing most Flux blocks to tau=1.
        self.curvature_contrast_gamma = 0.5
        self.min_anchor_gap = 8
        self.max_anchor_gap = 16
        self.anchor_gap_multiplier = 2
        self.anchor_policy = "dynamic"
        self.fixed_anchor_gap = 0

        self.min_block_size = 2
        self.max_block_size = 16
        self.min_block_tokens = 16
        self.max_block_tokens = 256
        self.cv_threshold = 0.5
        self.max_blocksize_candidates = 36
        self.partition_mode = "uniform"
        self.uniform_common_factor_blocks = True
        self.uniform_grid_h = 0
        self.uniform_grid_w = 0

        self.monitor = CurvatureMonitor(self)
        self.visualizer = CubicVisualizer(self)
        self.partitioner = AdaptivePartitioner(self)
        self.frequency_optimizer = UpdateFrequencyOptimizer(self)
        self.step_scheduler = SelectiveStepScheduler(self)
        self.selective_forward = FluxSelectiveForwardEngine(self)
        self.current_partition = None
        self.current_frequency_plan = None
        self.current_step_plan = None
        self.next_anchor_step = None
        self.current_anchor_gap = 0
        self.step_trace: list[dict[str, Any]] = []
        self.error_probe_enabled = False
        self.error_probe_max_steps = 0
        self.error_probe_output_path: Optional[str] = None
        self.error_probe_records: list[dict[str, Any]] = []
        self._error_probe_steps_done = 0
        self._block_age: dict[int, int] = {}
        self.anchor_curvature_drift: list[dict[str, Any]] = []
        self._previous_anchor_curvature: Optional[torch.Tensor] = None
        self._previous_anchor_step: Optional[int] = None

    def _set_next_anchor_step(self, step: int, total_steps: int, frequency_plan=None) -> None:
        if frequency_plan is None:
            self.current_anchor_gap = 1
            self.next_anchor_step = int(step) + 1
            return

        max_interval = max(1, int(frequency_plan.max_interval))
        if self.anchor_policy == "fixed":
            gap = max(1, int(self.fixed_anchor_gap))
        else:
            raw_gap = max(1, int(round(float(self.anchor_gap_multiplier) * max_interval)))
            min_gap = max(1, int(self.min_anchor_gap))
            max_gap = max(min_gap, int(self.max_anchor_gap))
            gap = min(max(raw_gap, min_gap), max_gap)
        self.current_anchor_gap = int(gap)

        cooldown_steps = max(0, int(self.phase_cooldown_steps))
        cooldown_start = max(0, int(total_steps) - cooldown_steps)
        next_anchor = int(step) + int(gap)
        if cooldown_steps > 0:
            next_anchor = min(next_anchor, cooldown_start)
        self.next_anchor_step = next_anchor if next_anchor < int(total_steps) else None

    def _get_inference_phase(self, step: int, total_steps: int) -> str:
        if total_steps <= 0:
            return "phase4_cooldown"

        warmup_steps, cooldown_steps = self._resolve_phase_steps(total_steps)
        cooldown_start = max(0, total_steps - cooldown_steps)

        if step < warmup_steps:
            return "phase1_warmup"
        if step >= cooldown_start:
            return "phase4_cooldown"

        if self.next_anchor_step is None:
            self.next_anchor_step = warmup_steps

        if step == self.next_anchor_step:
            return "phase2_cruise_anchor"
        return "phase3_cruise_selective"

    def _resolve_phase_steps(self, total_steps: int) -> tuple[int, int]:
        total_steps = max(0, int(total_steps))
        warmup_steps = max(0, int(self.phase_warmup_steps))
        cooldown_steps = max(0, int(self.phase_cooldown_steps))
        if warmup_steps + cooldown_steps > total_steps:
            cooldown_steps = max(0, total_steps - warmup_steps)
        return warmup_steps, cooldown_steps

    def _normalize_t_to_unit_interval(self, t: torch.Tensor | float) -> float:
        if isinstance(t, torch.Tensor):
            if t.numel() == 0:
                return 0.0
            t_val = float(t.detach().reshape(-1)[0].item())
        else:
            t_val = float(t)

        if t_val > 1.0:
            scheduler_steps = float(getattr(self.scheduler.config, "num_train_timesteps", 1000) - 1)
            t_val = t_val / max(1.0, scheduler_steps)
        return max(0.0, min(1.0, t_val))

    def _estimate_normalized_step_delta(self, timesteps: torch.Tensor, step_idx: int) -> float:
        if len(timesteps) <= 1:
            return 1.0

        idx = max(0, min(int(step_idx), len(timesteps) - 1))
        other_idx = idx - 1 if idx > 0 else 1
        t_cur = self._normalize_t_to_unit_interval(timesteps[idx])
        t_other = self._normalize_t_to_unit_interval(timesteps[other_idx])
        return max(abs(t_cur - t_other), 1e-8)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        cubic_warmup_steps: Optional[int] = None,
        cubic_cooldown_steps: Optional[int] = None,
        cubic_warmup_fraction: Optional[float] = None,
        cubic_cooldown_fraction: Optional[float] = None,
        cubic_anchor_interval: Optional[int] = None,
        cubic_anchor_policy: Optional[str] = None,
        cubic_anchor_gap_multiplier: Optional[float] = None,
        cubic_min_anchor_gap: Optional[int] = None,
        cubic_max_anchor_gap: Optional[int] = None,
        cubic_target_speedup: Optional[float] = None,
        cubic_curvature_error_tolerance: Optional[float] = None,
        cubic_alpha: Optional[float] = None,
        cubic_beta: Optional[float] = None,
        cubic_partition_mode: Optional[str] = None,
        cubic_uniform_grid_h: Optional[int] = None,
        cubic_uniform_grid_w: Optional[int] = None,
        cubic_uniform_common_factor_blocks: Optional[bool] = None,
        cubic_min_block_size: Optional[int] = None,
        cubic_max_block_size: Optional[int] = None,
        cubic_cv_threshold: Optional[float] = None,
        cubic_error_probe: bool = False,
        cubic_error_probe_max_steps: int = 0,
        cubic_error_probe_output_path: Optional[str] = None,
        cubic_enable_visualization: bool = True,
        cubic_results_dir: Optional[str] = None,
        cubic_visual_tag: Optional[str] = None,
    ):
        if cubic_warmup_steps is not None:
            self.phase_warmup_steps = int(cubic_warmup_steps)
        elif cubic_warmup_fraction is not None:
            self.phase_warmup_steps = int(num_inference_steps * float(cubic_warmup_fraction))
        if cubic_cooldown_steps is not None:
            self.phase_cooldown_steps = int(cubic_cooldown_steps)
        elif cubic_cooldown_fraction is not None:
            self.phase_cooldown_steps = int(num_inference_steps * float(cubic_cooldown_fraction))
        if cubic_anchor_policy is not None:
            policy = str(cubic_anchor_policy).lower()
            if policy not in {"dynamic", "fixed"}:
                raise ValueError(f"Unsupported Cubic anchor policy: {cubic_anchor_policy}")
            self.anchor_policy = policy
        if cubic_anchor_gap_multiplier is not None:
            self.anchor_gap_multiplier = float(cubic_anchor_gap_multiplier)
        if cubic_min_anchor_gap is not None:
            self.min_anchor_gap = int(cubic_min_anchor_gap)
        if cubic_max_anchor_gap is not None:
            self.max_anchor_gap = int(cubic_max_anchor_gap)
        if cubic_anchor_interval is not None:
            self.fixed_anchor_gap = max(1, int(cubic_anchor_interval))
            self.anchor_policy = "fixed"
        if cubic_target_speedup is not None:
            self.target_speedup = float(cubic_target_speedup)
        if cubic_curvature_error_tolerance is not None:
            self.curvature_error_tolerance = float(cubic_curvature_error_tolerance)
        if cubic_alpha is not None:
            self.alpha = float(cubic_alpha)
        if cubic_beta is not None:
            self.beta = float(cubic_beta)
        if cubic_partition_mode is not None:
            self.partition_mode = str(cubic_partition_mode)
        if cubic_uniform_grid_h is not None:
            self.uniform_grid_h = int(cubic_uniform_grid_h)
        if cubic_uniform_grid_w is not None:
            self.uniform_grid_w = int(cubic_uniform_grid_w)
        if cubic_uniform_common_factor_blocks is not None:
            self.uniform_common_factor_blocks = bool(cubic_uniform_common_factor_blocks)
        if cubic_min_block_size is not None:
            self.min_block_size = int(cubic_min_block_size)
        if cubic_max_block_size is not None:
            self.max_block_size = int(cubic_max_block_size)
        if cubic_cv_threshold is not None:
            self.cv_threshold = float(cubic_cv_threshold)

        self._sync_cubic_modules_config()

        self.monitor.reset()
        self.selective_forward.reset()
        self.current_partition = None
        self.current_frequency_plan = None
        self.current_step_plan = None
        self.next_anchor_step = None
        self.current_anchor_gap = 0
        self.step_trace = []
        self.error_probe_enabled = bool(cubic_error_probe)
        self.error_probe_max_steps = max(0, int(cubic_error_probe_max_steps))
        self.error_probe_output_path = cubic_error_probe_output_path
        self.error_probe_records = []
        self._error_probe_steps_done = 0
        self._block_age = {}
        self.anchor_curvature_drift = []
        self._previous_anchor_curvature = None
        self._previous_anchor_step = None

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        warmup_steps, cooldown_steps = self._resolve_phase_steps(len(timesteps))

        with get_timer("generate_e2e"):
            pbar = tqdm(
                timesteps,
                desc="Denoising",
                dynamic_ncols=True,
                position=0,
                leave=True,
            )
            for i, t in enumerate(pbar):
                if self.interrupt:
                    continue

                phase = self._get_inference_phase(i, len(timesteps))
                phase_enum = self._phase_to_enum(phase)

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                step_start = self._start_step_timer()

                if self.current_partition is not None and self.current_frequency_plan is not None:
                    selective_step_idx = max(0, i - warmup_steps)
                    self.current_step_plan = self.step_scheduler.build_step_plan(
                        partition=self.current_partition,
                        block_intervals=self.current_frequency_plan.block_intervals,
                        phase=phase_enum,
                        step_idx=i,
                        selective_step_idx=selective_step_idx,
                    )

                is_last_warmup = warmup_steps > 0 and phase == "phase1_warmup" and i == warmup_steps - 1
                is_cruise_anchor = phase == "phase2_cruise_anchor"
                if is_last_warmup or is_cruise_anchor:
                    with get_timer("anchor_step_total"):
                        noise_pred = self.selective_forward.forward(
                            transformer=self.transformer,
                            hidden_states=latents,
                            timestep=timestep,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            step_plan=None,
                            cache_key="flux",
                        )

                        with get_timer("anchor_step_monitor"):
                            latents_unpacked = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                            noise_unpacked = self._unpack_latents(noise_pred, height, width, self.vae_scale_factor)
                            curvature_map = self.monitor.update(
                                x=latents_unpacked.unsqueeze(2),
                                v=noise_unpacked.unsqueeze(2),
                                t=self._normalize_t_to_unit_interval(t),
                                step_idx=i,
                            )

                        if curvature_map is not None:
                            self._record_anchor_curvature_drift(curvature_map, step=i)
                            with get_timer("anchor_step_partitioner"):
                                ft, ht, wt = curvature_map.shape
                                partition = self.partitioner.partition(
                                    Ft=ft,
                                    Ht=ht,
                                    Wt=wt,
                                    curvature_map=curvature_map,
                                )
                                partition_debug = self.partitioner.describe_partition(partition, curvature_map)
                                print(
                                    "[flux-partitioner] "
                                    f"step={i} chosen_block_size={partition_debug.chosen_block_size} "
                                    f"blocks={partition_debug.block_count} "
                                    f"token_grid=({ft},{ht},{wt}) "
                                    f"token_stats(min={partition_debug.block_token_min} "
                                    f"max={partition_debug.block_token_max} "
                                    f"mean={partition_debug.block_token_mean:.2f}) "
                                    f"block_cv(mean={partition_debug.mean_block_cv:.4f} "
                                    f"p90={partition_debug.p90_block_cv:.4f})",
                                    flush=True,
                                )
                                block_curvatures = self.monitor.get_block_curvatures(
                                    partition,
                                    curvature_map=curvature_map,
                                )

                            with get_timer("anchor_step_optimizer"):
                                self.frequency_optimizer.time_step_size = self._estimate_normalized_step_delta(
                                    timesteps,
                                    i,
                                )
                                frequency_plan = self.frequency_optimizer.optimize(
                                    partition=partition,
                                    block_curvatures=block_curvatures,
                                )
                                self.current_partition = partition
                                self.current_frequency_plan = frequency_plan
                                self.current_step_plan = self.step_scheduler.build_step_plan(
                                    partition=partition,
                                    block_intervals=frequency_plan.block_intervals,
                                    phase=Phase.CRUISE_ANCHOR,
                                    step_idx=i,
                                    selective_step_idx=max(0, i - warmup_steps),
                                )
                                self._block_age = {block.block_id: 0 for block in partition}
                                self._set_next_anchor_step(
                                    step=i,
                                    total_steps=len(timesteps),
                                    frequency_plan=frequency_plan,
                                )

                            if cubic_enable_visualization and cubic_results_dir:
                                self.visualizer.visualize_partition_and_frequency(
                                    partition=partition,
                                    curvature_map=curvature_map,
                                    block_curvatures=block_curvatures,
                                    block_intervals=frequency_plan.block_intervals,
                                    step_idx=i,
                                    output_dir=cubic_results_dir,
                                    file_suffix=cubic_visual_tag,
                                )
                        else:
                            self._set_next_anchor_step(step=i, total_steps=len(timesteps), frequency_plan=None)

                        latents_dtype = latents.dtype
                        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                elif phase == "phase3_cruise_selective":
                    with get_timer("selective_forward"):
                        noise_pred = self.selective_forward.forward(
                            transformer=self.transformer,
                            hidden_states=latents,
                            timestep=timestep,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            step_plan=self.current_step_plan,
                            cache_key="flux",
                        )
                        self._maybe_record_error_probe(
                            noise_pred=noise_pred,
                            transformer=self.transformer,
                            hidden_states=latents,
                            timestep=timestep,
                            guidance=guidance,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            prompt_embeds=prompt_embeds,
                            text_ids=text_ids,
                            latent_image_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            step_idx=i,
                            t=t,
                            timesteps=timesteps,
                        )

                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                else:
                    with get_timer("warmup_cooldown_step"):
                        noise_pred = self.selective_forward.forward(
                            transformer=self.transformer,
                            hidden_states=latents,
                            timestep=timestep,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            step_plan=None,
                            cache_key="flux",
                        )

                        latents_dtype = latents.dtype
                        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                latency_ms = self._stop_step_timer(step_start)
                self._record_step_trace(
                    step=i,
                    phase=phase,
                    total_steps=len(timesteps),
                    latency_ms=latency_ms,
                    fallback_total_tokens=int(image_seq_len),
                )
                self._advance_block_ages()

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                pbar.update(0)

            pbar.close()

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if cubic_enable_visualization and cubic_results_dir and self.monitor.curvature_history:
            self.visualizer.export_curvature_history(
                self.monitor.curvature_history,
                cubic_results_dir,
                file_suffix=cubic_visual_tag,
            )

        if self.error_probe_output_path and self.error_probe_records:
            self._write_error_probe_records(self.error_probe_output_path)

        if not return_dict:
            return (image,)

        from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

        return FluxPipelineOutput(images=image)

    def get_step_trace(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self.step_trace]

    def get_error_probe_records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self.error_probe_records]

    def get_anchor_curvature_drift(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self.anchor_curvature_drift]

    def _record_anchor_curvature_drift(self, curvature_map: torch.Tensor, step: int) -> None:
        current = curvature_map.detach().float().reshape(-1)
        previous = self._previous_anchor_curvature
        previous_step = self._previous_anchor_step
        if previous is not None and previous.shape == current.shape and previous_step is not None:
            previous = previous.to(device=current.device)
            eps = 1e-8
            normalized_l1 = (
                torch.mean(torch.abs(current - previous))
                / torch.clamp(torch.mean(torch.abs(previous)), min=eps)
            )
            current_rank = torch.empty_like(current)
            previous_rank = torch.empty_like(previous)
            current_rank[torch.argsort(current)] = torch.arange(
                current.numel(), device=current.device, dtype=current.dtype
            )
            previous_rank[torch.argsort(previous)] = torch.arange(
                previous.numel(), device=previous.device, dtype=previous.dtype
            )
            current_centered = current_rank - current_rank.mean()
            previous_centered = previous_rank - previous_rank.mean()
            rank_denominator = torch.sqrt(
                torch.sum(current_centered.square()) * torch.sum(previous_centered.square())
            )
            rank_correlation = torch.sum(current_centered * previous_centered) / torch.clamp(
                rank_denominator, min=eps
            )
            self.anchor_curvature_drift.append(
                {
                    "previous_step": int(previous_step),
                    "step": int(step),
                    "observed_anchor_gap": int(step - previous_step),
                    "normalized_l1_drift": float(normalized_l1.item()),
                    "rank_correlation": float(rank_correlation.item()),
                    "token_count": int(current.numel()),
                }
            )
        self._previous_anchor_curvature = current.detach().cpu()
        self._previous_anchor_step = int(step)

    @staticmethod
    def _start_step_timer() -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    @staticmethod
    def _stop_step_timer(start_time: float) -> float:
        torch.cuda.synchronize()
        return (time.perf_counter() - start_time) * 1000.0

    def _record_step_trace(
        self,
        step: int,
        phase: str,
        total_steps: int,
        latency_ms: float,
        fallback_total_tokens: int,
    ) -> None:
        active_tokens, total_tokens, full_compute = self._get_step_activity_counts(fallback_total_tokens)
        self.step_trace.append(
            {
                "step": int(step),
                "step_text": f"{int(step):02d}/{int(total_steps) - 1:02d}",
                "phase": str(phase),
                "active_tokens": int(active_tokens),
                "total_tokens": int(total_tokens),
                "active_ratio": float(active_tokens / max(total_tokens, 1)),
                "full_compute": bool(full_compute),
                "anchor_gap": int(self.current_anchor_gap),
                "next_anchor_step": None if self.next_anchor_step is None else int(self.next_anchor_step),
                "latency_ms": round(float(latency_ms), 4),
                "text": (
                    f"step {int(step):02d}: "
                    f"active {int(active_tokens)}/{int(total_tokens)} "
                    f"latency {float(latency_ms):.4f} ms "
                    f"phase={phase} "
                    f"anchor_gap={int(self.current_anchor_gap)} "
                    f"next_anchor={self.next_anchor_step}"
                ),
            }
        )

    def _get_step_activity_counts(self, fallback_total_tokens: int) -> tuple[int, int, bool]:
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

    def _advance_block_ages(self) -> None:
        if self.current_partition is None or self.current_step_plan is None:
            return
        active = set(int(block_id) for block_id in self.current_step_plan.active_block_ids)
        ages = {}
        for block in self.current_partition:
            block_id = int(block.block_id)
            if self.current_step_plan.is_full_compute or block_id in active:
                ages[block_id] = 0
            else:
                ages[block_id] = int(self._block_age.get(block_id, 0)) + 1
        self._block_age = ages

    def _maybe_record_error_probe(
        self,
        noise_pred: torch.Tensor,
        transformer,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor],
        pooled_prompt_embeds: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        joint_attention_kwargs,
        step_idx: int,
        t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> None:
        if not self.error_probe_enabled:
            return
        if self.current_partition is None or self.current_frequency_plan is None or self.current_step_plan is None:
            return
        if self.current_step_plan.is_full_compute:
            return
        if self.error_probe_max_steps > 0 and self._error_probe_steps_done >= self.error_probe_max_steps:
            return

        cache_key = f"flux_error_probe_oracle_{int(step_idx)}"
        oracle = self.selective_forward._full_forward_and_refresh_cache(
            transformer=transformer,
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            cache_key=cache_key,
        )
        self.selective_forward.drop_cache(cache_key)
        self._error_probe_steps_done += 1

        diff = (noise_pred.detach().float() - oracle.detach().float()).squeeze(0)
        dt = self._estimate_normalized_step_delta(timesteps, step_idx)
        t_unit = self._normalize_t_to_unit_interval(t)
        active_blocks = set(int(block_id) for block_id in self.current_step_plan.active_block_ids)
        block_intervals = self.current_frequency_plan.block_intervals
        block_curvatures = {}
        if self.monitor.curvature_map is not None:
            block_curvatures = self.monitor.get_block_curvatures(
                self.current_partition,
                curvature_map=self.monitor.curvature_map,
            )
        token_indices_by_block = self.current_partition.get_all_token_indices()
        for block in self.current_partition:
            block_id = int(block.block_id)
            token_idx = token_indices_by_block[block_id].to(diff.device)
            if token_idx.numel() == 0:
                continue
            block_diff = diff.index_select(0, token_idx)
            l2_per_token = torch.linalg.vector_norm(block_diff, ord=2, dim=-1)
            rmse = torch.sqrt(torch.mean(block_diff * block_diff))
            age_before = int(self._block_age.get(block_id, 0))
            own_output_age = 0 if block_id in active_blocks else age_before + 1
            self.error_probe_records.append(
                {
                    "step": int(step_idx),
                    "t": float(t_unit),
                    "dt": float(dt),
                    "block_id": block_id,
                    "token_count": int(block.token_count),
                    "is_active": bool(block_id in active_blocks),
                    "age_before_step": age_before,
                    "age": own_output_age,
                    "age_dt": float(own_output_age * dt),
                    "tau": int(block_intervals.get(block_id, 1)),
                    "kappa": float(block_curvatures.get(block_id, 0.0)),
                    "l2_mean": float(l2_per_token.mean().item()),
                    "l2_max": float(l2_per_token.max().item()),
                    "rmse": float(rmse.item()),
                    "phase": "phase3_cruise_selective",
                }
            )

    def _write_error_probe_records(self, path: str) -> None:
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in self.error_probe_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _phase_to_enum(phase_name: str) -> Phase:
        if phase_name == "phase1_warmup":
            return Phase.WARMUP
        if phase_name == "phase2_cruise_anchor":
            return Phase.CRUISE_ANCHOR
        if phase_name == "phase3_cruise_selective":
            return Phase.CRUISE_SELECTIVE
        return Phase.COOLDOWN

    def _sync_cubic_modules_config(self) -> None:
        # Partitioner configs
        self.partitioner.min_block_size = int(self.min_block_size)
        self.partitioner.max_block_size = int(self.max_block_size)
        self.partitioner.min_block_tokens = int(self.min_block_tokens)
        self.partitioner.max_block_tokens = int(self.max_block_tokens)
        self.partitioner.cv_threshold = float(self.cv_threshold)
        self.partitioner.max_candidates = int(self.max_blocksize_candidates)
        self.partitioner.partition_mode = str(self.partition_mode).lower()
        self.partitioner.uniform_common_factor_blocks = bool(self.uniform_common_factor_blocks)
        self.partitioner.uniform_grid_h = int(self.uniform_grid_h)
        self.partitioner.uniform_grid_w = int(self.uniform_grid_w)

        # Optimizer configs
        self.frequency_optimizer.target_speedup = float(self.target_speedup)
        self.frequency_optimizer.use_target_speedup = bool(self.use_target_speedup)
        self.frequency_optimizer.curvature_error_tolerance = float(self.curvature_error_tolerance)
        self.frequency_optimizer.alpha = float(self.alpha)
        self.frequency_optimizer.beta = float(self.beta)
        self.frequency_optimizer.tau_min = int(self.tau_min)
        self.frequency_optimizer.tau_max = int(self.tau_max)
        self.frequency_optimizer.curvature_contrast_gamma = float(self.curvature_contrast_gamma)
