"""Fixed-frequency PAB baseline for Flux1 attention reuse."""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any

import torch


def _same_structure_and_shape(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return all(_same_structure_and_shape(a, b) for a, b in zip(left, right))
    return False


def _detach_cache(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_cache(item) for item in value)
    return value


@dataclass
class FluxPABManager:
    """Runtime state for fixed-frequency attention-output reuse."""

    total_steps: int
    skip_range: int
    warmup_steps: int = 0
    cooldown_steps: int = 0
    layer_interval: int = 1
    total_layers: int = 0
    current_step: int = -1
    cache: dict[int, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    _current_step_row: dict[str, Any] | None = None

    def reset(self) -> None:
        self.current_step = -1
        self.cache.clear()
        self.trace = []
        self._current_step_row = None

    @property
    def target_layers(self) -> list[int]:
        interval = max(1, int(self.layer_interval))
        return [idx for idx in range(int(self.total_layers)) if idx % interval == 0]

    def is_target_layer(self, layer_id: int) -> bool:
        return int(layer_id) % max(1, int(self.layer_interval)) == 0

    def should_compute_step(self, step: int) -> bool:
        if step < max(0, int(self.warmup_steps)):
            return True
        if step >= int(self.total_steps) - max(0, int(self.cooldown_steps)):
            return True
        local_step = step - max(0, int(self.warmup_steps))
        return local_step % max(1, int(self.skip_range)) == 0

    def begin_step(self) -> None:
        self.current_step += 1
        should_compute = self.should_compute_step(self.current_step)
        self._current_step_row = {
            "step": int(self.current_step),
            "should_compute": bool(should_compute),
            "target_layers": int(len(self.target_layers)),
            "computed_layers": 0,
            "reused_layers": 0,
            "cache_miss_layers": 0,
            "non_target_layers": int(self.total_layers - len(self.target_layers)),
        }

    def end_step(self) -> None:
        if self._current_step_row is not None:
            self.trace.append(dict(self._current_step_row))
        self._current_step_row = None

    def call_attention(self, layer_id: int, original_forward, *args, **kwargs):
        if self._current_step_row is None:
            # Fallback for any direct attention call outside the transformer wrapper.
            return original_forward(*args, **kwargs)

        layer_id = int(layer_id)
        if not self.is_target_layer(layer_id):
            return original_forward(*args, **kwargs)

        should_compute = bool(self._current_step_row["should_compute"])
        cached = self.cache.get(layer_id)
        if not should_compute and cached is not None:
            self._current_step_row["reused_layers"] += 1
            return cached

        output = original_forward(*args, **kwargs)
        if cached is not None and not _same_structure_and_shape(output, cached):
            self._current_step_row["cache_miss_layers"] += 1
        self.cache[layer_id] = _detach_cache(output)
        self._current_step_row["computed_layers"] += 1
        return output

    def budget_ledger(self) -> dict[str, Any]:
        total_steps = len(self.trace)
        target_layers = max(1, len(self.target_layers))
        target_events = total_steps * target_layers
        computed_events = sum(int(row["computed_layers"]) for row in self.trace)
        reused_events = sum(int(row["reused_layers"]) for row in self.trace)
        cache_miss_events = sum(int(row["cache_miss_layers"]) for row in self.trace)
        full_compute_steps = sum(1 for row in self.trace if int(row["computed_layers"]) >= target_layers)
        full_reuse_steps = sum(1 for row in self.trace if int(row["reused_layers"]) >= target_layers)
        attention_compute_ratio = computed_events / max(target_events, 1)
        return {
            "method": "pab",
            "budget_kind": "layer_attention_proxy",
            "budget_precision": "exact_for_target_attention_layers_proxy_for_dit",
            "total_steps": int(total_steps),
            "total_layers": int(self.total_layers),
            "target_layers": int(target_layers),
            "layer_interval": int(self.layer_interval),
            "skip_range": int(self.skip_range),
            "warmup_steps": int(self.warmup_steps),
            "cooldown_steps": int(self.cooldown_steps),
            "full_compute_steps": int(full_compute_steps),
            "reuse_steps": int(full_reuse_steps),
            "target_attention_events": int(target_events),
            "computed_attention_events": int(computed_events),
            "reused_attention_events": int(reused_events),
            "cache_miss_attention_events": int(cache_miss_events),
            "mean_step_compute_ratio": float(attention_compute_ratio),
            "mean_layer_compute_ratio": float(attention_compute_ratio),
            "mean_token_compute_ratio": 1.0,
            "effective_attention_compute_ratio": float(attention_compute_ratio),
            "effective_dit_compute_ratio": float(attention_compute_ratio),
            "effective_dit_compute_ratio_note": (
                "PAB only reuses attention outputs; FFN/projections outside target attention still run. "
                "Treat this as an attention-layer budget proxy until FLOPs profiling is added."
            ),
            "trace": [dict(row) for row in self.trace],
        }


def resolve_skip_range_for_target(
    total_steps: int,
    target_compute_ratio: float,
    warmup_steps: int,
    cooldown_steps: int,
) -> int:
    """Pick the fixed skip interval whose step compute ratio is closest to target."""
    total_steps = max(1, int(total_steps))
    target = min(1.0, max(0.0, float(target_compute_ratio)))
    warmup_steps = max(0, int(warmup_steps))
    cooldown_steps = max(0, int(cooldown_steps))
    best_range = 1
    best_delta = float("inf")
    for skip_range in range(1, total_steps + 1):
        manager = FluxPABManager(
            total_steps=total_steps,
            skip_range=skip_range,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            total_layers=1,
        )
        compute_steps = sum(1 for step in range(total_steps) if manager.should_compute_step(step))
        ratio = compute_steps / total_steps
        delta = abs(ratio - target)
        if delta < best_delta:
            best_delta = delta
            best_range = skip_range
    return best_range


def attach_flux_pab(transformer, manager: FluxPABManager) -> None:
    """Patch a Flux transformer instance with PAB attention-output reuse."""
    if hasattr(transformer, "_pab_original_forward"):
        return

    attention_layers = []
    for block in transformer.transformer_blocks:
        attention_layers.append(block.attn)
    for block in transformer.single_transformer_blocks:
        attention_layers.append(block.attn)

    manager.total_layers = len(attention_layers)
    transformer._pab_manager = manager
    transformer._pab_original_forward = transformer.forward

    original_transformer_forward = transformer.forward

    def transformer_forward_with_pab(self, *args, **kwargs):
        self._pab_manager.begin_step()
        try:
            return original_transformer_forward(*args, **kwargs)
        finally:
            self._pab_manager.end_step()

    transformer.forward = types.MethodType(transformer_forward_with_pab, transformer)

    for layer_id, attention in enumerate(attention_layers):
        if hasattr(attention, "_pab_original_forward"):
            continue
        original_attention_forward = attention.forward
        attention._pab_original_forward = original_attention_forward
        attention._pab_layer_id = layer_id
        attention._pab_manager = manager

        def attention_forward_with_pab(self, *args, _original=original_attention_forward, **kwargs):
            return self._pab_manager.call_attention(self._pab_layer_id, _original, *args, **kwargs)

        attention.forward = types.MethodType(attention_forward_with_pab, attention)
