#!/usr/bin/env python3
"""Summarize controlled Task 1 audit runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Task 1 controlled audit")
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", required=True, choices=["flux", "wan"])
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_params(condition_dir: Path) -> Path | None:
    paths = sorted(condition_dir.glob("*_params_metrics.json"))
    return paths[0] if paths else None


def _get(obj: dict, path: tuple[str, ...]):
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isinf(value):
        return value
    return value if math.isfinite(value) else None


def _quality_mean(quality: dict[str, Any]) -> dict[str, Any]:
    mean = quality.get("mean") if isinstance(quality.get("mean"), dict) else quality
    return mean if isinstance(mean, dict) else {}


def _parse_stage_and_name(path: Path) -> tuple[str, str]:
    parent = path.parent.name
    if parent in {"phase", "granularity", "frontier"}:
        return parent, path.name
    return "unknown", path.name


def _method_from_name(name: str) -> str:
    if "tc_" in name or "teacache" in name:
        return "teacache"
    if "pab" in name:
        return "pab"
    if "cubic" in name:
        return "cubic"
    return "unknown"


def _extract_blocks(log_path: Path | None) -> tuple[int | None, str | None]:
    if log_path is None or not log_path.exists():
        return None, None
    pattern = re.compile(r"blocks=(\d+).*?chosen_block_size=\(([^)]*)\)|chosen_block_size=\(([^)]*)\).*?blocks=(\d+)")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "partitioner" not in line:
            continue
        match = pattern.search(line)
        if not match:
            continue
        if match.group(1):
            return int(match.group(1)), match.group(2)
        return int(match.group(4)), match.group(3)
    return None, None


def summarize_condition(condition_dir: Path) -> dict[str, Any]:
    stage, name = _parse_stage_and_name(condition_dir)
    params_path = find_params(condition_dir)
    log_candidates = sorted(condition_dir.glob("run*.log"))
    log_path = log_candidates[0] if log_candidates else None
    row = {
        "stage": stage,
        "condition": name,
        "method": _method_from_name(name),
        "status": "missing",
        "params_path": str(params_path) if params_path else None,
        "log_path": str(log_path) if log_path else None,
    }
    if params_path is None:
        return row

    payload = load_json(params_path)
    params = payload.get("parameters") or {}
    quality = _quality_mean(payload.get("quality_metrics") or {})
    ledger = params.get("cache_budget_ledger") or {}
    timing = payload.get("timing_stats") or {}
    block_count, block_size = _extract_blocks(log_path)
    method = params.get("model") and _method_from_name(name)
    row.update(
        {
            "status": "ok",
            "method": method or row["method"],
            "tag": payload.get("tag"),
            "warmup_steps": params.get("warmup_steps", params.get("pab_warmup_steps", params.get("pab_warmup"))),
            "cooldown_steps": params.get("cooldown_steps", params.get("pab_cooldown_steps", params.get("pab_cooldown"))),
            "partition_mode": params.get("partition_mode"),
            "uniform_grid_h": params.get("uniform_grid_h"),
            "uniform_grid_w": params.get("uniform_grid_w"),
            "uniform_common_factor_blocks": params.get("uniform_common_factor_blocks"),
            "min_block_size": params.get("min_block_size"),
            "max_block_size": params.get("max_block_size"),
            "cv_threshold": params.get("cv_threshold"),
            "curvature_error_tolerance": params.get("curvature_error_tolerance"),
            "pab_skip_range": params.get("pab_skip_range"),
            "pab_self_range": params.get("pab_self_range"),
            "pab_cross_range": params.get("pab_cross_range"),
            "teacache_thresh": params.get("rel_l1_thresh", params.get("teacache_thresh")),
            "effective_dit_compute_ratio": ledger.get("effective_dit_compute_ratio"),
            "mean_step_compute_ratio": ledger.get("mean_step_compute_ratio"),
            "full_compute_steps": ledger.get("full_compute_steps"),
            "reuse_steps": ledger.get("reuse_steps"),
            "budget_kind": ledger.get("budget_kind"),
            "psnr": quality.get("psnr"),
            "ssim": quality.get("ssim"),
            "lpips": quality.get("lpips"),
            "fid": quality.get("fid"),
            "fvd": quality.get("fvd"),
            "generate_e2e_avg_ms": _get(timing, ("generate_e2e", "avg_ms")),
            "model_forward_avg_ms": _get(timing, ("model_forward", "avg_ms")),
            "dit_avg_ms": _get(timing, ("dit", "avg_ms")) or _get(timing, ("dit_forward", "avg_ms")),
            "block_count": block_count,
            "chosen_block_size": block_size,
        }
    )
    return row


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage_dir in [root / "phase", root / "granularity", root / "frontier"]:
        if not stage_dir.exists():
            continue
        for condition_dir in sorted(path for path in stage_dir.iterdir() if path.is_dir()):
            rows.append(summarize_condition(condition_dir))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "stage", "condition", "method", "status", "warmup_steps", "cooldown_steps",
        "partition_mode", "uniform_grid_h", "uniform_grid_w", "block_count", "chosen_block_size",
        "curvature_error_tolerance", "teacache_thresh", "pab_skip_range", "pab_self_range", "pab_cross_range",
        "effective_dit_compute_ratio", "full_compute_steps", "reuse_steps",
        "psnr", "ssim", "lpips", "fid", "fvd", "generate_e2e_avg_ms", "dit_avg_ms",
        "budget_kind", "params_path", "log_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_stage(rows: list[dict[str, Any]], stage: str, output_path: Path, title: str) -> None:
    stage_rows = [row for row in rows if row.get("stage") == stage and row.get("status") == "ok"]
    if not stage_rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    specs = [
        ("effective_dit_compute_ratio", "ssim", "SSIM"),
        ("effective_dit_compute_ratio", "lpips", "LPIPS"),
        ("effective_dit_compute_ratio", "generate_e2e_avg_ms", "e2e avg ms"),
    ]
    colors = {"teacache": "#4c78a8", "pab": "#f58518", "cubic": "#54a24b", "unknown": "#888888"}
    markers = {"teacache": "o", "pab": "s", "cubic": "^", "unknown": "x"}
    for ax, (x_key, y_key, ylabel) in zip(axes, specs):
        for method in ["teacache", "pab", "cubic", "unknown"]:
            points = []
            for row in stage_rows:
                if row.get("method") != method:
                    continue
                x = _float(row.get(x_key))
                y = _float(row.get(y_key))
                if x is not None and y is not None:
                    points.append((x, y, str(row["condition"])))
            if not points:
                continue
            ax.scatter(
                [p[0] for p in points],
                [p[1] for p in points],
                label=method,
                s=80,
                color=colors[method],
                marker=markers[method],
                alpha=0.9,
            )
            for x, y, label in points:
                ax.annotate(label, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("effective DiT compute ratio (proxy)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_granularity_bars(rows: list[dict[str, Any]], output_path: Path) -> None:
    gran = [row for row in rows if row.get("stage") == "granularity" and row.get("status") == "ok"]
    if not gran:
        return
    labels = [str(row["condition"]).replace("gran_cubic_", "") for row in gran]
    metrics = [
        ("ssim", "SSIM"),
        ("lpips", "LPIPS"),
        ("effective_dit_compute_ratio", "Compute proxy"),
        ("block_count", "Blocks"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), constrained_layout=True)
    for ax, (key, title) in zip(axes, metrics):
        values = [_float(row.get(key)) or 0.0 for row in gran]
        ax.bar(labels, values, color="#54a24b")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(root)
    summary = {"root": str(root), "model": args.model, "rows": rows}
    (output_dir / f"task1_controlled_{args.model}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(output_dir / f"task1_controlled_{args.model}_summary.csv", rows)
    _plot_stage(rows, "phase", output_dir / f"task1_controlled_{args.model}_phase.png", f"{args.model} phase calibration")
    _plot_stage(rows, "frontier", output_dir / f"task1_controlled_{args.model}_frontier.png", f"{args.model} frontier")
    _plot_granularity_bars(rows, output_dir / f"task1_controlled_{args.model}_granularity.png")
    print(output_dir / f"task1_controlled_{args.model}_summary.json")
    print(output_dir / f"task1_controlled_{args.model}_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
