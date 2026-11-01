#!/usr/bin/env python3
"""Summarize Wan1.3B Task 1 phase calibration sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Wan phase calibration sweep")
    parser.add_argument("--root", default="results/wan/1.3B/task1_phase_sweep_v2")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_params(condition_dir: Path) -> Path | None:
    paths = sorted(condition_dir.glob("*_params_metrics.json"))
    return paths[0] if paths else None


def _get(payload: dict, path: tuple[str, ...]):
    obj = payload
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_run_name(name: str) -> tuple[str, int | None, int | None]:
    if "_w" not in name or "_c" not in name:
        return name, None, None
    prefix, suffix = name.rsplit("_w", 1)
    warmup_text, cooldown_text = suffix.split("_c", 1)
    return prefix, int(warmup_text), int(cooldown_text)


def method_from_condition(condition: str) -> str:
    if condition.startswith("tc_"):
        return "teacache"
    if condition.startswith("cubic_"):
        return "cubic"
    if condition.startswith("pab_"):
        return "pab"
    return "unknown"


def summarize_condition(condition_dir: Path) -> dict[str, Any]:
    condition, warmup, cooldown = parse_run_name(condition_dir.name)
    params_path = find_params(condition_dir)
    row = {
        "run": condition_dir.name,
        "condition": condition,
        "method": method_from_condition(condition),
        "warmup_steps": warmup,
        "cooldown_steps": cooldown,
        "params_path": str(params_path) if params_path else None,
    }
    if params_path is None:
        row["status"] = "missing"
        return row
    payload = load_json(params_path)
    params = payload.get("parameters") or {}
    quality = payload.get("quality_metrics") or {}
    ledger = params.get("cache_budget_ledger") or {}
    timing = payload.get("timing_stats") or {}
    quality_mean = quality.get("mean") if isinstance(quality.get("mean"), dict) else quality
    if not isinstance(quality_mean, dict):
        quality_mean = {}
    row.update(
        {
            "status": "ok",
            "tag": payload.get("tag"),
            "warmup_steps": params.get("warmup_steps", row["warmup_steps"]),
            "cooldown_steps": params.get("cooldown_steps", row["cooldown_steps"]),
            "cache_reuse_mode": params.get("cache_reuse_mode"),
            "effective_dit_compute_ratio": ledger.get("effective_dit_compute_ratio"),
            "mean_step_compute_ratio": ledger.get("mean_step_compute_ratio"),
            "mean_layer_compute_ratio": ledger.get("mean_layer_compute_ratio"),
            "mean_token_compute_ratio": ledger.get("mean_token_compute_ratio"),
            "full_compute_steps": ledger.get("full_compute_steps"),
            "reuse_steps": ledger.get("reuse_steps"),
            "total_steps": ledger.get("total_steps"),
            "psnr": quality_mean.get("psnr"),
            "ssim": quality_mean.get("ssim"),
            "lpips": quality_mean.get("lpips"),
            "fid": quality_mean.get("fid"),
            "fvd": quality_mean.get("fvd"),
            "generate_e2e_avg_ms": _get(timing, ("generate_e2e", "avg_ms")),
            "model_forward_avg_ms": _get(timing, ("model_forward", "avg_ms")),
            "budget_kind": ledger.get("budget_kind"),
            "budget_precision": ledger.get("budget_precision"),
        }
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run",
        "condition",
        "method",
        "status",
        "warmup_steps",
        "cooldown_steps",
        "cache_reuse_mode",
        "effective_dit_compute_ratio",
        "mean_step_compute_ratio",
        "mean_layer_compute_ratio",
        "mean_token_compute_ratio",
        "full_compute_steps",
        "reuse_steps",
        "total_steps",
        "psnr",
        "ssim",
        "lpips",
        "fid",
        "fvd",
        "generate_e2e_avg_ms",
        "model_forward_avg_ms",
        "budget_kind",
        "budget_precision",
        "params_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_frontier(rows: list[dict[str, Any]], output_path: Path) -> None:
    rows = [row for row in rows if row.get("status") == "ok"]
    colors = {"teacache": "#4c78a8", "cubic": "#54a24b", "pab": "#f58518"}
    markers = {"teacache": "o", "cubic": "^", "pab": "s"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    specs = [
        ("ssim", "SSIM vs origin"),
        ("lpips", "LPIPS vs origin"),
        ("fid", "FID vs origin"),
    ]
    for ax, (y_key, ylabel) in zip(axes, specs):
        for method in ("teacache", "cubic", "pab"):
            points = []
            for row in rows:
                if row.get("method") != method:
                    continue
                x = _float(row.get("effective_dit_compute_ratio"))
                y = _float(row.get(y_key))
                if x is not None and y is not None:
                    points.append((x, y, row))
            if not points:
                continue
            ax.scatter(
                [item[0] for item in points],
                [item[1] for item in points],
                label=method,
                s=75,
                marker=markers[method],
                color=colors[method],
                alpha=0.9,
            )
            for x, y, row in points:
                label = f"{row['condition']} {row['warmup_steps']}/{row['cooldown_steps']}"
                ax.annotate(label, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("effective DiT compute ratio (proxy)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Wan1.3B phase-calibrated cache frontier")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_phase_heatmaps(rows: list[dict[str, Any]], output_path: Path) -> None:
    rows = [row for row in rows if row.get("status") == "ok"]
    conditions = sorted({str(row["condition"]) for row in rows})
    fig, axes = plt.subplots(len(conditions), 3, figsize=(13, 4.0 * max(len(conditions), 1)), constrained_layout=True)
    if len(conditions) == 1:
        axes = np.asarray([axes])
    metrics = [("ssim", "SSIM"), ("lpips", "LPIPS"), ("effective_dit_compute_ratio", "Compute")]
    for row_idx, condition in enumerate(conditions):
        cond_rows = [row for row in rows if row.get("condition") == condition]
        warmups = sorted({int(row["warmup_steps"]) for row in cond_rows if row.get("warmup_steps") is not None})
        cooldowns = sorted({int(row["cooldown_steps"]) for row in cond_rows if row.get("cooldown_steps") is not None})
        for col_idx, (metric, title) in enumerate(metrics):
            grid = np.full((len(warmups), len(cooldowns)), np.nan, dtype=np.float64)
            for i, warmup in enumerate(warmups):
                for j, cooldown in enumerate(cooldowns):
                    vals = [
                        _float(row.get(metric))
                        for row in cond_rows
                        if int(row["warmup_steps"]) == warmup and int(row["cooldown_steps"]) == cooldown
                    ]
                    vals = [val for val in vals if val is not None]
                    if vals:
                        grid[i, j] = vals[0]
            ax = axes[row_idx, col_idx]
            im = ax.imshow(grid, aspect="auto", interpolation="nearest")
            ax.set_xticks(np.arange(len(cooldowns)))
            ax.set_xticklabels(cooldowns)
            ax.set_yticks(np.arange(len(warmups)))
            ax.set_yticklabels(warmups)
            ax.set_xlabel("cooldown steps")
            ax.set_ylabel("warmup steps")
            ax.set_title(f"{condition}: {title}")
            fig.colorbar(im, ax=ax)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "summary"):
        rows.append(summarize_condition(condition_dir))
    summary = {"root": str(root), "rows": rows}
    (output_dir / "task1_wan_phase_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(output_dir / "task1_wan_phase_summary.csv", rows)
    plot_frontier(rows, output_dir / "task1_wan_phase_frontier.png")
    plot_phase_heatmaps(rows, output_dir / "task1_wan_phase_heatmaps.png")
    print(f"Wrote {output_dir / 'task1_wan_phase_summary.json'}")
    print(output_dir / "task1_wan_phase_frontier.png")
    print(output_dir / "task1_wan_phase_heatmaps.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
