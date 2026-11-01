#!/usr/bin/env python3
"""Summarize and visualize Wan Task 3 cache reuse mode experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


MODES = ("residual", "output", "kv_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Wan cache reuse modes")
    parser.add_argument("--root", default="results/wan/task3_v5_wan_cache_compare")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def iter_probe_rows(mode_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(mode_dir.glob("*error_probe*.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def finite_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def summarize_mode(root: Path, mode: str) -> dict[str, Any]:
    mode_dir = root / mode
    rows = iter_probe_rows(mode_dir)
    frozen_rows = [
        row
        for row in rows
        if not bool(row.get("is_active"))
        and int(row.get("age", 0)) > 0
        and finite_float(row.get("rmse")) is not None
    ]
    frozen_rmse = [float(row["rmse"]) for row in frozen_rows]
    fit = load_json(mode_dir / "analysis" / "error_probe_fit_summary.json")
    fit_main = fit.get("fits", {}).get("frozen_positive", {})
    fit_kappa = fit.get("fits", {}).get("frozen_kappa_only", {})
    fit_age = fit.get("fits", {}).get("frozen_age_dt_only", {})
    params = load_json(mode_dir / "cubic_params_metrics.json")
    quality = params.get("quality_metrics", {})
    trace = quality.get("cubic_step_trace", {}).get("steps", [])
    selective = [row for row in trace if "selective" in str(row.get("phase", ""))]
    trace_rows = selective or trace
    active_ratios = [finite_float(row.get("active_ratio")) for row in trace_rows]
    latencies = [finite_float(row.get("latency_ms")) for row in trace_rows]
    active_ratios = [val for val in active_ratios if val is not None]
    latencies = [val for val in latencies if val is not None]
    return {
        "mode": mode,
        "probe_rows": len(rows),
        "frozen_rows": len(frozen_rows),
        "mean_frozen_rmse": float(mean(frozen_rmse)) if frozen_rmse else None,
        "median_frozen_rmse": float(np.median(frozen_rmse)) if frozen_rmse else None,
        "p90_frozen_rmse": float(np.quantile(frozen_rmse, 0.9)) if frozen_rmse else None,
        "fit_r2": fit_main.get("r2"),
        "fit_kappa_slope": fit_main.get("kappa_slope"),
        "fit_age_dt_slope": fit_main.get("age_dt_slope"),
        "kappa_only_r2": fit_kappa.get("r2"),
        "age_dt_only_r2": fit_age.get("r2"),
        "mean_active_ratio": float(mean(active_ratios)) if active_ratios else None,
        "mean_selective_latency_ms": float(mean(latencies)) if latencies else None,
        "psnr": quality.get("psnr"),
        "ssim": quality.get("ssim"),
        "lpips": quality.get("lpips"),
        "fid": quality.get("fid"),
        "fvd": quality.get("fvd"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "probe_rows",
        "frozen_rows",
        "mean_frozen_rmse",
        "median_frozen_rmse",
        "p90_frozen_rmse",
        "fit_r2",
        "fit_kappa_slope",
        "fit_age_dt_slope",
        "kappa_only_r2",
        "age_dt_only_r2",
        "mean_active_ratio",
        "mean_selective_latency_ms",
        "psnr",
        "ssim",
        "lpips",
        "fid",
        "fvd",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bar_values(rows: list[dict[str, Any]], key: str) -> tuple[list[str], list[float]]:
    labels = []
    values = []
    for row in rows:
        val = finite_float(row.get(key))
        if val is not None:
            labels.append(str(row["mode"]))
            values.append(val)
    return labels, values


def plot_error_and_fit(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [str(row["mode"]) for row in rows]
    mean_rmse = [finite_float(row.get("mean_frozen_rmse")) or np.nan for row in rows]
    r2 = [finite_float(row.get("fit_r2")) or np.nan for row in rows]
    kappa = [finite_float(row.get("fit_kappa_slope")) or np.nan for row in rows]
    age = [finite_float(row.get("fit_age_dt_slope")) or np.nan for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    axes[0].bar(labels, mean_rmse, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].set_title("Frozen oracle RMSE")
    axes[0].set_ylabel("mean RMSE")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, r2, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[1].set_title("Power-law fit")
    axes[1].set_ylabel("R2")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.3)

    x = np.arange(len(labels))
    width = 0.35
    axes[2].bar(x - width / 2, kappa, width, label="kappa slope", color="#4c78a8")
    axes[2].bar(x + width / 2, age, width, label="age*dt slope", color="#e45756")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("Fitted exponents")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].legend()

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_quality_latency(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [str(row["mode"]) for row in rows]
    latency = [finite_float(row.get("mean_selective_latency_ms")) or np.nan for row in rows]
    psnr = [finite_float(row.get("psnr")) or np.nan for row in rows]
    lpips = [finite_float(row.get("lpips")) or np.nan for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    axes[0].bar(labels, latency, color="#72b7b2")
    axes[0].set_title("Selective latency")
    axes[0].set_ylabel("mean ms")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, psnr, color="#4c78a8")
    axes[1].set_title("PSNR vs origin ref")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(labels, lpips, color="#e45756")
    axes[2].set_title("LPIPS vs origin ref")
    axes[2].grid(axis="y", alpha=0.3)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_mode(root, mode) for mode in MODES if (root / mode).exists()]

    summary = {"root": str(root), "modes": rows}
    (output_dir / "cache_mode_comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(output_dir / "cache_mode_comparison.csv", rows)
    figures = [
        output_dir / "cache_mode_error_fit.png",
        output_dir / "cache_mode_quality_latency.png",
    ]
    plot_error_and_fit(rows, figures[0])
    plot_quality_latency(rows, figures[1])
    print(f"Wrote {output_dir / 'cache_mode_comparison.json'}")
    for path in figures:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
