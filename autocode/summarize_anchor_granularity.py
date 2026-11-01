#!/usr/bin/env python3
"""Aggregate anchor/granularity ablation JSONL records into tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Cubic anchor/granularity ablations")
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def finite(values) -> list[float]:
    output = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            output.append(number)
    return output


def stats(values) -> tuple[float | None, float | None]:
    values = finite(values)
    if not values:
        return None, None
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def nested(record: dict, *keys):
    current = record
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def load_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def aggregate(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        condition = record.get("condition") or {}
        groups[(condition.get("stage"), condition.get("name"))].append(record)
    rows = []
    for (stage, condition), items in sorted(groups.items()):
        row = {"stage": stage, "condition": condition, "samples": len(items)}
        metrics = {
            "normalized_speedup": [item.get("normalized_speedup") for item in items],
            "latency_ms": [item.get("latency_ms") for item in items],
            "psnr": [nested(item, "quality", "psnr") for item in items],
            "ssim": [nested(item, "quality", "ssim") for item in items],
            "lpips": [nested(item, "quality", "lpips") for item in items],
            "active_ratio": [nested(item, "trace_summary", "mean_active_ratio") for item in items],
            "anchor_count": [nested(item, "trace_summary", "anchor_count") for item in items],
            "anchor_gap": [nested(item, "trace_summary", "mean_anchor_gap") for item in items],
            "curvature_l1_drift": [
                nested(item, "trace_summary", "mean_normalized_l1_drift") for item in items
            ],
            "curvature_rank_correlation": [
                nested(item, "trace_summary", "mean_rank_correlation") for item in items
            ],
            "partition_overhead_ms": [
                nested(item, "timing_stats", "anchor_step_partitioner", "avg_ms") for item in items
            ],
            "optimizer_overhead_ms": [
                nested(item, "timing_stats", "anchor_step_optimizer", "avg_ms") for item in items
            ],
        }
        for name, values in metrics.items():
            avg, std = stats(values)
            row[name] = avg
            row[f"{name}_std"] = std
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_anchor(rows: list[dict], path: Path) -> None:
    rows = [row for row in rows if row["stage"] == "anchor"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for row in rows:
        label = row["condition"].replace("anchor_", "")
        axes[0].errorbar(
            row["normalized_speedup"],
            row["psnr"],
            xerr=row["normalized_speedup_std"],
            yerr=row["psnr_std"],
            marker="o",
            linestyle="none",
            label=label,
        )
        axes[1].errorbar(
            row["normalized_speedup"],
            row["lpips"],
            xerr=row["normalized_speedup_std"],
            yerr=row["lpips_std"],
            marker="o",
            linestyle="none",
            label=label,
        )
        axes[2].scatter(row["anchor_gap"], row["curvature_rank_correlation"], label=label)
    axes[0].set(xlabel="Normalized speedup (×)", ylabel="PSNR (dB)", title="Anchor quality–speed")
    axes[1].set(xlabel="Normalized speedup (×)", ylabel="LPIPS", title="Anchor perceptual error")
    axes[2].set(
        xlabel="Observed anchor gap (steps)",
        ylabel="Curvature rank correlation",
        title="Window stability",
    )
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_granularity(rows: list[dict], path: Path) -> None:
    rows = [row for row in rows if row["stage"] == "granularity"]
    if not rows:
        return
    labels = [row["condition"].replace("granularity_", "") for row in rows]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    specs = [
        ("psnr", "PSNR (dB)"),
        ("lpips", "LPIPS"),
        ("normalized_speedup", "Normalized speedup (×)"),
        ("partition_overhead_ms", "Partition overhead (ms)"),
    ]
    for axis, (key, title) in zip(axes, specs):
        values = [row[key] or 0.0 for row in rows]
        errors = [row[f"{key}_std"] or 0.0 for row in rows]
        axis.bar(labels, values, yerr=errors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=24)
        axis.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    records_paths = [Path(path) for path in args.records]
    output_dir = Path(args.output_dir) if args.output_dir else records_paths[0].parent / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = aggregate([record for path in records_paths for record in load_records(path)])
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(
            {"records": [str(path) for path in records_paths], "rows": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_csv(output_dir / "ablation_summary.csv", rows)
    plot_anchor(rows, output_dir / "anchor_ablation.png")
    plot_granularity(rows, output_dir / "granularity_ablation.png")
    print(output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
