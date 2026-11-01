#!/usr/bin/env python3
"""Stratified breakdown for Task 3 error-probe rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Break down Task 3 oracle error by step/age/tau")
    parser.add_argument("--input", nargs="+", required=True, help="Probe JSONL paths")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metric", default="rmse", choices=["rmse", "l2_mean", "l2_max"])
    return parser.parse_args()


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_str in paths:
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["source"] = str(path)
                rows.append(row)
    return rows


def finite_number(value: Any) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value)


def summarize_group(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    vals = [float(row[metric]) for row in rows if finite_number(row.get(metric))]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "mean": float(mean(vals)),
        "median": float(np.median(vals)),
        "p90": float(np.quantile(vals, 0.9)),
        "max": float(max(vals)),
    }


def grouped_stats(rows: list[dict[str, Any]], metric: str, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if finite_number(row.get(metric)):
            buckets[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for group_key, group_rows in sorted(buckets.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item.update(summarize_group(group_rows, metric))
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_step_age_heatmap(rows: list[dict[str, Any]], metric: str, output_path: Path) -> None:
    frozen = [
        row
        for row in rows
        if not bool(row.get("is_active"))
        and int(row.get("age", 0)) > 0
        and finite_number(row.get(metric))
    ]
    steps = sorted({int(row["step"]) for row in frozen})
    ages = sorted({int(row["age"]) for row in frozen})
    if not steps or not ages:
        return
    grid = np.full((len(ages), len(steps)), np.nan, dtype=np.float64)
    for age_idx, age in enumerate(ages):
        for step_idx, step in enumerate(steps):
            vals = [
                float(row[metric])
                for row in frozen
                if int(row["age"]) == age and int(row["step"]) == step
            ]
            if vals:
                grid[age_idx, step_idx] = float(mean(vals))

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    im = ax.imshow(grid, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels(steps, rotation=45)
    ax.set_yticks(np.arange(len(ages)))
    ax.set_yticklabels(ages)
    ax.set_xlabel("step")
    ax.set_ylabel("frozen output age")
    ax.set_title(f"Mean frozen {metric} by step and age")
    fig.colorbar(im, ax=ax, label=metric)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_step_mean(rows: list[dict[str, Any]], metric: str, output_path: Path) -> None:
    stats = grouped_stats(rows, metric, ("step",))
    if not stats:
        return
    steps = [int(item["step"]) for item in stats]
    means = [float(item["mean"]) for item in stats]
    p90 = [float(item["p90"]) for item in stats]
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    ax.plot(steps, means, marker="o", label="mean")
    ax.plot(steps, p90, marker="s", label="p90")
    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by probe step")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)
    frozen_positive = [
        row
        for row in rows
        if not bool(row.get("is_active"))
        and int(row.get("age", 0)) > 0
        and finite_number(row.get(args.metric))
    ]

    by_step = grouped_stats(frozen_positive, args.metric, ("step",))
    by_age = grouped_stats(frozen_positive, args.metric, ("age",))
    by_tau = grouped_stats(frozen_positive, args.metric, ("tau",))
    by_step_age = grouped_stats(frozen_positive, args.metric, ("step", "age"))
    write_csv(output_dir / "frozen_error_by_step.csv", by_step)
    write_csv(output_dir / "frozen_error_by_age.csv", by_age)
    write_csv(output_dir / "frozen_error_by_tau.csv", by_tau)
    write_csv(output_dir / "frozen_error_by_step_age.csv", by_step_age)

    heatmap_path = output_dir / f"heatmap_frozen_{args.metric}_by_step_age.png"
    step_path = output_dir / f"frozen_{args.metric}_by_step.png"
    plot_step_age_heatmap(frozen_positive, args.metric, heatmap_path)
    plot_step_mean(frozen_positive, args.metric, step_path)

    worst_steps = sorted(by_step, key=lambda item: float(item["mean"]), reverse=True)[:5]
    summary = {
        "inputs": args.input,
        "metric": args.metric,
        "raw_rows": len(rows),
        "frozen_positive_rows": len(frozen_positive),
        "by_step_csv": str(output_dir / "frozen_error_by_step.csv"),
        "by_age_csv": str(output_dir / "frozen_error_by_age.csv"),
        "by_tau_csv": str(output_dir / "frozen_error_by_tau.csv"),
        "by_step_age_csv": str(output_dir / "frozen_error_by_step_age.csv"),
        "visualizations": [str(heatmap_path), str(step_path)],
        "worst_steps_by_mean": worst_steps,
        "age_summary": by_age,
    }
    summary_path = output_dir / "error_probe_breakdown_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {summary_path}")
    for path in summary["visualizations"]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
