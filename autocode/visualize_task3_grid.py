#!/usr/bin/env python3
"""Create human-readable Task 3 grid visualizations from aggregate JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


METRICS = {
    "ssim": ("SSIM", "viridis", "{:.3f}"),
    "lpips": ("LPIPS", "magma_r", "{:.3f}"),
    "fid": ("FID", "magma_r", "{:.1f}"),
    "mean_active_ratio_selective": ("Selective Active Ratio", "cividis", "{:.3f}"),
    "condition_seconds": ("Condition Seconds", "plasma_r", "{:.1f}"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Task 3 aggregate grid summaries")
    parser.add_argument("--aggregate_json", default="results/flux/task3_v1_grid_inprocess/grid_aggregate_summary.json")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def metric_mean(group: dict[str, Any], metric: str) -> float | None:
    value = group.get(metric)
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def ab_label(group: dict[str, Any]) -> str:
    return f"a{group['alpha']:g}/b{group['beta']:g}"


def plot_heatmap(groups: list[dict[str, Any]], metric: str, output_path: Path, title: str) -> None:
    row_labels = sorted({ab_label(group) for group in groups})
    tolerances = sorted({float(group["tolerance"]) for group in groups})
    matrix = np.full((len(row_labels), len(tolerances)), np.nan, dtype=float)
    for group in groups:
        row = row_labels.index(ab_label(group))
        col = tolerances.index(float(group["tolerance"]))
        value = metric_mean(group, metric)
        if value is not None:
            matrix[row, col] = value

    label, cmap, fmt = METRICS[metric]
    fig_width = max(7.0, len(tolerances) * 1.65)
    fig_height = max(4.6, len(row_labels) * 0.62)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap=cmap, aspect="auto")
    ax.set_title(f"{title}: {label}")
    ax.set_xlabel("curvature_error_tolerance")
    ax.set_ylabel("(alpha, beta)")
    ax.set_xticks(range(len(tolerances)))
    ax.set_xticklabels([f"{tol:.0e}" for tol in tolerances])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    fig.colorbar(image, ax=ax, shrink=0.88)

    for row in range(len(row_labels)):
        for col in range(len(tolerances)):
            value = matrix[row, col]
            if math.isfinite(value):
                ax.text(col, row, fmt.format(value), ha="center", va="center", fontsize=8, color="white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pareto(groups: list[dict[str, Any]], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for ax, metric, ylabel, invert_y in (
        (axes[0], "ssim", "SSIM (higher better)", False),
        (axes[1], "lpips", "LPIPS (lower better)", True),
    ):
        for group in groups:
            x = metric_mean(group, "mean_active_ratio_selective")
            y = metric_mean(group, metric)
            if x is None or y is None:
                continue
            ax.scatter(x, y, s=90, alpha=0.78)
            ax.annotate(
                f"{ab_label(group)}\n{float(group['tolerance']):.0e}",
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xlabel("Selective active ratio")
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.4, alpha=0.4)
        if invert_y:
            ax.invert_yaxis()
    fig.suptitle(f"{title}: Quality vs Active Ratio")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    aggregate_path = Path(args.aggregate_json)
    data = json.loads(aggregate_path.read_text(encoding="utf-8"))
    groups = data.get("groups", [])
    if not groups:
        raise SystemExit(f"no groups in {aggregate_path}")
    output_dir = Path(args.output_dir) if args.output_dir else aggregate_path.parent / "visualizations"
    title = args.title or aggregate_path.parent.name

    outputs = []
    for metric in METRICS:
        path = output_dir / f"heatmap_{metric}.png"
        plot_heatmap(groups, metric, path, title)
        outputs.append(path)

    pareto_path = output_dir / "pareto_quality_active.png"
    plot_pareto(groups, pareto_path, title)
    outputs.append(pareto_path)

    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "aggregate_json": str(aggregate_path),
                "outputs": [str(path) for path in outputs],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
