#!/usr/bin/env python3
"""Summarize Wan1.3B Task 1 matched baseline sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Wan Task 1 baseline sweep")
    parser.add_argument("--root", default="results/wan/1.3B/task1_matched_sweep_v1")
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


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def method_from_condition(name: str) -> str:
    if name.startswith("tc_"):
        return "teacache"
    if name.startswith("pab_"):
        return "pab"
    if name.startswith("cubic_"):
        return "cubic"
    return "unknown"


def summarize_condition(condition_dir: Path) -> dict[str, Any]:
    params_path = find_params(condition_dir)
    row = {
        "condition": condition_dir.name,
        "method": method_from_condition(condition_dir.name),
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
            "warmup_steps": params.get("warmup_steps", params.get("pab_warmup")),
            "cooldown_steps": params.get("cooldown_steps", params.get("pab_cooldown")),
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
        "condition",
        "method",
        "status",
        "warmup_steps",
        "cooldown_steps",
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


def _scatter(ax, rows: list[dict[str, Any]], x_key: str, y_key: str, ylabel: str) -> None:
    colors = {"teacache": "#4c78a8", "pab": "#f58518", "cubic": "#54a24b"}
    markers = {"teacache": "o", "pab": "s", "cubic": "^"}
    for method in ("teacache", "pab", "cubic"):
        points = []
        for row in rows:
            if row.get("method") != method:
                continue
            x = _float(row.get(x_key))
            y = _float(row.get(y_key))
            if x is not None and y is not None:
                points.append((x, y, row.get("condition")))
        if not points:
            continue
        ax.scatter(
            [item[0] for item in points],
            [item[1] for item in points],
            label=method,
            s=80,
            marker=markers[method],
            color=colors[method],
            alpha=0.9,
        )
        for x, y, label in points:
            ax.annotate(str(label), (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("effective DiT compute ratio (proxy)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_frontier(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    _scatter(axes[0], rows, "effective_dit_compute_ratio", "ssim", "SSIM vs origin")
    _scatter(axes[1], rows, "effective_dit_compute_ratio", "lpips", "LPIPS vs origin")
    _scatter(axes[2], rows, "effective_dit_compute_ratio", "generate_e2e_avg_ms", "generation e2e avg ms")
    fig.suptitle("Wan1.3B Task 1 matched baseline frontier")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_budget(rows: list[dict[str, Any]], output_path: Path) -> None:
    rows = [row for row in rows if row.get("status") == "ok"]
    labels = [row["condition"] for row in rows]
    values = [_float(row.get("effective_dit_compute_ratio")) or 0.0 for row in rows]
    colors = [
        {"teacache": "#4c78a8", "pab": "#f58518", "cubic": "#54a24b"}.get(row["method"], "#999999")
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("effective DiT compute ratio (proxy)")
    ax.set_title("Wan1.3B budget proxy by condition")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
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
    (output_dir / "task1_wan_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(output_dir / "task1_wan_summary.csv", rows)
    plot_frontier(rows, output_dir / "task1_wan_frontier.png")
    plot_budget(rows, output_dir / "task1_wan_budget.png")
    print(f"Wrote {output_dir / 'task1_wan_summary.json'}")
    print(output_dir / "task1_wan_frontier.png")
    print(output_dir / "task1_wan_budget.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
