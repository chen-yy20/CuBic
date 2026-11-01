#!/usr/bin/env python3
"""Summarize Wan1.3B phase-calibrated CuBic granularity sweep."""

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
    parser = argparse.ArgumentParser(description="Summarize Wan CuBic granularity sweep")
    parser.add_argument("--root", default="results/wan/1.3B/task1_granularity_sweep_v2")
    parser.add_argument("--phase_summary", default="results/wan/1.3B/task1_phase_sweep_v2/summary/task1_wan_phase_summary.csv")
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


def summarize_condition(condition_dir: Path) -> dict[str, Any]:
    params_path = find_params(condition_dir)
    row = {
        "condition": condition_dir.name,
        "method": "cubic",
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
    trace = quality.get("cubic_step_trace", {}).get("steps", [])
    anchor_blocks = [
        row.get("text", "")
        for row in trace
        if "phase2_cruise_anchor" in str(row.get("phase", ""))
    ]
    row.update(
        {
            "status": "ok",
            "warmup_steps": params.get("warmup_steps"),
            "cooldown_steps": params.get("cooldown_steps"),
            "cache_reuse_mode": params.get("cache_reuse_mode"),
            "partition_mode": params.get("partition_mode"),
            "min_block_size": params.get("min_block_size"),
            "max_block_size": params.get("max_block_size"),
            "cv_threshold": params.get("cv_threshold"),
            "foreground_focus_block_size": params.get("foreground_focus_block_size"),
            "effective_dit_compute_ratio": ledger.get("effective_dit_compute_ratio"),
            "full_compute_steps": ledger.get("full_compute_steps"),
            "reuse_steps": ledger.get("reuse_steps"),
            "psnr": quality.get("psnr"),
            "ssim": quality.get("ssim"),
            "lpips": quality.get("lpips"),
            "fid": quality.get("fid"),
            "fvd": quality.get("fvd"),
            "generate_e2e_avg_ms": _get(timing, ("generate_e2e", "avg_ms")),
            "anchor_step_total_avg_ms": _get(timing, ("anchor_step_total", "avg_ms")),
            "selective_forward_avg_ms": _get(timing, ("selective_forward", "avg_ms")),
            "anchor_step_count": len(anchor_blocks),
        }
    )
    return row


def load_phase_references(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("condition") in {"tc_0p07", "tc_0p10"} and row.get("warmup_steps") in {"8"}:
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "condition",
        "method",
        "status",
        "warmup_steps",
        "cooldown_steps",
        "cache_reuse_mode",
        "partition_mode",
        "min_block_size",
        "max_block_size",
        "cv_threshold",
        "foreground_focus_block_size",
        "effective_dit_compute_ratio",
        "full_compute_steps",
        "reuse_steps",
        "psnr",
        "ssim",
        "lpips",
        "fid",
        "fvd",
        "generate_e2e_avg_ms",
        "anchor_step_total_avg_ms",
        "selective_forward_avg_ms",
        "anchor_step_count",
        "params_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_frontier(rows: list[dict[str, Any]], references: list[dict[str, Any]], output_path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    specs = [("ssim", "SSIM"), ("lpips", "LPIPS"), ("fid", "FID")]
    for ax, (metric, ylabel) in zip(axes, specs):
        for row in ok_rows:
            x = _float(row.get("effective_dit_compute_ratio"))
            y = _float(row.get(metric))
            if x is None or y is None:
                continue
            ax.scatter(x, y, s=90, marker="^", color="#54a24b")
            ax.annotate(str(row["condition"]), (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
        for ref in references:
            x = _float(ref.get("effective_dit_compute_ratio"))
            y = _float(ref.get(metric))
            if x is None or y is None:
                continue
            ax.scatter(x, y, s=80, marker="o", color="#4c78a8", alpha=0.75)
            label = f"{ref.get('condition')} {ref.get('warmup_steps')}/{ref.get('cooldown_steps')}"
            ax.annotate(label, (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("effective DiT compute ratio (proxy)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Wan1.3B phase-calibrated CuBic granularity sweep")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_bar(rows: list[dict[str, Any]], output_path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    labels = [str(row["condition"]) for row in ok_rows]
    ssim = [_float(row.get("ssim")) or 0.0 for row in ok_rows]
    lpips = [_float(row.get("lpips")) or 0.0 for row in ok_rows]
    compute = [_float(row.get("effective_dit_compute_ratio")) or 0.0 for row in ok_rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for ax, values, title in zip(axes, [ssim, lpips, compute], ["SSIM", "LPIPS", "Compute proxy"]):
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
    rows = []
    for condition_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "summary"):
        rows.append(summarize_condition(condition_dir))
    refs = load_phase_references(Path(args.phase_summary))
    summary = {"root": str(root), "rows": rows, "references": refs}
    (output_dir / "task1_wan_granularity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(output_dir / "task1_wan_granularity_summary.csv", rows)
    plot_frontier(rows, refs, output_dir / "task1_wan_granularity_frontier.png")
    plot_bar(rows, output_dir / "task1_wan_granularity_bars.png")
    print(f"Wrote {output_dir / 'task1_wan_granularity_summary.json'}")
    print(output_dir / "task1_wan_granularity_frontier.png")
    print(output_dir / "task1_wan_granularity_bars.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
