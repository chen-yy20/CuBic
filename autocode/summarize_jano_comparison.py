#!/usr/bin/env python3
"""Build normalized CUBIC-vs-Jano Pareto tables from real experiment outputs."""

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
    parser = argparse.ArgumentParser(description="Summarize CUBIC and Jano comparison")
    parser.add_argument("--cubic_records", nargs="+", required=True)
    parser.add_argument("--jano_root", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def valid(values) -> list[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def mean_std(values) -> tuple[float | None, float | None]:
    values = valid(values)
    if not values:
        return None, None
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def flatten_quality(value) -> list[dict]:
    if not isinstance(value, dict):
        return []
    if any(key in value for key in ("psnr", "ssim", "lpips")):
        return [value]
    output = []
    for child in value.values():
        output.extend(flatten_quality(child))
    return output


def timing(payload: dict) -> float | None:
    try:
        return float(payload["timing_stats"]["generate_e2e"]["avg_ms"])
    except (KeyError, TypeError, ValueError):
        return None


def timing_samples(payload: dict) -> int:
    try:
        return int(payload["timing_stats"]["generate_e2e"]["samples"])
    except (KeyError, TypeError, ValueError):
        return 1


def load_cubic(paths: list[Path]) -> list[dict]:
    groups = defaultdict(list)
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            condition = record.get("condition") or {}
            if condition.get("stage") != "frontier":
                continue
            name = condition.get("name", "cubic")
            if any(
                marker in part
                for part in path.parts
                for marker in ("matched_boundary", "warmed_final")
            ):
                name = f"matched_boundary_{name}"
            groups[name].append(record)
    rows = []
    for condition, records in sorted(groups.items()):
        row = {"method": "CUBIC", "condition": condition, "samples": len(records)}
        for key, getter in {
            "speedup": lambda item: item.get("normalized_speedup"),
            "psnr": lambda item: (item.get("quality") or {}).get("psnr"),
            "ssim": lambda item: (item.get("quality") or {}).get("ssim"),
            "lpips": lambda item: (item.get("quality") or {}).get("lpips"),
        }.items():
            row[key], row[f"{key}_std"] = mean_std([getter(item) for item in records])
        rows.append(row)
    return rows


def load_jano(roots: list[Path]) -> list[dict]:
    groups = defaultdict(list)
    for root in roots:
        origin_timings = {}
        for path in root.glob("ori/**/*_params_metrics.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            params = payload.get("parameters") or {}
            prompts = params.get("prompts") or [params.get("prompt")]
            for prompt in prompts:
                if prompt:
                    origin_timings[str(prompt)] = timing(payload)

        for path in root.glob("**/*_params_metrics.json"):
            relative = path.relative_to(root)
            if relative.parts[0] == "ori":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            params = payload.get("parameters") or {}
            config = relative.parts[0]
            prompt_values = params.get("prompts") or [params.get("prompt")]
            available_origins = valid([origin_timings.get(str(prompt)) for prompt in prompt_values if prompt])
            origin_ms = mean(available_origins) if available_origins else None
            method_ms = timing(payload)
            groups[config].append(
                {
                    "origin_ms": origin_ms,
                    "method_ms": method_ms,
                    "timing_samples": timing_samples(payload),
                    "qualities": flatten_quality(payload.get("quality_metrics") or {}),
                }
            )

    rows = []
    for condition, items in sorted(groups.items()):
        qualities = [quality for item in items for quality in item["qualities"]]
        row = {
            "method": "Jano",
            "condition": condition,
            "samples": len(qualities),
        }
        total_origin = sum(
            item["origin_ms"] * item["timing_samples"]
            for item in items
            if item["origin_ms"] is not None and item["method_ms"] is not None
        )
        total_method = sum(
            item["method_ms"] * item["timing_samples"]
            for item in items
            if item["origin_ms"] is not None and item["method_ms"] is not None
        )
        row["speedup"] = total_origin / total_method if total_method else None
        per_run_speedups = [
            item["origin_ms"] / item["method_ms"]
            for item in items
            if item["origin_ms"] is not None and item["method_ms"] is not None
        ]
        # A sidecar containing multiple samples stores only aggregate timing
        # statistics, so a per-sample standard deviation cannot be recovered.
        _, row["speedup_std"] = (
            mean_std(per_run_speedups) if len(per_run_speedups) > 1 else (None, None)
        )
        for key in ("psnr", "ssim", "lpips"):
            row[key], row[f"{key}_std"] = mean_std([quality.get(key) for quality in qualities])
        rows.append(row)
    return rows


def plot(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    styles = {"CUBIC": ("o", "tab:green"), "Jano": ("s", "tab:blue")}
    for method in ("CUBIC", "Jano"):
        method_rows = [row for row in rows if row["method"] == method]
        marker, color = styles[method]
        for axis, metric, ylabel in zip(axes, ("psnr", "ssim", "lpips"), ("PSNR (dB)", "SSIM", "LPIPS")):
            axis.errorbar(
                [row["speedup"] for row in method_rows],
                [row[metric] for row in method_rows],
                xerr=[row["speedup_std"] or 0.0 for row in method_rows],
                yerr=[row[f"{metric}_std"] or 0.0 for row in method_rows],
                marker=marker,
                color=color,
                linestyle="-",
                label=method,
            )
            axis.set_xlabel("Normalized speedup (×)")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.3)
    for axis in axes:
        axis.legend()
    fig.suptitle("CUBIC vs Jano on matched prompts and seeds")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_cubic([Path(path) for path in args.cubic_records])
    rows.extend(load_jano([Path(path) for path in args.jano_root]))
    (output_dir / "cubic_vs_jano.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if rows:
        with (output_dir / "cubic_vs_jano.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        plot(rows, output_dir / "cubic_vs_jano_pareto.png")
    print(output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
