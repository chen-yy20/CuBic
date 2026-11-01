#!/usr/bin/env python3
"""Summarize Task 3 v1 Flux1 CuBic in-process grid results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRIC_KEYS = ("psnr", "ssim", "lpips", "fid", "fvd")
TRACE_KEYS = (
    "full_compute_steps",
    "mean_active_ratio_all",
    "mean_active_ratio_selective",
    "zero_active_selective_steps",
    "selective_steps",
)
TIMING_KEYS = ("condition_seconds", "generate_e2e_ms", "selective_forward_ms", "anchor_step_total_ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Task 3 v1 grid result JSON files")
    parser.add_argument("--input_root", default="results/flux/task3_v1_grid_inprocess")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None)
    return parser.parse_args()


def finite_stats(values: list[float]) -> dict[str, Any]:
    finite = [v for v in values if math.isfinite(v)]
    return {
        "mean": mean(finite) if finite else None,
        "stdev": stdev(finite) if len(finite) > 1 else 0.0 if len(finite) == 1 else None,
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "count": len(values),
        "finite_count": len(finite),
        "inf_count": sum(1 for v in values if math.isinf(v)),
        "nan_count": sum(1 for v in values if math.isnan(v)),
    }


def get_nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def load_record(path: Path) -> dict[str, Any] | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    params = data.get("parameters", {})
    quality_mean = get_nested(data, ("quality_metrics", "mean")) or {}
    trace = params.get("trace_summary") or {}
    timing = data.get("timing_stats", {})

    try:
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        tolerance = float(params["curvature_error_tolerance"])
        seed = int((params.get("completed_seeds") or params.get("seeds") or [None])[0])
    except (KeyError, TypeError, ValueError):
        return None

    record: dict[str, Any] = {
        "path": str(path),
        "alpha": alpha,
        "beta": beta,
        "tolerance": tolerance,
        "seed": seed,
        "condition_seconds": params.get("condition_seconds"),
        "generate_e2e_ms": get_nested(timing, ("generate_e2e", "avg_ms")),
        "selective_forward_ms": get_nested(timing, ("selective_forward", "avg_ms")),
        "anchor_step_total_ms": get_nested(timing, ("anchor_step_total", "avg_ms")),
    }
    for key in METRIC_KEYS:
        record[key] = quality_mean.get(key)
    for key in TRACE_KEYS:
        record[key] = trace.get(key)
    return record


def append_value(bucket: dict[str, list[float]], key: str, value: Any) -> None:
    if value is None:
        return
    try:
        bucket[key].append(float(value))
    except (TypeError, ValueError):
        return


def main() -> int:
    args = parse_args()
    root = Path(args.input_root)
    if not root.exists():
        raise SystemExit(f"missing input root: {root}")

    records = []
    for path in sorted(root.glob("*/cubic_curvonly_balanced_params_metrics.json")):
        record = load_record(path)
        if record is not None:
            records.append(record)

    grouped: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["alpha"], record["beta"], record["tolerance"])].append(record)

    groups = []
    for (alpha, beta, tolerance), group_records in sorted(grouped.items()):
        bucket: dict[str, list[float]] = defaultdict(list)
        for record in group_records:
            for key in (*METRIC_KEYS, *TRACE_KEYS, *TIMING_KEYS):
                append_value(bucket, key, record.get(key))

        summary: dict[str, Any] = {
            "alpha": alpha,
            "beta": beta,
            "tolerance": tolerance,
            "count": len(group_records),
            "seeds": sorted(record["seed"] for record in group_records),
        }
        for key in (*METRIC_KEYS, *TRACE_KEYS, *TIMING_KEYS):
            summary[key] = finite_stats(bucket.get(key, []))

        ssim = summary["ssim"]["mean"]
        lpips = summary["lpips"]["mean"]
        fid = summary["fid"]["mean"]
        active = summary["mean_active_ratio_selective"]["mean"]
        seconds = summary["condition_seconds"]["mean"]
        summary["quick_score"] = None
        if all(v is not None for v in (ssim, lpips, fid, active, seconds)):
            summary["quick_score"] = float(ssim) - float(lpips) - 0.001 * float(fid) - 0.05 * float(active) - 0.001 * float(seconds)
        groups.append(summary)

    ranked = sorted(
        groups,
        key=lambda item: (
            item["quick_score"] is None,
            -(item["quick_score"] or -1e9),
            item["mean_active_ratio_selective"]["mean"] or 1e9,
        ),
    )
    output = {
        "input_root": str(root),
        "records": len(records),
        "groups": groups,
        "ranked_by_quick_score": ranked,
    }

    output_json = Path(args.output_json) if args.output_json else root / "grid_aggregate_summary.json"
    output_csv = Path(args.output_csv) if args.output_csv else root / "grid_aggregate_summary.csv"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_fields = [
        "alpha",
        "beta",
        "tolerance",
        "count",
        "seeds",
        "ssim_mean",
        "lpips_mean",
        "fid_mean",
        "psnr_mean_finite",
        "psnr_inf_count",
        "active_selective_mean",
        "active_all_mean",
        "full_compute_steps_mean",
        "zero_active_selective_steps_mean",
        "condition_seconds_mean",
        "generate_e2e_ms_mean",
        "quick_score",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for item in ranked:
            writer.writerow(
                {
                    "alpha": item["alpha"],
                    "beta": item["beta"],
                    "tolerance": item["tolerance"],
                    "count": item["count"],
                    "seeds": " ".join(str(seed) for seed in item["seeds"]),
                    "ssim_mean": item["ssim"]["mean"],
                    "lpips_mean": item["lpips"]["mean"],
                    "fid_mean": item["fid"]["mean"],
                    "psnr_mean_finite": item["psnr"]["mean"],
                    "psnr_inf_count": item["psnr"]["inf_count"],
                    "active_selective_mean": item["mean_active_ratio_selective"]["mean"],
                    "active_all_mean": item["mean_active_ratio_all"]["mean"],
                    "full_compute_steps_mean": item["full_compute_steps"]["mean"],
                    "zero_active_selective_steps_mean": item["zero_active_selective_steps"]["mean"],
                    "condition_seconds_mean": item["condition_seconds"]["mean"],
                    "generate_e2e_ms_mean": item["generate_e2e_ms"]["mean"],
                    "quick_score": item["quick_score"],
                }
            )

    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    print(f"records={len(records)} groups={len(groups)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
