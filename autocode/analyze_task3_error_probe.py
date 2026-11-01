#!/usr/bin/env python3
"""Analyze Task 3 block-level error probe JSONL files."""

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


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit cache error power laws from probe JSONL")
    parser.add_argument("--input", nargs="+", required=True, help="Probe JSONL paths")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metric", default="rmse", choices=["rmse", "l2_mean", "l2_max"])
    return parser.parse_args()


def load_records(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
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


def finite_positive(value: Any) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def prepare_matrix(
    rows: list[dict[str, Any]],
    metric: str,
    frozen_only: bool,
    include_time: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    used = []
    xs = []
    ys = []
    for row in rows:
        if frozen_only and bool(row.get("is_active")):
            continue
        if not finite_positive(row.get(metric)):
            continue
        if not finite_positive(row.get("kappa")):
            continue
        if not finite_positive(row.get("age_dt")):
            continue
        log_kappa = math.log(float(row["kappa"]) + EPS)
        log_age_dt = math.log(float(row["age_dt"]) + EPS)
        log_error = math.log(float(row[metric]) + EPS)
        features = [1.0, log_kappa, log_age_dt]
        if include_time:
            t_rem = max(1.0 - float(row.get("t", 0.0)), EPS)
            features.append(math.log(t_rem + EPS))
        xs.append(features)
        ys.append(log_error)
        used.append(row)
    if not xs:
        return np.empty((0, 3)), np.empty((0,)), []
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), used


def prepare_single_matrix(rows: list[dict[str, Any]], metric: str, x_key: str, frozen_only: bool) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    used = []
    xs = []
    ys = []
    for row in rows:
        if frozen_only and bool(row.get("is_active")):
            continue
        if not finite_positive(row.get(metric)):
            continue
        if not finite_positive(row.get(x_key)):
            continue
        xs.append([1.0, math.log(float(row[x_key]) + EPS)])
        ys.append(math.log(float(row[metric]) + EPS))
        used.append(row)
    if not xs:
        return np.empty((0, 2)), np.empty((0,)), []
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), used


def fit_model(rows: list[dict[str, Any]], metric: str, frozen_only: bool, include_time: bool = False) -> dict[str, Any]:
    x, y, used = prepare_matrix(rows, metric=metric, frozen_only=frozen_only, include_time=include_time)
    if x.shape[0] < 3:
        return {"count": int(x.shape[0]), "error": "not enough positive rows"}
    coeff, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coeff
    residual = y - pred
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {
        "count": int(x.shape[0]),
        "intercept": float(coeff[0]),
        "kappa_slope": float(coeff[1]),
        "age_dt_slope": float(coeff[2]),
        "remaining_time_slope": float(coeff[3]) if include_time and len(coeff) > 3 else None,
        "r2": r2,
        "rmse_log": float(np.sqrt(np.mean(residual * residual))),
        "rows": used,
        "pred": pred.tolist(),
        "obs": y.tolist(),
        "residual": residual.tolist(),
    }


def fit_single_model(rows: list[dict[str, Any]], metric: str, x_key: str, frozen_only: bool) -> dict[str, Any]:
    x, y, used = prepare_single_matrix(rows, metric=metric, x_key=x_key, frozen_only=frozen_only)
    if x.shape[0] < 2:
        return {"count": int(x.shape[0]), "error": "not enough positive rows"}
    coeff, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coeff
    residual = y - pred
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {
        "count": int(x.shape[0]),
        "intercept": float(coeff[0]),
        "slope": float(coeff[1]),
        "x_key": x_key,
        "r2": r2,
        "rmse_log": float(np.sqrt(np.mean(residual * residual))),
    }


def write_used_rows(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fields = [
        "source",
        "step",
        "block_id",
        "is_active",
        "age",
        "age_dt",
        "tau",
        "kappa",
        "rmse",
        "l2_mean",
        "l2_max",
        "token_count",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_fit(fit: dict[str, Any], output_dir: Path, prefix: str) -> list[str]:
    if "rows" not in fit:
        return []
    rows = fit["rows"]
    obs = np.asarray(fit["obs"], dtype=np.float64)
    pred = np.asarray(fit["pred"], dtype=np.float64)
    residual = np.asarray(fit["residual"], dtype=np.float64)
    log_kappa = np.asarray([math.log(float(row["kappa"]) + EPS) for row in rows])
    log_age_dt = np.asarray([math.log(float(row["age_dt"]) + EPS) for row in rows])
    steps = np.asarray([int(row["step"]) for row in rows])

    outputs = []
    specs = [
        ("log_error_vs_log_kappa", log_kappa, obs, "log(kappa)", "log(error)"),
        ("log_error_vs_log_age_dt", log_age_dt, obs, "log(age * dt)", "log(error)"),
        ("predicted_vs_observed", pred, obs, "predicted log(error)", "observed log(error)"),
        ("residual_by_step", steps, residual, "step", "log residual"),
    ]
    for name, x, y, xlabel, ylabel in specs:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax.scatter(x, y, s=18, alpha=0.55)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{prefix}: {name}")
        ax.grid(True, alpha=0.35)
        path = output_dir / f"{prefix}_{name}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(str(path))
    return outputs


def public_fit(fit: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in fit.items() if k not in {"rows", "pred", "obs", "residual"}}


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_records(args.input)
    write_used_rows(rows, output_dir / "probe_rows.csv")

    fits = {
        "all_positive": fit_model(rows, metric=args.metric, frozen_only=False),
        "frozen_positive": fit_model(rows, metric=args.metric, frozen_only=True),
        "all_positive_with_time": fit_model(rows, metric=args.metric, frozen_only=False, include_time=True),
        "frozen_positive_with_time": fit_model(rows, metric=args.metric, frozen_only=True, include_time=True),
        "frozen_kappa_only": fit_single_model(rows, metric=args.metric, x_key="kappa", frozen_only=True),
        "frozen_age_dt_only": fit_single_model(rows, metric=args.metric, x_key="age_dt", frozen_only=True),
        "all_kappa_only": fit_single_model(rows, metric=args.metric, x_key="kappa", frozen_only=False),
        "all_age_dt_only": fit_single_model(rows, metric=args.metric, x_key="age_dt", frozen_only=False),
    }
    outputs = []
    for name, fit in fits.items():
        outputs.extend(plot_fit(fit, output_dir, name))

    summary = {
        "inputs": args.input,
        "metric": args.metric,
        "raw_rows": len(rows),
        "fits": {name: public_fit(fit) for name, fit in fits.items()},
        "visualizations": outputs,
    }
    summary_path = output_dir / "error_probe_fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {summary_path}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
