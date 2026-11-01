#!/usr/bin/env python3
"""Visualize a single Task 3 run's active-token trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one Cubic params/metrics JSON")
    parser.add_argument("--params_json", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params_path = Path(args.params_json)
    data = json.loads(params_path.read_text(encoding="utf-8"))
    trace = (((data.get("quality_metrics") or {}).get("cubic_step_trace") or {}).get("steps") or [])
    if not trace:
        raise SystemExit(f"missing cubic_step_trace.steps in {params_path}")

    output_dir = Path(args.output_dir) if args.output_dir else params_path.parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    title = args.title or params_path.parent.name

    steps = [int(item["step"]) for item in trace]
    ratios = [float(item["active_ratio"]) for item in trace]
    latencies = [float(item["latency_ms"]) for item in trace]
    phases = [str(item["phase"]) for item in trace]

    colors = []
    for phase in phases:
        if "anchor" in phase:
            colors.append("#d55e00")
        elif "selective" in phase:
            colors.append("#0072b2")
        else:
            colors.append("#009e73")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].bar(steps, ratios, color=colors, width=0.82)
    axes[0].set_ylabel("Active token ratio")
    axes[0].set_ylim(0, 1.08)
    axes[0].grid(True, axis="y", alpha=0.35)
    axes[0].set_title(f"{title}: active ratio by denoising step")

    axes[1].plot(steps, latencies, color="#444444", linewidth=1.5)
    axes[1].scatter(steps, latencies, c=colors, s=18)
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_xlabel("Denoising step")
    axes[1].grid(True, alpha=0.35)

    output_path = output_dir / "single_run_active_ratio_latency.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "params_json": str(params_path),
                "outputs": [str(output_path)],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
