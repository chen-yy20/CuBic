#!/usr/bin/env python3
"""Summarize one-prompt Wan CUBIC-vs-Jano comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Wan CUBIC and Jano outputs")
    parser.add_argument("--cubic_origin", required=True)
    parser.add_argument("--cubic_result", required=True)
    parser.add_argument("--jano_origin", required=True)
    parser.add_argument("--jano_result", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latency(payload: dict) -> float:
    return float(payload["timing_stats"]["generate_e2e"]["avg_ms"])


def row(method: str, origin: dict, result: dict) -> dict:
    quality = result.get("quality_metrics") or {}
    return {
        "method": method,
        "condition": "default",
        "samples": 1,
        "speedup": latency(origin) / latency(result),
        "speedup_std": 0.0,
        "psnr": quality.get("psnr"),
        "psnr_std": 0.0,
        "ssim": quality.get("ssim"),
        "ssim_std": 0.0,
        "lpips": quality.get("lpips"),
        "lpips_std": 0.0,
    }


def main() -> int:
    args = parse_args()
    rows = [
        row("CUBIC", load(args.cubic_origin), load(args.cubic_result)),
        row("Jano", load(args.jano_origin), load(args.jano_result)),
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
