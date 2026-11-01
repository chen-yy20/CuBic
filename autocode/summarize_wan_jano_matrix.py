#!/usr/bin/env python3
"""Aggregate paired Wan CUBIC-vs-Jano prompt/seed runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def finite(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    numbers = finite(values)
    if not numbers:
        return None, None
    return mean(numbers), stdev(numbers) if len(numbers) > 1 else 0.0


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latency(payload: dict) -> float:
    return float(payload["timing_stats"]["generate_e2e"]["avg_ms"])


def aggregate(method: str, samples: list[dict], condition: str) -> dict:
    row = {"method": method, "condition": condition, "samples": len(samples)}
    for key in ("speedup", "psnr", "ssim", "lpips"):
        row[key], row[f"{key}_std"] = mean_std([sample.get(key) for sample in samples])
    return row


def main() -> int:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    payloads = {}
    for run in manifest["runs"]:
        key = (run["repo"], run["method"], run["prompt_id"], int(run["seed"]))
        payloads[key] = load(run["metrics"])

    samples_by_method = defaultdict(list)
    samples_by_prompt = defaultdict(list)
    method_specs = (("CUBIC", "cubic", "cubic"), ("Jano", "jano", "jano"))
    for method_name, repo, accelerated_method in method_specs:
        for prompt in manifest["prompts"]:
            prompt_words = [
                word
                for word in prompt.lower().split()
                if word not in {"a", "an", "the", "in", "on", "at", "and", "or", "of", "with", "by"}
            ]
            prompt_key = "_".join(prompt_words[:4])
            for seed in manifest["seeds"]:
                origin = payloads[(repo, "ori", prompt_key, int(seed))]
                result = payloads[(repo, accelerated_method, prompt_key, int(seed))]
                quality = result.get("quality_metrics") or {}
                sample = {
                    "method": method_name,
                    "prompt": prompt,
                    "prompt_id": prompt_key,
                    "seed": int(seed),
                    "origin_ms": latency(origin),
                    "method_ms": latency(result),
                    "speedup": latency(origin) / latency(result),
                    "psnr": quality.get("psnr"),
                    "ssim": quality.get("ssim"),
                    "lpips": quality.get("lpips"),
                }
                samples_by_method[method_name].append(sample)
                samples_by_prompt[(method_name, prompt_key)].append(sample)

    rows = [
        aggregate(method, samples_by_method[method], "all_3_prompts_3_seeds")
        for method in ("CUBIC", "Jano")
    ]
    per_prompt = [
        aggregate(method, samples, prompt_key)
        for (method, prompt_key), samples in sorted(samples_by_prompt.items())
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("wan_cubic_vs_jano.json").write_text(
        json.dumps(
            {
                "protocol": {"prompts": manifest["prompts"], "seeds": manifest["seeds"]},
                "rows": rows,
                "per_prompt": per_prompt,
                "samples": samples_by_method,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with output_dir.joinpath("wan_cubic_vs_jano.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
