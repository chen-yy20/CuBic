#!/usr/bin/env python3
"""Collect speedup and fidelity for every condition of autocode/demo_tradeoff_sweep.py.

Each method is compared with the uncached run of its own code path: the CUBIC repo origin,
the Jano repo origin, or the in-process origin of the Flux matrix and TaylorSeer scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.quality_metric import QualityEvaluator  # noqa: E402

METHOD_NAMES = {"teacache": "TeaCache", "pab": "PAB", "toca": "ToCa", "taylorseer": "TaylorSeer",
                "jano": "Jano", "cubic": "CUBIC"}


def pretty(method: str, label: str) -> str:
    if method == "jano":
        return "default" if label == "default" else "intervals 8/4"
    if method == "teacache":
        return f"\u03b4={label}"
    if method == "toca":
        return f"fresh ratio {label}"
    if method == "pab":
        if label.startswith("s"):
            s, c = label[1:].split("c")
            return f"self {s} / cross {c}"
        return f"range {label}"
    if method == "cubic":
        if label.startswith("r"):
            return f"\u03c1={label[1:]}"
        s, c = label[1:].split("c")
        return f"\u03c1={s}, ref \u03c1={c}"
    return label


def media(directory: Path, suffix: str, prefix: str | None = None) -> Path:
    files = [p for p in directory.rglob(f"*{suffix}")
             if "vis" not in p.parent.name and "logs" not in p.parts and (prefix is None or p.name.startswith(prefix))]
    if not files:
        raise FileNotFoundError(f"no {suffix} under {directory}")
    return max(files, key=lambda p: p.stat().st_mtime)


def e2e_ms(directory: Path) -> float:
    for path in directory.rglob("*params_metrics.json"):
        stats = json.loads(path.read_text()).get("timing_stats", {}).get("generate_e2e")
        if stats:
            return float(stats["min_ms"])
    raise FileNotFoundError(f"no timing under {directory}")


def collect(model: str, root: Path) -> list[dict]:
    suffix = ".mp4" if model == "wan" else ".png"
    points = []

    def add(name, method, out, ref, latency, ref_latency, label):
        points.append({"name": name, "method": METHOD_NAMES[method], "label": label, "output": str(out),
                       "reference": str(ref), "latency_ms": latency, "speedup": ref_latency / latency})

    for cond in sorted(p for p in root.iterdir() if (p / "DONE").exists()):
        name = cond.name
        if name in ("ori", "jano_ori"):
            continue
        method = name.split("_")[0]
        label = pretty(method, name[len(method) + 1:] or "default")
        if name == "taylorseer":
            records = json.loads((cond / "taylorseer_records.json").read_text())["records"]
            ref = media(cond, suffix, "ori_")
            for r in records[1:]:
                add(r["name"], "taylorseer", media(cond, suffix, r["name"] + "_"), ref,
                    r["latency_ms"], records[0]["latency_ms"], f"N={r['cache_interval']}")
            continue
        if model == "flux" and method == "cubic":
            record = json.loads(next(cond.rglob("records.jsonl")).read_text().splitlines()[-1])
            out = Path(record["image_path"]) if "image_path" in record else media(cond, suffix, "frontier")
            ref = Path(record["origin_path"]) if "origin_path" in record else media(cond, suffix, "origin")
            add(name, method, out, ref, record["latency_ms"], record["origin_latency_ms"], label)
            continue
        ref_dir = root / ("jano_ori" if method == "jano" and model != "firered" else "ori")
        add(name, method, media(cond, suffix), media(ref_dir, suffix), e2e_ms(cond), e2e_ms(ref_dir), label)
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO / "results/demo_tradeoff"))
    parser.add_argument("--models", nargs="+", default=["flux", "firered", "wan"])
    args = parser.parse_args()
    evaluator = QualityEvaluator(enable_distribution_metrics=False)
    summary = {}
    for model in args.models:
        points = collect(model, Path(args.root) / model)
        for p in points:
            if model == "wan":
                q = evaluator.evaluate_videos(p["output"], p["reference"])
            else:
                q = evaluator.evaluate_images(p["output"], p["reference"])
            p.update({k: q.get(k) for k in ("psnr", "ssim", "lpips")})
            print(f"{model:8s} {p['name']:18s} {p['speedup']:.2f}x  PSNR {p['psnr']:.2f}  LPIPS {p['lpips']}", flush=True)
        summary[model] = points
    out = Path(args.root) / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(out)


if __name__ == "__main__":
    main()
