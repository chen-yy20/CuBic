#!/usr/bin/env python3
"""Export the demo trade-off sweep into web assets for docs/index.html."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "results/demo_tradeoff/summary.json"
# The FireRed Jano entry point runs without acceleration (~1.0x), and the paper has no FireRed Jano baseline.
EXCLUDE = {("firered", "jano"), ("firered", "jano_fast")}
OUT = REPO / "docs/assets/demo"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
IMAGE_WIDTH = 768
VIDEO_WIDTH = 576


def export_image(src: str, dst: Path) -> None:
    image = cv2.imread(src)
    h, w = image.shape[:2]
    image = cv2.resize(image, (IMAGE_WIDTH, round(h * IMAGE_WIDTH / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), image, [cv2.IMWRITE_JPEG_QUALITY, 86, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])


def export_diff(src: str, ref: str, dst: Path, gain: float = 4.0) -> None:
    a = cv2.imread(src).astype("float32")
    b = cv2.imread(ref).astype("float32")
    if a.shape != b.shape:
        a = cv2.resize(a, (b.shape[1], b.shape[0]))
    err = (abs(a - b).mean(axis=2) * gain).clip(0, 255).astype("uint8")
    heat = cv2.applyColorMap(err, cv2.COLORMAP_INFERNO)
    h, w = heat.shape[:2]
    heat = cv2.resize(heat, (IMAGE_WIDTH, round(h * IMAGE_WIDTH / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), heat, [cv2.IMWRITE_JPEG_QUALITY, 82, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])


def export_video(src: str, dst: Path) -> None:
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                    "-vf", f"scale={VIDEO_WIDTH}:-2", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "29",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-ss", "1", "-i", str(dst),
                    "-frames:v", "1", "-q:v", "4", str(dst.with_suffix(".jpg"))], check=True)


def main():
    summary = json.loads(SUMMARY.read_text())
    data = {}
    for model, points in summary.items():
        folder = OUT / model
        folder.mkdir(parents=True, exist_ok=True)
        video = model == "wan"
        ext = ".mp4" if video else ".jpg"
        export = export_video if video else export_image
        refs = {}
        entries = []
        for p in points:
            if (model, p["name"]) in EXCLUDE:
                continue
            key = hashlib.md5(Path(p["reference"]).read_bytes()).hexdigest()
            if key not in refs:
                refs[key] = f"ref{len(refs)}"
                export(p["reference"], folder / f"{refs[key]}{ext}")
            ref = key
            export(p["output"], folder / f"{p['name']}{ext}")
            if not video:
                export_diff(p["output"], p["reference"], folder / f"{p['name']}_diff.jpg")
            entries.append({
                "name": p["name"], "method": p["method"], "label": p["label"],
                "speedup": round(p["speedup"], 3), "latency_s": round(p["latency_ms"] / 1000, 2),
                "psnr": round(p["psnr"], 2), "ssim": round(p["ssim"], 4), "lpips": round(p["lpips"], 4),
                "src": f"assets/demo/{model}/{p['name']}{ext}", "ref": f"assets/demo/{model}/{refs[ref]}{ext}",
                **({} if video else {"diff": f"assets/demo/{model}/{p['name']}_diff.jpg"}),
            })
        data[model] = entries
        print(model, len(entries), "points", len(refs), "references")
    (OUT / "data.js").write_text("window.DEMO_DATA = " + json.dumps(data, indent=1) + ";\n")


if __name__ == "__main__":
    main()
