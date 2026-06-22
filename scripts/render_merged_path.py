#!/usr/bin/env python3
"""CPU fallback point-cloud renderer for merged Gaussian PLY previews.

This is not a replacement for the official 3DGS renderer. It is provided as a
sanity/fallback preview path and is labelled as such in generated reports.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ply_utils import read_ply

SH_C0 = 0.28209479177387814


def colors_from_arrays(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if all(k in arrays for k in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        rgb = np.stack([arrays["f_dc_0"], arrays["f_dc_1"], arrays["f_dc_2"]], axis=1) * SH_C0 + 0.5
        return np.clip(rgb, 0, 1)
    return np.full((len(arrays["x"]), 3), 0.8, dtype=np.float32)


def look_at(points: np.ndarray, center: np.ndarray, angle: float, elev: float, radius: float) -> np.ndarray:
    eye = center + np.array([math.cos(angle) * radius, math.sin(angle) * radius, math.sin(elev) * radius * 0.35], dtype=np.float32)
    forward = center - eye
    forward /= np.linalg.norm(forward) + 1e-12
    up = np.array([0, 0, 1], dtype=np.float32)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    up2 = np.cross(right, forward)
    rel = points - eye
    return np.stack([rel @ right, rel @ up2, rel @ forward], axis=1)


def render_frame(cam: np.ndarray, rgb: np.ndarray, width: int, height: int, point_radius: int) -> Image.Image:
    z = cam[:, 2]
    valid = z > 0.05
    cam = cam[valid]
    rgb = rgb[valid]
    z = cam[:, 2]
    f = min(width, height) * 1.35
    x = (cam[:, 0] / (z + 1e-6) * f + width / 2).astype(np.int32)
    y = (-cam[:, 1] / (z + 1e-6) * f + height / 2).astype(np.int32)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    order = np.argsort(z[inside])[::-1]
    x = x[inside][order]
    y = y[inside][order]
    c = (rgb[inside][order] * 255).astype(np.uint8)
    img = Image.new("RGB", (width, height), (18, 18, 20))
    draw = ImageDraw.Draw(img)
    r = max(1, point_radius)
    for px, py, col in zip(x.tolist(), y.tolist(), c.tolist()):
        draw.rectangle((px - r, py - r, px + r, py + r), fill=tuple(col))
    return img


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/task1/merged_scene/point_cloud.ply")
    parser.add_argument("--video-output", default="outputs/task1/videos/merged_scene_walkthrough.mp4")
    parser.add_argument("--frames-dir", default="outputs/task1/merged_scene/preview_frames")
    parser.add_argument("--report-figs-dir", default="report/figs")
    parser.add_argument("--num-frames", type=int, default=96)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--point-radius", type=int, default=2)
    args = parser.parse_args()
    props, arrays = read_ply(Path(args.input))
    points = np.stack([arrays["x"], arrays["y"], arrays["z"]], axis=1).astype(np.float32)
    rgb = colors_from_arrays(arrays)
    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_figs_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    robust_min = np.percentile(points, 1, axis=0)
    robust_max = np.percentile(points, 99, axis=0)
    center = ((robust_min + robust_max) * 0.5).astype(np.float32)
    diag = float(np.linalg.norm(robust_max - robust_min)) or 1.0
    for i in range(args.num_frames):
        angle = 2 * math.pi * i / args.num_frames
        cam = look_at(points, center, angle, elev=0.25, radius=diag * 1.05)
        img = render_frame(cam, rgb, args.width, args.height, args.point_radius)
        frame_path = frames_dir / f"frame_{i:04d}.png"
        img.save(frame_path)
        if i in np.linspace(0, args.num_frames - 1, 8, dtype=int).tolist():
            img.save(report_dir / f"task1_merged_{i:04d}.png")
    Path(args.video_output).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "24",
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-pix_fmt",
            "yuv420p",
            str(args.video_output),
        ],
        check=True,
    )
    print(f"wrote fallback preview video: {args.video_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
