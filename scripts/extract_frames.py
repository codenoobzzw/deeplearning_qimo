#!/usr/bin/env python3
"""Prepare object-A images from a video or a photo directory.

The script deliberately keeps a JSON audit trail: every input frame/image has a
blur score, output path, and selection decision. It uses only Pillow and NumPy so
it can run in the provided PG environment.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def laplacian_variance(img: Image.Image) -> float:
    gray = ImageOps.grayscale(img).resize((256, 256))
    arr = np.asarray(gray, dtype=np.float32)
    padded = np.pad(arr, 1, mode="edge")
    lap = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(np.var(lap))


def perceptual_thumb(img: Image.Image) -> np.ndarray:
    thumb = ImageOps.grayscale(img).resize((32, 32))
    arr = np.asarray(thumb, dtype=np.float32)
    arr -= arr.mean()
    denom = float(np.linalg.norm(arr)) or 1.0
    return arr.reshape(-1) / denom


def resize_keep_aspect(img: Image.Image, max_width: int) -> Image.Image:
    if max_width <= 0 or img.width <= max_width:
        return img
    scale = max_width / float(img.width)
    size = (max_width, max(1, int(round(img.height * scale))))
    return img.resize(size, Image.Resampling.LANCZOS)


def discover_inputs(input_path: Path) -> tuple[str, list[Path]]:
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTS:
        return "video", [input_path]
    if input_path.is_dir():
        videos = sorted(p for p in input_path.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if videos:
            return "video", [videos[0]]
        images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        return "images", images
    return "missing", []


def extract_video_frames(video: Path, fps: float, tmp_dir: Path) -> list[Path]:
    pattern = tmp_dir / "frame_%05d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(pattern),
    ]
    subprocess.run(cmd, check=True)
    return sorted(tmp_dir.glob("frame_*.jpg"))


def select_and_write(
    candidates: list[Path],
    output_dir: Path,
    max_width: int,
    blur_quantile: float,
    similarity_threshold: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*"):
        if old.is_file():
            old.unlink()

    records: list[dict] = []
    scored: list[tuple[Path, Image.Image, float, np.ndarray]] = []
    for path in candidates:
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img).convert("RGB")
            score = laplacian_variance(img)
            scored.append((path, img, score, perceptual_thumb(img)))
        except Exception as exc:  # noqa: BLE001
            records.append({"input": str(path), "selected": False, "reason": f"read_failed: {exc}"})

    if scored:
        scores = np.array([s[2] for s in scored], dtype=np.float32)
        threshold = float(np.quantile(scores, blur_quantile)) if len(scores) > 3 else -math.inf
    else:
        threshold = math.inf

    selected_thumbs: list[np.ndarray] = []
    selected_count = 0
    for idx, (path, img, score, thumb) in enumerate(scored):
        reason = "selected"
        selected = True
        if score < threshold:
            selected = False
            reason = f"blur_score_below_quantile_{blur_quantile:.2f}"
        if selected and selected_thumbs:
            max_sim = max(float(np.dot(thumb, prev)) for prev in selected_thumbs)
            if max_sim > similarity_threshold:
                selected = False
                reason = f"near_duplicate_similarity_{max_sim:.3f}"
        out_path = None
        if selected:
            selected_count += 1
            selected_thumbs.append(thumb)
            out_img = resize_keep_aspect(img, max_width=max_width)
            out_path = output_dir / f"{selected_count:04d}.jpg"
            out_img.save(out_path, quality=95)
        records.append(
            {
                "input": str(path),
                "index": idx,
                "selected": selected,
                "reason": reason,
                "laplacian_variance": score,
                "output": str(out_path) if out_path else None,
                "original_size": [img.width, img.height],
            }
        )

    return {
        "candidate_count": len(candidates),
        "selected_count": selected_count,
        "blur_quantile": blur_quantile,
        "similarity_threshold": similarity_threshold,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="inputs/object_A", help="Photo directory or video file.")
    parser.add_argument("--output", default="data/object_A/input", help="Output image directory.")
    parser.add_argument("--stats", default="outputs/task1/object_A_frame_stats.json")
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--max-width", type=int, default=1400)
    parser.add_argument("--blur-quantile", type=float, default=0.05)
    parser.add_argument("--similarity-threshold", type=float, default=0.995)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    mode, paths = discover_inputs(input_path)
    if mode == "missing" or not paths:
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = {
            "status": "missing_input",
            "input": str(input_path),
            "message": "Place object A photos or a video under inputs/object_A/.",
        }
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 1

    with tempfile.TemporaryDirectory(prefix="object_A_frames_") as tmp:
        if mode == "video":
            candidates = extract_video_frames(paths[0], args.fps, Path(tmp))
        else:
            candidates = paths
        stats = select_and_write(
            candidates,
            output_dir,
            args.max_width,
            args.blur_quantile,
            args.similarity_threshold,
        )
        stats.update({"status": "ok", "mode": mode, "source": [str(p) for p in paths]})
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: stats[k] for k in ["status", "mode", "candidate_count", "selected_count"]}, ensure_ascii=False))
        return 0 if stats["selected_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
