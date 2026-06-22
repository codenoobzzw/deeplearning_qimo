#!/usr/bin/env python3
"""Remove the background from the cup photo and create a square RGBA asset.

When cv2/rembg/SAM are unavailable, the script uses a conservative foreground
heuristic: it flood-fills dark regions connected to the image border, keeps the
largest non-border dark component around the center, dilates it to include the
white handle and white line art, and saves every intermediate artifact.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


def flood_border(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        if mask[0, x]:
            q.append((0, x))
        if mask[h - 1, x]:
            q.append((h - 1, x))
    for y in range(h):
        if mask[y, 0]:
            q.append((y, 0))
        if mask[y, w - 1]:
            q.append((y, w - 1))
    while q:
        y, x = q.popleft()
        if visited[y, x] or not mask[y, x]:
            continue
        visited[y, x] = True
        if y > 0:
            q.append((y - 1, x))
        if y + 1 < h:
            q.append((y + 1, x))
        if x > 0:
            q.append((y, x - 1))
        if x + 1 < w:
            q.append((y, x + 1))
    return visited


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        q: deque[tuple[int, int]] = deque([(sy, sx)])
        coords: list[tuple[int, int]] = []
        visited[sy, sx] = True
        while q:
            y, x = q.popleft()
            coords.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
        if len(coords) > 20:
            comp = np.zeros_like(mask, dtype=bool)
            yy, xx = zip(*coords)
            comp[np.array(yy), np.array(xx)] = True
            comps.append(comp)
    return comps


def make_mask(img: Image.Image) -> tuple[Image.Image, dict]:
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    dark = gray < 105.0

    border_dark = flood_border(dark)
    seed = dark & ~border_dark
    comps = connected_components(seed)
    if comps:
        cx, cy = w / 2.0, h / 2.0
        scored = []
        for comp in comps:
            ys, xs = np.nonzero(comp)
            area = len(xs)
            dist = abs(float(xs.mean()) - cx) / w + abs(float(ys.mean()) - cy) / h
            scored.append((area - 0.25 * area * dist, comp))
        base = max(scored, key=lambda item: item[0])[1]
    else:
        base_img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(base_img)
        draw.ellipse((int(0.22 * w), int(0.02 * h), int(0.80 * w), int(0.98 * h)), fill=1)
        base = np.asarray(base_img) > 0

    base_img = Image.fromarray((base.astype(np.uint8) * 255), mode="L")
    dilated = base_img.filter(ImageFilter.MaxFilter(95))
    dilated_np = np.asarray(dilated) > 0

    # Keep white handle and white doodles only near the dark body, avoiding the
    # far white desktop background.
    bright = gray > 155.0
    alpha_np = (base | (bright & dilated_np)).astype(np.uint8) * 255
    alpha = Image.fromarray(alpha_np, mode="L")
    alpha = alpha.filter(ImageFilter.MaxFilter(31)).filter(ImageFilter.MinFilter(17)).filter(ImageFilter.GaussianBlur(3.0))

    # A soft central prior prevents the mask from growing into far background.
    prior = Image.new("L", (w, h), 0)
    pdraw = ImageDraw.Draw(prior)
    pdraw.rounded_rectangle(
        (int(0.20 * w), int(0.01 * h), int(0.82 * w), int(0.985 * h)),
        radius=int(0.20 * w),
        fill=255,
    )
    alpha = ImageChops_multiply(alpha, prior)
    bbox = alpha.getbbox()
    used_shape_prior = False
    if bbox is None or (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 0.18 * w * h:
        prior = Image.new("L", (w, h), 0)
        pdraw = ImageDraw.Draw(prior)
        # Conservative full-cup prior for the provided thermos photo. It keeps
        # the lid, cylindrical body, base, and the white front handle.
        pdraw.rounded_rectangle(
            (int(0.25 * w), int(0.03 * h), int(0.77 * w), int(0.965 * h)),
            radius=int(0.15 * w),
            fill=255,
        )
        pdraw.ellipse((int(0.22 * w), int(0.12 * h), int(0.53 * w), int(0.38 * h)), fill=255)
        pdraw.ellipse((int(0.30 * w), int(0.91 * h), int(0.78 * w), int(0.99 * h)), fill=255)
        prior_np = np.asarray(prior) > 0
        darkish = ((gray < 128.0) & prior_np).astype(np.uint8) * 255
        darkish_img = Image.fromarray(darkish, mode="L").filter(ImageFilter.MaxFilter(51))
        near_dark = np.asarray(darkish_img) > 0
        bright_near = ((gray > 150.0) & near_dark & prior_np).astype(np.uint8) * 255
        combined = np.maximum(darkish, bright_near)
        alpha = Image.fromarray(combined, mode="L")
        alpha = alpha.filter(ImageFilter.MaxFilter(61)).filter(ImageFilter.MinFilter(15)).filter(ImageFilter.GaussianBlur(2.0))
        alpha = ImageChops_multiply(alpha, prior)
        if alpha.getbbox() is None or (alpha.getbbox()[2] - alpha.getbbox()[0]) * (alpha.getbbox()[3] - alpha.getbbox()[1]) < 0.12 * w * h:
            alpha = prior.filter(ImageFilter.GaussianBlur(2.0))
        bbox = alpha.getbbox()
        used_shape_prior = True
    meta = {
        "method": "heuristic_dark_component_with_soft_prior",
        "cv2_available": False,
        "bbox": list(bbox) if bbox else None,
        "used_shape_prior_fallback": used_shape_prior,
        "note": "rembg/SAM/cv2 were not required for this fallback path; inspect before_after before using for Zero123.",
    }
    return alpha, meta


def ImageChops_multiply(a: Image.Image, b: Image.Image) -> Image.Image:
    arr = (np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32) / 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def square_rgba(img: Image.Image, alpha: Image.Image, size: int, padding: int) -> tuple[Image.Image, Image.Image]:
    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)
    bbox = alpha.getbbox()
    if not bbox:
        bbox = (0, 0, img.width, img.height)
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(img.width, right + padding)
    bottom = min(img.height, bottom + padding)
    crop = rgba.crop((left, top, right, bottom))
    side = max(crop.width, crop.height)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.alpha_composite(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    canvas = canvas.resize((size, size), Image.Resampling.LANCZOS)
    preview = Image.new("RGB", (side, side), (235, 235, 235))
    preview.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2), crop)
    preview = preview.resize((size, size), Image.Resampling.LANCZOS)
    return canvas, preview


def before_after(original: Image.Image, rgba: Image.Image, out_path: Path) -> None:
    h = 720
    left = original.convert("RGB")
    left = left.resize((int(left.width * h / left.height), h), Image.Resampling.LANCZOS)
    checker = Image.new("RGB", rgba.size, (225, 225, 225))
    draw = ImageDraw.Draw(checker)
    step = 32
    for y in range(0, rgba.height, step):
        for x in range(0, rgba.width, step):
            if (x // step + y // step) % 2 == 0:
                draw.rectangle((x, y, x + step - 1, y + step - 1), fill=(245, 245, 245))
    right = checker.convert("RGBA")
    right.alpha_composite(rgba)
    right = right.convert("RGB").resize((h, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (left.width + right.width, h), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="inputs/object_C/cup.jpg")
    parser.add_argument("--rgba-output", default="third_party/threestudio/load/images/cup_rgba.png")
    parser.add_argument("--copy-output", default="outputs/task1/object_C_image3d/cup_rgba.png")
    parser.add_argument("--mask-output", default="outputs/task1/object_C_image3d/mask.png")
    parser.add_argument("--preview-output", default="outputs/task1/object_C_image3d/crop_preview.png")
    parser.add_argument("--figure-output", default="outputs/task1/figures/object_C_bg_removal.png")
    parser.add_argument("--meta-output", default="outputs/task1/object_C_image3d/preprocess_meta.json")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--padding", type=int, default=24)
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Missing cup image: {inp}")
    img = ImageOps.exif_transpose(Image.open(inp)).convert("RGB")
    alpha, meta = make_mask(img)
    rgba, preview = square_rgba(img, alpha, args.size, args.padding)

    for path in [args.rgba_output, args.copy_output, args.mask_output, args.preview_output, args.meta_output]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    rgba.save(args.rgba_output)
    rgba.save(args.copy_output)
    alpha.save(args.mask_output)
    preview.save(args.preview_output)
    before_after(img, rgba, Path(args.figure_output))
    meta.update({"input": str(inp), "rgba_output": args.rgba_output, "size": args.size, "padding": args.padding})
    Path(args.meta_output).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
