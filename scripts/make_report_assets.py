#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def contact_sheet(input_dir: Path, output: Path, title: str) -> None:
    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        return
    thumb_w, thumb_h = 260, 350
    pad = 16
    title_h = 48
    cols = 3
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_w + (cols + 1) * pad, rows * (thumb_h + 34) + (rows + 1) * pad + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(20)
    small = load_font(14)
    draw.text((pad, 12), title, fill=(20, 20, 20), font=font)
    for idx, path in enumerate(images):
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = pad + (idx % cols) * (thumb_w + pad)
        y = title_h + pad + (idx // cols) * (thumb_h + 34 + pad)
        frame = Image.new("RGB", (thumb_w, thumb_h), (245, 245, 245))
        frame.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
        canvas.paste(frame, (x, y))
        draw.text((x, y + thumb_h + 6), path.name, fill=(40, 40, 40), font=small)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def status_figure(output: Path) -> None:
    items = [
        ("3DGS deps", "ready"),
        ("Object A COLMAP", "failed: no initial pair"),
        ("Object C RGBA", "heuristic done"),
        ("threestudio", "repo/config ready"),
        ("CALVIN HF", "blocked: SSL"),
        ("LeRobot CLI", "help ready"),
    ]
    w, h = 1000, 360
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(22)
    small = load_font(16)
    draw.text((24, 18), "实验状态摘要（来自真实运行日志）", fill=(20, 20, 20), font=font)
    y = 70
    colors = {"ready": (44, 132, 84), "heuristic": (166, 112, 28), "repo": (44, 92, 160), "failed": (180, 60, 50), "blocked": (180, 60, 50), "help": (44, 132, 84)}
    for name, status in items:
        key = status.split(":")[0].split()[0]
        color = colors.get(key, (80, 80, 80))
        draw.rounded_rectangle((24, y, 250, y + 34), radius=6, fill=(238, 238, 238))
        draw.text((36, y + 6), name, fill=(20, 20, 20), font=small)
        draw.rounded_rectangle((270, y, 960, y + 34), radius=6, fill=color)
        draw.text((284, y + 6), status, fill="white", font=small)
        y += 46
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    contact_sheet(ROOT / "inputs/object_A", ROOT / "report/figs/object_A_uploaded_photos.png", "Object A uploaded photos")
    contact_sheet(ROOT / "data/object_A_all/input", ROOT / "report/figs/object_A_prepared_photos.png", "Object A prepared images used for COLMAP retry")
    status_figure(ROOT / "report/figs/experiment_status.png")
    print("report assets generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
