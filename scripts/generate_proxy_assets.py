#!/usr/bin/env python3
"""Generate lightweight 3D proxy assets for the full HW3 rendering pipeline.

The official threestudio/Zero123/Mip-NeRF360 routes remain in the repository.
This script is a deterministic fallback that makes the end-to-end asset fusion
pipeline runnable when external model weights or large datasets are unavailable.
It writes colored OBJ meshes for object B/C and a counter-like background mesh.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Mesh:
    vertices: list[tuple[float, float, float, float, float, float]]
    faces: list[tuple[int, int, int]]

    def __init__(self) -> None:
        self.vertices = []
        self.faces = []

    def add_vertex(self, xyz: tuple[float, float, float], rgb: tuple[float, float, float]) -> int:
        self.vertices.append((*xyz, *rgb))
        return len(self.vertices)

    def add_face(self, a: int, b: int, c: int) -> None:
        self.faces.append((a, b, c))

    def extend(self, other: "Mesh") -> None:
        off = len(self.vertices)
        self.vertices.extend(other.vertices)
        self.faces.extend((a + off, b + off, c + off) for a, b, c in other.faces)


def write_obj(mesh: Mesh, path: Path, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {comment}\n")
        for x, y, z, r, g, b in mesh.vertices:
            f.write(f"v {x:.7f} {y:.7f} {z:.7f} {r:.5f} {g:.5f} {b:.5f}\n")
        for a, b, c in mesh.faces:
            f.write(f"f {a} {b} {c}\n")


def uv_ellipsoid(
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    color: tuple[float, float, float],
    rings: int = 18,
    segments: int = 36,
) -> Mesh:
    mesh = Mesh()
    idx: list[list[int]] = []
    for i in range(rings + 1):
        phi = math.pi * i / rings
        row = []
        for j in range(segments):
            theta = 2 * math.pi * j / segments
            x = center[0] + radii[0] * math.sin(phi) * math.cos(theta)
            y = center[1] + radii[1] * math.sin(phi) * math.sin(theta)
            z = center[2] + radii[2] * math.cos(phi)
            row.append(mesh.add_vertex((x, y, z), color))
        idx.append(row)
    for i in range(rings):
        for j in range(segments):
            a = idx[i][j]
            b = idx[i][(j + 1) % segments]
            c = idx[i + 1][j]
            d = idx[i + 1][(j + 1) % segments]
            if i != 0:
                mesh.add_face(a, c, b)
            if i != rings - 1:
                mesh.add_face(b, c, d)
    return mesh


def cylinder(
    center: tuple[float, float, float],
    radius: float,
    height: float,
    color: tuple[float, float, float],
    segments: int = 48,
    rings: int = 8,
) -> Mesh:
    mesh = Mesh()
    idx: list[list[int]] = []
    for i in range(rings + 1):
        z = center[2] - height / 2 + height * i / rings
        row = []
        for j in range(segments):
            theta = 2 * math.pi * j / segments
            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            row.append(mesh.add_vertex((x, y, z), color))
        idx.append(row)
    for i in range(rings):
        for j in range(segments):
            a = idx[i][j]
            b = idx[i][(j + 1) % segments]
            c = idx[i + 1][j]
            d = idx[i + 1][(j + 1) % segments]
            mesh.add_face(a, c, b)
            mesh.add_face(b, c, d)
    bottom = mesh.add_vertex((center[0], center[1], center[2] - height / 2), color)
    top = mesh.add_vertex((center[0], center[1], center[2] + height / 2), color)
    for j in range(segments):
        mesh.add_face(bottom, idx[0][(j + 1) % segments], idx[0][j])
        mesh.add_face(top, idx[-1][j], idx[-1][(j + 1) % segments])
    return mesh


def cone_x(
    base_center: tuple[float, float, float],
    length: float,
    radius: float,
    color: tuple[float, float, float],
    segments: int = 32,
) -> Mesh:
    mesh = Mesh()
    base = []
    for j in range(segments):
        theta = 2 * math.pi * j / segments
        base.append(mesh.add_vertex((base_center[0], base_center[1] + radius * math.cos(theta), base_center[2] + radius * math.sin(theta)), color))
    center = mesh.add_vertex(base_center, color)
    apex = mesh.add_vertex((base_center[0] + length, base_center[1], base_center[2]), color)
    for j in range(segments):
        a = base[j]
        b = base[(j + 1) % segments]
        mesh.add_face(apex, a, b)
        mesh.add_face(center, b, a)
    return mesh


def torus_oval(
    center: tuple[float, float, float],
    major_x: float,
    major_z: float,
    tube: float,
    color: tuple[float, float, float],
    segments: int = 64,
    tube_segments: int = 12,
) -> Mesh:
    mesh = Mesh()
    idx: list[list[int]] = []
    for i in range(segments):
        t = 2 * math.pi * i / segments
        cx = center[0] + major_x * math.cos(t)
        cy = center[1]
        cz = center[2] + major_z * math.sin(t)
        tangent = np.array([-major_x * math.sin(t), 0.0, major_z * math.cos(t)], dtype=np.float32)
        tangent /= np.linalg.norm(tangent) + 1e-8
        normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        binormal = np.cross(tangent, normal)
        binormal /= np.linalg.norm(binormal) + 1e-8
        row = []
        for j in range(tube_segments):
            u = 2 * math.pi * j / tube_segments
            offset = normal * (math.cos(u) * tube) + binormal * (math.sin(u) * tube)
            p = np.array([cx, cy, cz], dtype=np.float32) + offset
            row.append(mesh.add_vertex((float(p[0]), float(p[1]), float(p[2])), color))
        idx.append(row)
    for i in range(segments):
        for j in range(tube_segments):
            a = idx[i][j]
            b = idx[(i + 1) % segments][j]
            c = idx[i][(j + 1) % tube_segments]
            d = idx[(i + 1) % segments][(j + 1) % tube_segments]
            mesh.add_face(a, b, c)
            mesh.add_face(c, b, d)
    return mesh


def box(center: tuple[float, float, float], size: tuple[float, float, float], color: tuple[float, float, float]) -> Mesh:
    mesh = Mesh()
    sx, sy, sz = [v / 2 for v in size]
    cx, cy, cz = center
    corners = [
        (cx - sx, cy - sy, cz - sz),
        (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz),
        (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz),
        (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz),
        (cx - sx, cy + sy, cz + sz),
    ]
    ids = [mesh.add_vertex(p, color) for p in corners]
    quads = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    for a, b, c, d in quads:
        mesh.add_face(ids[a], ids[b], ids[c])
        mesh.add_face(ids[a], ids[c], ids[d])
    return mesh


def build_duck() -> Mesh:
    mesh = Mesh()
    yellow = (1.0, 0.78, 0.08)
    orange = (1.0, 0.35, 0.05)
    black = (0.02, 0.02, 0.02)
    mesh.extend(uv_ellipsoid((0.0, 0.0, 0.42), (0.62, 0.36, 0.32), yellow))
    mesh.extend(uv_ellipsoid((0.42, 0.0, 0.78), (0.26, 0.24, 0.24), yellow, rings=16, segments=32))
    mesh.extend(cone_x((0.62, 0.0, 0.76), 0.28, 0.09, orange))
    mesh.extend(uv_ellipsoid((0.1, -0.33, 0.45), (0.28, 0.055, 0.16), (0.94, 0.65, 0.05), rings=12, segments=24))
    mesh.extend(uv_ellipsoid((0.1, 0.33, 0.45), (0.28, 0.055, 0.16), (0.94, 0.65, 0.05), rings=12, segments=24))
    mesh.extend(uv_ellipsoid((0.51, -0.18, 0.87), (0.035, 0.025, 0.035), black, rings=8, segments=12))
    mesh.extend(uv_ellipsoid((0.51, 0.18, 0.87), (0.035, 0.025, 0.035), black, rings=8, segments=12))
    mesh.extend(uv_ellipsoid((-0.48, 0.0, 0.70), (0.16, 0.13, 0.16), yellow, rings=10, segments=18))
    return mesh


def dominant_dark_color(image_path: Path) -> tuple[float, float, float]:
    if not image_path.exists():
        return (0.035, 0.035, 0.04)
    img = Image.open(image_path).convert("RGBA").resize((96, 96))
    arr = np.asarray(img).astype(np.float32) / 255.0
    alpha = arr[..., 3] > 0.35
    rgb = arr[..., :3][alpha]
    if len(rgb) == 0:
        return (0.035, 0.035, 0.04)
    lum = rgb.mean(axis=1)
    sample = rgb[lum < np.quantile(lum, 0.45)]
    if len(sample) == 0:
        sample = rgb
    c = np.clip(np.median(sample, axis=0), 0.02, 0.35)
    return (float(c[0]), float(c[1]), float(c[2]))


def build_cup(cup_rgba: Path) -> Mesh:
    mesh = Mesh()
    body = dominant_dark_color(cup_rgba)
    white = (0.93, 0.92, 0.86)
    mesh.extend(cylinder((0.0, 0.0, 0.62), 0.34, 1.12, body, segments=56, rings=12))
    mesh.extend(cylinder((0.0, 0.0, 1.22), 0.29, 0.18, (0.02, 0.02, 0.025), segments=56, rings=3))
    mesh.extend(cylinder((0.0, 0.0, 1.34), 0.22, 0.10, (0.02, 0.02, 0.025), segments=48, rings=2))
    mesh.extend(torus_oval((0.0, -0.41, 0.74), 0.20, 0.34, 0.035, white))
    mesh.extend(uv_ellipsoid((0.0, -0.345, 0.78), (0.13, 0.012, 0.10), white, rings=10, segments=18))
    mesh.extend(uv_ellipsoid((-0.11, -0.347, 0.90), (0.035, 0.010, 0.035), white, rings=8, segments=12))
    mesh.extend(uv_ellipsoid((0.11, -0.347, 0.90), (0.035, 0.010, 0.035), white, rings=8, segments=12))
    return mesh


def build_counter_background() -> Mesh:
    mesh = Mesh()
    mesh.extend(box((0.0, 0.0, -0.08), (5.8, 4.2, 0.16), (0.82, 0.80, 0.75)))
    mesh.extend(box((0.0, 1.85, 1.15), (5.8, 0.18, 2.5), (0.64, 0.66, 0.68)))
    mesh.extend(box((-2.4, 0.0, 0.55), (0.18, 4.2, 1.25), (0.72, 0.70, 0.66)))
    mesh.extend(box((2.3, -1.1, 0.45), (0.16, 1.6, 0.9), (0.45, 0.48, 0.50)))
    mesh.extend(box((-1.2, 1.55, 0.7), (0.65, 0.08, 0.5), (0.30, 0.33, 0.36)))
    mesh.extend(box((1.15, 1.52, 0.55), (0.75, 0.08, 0.36), (0.55, 0.57, 0.60)))
    return mesh


def mesh_stats(mesh: Mesh) -> dict[str, int]:
    return {"vertices": len(mesh.vertices), "faces": len(mesh.faces)}


def main() -> int:
    object_b = build_duck()
    object_c = build_cup(ROOT / "outputs/task1/object_C_image3d/cup_rgba.png")
    background = build_counter_background()

    b_mesh = ROOT / "outputs/task1/object_B_text3d/export/mesh.obj"
    c_mesh = ROOT / "outputs/task1/object_C_image3d/export/mesh.obj"
    bg_mesh = ROOT / "outputs/task1/background_3dgs/counter_proxy/export/mesh.obj"
    write_obj(object_b, b_mesh, "Procedural proxy for the text prompt: small yellow rubber duck toy")
    write_obj(object_c, c_mesh, "Single-image proxy mesh inferred from the prepared cup RGBA foreground")
    write_obj(background, bg_mesh, "Counter-like background proxy mesh for Gaussian fusion fallback")

    meta = {
        "status": "proxy_assets_generated",
        "honesty_note": "These assets make the full fusion pipeline executable when external AIGC/model/dataset dependencies are unavailable; they are not claimed as official SDS, Zero123, or Mip-NeRF360 training outputs.",
        "object_B": {"prompt": "a small yellow rubber duck toy, smooth plastic surface, single object, full body, centered, clean 3D asset, no text, no watermark", **mesh_stats(object_b), "mesh": str(b_mesh.relative_to(ROOT))},
        "object_C": {"source": "outputs/task1/object_C_image3d/cup_rgba.png", **mesh_stats(object_c), "mesh": str(c_mesh.relative_to(ROOT))},
        "background": {"source": "procedural counter proxy", **mesh_stats(background), "mesh": str(bg_mesh.relative_to(ROOT))},
    }
    out = ROOT / "outputs/task1/proxy_assets_meta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "outputs/task1/object_B_text3d/prompt.txt").write_text(meta["object_B"]["prompt"] + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
