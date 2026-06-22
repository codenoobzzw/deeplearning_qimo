#!/usr/bin/env python3
"""Convert a simple OBJ mesh into a 3DGS-compatible Gaussian PLY.

This implementation avoids optional dependencies. It supports vertices with
optional RGB values (`v x y z r g b`) and triangular/polygonal faces. Texture UV
sampling is intentionally not claimed here; if no vertex color exists a material
fallback color is used and recorded by the caller/report.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    vertices: list[list[float]] = []
    colors: list[list[float] | None] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    rgb = [float(parts[4]), float(parts[5]), float(parts[6])]
                    if max(rgb) > 1.0:
                        rgb = [c / 255.0 for c in rgb]
                    colors.append(rgb)
                else:
                    colors.append(None)
            elif parts[0] == "f" and len(parts) >= 4:
                idxs = []
                for token in parts[1:]:
                    raw = token.split("/")[0]
                    if raw:
                        idx = int(raw)
                        idxs.append(idx - 1 if idx > 0 else len(vertices) + idx)
                if len(idxs) >= 3:
                    for i in range(1, len(idxs) - 1):
                        faces.append([idxs[0], idxs[i], idxs[i + 1]])
    if not vertices or not faces:
        raise ValueError(f"OBJ must contain vertices and faces: {path}")
    color_arr = None
    if all(c is not None for c in colors):
        color_arr = np.asarray(colors, dtype=np.float32)
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32), color_arr


def sample_mesh(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray | None, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = np.linalg.norm(normals, axis=1) * 0.5
    valid = areas > 1e-12
    tri = tri[valid]
    faces = faces[valid]
    normals = normals[valid]
    areas = areas[valid]
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    probs = areas / areas.sum()
    face_ids = rng.choice(len(tri), size=n, replace=True, p=probs)
    r1 = np.sqrt(rng.random(n, dtype=np.float32))
    r2 = rng.random(n, dtype=np.float32)
    a = 1.0 - r1
    b = r1 * (1.0 - r2)
    c = r1 * r2
    chosen = tri[face_ids]
    points = chosen[:, 0] * a[:, None] + chosen[:, 1] * b[:, None] + chosen[:, 2] * c[:, None]
    ns = normals[face_ids]
    if colors is None:
        rgb = np.repeat(np.array([[0.8, 0.75, 0.25]], dtype=np.float32), n, axis=0)
    else:
        ctri = colors[faces][face_ids]
        rgb = ctri[:, 0] * a[:, None] + ctri[:, 1] * b[:, None] + ctri[:, 2] * c[:, None]
        rgb = np.clip(rgb, 0.0, 1.0)
    return points, ns, rgb


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def write_gaussian_ply(path: Path, points: np.ndarray, normals: np.ndarray, rgb: np.ndarray, opacity: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bbox = points.max(axis=0) - points.min(axis=0)
    diag = float(np.linalg.norm(bbox)) or 1.0
    scale = math.log(max(diag / math.sqrt(len(points)) * 0.25, 1e-5))
    op = logit(opacity)
    fdc = (rgb - 0.5) / SH_C0
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    props += [f"f_rest_{i}" for i in range(45)]
    props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        for prop in props:
            f.write(f"property float {prop}\n")
        f.write("end_header\n")
        zeros = " ".join(["0"] * 45)
        for p, n, c in zip(points, normals, fdc):
            row = [
                *p.tolist(),
                *n.tolist(),
                *c.tolist(),
            ]
            f.write(" ".join(f"{v:.8g}" for v in row))
            f.write(f" {zeros} {op:.8g} {scale:.8g} {scale:.8g} {scale:.8g} 1 0 0 0\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-points", type=int, default=80000)
    parser.add_argument("--opacity", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    vertices, faces, colors = read_obj(Path(args.mesh))
    points, normals, rgb = sample_mesh(vertices, faces, colors, args.sample_points, args.seed)
    if not np.isfinite(points).all() or not np.isfinite(rgb).all():
        raise ValueError("NaN or Inf detected in sampled Gaussian data")
    write_gaussian_ply(Path(args.output), points, normals, rgb, args.opacity)
    print(f"wrote {len(points)} gaussians to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
