#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ply_utils import read_ply, write_ascii_ply


def euler_matrix(deg_xyz: list[float]) -> np.ndarray:
    ax, ay, az = [math.radians(v) for v in deg_xyz]
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    tr = float(np.trace(m))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s], dtype=np.float32)
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s], dtype=np.float32)
    if i == 1:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s], dtype=np.float32)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s], dtype=np.float32)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def transform(input_path: Path, output_path: Path, scale: float, rotation_deg_xyz: list[float], translation: list[float]) -> None:
    props, arrays = read_ply(input_path)
    xyz = np.stack([arrays["x"], arrays["y"], arrays["z"]], axis=1).astype(np.float32)
    rot = euler_matrix(rotation_deg_xyz)
    trans = np.asarray(translation, dtype=np.float32)
    out = (xyz * scale) @ rot.T + trans
    arrays["x"], arrays["y"], arrays["z"] = out[:, 0], out[:, 1], out[:, 2]
    if all(k in arrays for k in ["nx", "ny", "nz"]):
        n = np.stack([arrays["nx"], arrays["ny"], arrays["nz"]], axis=1).astype(np.float32) @ rot.T
        arrays["nx"], arrays["ny"], arrays["nz"] = n[:, 0], n[:, 1], n[:, 2]
    if scale > 0:
        delta = math.log(scale)
        for key in ["scale_0", "scale_1", "scale_2"]:
            if key in arrays:
                arrays[key] = arrays[key].astype(np.float32) + delta
    if all(k in arrays for k in ["rot_0", "rot_1", "rot_2", "rot_3"]):
        gq = quat_from_matrix(rot)
        qs = np.stack([arrays["rot_0"], arrays["rot_1"], arrays["rot_2"], arrays["rot_3"]], axis=1).astype(np.float32)
        out_q = np.array([quat_mul(gq, q) for q in qs], dtype=np.float32)
        norm = np.maximum(np.linalg.norm(out_q, axis=1, keepdims=True), 1e-12)
        out_q = out_q / norm
        arrays["rot_0"], arrays["rot_1"], arrays["rot_2"], arrays["rot_3"] = out_q[:, 0], out_q[:, 1], out_q[:, 2], out_q[:, 3]
    write_ascii_ply(output_path, props, arrays)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--rotation-deg-xyz", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--translation", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    args = parser.parse_args()
    transform(Path(args.input), Path(args.output), args.scale, args.rotation_deg_xyz, args.translation)
    print(f"wrote transformed gaussians: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
