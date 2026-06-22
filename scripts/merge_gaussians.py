#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ply_utils import read_ply, write_ascii_ply
from transform_gaussians import transform


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def append_arrays(all_arrays: list[dict[str, np.ndarray]], props):
    names = [p[0] for p in props]
    merged = {}
    for name in names:
        merged[name] = np.concatenate([arr[name] for arr in all_arrays], axis=0)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/placements.yaml")
    parser.add_argument("--output", default="outputs/task1/merged_scene/point_cloud.ply")
    parser.add_argument("--workdir", default="outputs/task1/merged_scene/intermediate")
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    entries = []
    if "background" in cfg and cfg["background"].get("ply"):
        entries.append(("background", cfg["background"], "gaussian"))
    for name, item in cfg.get("objects", {}).items():
        entries.append((name, item, item.get("type", "gaussian")))

    arrays_list = []
    base_props = None
    for name, item, typ in entries:
        scale = float(item.get("scale", 1.0))
        rotation = item.get("rotation_deg_xyz", [0, 0, 0])
        translation = item.get("translation", [0, 0, 0])
        if typ == "mesh":
            mesh = Path(item["mesh"])
            if not mesh.exists():
                raise FileNotFoundError(f"Missing mesh for {name}: {mesh}")
            gaussian = workdir / f"{name}_sampled.ply"
            cmd = [
                sys.executable,
                "scripts/mesh_to_gaussians.py",
                "--mesh",
                str(mesh),
                "--output",
                str(gaussian),
                "--sample-points",
                str(item.get("sample_points", 80000)),
            ]
            subprocess.run(cmd, check=True)
            source = gaussian
        else:
            source = Path(item["ply"])
            if not source.exists():
                raise FileNotFoundError(f"Missing gaussian PLY for {name}: {source}")
        transformed = workdir / f"{name}_transformed.ply"
        transform(source, transformed, scale, rotation, translation)
        props, arrays = read_ply(transformed)
        if base_props is None:
            base_props = props
        elif [p[0] for p in props] != [p[0] for p in base_props]:
            raise ValueError(f"PLY property mismatch for {name}; cannot safely concatenate")
        arrays_list.append(arrays)

    if not arrays_list or base_props is None:
        raise ValueError("No gaussian assets were found to merge")
    merged = append_arrays(arrays_list, base_props)
    write_ascii_ply(Path(args.output), base_props, merged)
    print(f"merged {sum(len(a[base_props[0][0]]) for a in arrays_list)} gaussians into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
