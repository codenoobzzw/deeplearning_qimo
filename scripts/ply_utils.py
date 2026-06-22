from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


PLY_TYPES = {
    "char": ("b", np.int8),
    "uchar": ("B", np.uint8),
    "int8": ("b", np.int8),
    "uint8": ("B", np.uint8),
    "short": ("h", np.int16),
    "ushort": ("H", np.uint16),
    "int16": ("h", np.int16),
    "uint16": ("H", np.uint16),
    "int": ("i", np.int32),
    "uint": ("I", np.uint32),
    "int32": ("i", np.int32),
    "uint32": ("I", np.uint32),
    "float": ("f", np.float32),
    "float32": ("f", np.float32),
    "double": ("d", np.float64),
    "float64": ("d", np.float64),
}


def read_ply(path: Path) -> tuple[list[tuple[str, str]], dict[str, np.ndarray]]:
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading PLY header")
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        if header_lines[0] != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt = None
        vertex_count = None
        props: list[tuple[str, str]] = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError("List properties in vertex element are not supported")
                props.append((parts[2], parts[1]))
        if fmt is None or vertex_count is None:
            raise ValueError("PLY missing format or vertex count")
        arrays = {name: np.empty(vertex_count, dtype=PLY_TYPES[typ][1]) for name, typ in props}
        if fmt == "ascii":
            for i in range(vertex_count):
                vals = f.readline().decode("ascii", errors="replace").split()
                for (name, typ), value in zip(props, vals):
                    arrays[name][i] = value
        elif fmt == "binary_little_endian":
            fmt_str = "<" + "".join(PLY_TYPES[typ][0] for _, typ in props)
            size = struct.calcsize(fmt_str)
            for i in range(vertex_count):
                vals = struct.unpack(fmt_str, f.read(size))
                for (name, _), value in zip(props, vals):
                    arrays[name][i] = value
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")
    return props, arrays


def write_ascii_ply(path: Path, props: list[tuple[str, str]], arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not props:
        raise ValueError("No properties to write")
    n = len(arrays[props[0][0]])
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        for name, typ in props:
            out_type = "float" if np.issubdtype(arrays[name].dtype, np.floating) else typ
            f.write(f"property {out_type} {name}\n")
        f.write("end_header\n")
        names = [p[0] for p in props]
        for i in range(n):
            f.write(" ".join(f"{float(arrays[name][i]):.8g}" for name in names))
            f.write("\n")
