#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--x", default="step")
    parser.add_argument("--y", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    rows = read_rows(Path(args.csv))
    if not rows:
        raise ValueError(f"No rows in {args.csv}")
    xs = [float(r[args.x]) for r in rows if r.get(args.x)]
    plt.figure(figsize=(6, 4))
    for ykey in args.y:
        ys = [float(r[ykey]) for r in rows if r.get(ykey)]
        if len(ys) == len(xs):
            plt.plot(xs, ys, label=ykey)
    plt.xlabel(args.x)
    plt.ylabel("value")
    plt.title(args.title or Path(args.csv).stem)
    plt.grid(True, alpha=0.3)
    plt.legend()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
