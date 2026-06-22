#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def latest_point_cloud(root: Path) -> str:
    if not root.exists():
        return "missing"
    matches = sorted(root.glob("**/point_cloud.ply"))
    return str(matches[-1].relative_to(ROOT)) if matches else "missing"


def file_size(rel: str) -> str:
    path = ROOT / rel
    if not path.exists():
        return "missing"
    return str(path.stat().st_size)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def ply_vertex_count(path: Path) -> str:
    if not path.exists():
        return "NA"
    with path.open("rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return line.split()[-1]
            if line == "end_header":
                break
    return "NA"


def parse_colmap_stats(path: Path) -> dict[str, str]:
    stats: dict[str, str] = {}
    if not path.exists():
        return stats
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        stats[key.strip()] = value.strip()
    return stats


def object_a_metrics() -> str:
    metrics = read_json(ROOT / "outputs/task1/object_A_3dgs/results.json")
    if not metrics:
        return "NA"
    values = metrics.get("ours_1000")
    if not isinstance(values, dict):
        values = next((v for v in metrics.values() if isinstance(v, dict)), {})
    if not values:
        return "NA"
    parts = []
    for key in ["PSNR", "SSIM", "LPIPS"]:
        value = values.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key} {value:.4f}")
    return "; ".join(parts) if parts else "NA"


def write_task1_assets() -> None:
    video_stats = read_json(ROOT / "outputs/task1/object_A_video_frame_stats.json")
    colmap_stats = parse_colmap_stats(ROOT / "outputs/task1/object_A_video_colmap_stats.txt")
    object_a_pc = ROOT / "outputs/task1/object_A_3dgs/point_cloud/iteration_1000/point_cloud.ply"
    object_a_success = object_a_pc.exists()
    if video_stats:
        selected = video_stats.get("selected_count", count_images(ROOT / "data/object_A_video/input"))
        registered = colmap_stats.get("Registered images")
        object_a_images = f"{selected} selected video frames"
        if registered:
            object_a_images += f"; {registered} registered"
    else:
        stats_path = ROOT / "outputs/task1/object_A_frame_stats_all.json"
        object_a_images = count_images(ROOT / "data/object_A_all/input")
        if not stats_path.exists():
            stats_path = ROOT / "outputs/task1/object_A_frame_stats.json"
            object_a_images = count_images(ROOT / "data/object_A/input")
        if stats_path.exists():
            object_a_images = read_json(stats_path).get("selected_count", object_a_images)

    if object_a_success:
        object_a_row = {
            "Asset": "A",
            "Input source": "student A_video.mp4; original 6 photos kept as failed baseline",
            "Method": "COLMAP + 3DGS",
            "Output representation": latest_point_cloud(ROOT / "outputs/task1/object_A_3dgs"),
            "Training steps or iterations": "COLMAP SfM + 3DGS 1000 sanity iterations",
            "Number of input images": object_a_images,
            "Number of Gaussians or mesh vertices/faces": ply_vertex_count(object_a_pc),
            "Training time": "extract 1s; COLMAP/convert 269s; train 19s; render 15s; metrics 78s",
            "Peak GPU memory": "not sampled continuously; completed on RTX A6000",
            "Main quality observations": "Video fixed COLMAP init; 1000-iter render is coherent but still slightly blurry.",
            "Metrics": object_a_metrics(),
        }
    else:
        object_a_row = {
            "Asset": "A",
            "Input source": "multi-view photos uploaded by student",
            "Method": "COLMAP + 3DGS",
            "Output representation": "No Gaussian PLY; COLMAP sparse model missing",
            "Training steps or iterations": "COLMAP feature/match/mapper attempted; 3DGS training not started",
            "Number of input images": object_a_images,
            "Number of Gaussians or mesh vertices/faces": "NA",
            "Training time": "COLMAP mapper 72s; relaxed retry 142s; official convert patched retry 94s",
            "Peak GPU memory": "3DGS training not started",
            "Main quality observations": "COLMAP repeatedly reported no good initial image pair; likely too few or insufficient-baseline laptop views.",
            "Metrics": "NA",
        }

    rows = [
        object_a_row,
        {
            "Asset": "B",
            "Input source": "text prompt",
            "Method": "DreamFusion SDS via threestudio",
            "Output representation": "Mesh, then Gaussian-converted mesh",
            "Training steps or iterations": "not_run",
            "Number of input images": 0,
            "Number of Gaussians or mesh vertices/faces": "NA",
            "Training time": "NA",
            "Peak GPU memory": "NA",
            "Main quality observations": "No generated mesh found in outputs.",
            "Metrics": "NA",
        },
        {
            "Asset": "C",
            "Input source": "single cup image uploaded by student",
            "Method": "background removal + Stable Zero123/Magic123 via threestudio",
            "Output representation": "RGBA prepared; mesh if training succeeds",
            "Training steps or iterations": "preprocess_only" if exists("outputs/task1/object_C_image3d/cup_rgba.png") else "not_run",
            "Number of input images": 1 if exists("inputs/object_C/cup.jpg") else 0,
            "Number of Gaussians or mesh vertices/faces": "NA",
            "Training time": "NA",
            "Peak GPU memory": "NA",
            "Main quality observations": "Heuristic RGBA exists; inspect handle and cup lid before Zero123 training." if exists("outputs/task1/object_C_image3d/cup_rgba.png") else "RGBA missing.",
            "Metrics": "NA",
        },
        {
            "Asset": "Background",
            "Input source": "Mip-NeRF 360 counter preferred",
            "Method": "3DGS",
            "Output representation": "Gaussian PLY if training succeeds",
            "Training steps or iterations": "not_run",
            "Number of input images": count_images(ROOT / "data/mipnerf360/counter/images"),
            "Number of Gaussians or mesh vertices/faces": "NA",
            "Training time": "NA",
            "Peak GPU memory": "NA",
            "Main quality observations": "No Mip-NeRF 360 scene found locally yet.",
            "Metrics": "NA",
        },
    ]
    out = ROOT / "report/tables/task1_assets.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def csv_to_latex(csv_path: Path, tex_path: Path, caption: str) -> None:
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if not rows:
        return
    cols = list(rows[0].keys())
    def esc(value: str) -> str:
        return (
            str(value)
            .replace("\\", "\\textbackslash{}")
            .replace("_", "\\_")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("#", "\\#")
        )
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{%s}\n\\hline\n" % ("l" * len(cols)))
        f.write(" & ".join(esc(c) for c in cols) + " \\\\\n\\hline\n")
        for row in rows:
            vals = [esc(str(row.get(c, "NA"))[:80]) for c in cols]
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}}\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write("\\end{table*}\n")


def write_task2_eval_placeholder() -> None:
    out = ROOT / "report/tables/task2_eval_D.csv"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "metric_type", "mean_action_l1", "median_action_l1", "success_rate", "status", "note"])
        writer.writeheader()
        writer.writerow({"model": "ACT-A", "metric_type": "not_available", "status": "not_run", "note": "No trained checkpoint/eval output found."})
        writer.writerow({"model": "ACT-ABC", "metric_type": "not_available", "status": "not_run", "note": "No trained checkpoint/eval output found."})


def ensure_calvin_table() -> None:
    json_path = ROOT / "outputs/task2/calvin_split_summary.json"
    csv_path = ROOT / "report/tables/calvin_split_summary.csv"
    if not json_path.exists():
        return
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    rows = summary.get("splits") or []
    if rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "episodes", "frames", "action_dim", "state_dim", "camera_keys", "task_count", "env_distribution", "status"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "split": "A/B/C/D",
                "episodes": "NA",
                "frames": "NA",
                "action_dim": "NA",
                "state_dim": "NA",
                "camera_keys": "NA",
                "task_count": "NA",
                "env_distribution": "NA",
                "status": summary.get("status", "blocked"),
            }
        )


def zip_weights() -> None:
    weights_dir = ROOT / "outputs/weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    zip_path = weights_dir / "hw3_weights.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        files = [p for p in weights_dir.rglob("*") if p.is_file() and p != zip_path]
        if not files:
            note = weights_dir / "NO_TRAINED_WEIGHTS.txt"
            note.write_text("No trained weights were produced in this run. See run_manifest.md and final_checklist.md.\n", encoding="utf-8")
            files = [note]
        for p in files:
            zf.write(p, p.relative_to(ROOT))


def write_checklist() -> None:
    checks = [
        ("object_A 3DGS point_cloud.ply exists", latest_point_cloud(ROOT / "outputs/task1/object_A_3dgs") != "missing"),
        ("object_A render/metrics exists", exists("outputs/task1/object_A_3dgs/results.json") or bool(list((ROOT / "outputs/task1/object_A_3dgs").glob("**/*.png")))),
        ("object_B mesh exists", bool(list((ROOT / "outputs/task1/object_B_text3d").glob("**/*.obj")))),
        ("object_C RGBA exists", exists("outputs/task1/object_C_image3d/cup_rgba.png")),
        ("object_C mesh exists", bool(list((ROOT / "outputs/task1/object_C_image3d").glob("**/*.obj")))),
        ("background 3DGS point_cloud.ply exists", latest_point_cloud(ROOT / "outputs/task1/background_3dgs") != "missing"),
        ("merged scene point_cloud.ply or fallback exists", exists("outputs/task1/merged_scene/point_cloud.ply")),
        ("merged_scene_walkthrough.mp4 exists", exists("outputs/task1/videos/merged_scene_walkthrough.mp4")),
        ("key frames exist", bool(list((ROOT / "report/figs").glob("task1_merged_*.png")))),
        ("task1_assets.csv exists", exists("report/tables/task1_assets.csv")),
        ("calvin split summary exists", exists("outputs/task2/calvin_split_summary.json")),
        ("ACT-A checkpoint exists", exists("outputs/weights/act_A_best") or bool(list((ROOT / "outputs/task2/act_A").glob("**/*.safetensors")))),
        ("ACT-ABC checkpoint exists", exists("outputs/weights/act_ABC_best") or bool(list((ROOT / "outputs/task2/act_ABC").glob("**/*.safetensors")))),
        ("ACT-A curve exists", bool(list((ROOT / "outputs/task2/act_A").glob("**/*.png")))),
        ("ACT-ABC curve exists", bool(list((ROOT / "outputs/task2/act_ABC").glob("**/*.png")))),
        ("environment D eval table exists", exists("report/tables/task2_eval_D.csv")),
        ("README.md exists", exists("README.md")),
        ("report/report.pdf exists", exists("report/report.pdf")),
        ("refs.bib exists", exists("report/refs.bib")),
        ("outputs/weights/hw3_weights.zip exists", exists("outputs/weights/hw3_weights.zip")),
        ("run_manifest.md exists", exists("run_manifest.md")),
        ("no obvious API key/token in text files", not scan_secrets()),
    ]
    lines = ["# Final Checklist", ""]
    for label, ok in checks:
        lines.append(f"- [{'x' if ok else ' '}] {label}")
    lines.append("")
    lines.append("This checklist is generated from actual file existence checks; unchecked items were not fabricated as complete.")
    (ROOT / "final_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def scan_secrets() -> bool:
    pattern = re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})")
    for rel in ["README.md", "run_manifest.md", "report/main.tex"]:
        path = ROOT / rel
        if path.exists() and pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            return True
    return False


def main() -> int:
    write_task1_assets()
    write_task2_eval_placeholder()
    ensure_calvin_table()
    csv_to_latex(ROOT / "report/tables/task1_assets.csv", ROOT / "report/tables/task1_assets.tex", "题目一资产生成结果汇总。")
    csv_to_latex(ROOT / "report/tables/calvin_split_summary.csv", ROOT / "report/tables/calvin_split_summary.tex", "CALVIN/LeRobot 数据切分探测结果。")
    csv_to_latex(ROOT / "report/tables/task2_eval_D.csv", ROOT / "report/tables/task2_eval_D.tex", "环境 D zero-shot 评估结果。")
    zip_weights()
    write_checklist()
    print("generated report tables, weights zip, and final_checklist.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
