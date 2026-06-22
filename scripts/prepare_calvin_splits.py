#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def write_outputs(summary: dict, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = summary.get("splits", [])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "episodes", "frames", "action_dim", "state_dim", "camera_keys", "task_count", "env_distribution", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "NA") for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="xiaoma26/calvin-lerobot")
    parser.add_argument("--output-json", default="outputs/task2/calvin_split_summary.json")
    parser.add_argument("--output-csv", default="report/tables/calvin_split_summary.csv")
    parser.add_argument("--file-tree-output", default="outputs/task2/calvin_hf_file_tree.txt")
    parser.add_argument("--download", action="store_true", help="Download the dataset snapshot. Default only probes metadata/file tree.")
    args = parser.parse_args()

    summary: dict = {
        "repo_id": args.repo_id,
        "status": "started",
        "splits": [],
        "notes": [],
    }
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except Exception as exc:  # noqa: BLE001
        summary.update({"status": "blocked", "blocker": f"huggingface_hub import failed: {exc}"})
        write_outputs(summary, Path(args.output_json), Path(args.output_csv))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1

    try:
        api = HfApi()
        info = api.dataset_info(args.repo_id)
        files = api.list_repo_files(args.repo_id, repo_type="dataset")
        Path(args.file_tree_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.file_tree_output).write_text("\n".join(files) + "\n", encoding="utf-8")
        summary["dataset_info"] = {
            "id": getattr(info, "id", args.repo_id),
            "sha": getattr(info, "sha", None),
            "siblings_count": len(files),
            "tags": getattr(info, "tags", []),
        }
        summary["file_tree_output"] = args.file_tree_output
        summary["top_level_files_preview"] = files[:80]

        for candidate in ["README.md", "meta/info.json", "meta/episodes.jsonl", "meta/episodes_stats.jsonl"]:
            if candidate in files:
                try:
                    local = hf_hub_download(args.repo_id, candidate, repo_type="dataset")
                    text = Path(local).read_text(encoding="utf-8", errors="replace")
                    summary.setdefault("downloaded_metadata", {})[candidate] = text[:20000]
                except Exception as exc:  # noqa: BLE001
                    summary.setdefault("metadata_errors", {})[candidate] = str(exc)

        env_tokens = Counter()
        for name in files:
            low = name.lower()
            for env in ["env_a", "env_b", "env_c", "env_d", "environment_a", "environment_b", "environment_c", "environment_d", "calvin_a", "calvin_b", "calvin_c", "calvin_d"]:
                if env in low:
                    env_tokens[env] += 1
        summary["environment_token_counts_in_paths"] = dict(env_tokens)

        if args.download:
            local_dir = snapshot_download(args.repo_id, repo_type="dataset", local_dir="data/calvin/calvin-lerobot")
            summary["snapshot_dir"] = local_dir
            summary["notes"].append("Snapshot downloaded, but this script still requires explicit env metadata to create splits safely.")

        if env_tokens:
            summary["status"] = "metadata_probe_ok_split_requires_dataset_loading"
            summary["notes"].append("Environment tokens were found in paths; inspect metadata before constructing split symlinks/copies.")
        else:
            summary["status"] = "blocked_no_explicit_env_split_detected_from_file_tree"
            summary["blocker"] = "No explicit A/B/C/D environment token was detected in file paths. Per assignment rules, the script did not invent episode-index splits."
        summary["splits"] = [
            {"split": "train_A", "status": "not_created", "episodes": "NA", "frames": "NA", "env_distribution": "needs metadata"},
            {"split": "train_ABC", "status": "not_created", "episodes": "NA", "frames": "NA", "env_distribution": "needs metadata"},
            {"split": "test_D", "status": "not_created", "episodes": "NA", "frames": "NA", "env_distribution": "needs metadata"},
        ]
        write_outputs(summary, Path(args.output_json), Path(args.output_csv))
        print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])
        return 0 if summary["status"].startswith("metadata_probe_ok") else 1
    except Exception as exc:  # noqa: BLE001
        summary.update({"status": "blocked_hf_probe_failed", "blocker": str(exc)})
        write_outputs(summary, Path(args.output_json), Path(args.output_csv))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
