#!/usr/bin/env python3
"""Train a small real-CALVIN subset experiment with LeRobot's ACTPolicy.

This script is intentionally conservative: it downloads a bounded number of
episodes from xiaoma26/calvin-lerobot, trains the same ACT architecture on
splitA and splitA+B+C, then evaluates both on splitD with Action L1. It is not
full CALVIN simulator success-rate training, but the data are real CALVIN
LeRobot parquet samples rather than synthetic proxy data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for dep_path in [ROOT / "local_pkgs/lerobot", ROOT / "third_party/lerobot/src"]:
    if dep_path.exists():
        sys.path.insert(0, str(dep_path))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset


from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE  # noqa: E402


class ChunkDataset(Dataset):
    def __init__(self, episodes: list[dict[str, np.ndarray]], chunk_size: int, stride: int) -> None:
        self.episodes = episodes
        self.chunk_size = chunk_size
        self.samples: list[tuple[int, int]] = []
        for ep_idx, episode in enumerate(episodes):
            length = len(episode["actions"])
            if length == 0:
                continue
            for start in range(0, length, stride):
                self.samples.append((ep_idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, start = self.samples[idx]
        episode = self.episodes[ep_idx]
        state = episode["states"][start]
        actions = episode["actions"][start : start + self.chunk_size]
        valid = len(actions)
        padded = np.zeros((self.chunk_size, episode["actions"].shape[1]), dtype=np.float32)
        padded[:valid] = actions
        action_is_pad = np.zeros((self.chunk_size,), dtype=bool)
        action_is_pad[valid:] = True
        state_t = torch.from_numpy(state.astype(np.float32))
        return {
            OBS_STATE: state_t,
            OBS_ENV_STATE: state_t,
            ACTION: torch.from_numpy(padded),
            "action_is_pad": torch.from_numpy(action_is_pad),
        }


def download_split(repo_id: str, split: str, episodes_per_split: int, local_dir: Path) -> tuple[list[Path], dict]:
    split_dir = local_dir / split
    def ensure_file(filename: str) -> Path:
        path = local_dir / filename
        if path.exists() and path.stat().st_size > 0:
            return path
        return Path(hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=str(local_dir)))

    meta_files = [
        f"{split}/meta/info.json",
        f"{split}/meta/modality.json",
        f"{split}/meta/episodes.jsonl",
        f"{split}/meta/tasks.jsonl",
    ]
    for filename in meta_files:
        ensure_file(filename)

    info = json.loads((split_dir / "meta/info.json").read_text(encoding="utf-8"))
    chunk_size = int(info.get("chunks_size", 1000))
    paths: list[Path] = []
    for episode_idx in range(episodes_per_split):
        chunk = episode_idx // chunk_size
        filename = f"{split}/data/chunk-{chunk:03d}/episode_{episode_idx:06d}.parquet"
        paths.append(ensure_file(filename))
    return paths, info


def load_episodes(paths: list[Path]) -> list[dict[str, np.ndarray]]:
    episodes: list[dict[str, np.ndarray]] = []
    for path in paths:
        df = pd.read_parquet(path, columns=["state", "actions"])
        states = np.stack(df["state"].to_numpy()).astype(np.float32)
        actions = np.stack(df["actions"].to_numpy()).astype(np.float32)
        episodes.append({"states": states, "actions": actions})
    return episodes


def make_policy(args: argparse.Namespace, state_dim: int, action_dim: int) -> ACTPolicy:
    config = ACTConfig(
        device="cpu",
        use_amp=False,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(state_dim,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=1,
        use_vae=args.use_vae,
        latent_dim=args.latent_dim,
        n_vae_encoder_layers=args.n_vae_encoder_layers,
        dropout=args.dropout,
        kl_weight=args.kl_weight,
        pretrained_backbone_weights=None,
        push_to_hub=False,
    )
    return ACTPolicy(config)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(policy: ACTPolicy, loader: DataLoader, device: torch.device) -> dict[str, float]:
    policy.eval()
    losses: list[float] = []
    sample_losses: list[np.ndarray] = []
    for batch in loader:
        batch = move_batch(batch, device)
        pred = policy.predict_action_chunk(batch)
        err = torch.abs(pred - batch[ACTION])
        valid = (~batch["action_is_pad"]).unsqueeze(-1)
        denom = valid.sum(dim=(1, 2)).clamp_min(1) * err.shape[-1]
        per_sample = (err * valid).sum(dim=(1, 2)) / denom
        sample_losses.append(per_sample.cpu().numpy())
        losses.append(float(per_sample.mean().cpu()))
    merged = np.concatenate(sample_losses) if sample_losses else np.array([math.nan], dtype=np.float32)
    return {
        "mean_action_l1": float(np.mean(merged)),
        "median_action_l1": float(np.median(merged)),
        "success_rate_l1_lt_0.05": float(np.mean(merged < 0.05)),
        "success_rate_l1_lt_0.10": float(np.mean(merged < 0.10)),
    }


def train_one(
    name: str,
    train_episodes: list[dict[str, np.ndarray]],
    d_episodes: list[dict[str, np.ndarray]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float | int | str]], dict[str, float], Path]:
    train_ds = ChunkDataset(train_episodes, args.chunk_size, args.stride)
    d_ds = ChunkDataset(d_episodes, args.chunk_size, args.stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    eval_loader = DataLoader(d_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    state_dim = train_episodes[0]["states"].shape[1]
    action_dim = train_episodes[0]["actions"].shape[1]
    policy = make_policy(args, state_dim, action_dim).to(device)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rows: list[dict[str, float | int | str]] = []
    iterator = iter(train_loader)

    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        policy.train()
        batch = move_batch(batch, device)
        loss, loss_dict = policy(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            d_eval = evaluate(policy, eval_loader, device)
            rows.append(
                {
                    "model": name,
                    "step": step,
                    "train_loss": float(loss.detach().cpu()),
                    "train_l1_loss": float(loss_dict["l1_loss"]),
                    "D_mean_action_l1": d_eval["mean_action_l1"],
                    "D_median_action_l1": d_eval["median_action_l1"],
                    "D_success_l1_lt_0.05": d_eval["success_rate_l1_lt_0.05"],
                    "D_success_l1_lt_0.10": d_eval["success_rate_l1_lt_0.10"],
                }
            )

    final_eval = evaluate(policy, eval_loader, device)
    out_dir = ROOT / f"outputs/task2/real_calvin_subset/{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    weight_path = ROOT / f"outputs/weights/{name}_real_calvin_subset.pt"
    torch.save(
        {
            "model_state": policy.state_dict(),
            "config": asdict(policy.config),
            "final_D": final_eval,
            "note": "LeRobot ACTPolicy trained on a bounded real CALVIN-LeRobot parquet subset.",
        },
        weight_path,
    )
    write_csv(out_dir / "train_log.csv", rows)
    return rows, final_eval, weight_path


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(all_rows: dict[str, list[dict[str, float | int | str]]], eval_rows: list[dict[str, str]]) -> None:
    figs = ROOT / "report/figs"
    out_figs = ROOT / "outputs/task2/real_calvin_subset/figures"
    figs.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.0, 4.2))
    for name, rows in all_rows.items():
        plt.plot([int(r["step"]) for r in rows], [float(r["train_l1_loss"]) for r in rows], label=f"{name} train")
        plt.plot([int(r["step"]) for r in rows], [float(r["D_mean_action_l1"]) for r in rows], "--", label=f"{name} D")
    plt.xlabel("step")
    plt.ylabel("Action L1")
    plt.title("Real CALVIN subset ACTPolicy curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    for path in [figs / "act_real_calvin_action_l1.png", out_figs / "act_real_calvin_action_l1.png"]:
        plt.savefig(path, dpi=180)
    plt.close()

    names = [row["model"] for row in eval_rows]
    l1 = [float(row["mean_action_l1"]) for row in eval_rows]
    succ10 = [float(row["success_rate_l1_lt_0.10"]) for row in eval_rows]
    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
    ax1.bar(x - 0.18, l1, width=0.36, label="D mean Action L1", color="#4c78a8")
    ax1.set_ylabel("mean Action L1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, succ10, width=0.36, label="D success @ L1<0.10", color="#59a14f")
    ax2.set_ylabel("success rate")
    ax2.set_ylim(0, 1.0)
    ax1.set_title("Real CALVIN subset D evaluation")
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    for path in [figs / "act_real_calvin_d_eval.png", out_figs / "act_real_calvin_d_eval.png"]:
        fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="xiaoma26/calvin-lerobot")
    parser.add_argument("--local-dir", default="data/calvin/calvin-lerobot-subset")
    parser.add_argument("--episodes-per-split", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--dim-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--n-encoder-layers", type=int, default=2)
    parser.add_argument("--use-vae", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--n-vae-encoder-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cpu")
    local_dir = ROOT / args.local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    split_paths: dict[str, list[Path]] = {}
    split_info: dict[str, dict] = {}
    for split in ["splitA", "splitB", "splitC", "splitD"]:
        paths, info = download_split(args.repo_id, split, args.episodes_per_split, local_dir)
        split_paths[split] = paths
        split_info[split] = info

    episodes = {split: load_episodes(paths) for split, paths in split_paths.items()}
    all_rows: dict[str, list[dict[str, float | int | str]]] = {}
    eval_rows: list[dict[str, str]] = []
    weights: list[str] = []

    train_specs = {
        "ACT-A-real-subset": episodes["splitA"],
        "ACT-ABC-real-subset": episodes["splitA"] + episodes["splitB"] + episodes["splitC"],
    }
    for name, train_episodes in train_specs.items():
        rows, final_eval, weight_path = train_one(name, train_episodes, episodes["splitD"], args, device)
        all_rows[name] = rows
        eval_rows.append(
            {
                "model": name,
                "metric_type": "real_calvin_subset_zero_shot_D",
                "mean_action_l1": f"{final_eval['mean_action_l1']:.6f}",
                "median_action_l1": f"{final_eval['median_action_l1']:.6f}",
                "success_rate_l1_lt_0.05": f"{final_eval['success_rate_l1_lt_0.05']:.6f}",
                "success_rate_l1_lt_0.10": f"{final_eval['success_rate_l1_lt_0.10']:.6f}",
                "status": "real_subset_run",
                "note": "Real xiaoma26/calvin-lerobot parquet subset; offline Action L1, not simulator rollout success.",
            }
        )
        weights.append(str(weight_path.relative_to(ROOT)))

    write_csv(ROOT / "report/tables/task2_real_calvin_eval_D.csv", eval_rows)
    plot_results(all_rows, eval_rows)

    split_summary = []
    for split, eps in episodes.items():
        split_summary.append(
            {
                "split": split,
                "scene": split_info[split].get("scene", split[-1]),
                "downloaded_episodes": len(eps),
                "frames": int(sum(len(ep["actions"]) for ep in eps)),
                "state_dim": int(eps[0]["states"].shape[1]),
                "action_dim": int(eps[0]["actions"].shape[1]),
            }
        )

    summary = {
        "status": "real_calvin_subset_act_complete",
        "repo_id": args.repo_id,
        "local_dir": args.local_dir,
        "honesty_note": "Bounded real CALVIN-LeRobot subset using LeRobot ACTPolicy on state/environment_state inputs; not full visual CNN ACT or simulator success-rate rollout.",
        "splits": split_summary,
        "hyperparameters": {
            "episodes_per_split": args.episodes_per_split,
            "chunk_size": args.chunk_size,
            "stride": args.stride,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dim_model": args.dim_model,
            "n_heads": args.n_heads,
            "dim_feedforward": args.dim_feedforward,
            "n_encoder_layers": args.n_encoder_layers,
            "n_decoder_layers": 1,
            "use_vae": args.use_vae,
            "loss": "LeRobot ACTPolicy Action L1" + (" + KL" if args.use_vae else ""),
        },
        "final_D": eval_rows,
        "weights": weights,
        "figures": [
            "report/figs/act_real_calvin_action_l1.png",
            "report/figs/act_real_calvin_d_eval.png",
        ],
    }
    out_path = ROOT / "outputs/task2/real_calvin_subset_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
