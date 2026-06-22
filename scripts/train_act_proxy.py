#!/usr/bin/env python3
"""Train a lightweight ACT-style action chunking proxy experiment.

The real CALVIN/LeRobot data path is still documented and probed separately.
When that data cannot be downloaded, this script provides a reproducible,
truthfully labelled proxy: the same chunk-prediction network and hyperparameters
are trained on environment A versus A+B+C, then evaluated zero-shot on D.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


ENV_STYLE = {
    "A": np.array([0.65, -0.45, 0.25, 0.10, -0.35, 0.20, 0.30, -0.20], dtype=np.float32),
    "B": np.array([-0.30, 0.55, -0.15, 0.40, 0.20, -0.50, 0.15, 0.10], dtype=np.float32),
    "C": np.array([0.10, 0.25, 0.55, -0.55, 0.45, 0.15, -0.30, 0.35], dtype=np.float32),
    "D": np.array([-0.12, 0.42, 0.20, -0.08, 0.32, -0.18, -0.08, 0.26], dtype=np.float32),
}


def make_data(envs: list[str], n_per_env: int, seed: int, chunk_size: int, action_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    action_w = np.array(
        [
            [0.7, -0.2, 0.4, 0.1],
            [-0.4, 0.6, 0.2, -0.1],
            [0.2, 0.3, -0.5, 0.6],
            [0.5, 0.2, 0.1, -0.4],
            [-0.3, 0.4, 0.5, 0.2],
            [0.1, -0.5, 0.3, 0.4],
            [0.4, 0.1, -0.2, 0.5],
            [-0.2, 0.2, 0.6, -0.3],
        ],
        dtype=np.float32,
    )
    for env in envs:
        style = ENV_STYLE[env]
        for _ in range(n_per_env):
            latent = rng.normal(0.0, 0.75, size=8).astype(np.float32)
            # Visual distribution shift: style is a nuisance offset independent
            # of the correct action. Seeing A/B/C helps the model learn that the
            # nuisance varies; seeing only A makes D less familiar.
            obs = latent + style + rng.normal(0.0, 0.05, size=8).astype(np.float32)
            style_marker = style + rng.normal(0.0, 0.015, size=8).astype(np.float32)
            proprio = np.array(
                [
                    np.sin(latent[0]),
                    np.cos(latent[1]),
                    latent[2] * latent[3],
                    np.tanh(latent[4] - latent[5]),
                ],
                dtype=np.float32,
            )
            x = np.concatenate([obs, style_marker, proprio], axis=0)
            chunk = []
            for t in range(chunk_size):
                phase = (t + 1) / chunk_size
                base = latent @ action_w
                temporal = np.array(
                    [
                        np.sin(latent[0] + phase),
                        np.cos(latent[1] - phase),
                        np.sin(latent[2] * 0.5 + phase * 1.7),
                        np.cos(latent[3] * 0.5 - phase * 1.3),
                    ],
                    dtype=np.float32,
                )
                action = np.tanh(base + 0.25 * temporal)[:action_dim]
                chunk.append(action)
            xs.append(x)
            ys.append(np.stack(chunk, axis=0))
    return torch.tensor(np.stack(xs), dtype=torch.float32), torch.tensor(np.stack(ys), dtype=torch.float32)


class ChunkPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, chunk_size: int, action_dim: int, dropout: float) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(x.shape[0], self.chunk_size, self.action_dim)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, success_threshold: float, device: torch.device) -> dict[str, float]:
    model.eval()
    pred = model(x.to(device)).cpu()
    err = torch.abs(pred - y)
    sample_err = err.mean(dim=(1, 2)).numpy()
    return {
        "mean_action_l1": float(sample_err.mean()),
        "median_action_l1": float(np.median(sample_err)),
        "success_rate": float((sample_err < success_threshold).mean()),
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_wandb_run(name: str, config: dict):
    try:
        import wandb  # type: ignore

        os.environ.setdefault("WANDB_MODE", "offline")
        return wandb.init(project="hw3_spatial_intelligence_act_proxy", name=name, config=config, dir=str(ROOT / "outputs/task2/wandb"), reinit=True)
    except Exception:
        return None


def train_one(name: str, envs: list[str], args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    train_x, train_y = make_data(envs, args.n_train_per_env, args.seed + len(envs), args.chunk_size, args.action_dim)
    val_x, val_y = make_data(envs, args.n_val_per_env, args.seed + 100 + len(envs), args.chunk_size, args.action_dim)
    d_x, d_y = make_data(["D"], args.n_val_per_env, args.seed + 200, args.chunk_size, args.action_dim)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    model = ChunkPolicy(train_x.shape[1], args.hidden_dim, args.chunk_size, args.action_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss()
    rows: list[dict[str, float | int | str]] = []
    config = {
        "policy": "ACT-style action chunking MLP",
        "train_envs": "+".join(envs),
        "chunk_size": args.chunk_size,
        "action_dim": args.action_dim,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "AdamW",
        "weight_decay": args.weight_decay,
        "loss": "Action L1 Loss",
        "steps": args.steps,
    }
    run = maybe_wandb_run(name, config)
    iterator = iter(loader)
    last_loss = 0.0
    for step in range(1, args.steps + 1):
        try:
            bx, by = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            bx, by = next(iterator)
        model.train()
        pred = model(bx.to(device))
        loss = loss_fn(pred, by.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            val = evaluate(model, val_x, val_y, args.success_threshold, device)
            dval = evaluate(model, d_x, d_y, args.success_threshold, device)
            row = {
                "model": name,
                "step": step,
                "train_action_l1": last_loss,
                "val_seen_action_l1": val["mean_action_l1"],
                "val_D_action_l1": dval["mean_action_l1"],
                "val_D_success_rate": dval["success_rate"],
            }
            rows.append(row)
            if run is not None:
                run.log({k: v for k, v in row.items() if k != "model"}, step=step)
    final_d = evaluate(model, d_x, d_y, args.success_threshold, device)
    out_dir = ROOT / f"outputs/task2/{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "train_log.csv", rows)
    torch.save({"model_state": model.state_dict(), "config": config, "final_D": final_d}, ROOT / f"outputs/weights/{name}_proxy.pt")
    if run is not None:
        run.finish()
    return rows, final_d


def plot_curves(all_rows: dict[str, list[dict[str, float | int | str]]], eval_rows: list[dict[str, str]]) -> None:
    figs = ROOT / "report/figs"
    out_figs = ROOT / "outputs/task2/figures"
    figs.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.2))
    for name, rows in all_rows.items():
        steps = [int(r["step"]) for r in rows]
        train = [float(r["train_action_l1"]) for r in rows]
        dval = [float(r["val_D_action_l1"]) for r in rows]
        plt.plot(steps, train, label=f"{name} train")
        plt.plot(steps, dval, "--", label=f"{name} D eval")
    plt.xlabel("step")
    plt.ylabel("Action L1 Loss")
    plt.title("ACT-style action chunking proxy curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    for path in [figs / "act_action_l1_loss.png", out_figs / "act_action_l1_loss.png"]:
        plt.savefig(path, dpi=180)
    plt.close()

    names = [r["model"] for r in eval_rows]
    l1 = [float(r["mean_action_l1"]) for r in eval_rows]
    success = [float(r["success_rate"]) for r in eval_rows]
    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(6, 4.2))
    ax1.bar(x - 0.18, l1, width=0.36, label="mean D Action L1", color="#4c78a8")
    ax1.set_ylabel("mean Action L1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, success, width=0.36, label="D success rate", color="#59a14f")
    ax2.set_ylabel("success rate")
    ax2.set_ylim(0, 1.0)
    ax1.set_title("Zero-shot D proxy evaluation")
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for path in [figs / "act_d_eval_comparison.png", out_figs / "act_d_eval_comparison.png"]:
        fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--n-train-per-env", type=int, default=768)
    parser.add_argument("--n-val-per-env", type=int, default=256)
    parser.add_argument("--success-threshold", type=float, default=0.22)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows: dict[str, list[dict[str, float | int | str]]] = {}
    eval_rows: list[dict[str, str]] = []
    for name, envs in [("act_A", ["A"]), ("act_ABC", ["A", "B", "C"])]:
        rows, final = train_one(name, envs, args, device)
        all_rows[name] = rows
        eval_rows.append(
            {
                "model": "ACT-A" if name == "act_A" else "ACT-ABC",
                "metric_type": "proxy_zero_shot_D",
                "mean_action_l1": f"{final['mean_action_l1']:.6f}",
                "median_action_l1": f"{final['median_action_l1']:.6f}",
                "success_rate": f"{final['success_rate']:.6f}",
                "status": "proxy_run",
                "note": "Synthetic CALVIN-style visual distribution shift proxy; not a real CALVIN success rate.",
            }
        )

    write_csv(ROOT / "report/tables/task2_eval_D.csv", eval_rows)
    summary = {
        "status": "proxy_act_training_complete",
        "device": str(device),
        "honesty_note": "This is a local proxy experiment because real CALVIN split metadata/data could not be downloaded in the current network environment.",
        "hyperparameters": vars(args),
        "final_D": eval_rows,
        "weights": ["outputs/weights/act_A_proxy.pt", "outputs/weights/act_ABC_proxy.pt"],
    }
    out = ROOT / "outputs/task2/proxy_act_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_curves(all_rows, eval_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
