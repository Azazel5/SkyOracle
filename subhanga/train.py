from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from subhanga.data import DatasetPaths, WeatherForecastDataset, build_time_index, year_mask
from subhanga.models import WeatherResNetGAP


@dataclass(frozen=True)
class TrainConfig:
    dataset_dir: str
    checkpoint_dir: str
    lead_hours: int = 24
    train_years: tuple[int, ...] = (2018, 2019, 2020)
    val_years: tuple[int, ...] = (2021,)
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-2
    num_workers: int = 2
    base_channels: int = 32
    seed: int = 137


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_target_stats(y: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y_valid = y[valid_mask]
    mean = y_valid.mean(dim=0)
    std = y_valid.std(dim=0)
    std[std < 1e-6] = 1.0
    return mean, std


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset")
    p.add_argument("--checkpoint-dir", default="checkpoints/subhanga")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--train-years", default="2018,2019,2020")
    p.add_argument("--val-years", default="2021")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=137)
    args = p.parse_args()

    cfg = TrainConfig(
        dataset_dir=args.dataset_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        base_channels=args.base_channels,
        train_years=tuple(int(x) for x in args.train_years.split(",") if x.strip()),
        val_years=tuple(int(x) for x in args.val_years.split(",") if x.strip()),
        num_workers=args.num_workers,
        seed=args.seed,
    )

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paths = DatasetPaths(dataset_dir=Path(cfg.dataset_dir))
    metadata = torch.load(paths.metadata_path, weights_only=False)
    targets = torch.load(paths.targets_path, weights_only=False)

    times = targets["time"]  # numpy datetime64
    y_reg = targets["values"].float()  # (T, 6)
    y_cls = targets["binary_label"]  # (T,)

    t_indices_all = build_time_index(times, lead_hours=cfg.lead_hours)
    times_np = np.asarray(times)

    train_mask = year_mask(times_np, set(cfg.train_years))
    val_mask = year_mask(times_np, set(cfg.val_years))
    train_t = t_indices_all[train_mask[t_indices_all]]
    val_t = t_indices_all[val_mask[t_indices_all]]

    print(f"Total time steps: {len(times_np)}")
    print(f"Total usable inputs (t with t+{cfg.lead_hours} in range): {len(t_indices_all)}")
    print(f"Train inputs: {len(train_t)}  years={cfg.train_years}")
    print(f"Val inputs  : {len(val_t)}  years={cfg.val_years}")
    print(f"Targets shape: {tuple(y_reg.shape)}")
    print(f"Binary label positives: {int(y_cls.sum().item())}/{len(y_cls)}")

    # Remove any NaN rows in targets when computing stats (robust to older datasets)
    valid_target_rows = ~y_reg.isnan().any(dim=1)
    y_mean, y_std = compute_target_stats(y_reg, valid_target_rows)
    y_reg_norm = (y_reg - y_mean) / y_std

    # Load precomputed channel stats if present (work.py produces channel_stats.pt)
    channel_mean = channel_std = None
    channel_stats_path = Path("channel_stats.pt")
    if channel_stats_path.exists():
        channel_stats = torch.load(channel_stats_path, weights_only=False)
        channel_mean = channel_stats["mean"].float()
        channel_std = channel_stats["std"].float()
        print("Loaded channel normalization stats from channel_stats.pt")
    else:
        print("channel_stats.pt not found; training without input normalization")

    train_ds = WeatherForecastDataset(
        paths=paths,
        times=times_np,
        t_indices=train_t,
        y_reg_norm=y_reg_norm,
        y_binary=y_cls,
        lead_hours=cfg.lead_hours,
        channel_mean=channel_mean,
        channel_std=channel_std,
    )
    val_ds = WeatherForecastDataset(
        paths=paths,
        times=times_np,
        t_indices=val_t,
        y_reg_norm=y_reg_norm,
        y_binary=y_cls,
        lead_hours=cfg.lead_hours,
        channel_mean=channel_mean,
        channel_std=channel_std,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )

    model = WeatherResNetGAP(in_channels=metadata["n_vars"], base_channels=cfg.base_channels, out_dim=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    best_val = float("inf")

    def run_epoch(loader: DataLoader, *, train: bool) -> float:
        model.train(train)
        total = 0.0
        n = 0
        for x, y, _ycls in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total += float(loss.item()) * x.size(0)
            n += int(x.size(0))
        return total / max(n, 1)

    for epoch in range(cfg.epochs):
        train_mse = run_epoch(train_loader, train=True)
        val_mse = run_epoch(val_loader, train=False) if len(val_ds) else float("nan")

        train_rmse = train_mse**0.5
        val_rmse = val_mse**0.5 if np.isfinite(val_mse) else float("nan")
        print(f"Epoch {epoch:03d}  train_rmse(norm)={train_rmse:.4f}  val_rmse(norm)={val_rmse:.4f}")

        # Save last
        last_path = ckpt_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "y_mean": y_mean,
                "y_std": y_std,
                "channel_mean": channel_mean,
                "channel_std": channel_std,
                "config": asdict(cfg),
            },
            last_path,
        )

        # Save best by val (or by train if no val set)
        score = val_mse if np.isfinite(val_mse) else train_mse
        if score < best_val:
            best_val = score
            best_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "channel_mean": channel_mean,
                    "channel_std": channel_std,
                    "config": asdict(cfg),
                },
                best_path,
            )
            print(f"  saved new best → {best_path} (mse={best_val:.6f})")

    print("Done.")


if __name__ == "__main__":
    main()

