from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from subhanga.data import DatasetPaths, WeatherForecastDataset, build_time_index, year_mask
from subhanga.models import REG_HEAD_DIM, WeatherResNetGAP


@dataclass(frozen=True)
class TrainConfig:
    dataset_dir: str
    checkpoint_dir: str
    lead_hours: int = 24
    train_years: tuple[int, ...] = (2018, 2019, 2020)
    val_years: tuple[int, ...] = (2021,)
    batch_size: int = 16
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-2
    num_workers: int = 2
    base_channels: int = 16
    seed: int = 137
    # 7-output ablation: 6 regression + 1 binary logit; best.pt still chosen by val regression MSE only.
    multitask7: bool = False
    bce_weight: float = 1.0


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
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--train-years", default="2018,2019,2020")
    p.add_argument("--val-years", default="2021")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=137)
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision (uses more VRAM).",
    )
    p.add_argument(
        "--multitask7",
        action="store_true",
        help="7 outputs: 6 regression (MSE) + binary logit (BCE). Best checkpoint still by val regression MSE only.",
    )
    p.add_argument(
        "--bce-weight",
        type=float,
        default=1.0,
        help="Multiplies BCE term when --multitask7 (default 1.0, same as baseline coupling strength).",
    )
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
        multitask7=args.multitask7,
        bce_weight=args.bce_weight,
    )

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if cfg.multitask7:
        print(
            "[7feat multitask] 6×regression MSE(norm) + BCEWithLogits(binary@t+lead); "
            "best.pt selected by val regression MSE only (baseline-fair)."
        )
    else:
        print("[6feat baseline] regression MSE(norm) only.")

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

    # Filter t_indices to exclude any t where the target at t+lead_hours is NaN.
    # y_reg_norm still holds NaN for bad rows; if such a row enters a batch the
    # MSE loss becomes NaN and corrupts all model parameters for the rest of training.
    valid_target_at_lead = ~y_reg_norm.isnan().any(dim=1)  # (T,) bool
    train_t = train_t[valid_target_at_lead[train_t + cfg.lead_hours]]
    val_t   = val_t[valid_target_at_lead[val_t   + cfg.lead_hours]]
    print(f"Train inputs after NaN-target filter: {len(train_t)}")
    print(f"Val inputs after NaN-target filter  : {len(val_t)}")

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

    out_dim = REG_HEAD_DIM + 1 if cfg.multitask7 else REG_HEAD_DIM
    model = WeatherResNetGAP(
        in_channels=metadata["n_vars"], base_channels=cfg.base_channels, out_dim=out_dim
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse_fn = nn.MSELoss()

    bce_fn: nn.BCEWithLogitsLoss | None = None
    bce_pos_weight_cpu: torch.Tensor | None = None
    if cfg.multitask7:
        lead_idx = torch.as_tensor(train_t, dtype=torch.long) + cfg.lead_hours
        y_train_cls = y_cls[lead_idx].float()
        n_pos = float(y_train_cls.sum().item())
        n_neg = float(len(y_train_cls) - n_pos)
        if n_pos < 1.0:
            pw = 1.0
        else:
            pw = n_neg / n_pos
        bce_pos_weight_cpu = torch.tensor(pw, dtype=torch.float32)
        bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
        print(f"[7feat] BCE pos_weight={pw:.4f} (train positives={int(n_pos)}, negatives={int(n_neg)})")

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    best_val = float("inf")
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
        if use_amp
        else torch.float32
    )
    if use_amp:
        print(f"Mixed precision (autocast): enabled, dtype={amp_dtype}")

    def run_epoch(loader: DataLoader, *, train: bool) -> tuple[float, float]:
        """Returns (mean regression MSE on first REG_HEAD_DIM outputs, mean BCE or nan)."""
        model.train(train)
        total_reg = 0.0
        total_bce = 0.0
        n = 0
        infer_ctx = torch.inference_mode if not train else torch.enable_grad
        with infer_ctx():
            for x, y, y_cls_b in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_cls_b = y_cls_b.to(device, non_blocking=True).float()
                if train:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    enabled=use_amp and device.type == "cuda",
                    dtype=amp_dtype,
                ):
                    pred = model(x)
                pred_f = pred.float()
                reg_mse = mse_fn(pred_f[:, :REG_HEAD_DIM], y.float())
                if cfg.multitask7 and bce_fn is not None:
                    bce = bce_fn(pred_f[:, REG_HEAD_DIM], y_cls_b)
                    loss = reg_mse + cfg.bce_weight * bce
                    total_bce += float(bce.detach().item()) * x.size(0)
                else:
                    loss = reg_mse
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                total_reg += float(reg_mse.detach().item()) * x.size(0)
                n += int(x.size(0))
        if device.type == "cuda":
            torch.cuda.empty_cache()
        reg_mean = total_reg / max(n, 1)
        bce_mean = total_bce / max(n, 1) if cfg.multitask7 else float("nan")
        return reg_mean, bce_mean

    for epoch in range(cfg.epochs):
        train_mse, train_bce = run_epoch(train_loader, train=True)
        if len(val_ds):
            val_mse, val_bce = run_epoch(val_loader, train=False)
        else:
            val_mse, val_bce = float("nan"), float("nan")

        train_rmse = train_mse**0.5
        val_rmse = val_mse**0.5 if np.isfinite(val_mse) else float("nan")
        if cfg.multitask7:
            print(
                f"Epoch {epoch:03d} [7feat]  train_rmse(norm)={train_rmse:.4f}  val_rmse(norm)={val_rmse:.4f}  "
                f"| aux BCE: train={train_bce:.4f} val={val_bce:.4f}"
            )
        else:
            print(f"Epoch {epoch:03d}  train_rmse(norm)={train_rmse:.4f}  val_rmse(norm)={val_rmse:.4f}")

        # Save last
        last_path = ckpt_dir / "last.pt"
        ckpt_extra: dict = {}
        if cfg.multitask7 and bce_pos_weight_cpu is not None:
            ckpt_extra["bce_pos_weight"] = bce_pos_weight_cpu.clone()

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
                **ckpt_extra,
            },
            last_path,
        )

        # Save best by val regression MSE only (same criterion as 6-feature baseline).
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
                    **ckpt_extra,
                },
                best_path,
            )
            tag = "[7feat] " if cfg.multitask7 else ""
            print(f"  {tag}saved new best → {best_path} (val_reg_mse={best_val:.6f})")

    print("Done.")


if __name__ == "__main__":
    main()

