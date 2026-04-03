"""
Channel-Level Ablation Study for WeatherCNN
=============================================
For each of the 42 input channels, replace it with its mean value
(i.e. a flat, uninformative field) across the validation set, run
inference, and measure the increase in regression MSE vs. baseline.

A large ΔMSE → the model depends heavily on that channel.
"""

import os
import csv
import torch
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from work import WeatherCNN, val_dataset, y_reg_mean, y_reg_std


# ── Paste / import your definitions ───────────────────────────────────────────
# from my_module import WeatherDataset, WeatherCNN, DATASET_DIR
# (or paste them inline below)

DATASET_DIR = "/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR    = "ablation_results"
BATCH_SIZE  = 128


# Human-readable names for the 42 channels — edit to match your data.
# If None, channels are labelled Ch_00 … Ch_41.
metadata = torch.load(f"{DATASET_DIR}/metadata.pt", weights_only=False)
targets = torch.load(f"{DATASET_DIR}/targets.pt", weights_only=False)

CHANNEL_NAMES = list(metadata["variable_names"])
assert(len(CHANNEL_NAMES) == 42)

# Human-readable names for the 6 regression targets.
OUTPUT_NAMES = targets["variable_names"]

# ── 1. Load model ──────────────────────────────────────────────────────────────
model = WeatherCNN().to(DEVICE)
model.load_state_dict(torch.load("./checkpoints/best_model.pt", map_location=DEVICE))
model.eval()
print("Model loaded.")


# ── 2. Build validation DataLoader ────────────────────────────────────────────
channel_stats = torch.load("channel_stats.pt")
channel_mean  = channel_stats["mean"]   # shape (42,)
channel_std   = channel_stats["std"]    # shape (42,)

# ── Paste the same setup you used before training ─────────────────────────────
# file_names, metadata, y_reg_norm, y_cls, valid_indices = ...

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── 3. Helpers ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(loader, channel_to_ablate: Optional[int] = None, ablate_value: float = 0.0, debug: bool = True):
    """
    Run inference on loader.  If channel_to_ablate is not None, replace that
    channel with ablate_value (in normalised space, 0.0 = channel mean).

    Args:
        loader             : DataLoader to evaluate on
        channel_to_ablate  : if set, zero out this input channel for all batches
        ablate_value       : value to replace the ablated channel with (default 0.0 = channel mean)
        debug              : if True, print diagnostic information during evaluation

    Returns:
        total_mse       : scalar MSE over all samples (normalised targets)
        per_output_mse  : list[float] length 6
        total_rmse_real : scalar RMSE in real units (or None if scaling unavailable)
    """
    sum_sq      = torch.zeros(6)
    sum_sq_real = torch.zeros(6) if (y_reg_mean is not None) else None
    n_samples   = 0

    if debug:
        mode = f"ablating channel {channel_to_ablate} (value={ablate_value})" \
               if channel_to_ablate is not None else "baseline (no ablation)"
        print(f"[evaluate] Starting evaluation — {mode}")
        print(f"[evaluate] Device: {DEVICE} | Real-unit scaling available: {y_reg_mean is not None}")

    for batch_idx, (batch_x, batch_y_reg, _) in enumerate(loader):
        batch_x     = batch_x.to(DEVICE)
        batch_y_reg = batch_y_reg.cpu()

        if debug and batch_idx == 0:
            print(f"[evaluate] Input  shape : {batch_x.shape}  dtype={batch_x.dtype}")
            print(f"[evaluate] Target shape : {batch_y_reg.shape}  dtype={batch_y_reg.dtype}")
            print(f"[evaluate] Input  stats — min={batch_x.min():.4f}  max={batch_x.max():.4f}  mean={batch_x.mean():.4f}")

        if channel_to_ablate is not None:
            if debug and batch_idx == 0:
                pre_ablate_mean = batch_x[:, channel_to_ablate].mean().item()
                print(f"[evaluate] Channel {channel_to_ablate} mean before ablation: {pre_ablate_mean:.4f}")
            batch_x[:, channel_to_ablate, :, :] = ablate_value
            if debug and batch_idx == 0:
                print(f"[evaluate] Channel {channel_to_ablate} set to {ablate_value} for all spatial positions")

        preds = model(batch_x)[:, :6].cpu()

        if debug and batch_idx == 0:
            print(f"[evaluate] Preds  stats — min={preds.min():.4f}  max={preds.max():.4f}  mean={preds.mean():.4f}")
            print(f"[evaluate] Target stats — min={batch_y_reg.min():.4f}  max={batch_y_reg.max():.4f}  mean={batch_y_reg.mean():.4f}")
            batch_mse = ((preds - batch_y_reg) ** 2).mean().item()
            print(f"[evaluate] Batch 0 MSE (normalised): {batch_mse:.6f}")

        batch_sq   = (preds - batch_y_reg) ** 2
        sum_sq    += batch_sq.sum(dim=0)
        n_samples += batch_x.shape[0]

        if sum_sq_real is not None:
            preds_real   = preds       * y_reg_std + y_reg_mean
            targets_real = batch_y_reg * y_reg_std + y_reg_mean
            sum_sq_real += ((preds_real - targets_real) ** 2).sum(dim=0)

    if debug:
        print(f"[evaluate] Finished — {n_samples} total samples across {batch_idx + 1} batches")

    per_output_mse  = (sum_sq / n_samples).tolist()
    total_mse       = float((sum_sq / n_samples).mean())
    total_rmse_real = float((sum_sq_real / n_samples).mean() ** 0.5) \
                      if sum_sq_real is not None else None

    if debug:
        print(f"[evaluate] Per-output MSE : {[f'{v:.6f}' for v in per_output_mse]}")
        print(f"[evaluate] Total MSE      : {total_mse:.6f}")
        if total_rmse_real is not None:
            print(f"[evaluate] Total RMSE (real units): {total_rmse_real:.4f}")

    return total_mse, per_output_mse, total_rmse_real


# ── 4. Baseline ────────────────────────────────────────────────────────────────
print("Computing baseline …")
baseline_mse, baseline_per_out, baseline_rmse_r = evaluate(val_loader)
print(f"  Baseline MSE  (normalised) : {baseline_mse:.6f}")
if baseline_rmse_r is not None:
    print(f"  Baseline RMSE (real units) : {baseline_rmse_r:.4f}")


# ── 5. Ablation loop ───────────────────────────────────────────────────────────
num_channels = 42
if CHANNEL_NAMES is None:
    CHANNEL_NAMES = [f"Ch_{i:02d}" for i in range(num_channels)]

# In normalised space, ablating to 0.0 is equivalent to setting the channel
# to its training mean — the most "uninformative but safe" replacement.
ABLATE_VALUE = 0.0

results = []
print(f"\nAblating {num_channels} channels …")
# for ch in range(num_channels):
for ch in range(42):
    abl_mse, abl_per_out, _ = evaluate(val_loader, channel_to_ablate=ch,
                                        ablate_value=ABLATE_VALUE)
    delta_mse    = abl_mse - baseline_mse
    pct_increase = 100.0 * delta_mse / (baseline_mse + 1e-12)

    row = {
        "channel"      : CHANNEL_NAMES[ch],
        "ablated_mse"  : abl_mse,
        "delta_mse"    : delta_mse,
        "pct_increase" : pct_increase,
    }
    for i, name in enumerate(OUTPUT_NAMES):
        row[f"delta_mse_{name}"] = abl_per_out[i] - baseline_per_out[i]

    results.append(row)
    print(f"  [{ch:02d}] {CHANNEL_NAMES[ch]:25s}  ΔMSE={delta_mse:+.6f}  ({pct_increase:+.1f}%)")


# ── 6. Save CSV ────────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)
results_sorted = sorted(results, key=lambda r: r["delta_mse"], reverse=True)

csv_path = os.path.join(SAVE_DIR, "ablation_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results_sorted[0].keys())
    writer.writeheader()
    writer.writerows(results_sorted)
print(f"\nCSV saved → {csv_path}")
