#!/usr/bin/env python3
"""
saliency.py  –  Gradient saliency maps for SkyOracle (Part 2)

For each of the 6 continuous output variables, compute the mean absolute
gradient of the model output w.r.t. the input tensor, aggregated as
    s_v(i,j) = mean_t  max_c  |∂ŷ_v / ∂X_{t,c,i,j}|
over ~200 stratified 2021 samples (half rainy, half dry).

Saves one 6-panel cartopy figure to the figures directory.

Run from repo root:
    module load pytorch/2.8.0-cuda12.9-cudnn9
    python3 -m subhanga.saliency
or directly:
    python3 subhanga/saliency.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_DIR = Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset")
# best.pt is our trained WeatherResNetGAP (base_channels=16, 6 outputs).
# Note: checkpoints/best_model.pt is a different architecture (WeatherCNN, 7 outputs)
# and is NOT compatible with WeatherResNetGAP — do not use it here.
CHECKPOINT  = Path("/cluster/tufts/c26sp1cs0137/supadh03/SkyOracle/checkpoints/subhanga/best.pt")
FIGURES_DIR = Path("/cluster/tufts/c26sp1cs0137/supadh03/SkyOracle/subhanga/figures")
INPUTS_DIR  = DATASET_DIR / "inputs"

# ── Saliency settings ─────────────────────────────────────────────────────────
N_RAINY    = 100   # rainy samples (binary_label == True at lead time)
N_DRY      = 100   # dry samples
LEAD_HOURS = 24
TEST_YEAR  = 2021
SEED       = 137

# From data_spec.py — Jumbo Statue at grid row 177, col 263
JUMBO_Y_IDX = 177
JUMBO_X_IDX = 263
DSWRF_CH    = 5  # channel index for DSWRF (NaN at night → fill 0)

TARGET_NAMES = [
    "TMP@2m_above_ground",
    "RH@2m_above_ground",
    "UGRD@10m_above_ground",
    "VGRD@10m_above_ground",
    "GUST@surface",
    "DSWRF@surface",
]
TARGET_LABELS = [
    "2-m Temperature (K)",
    "2-m Relative Humidity (%)",
    "10-m U-Wind (m/s)",
    "10-m V-Wind (m/s)",
    "Surface Gust (m/s)",
    "SW Radiation (W/m²)",
]


# ── Model definitions ─────────────────────────────────────────────────────────
# WeatherResNetGAP lives in subhanga/models.py — import it directly so any
# future changes there (BN config, layer shapes, etc.) are automatically
# reflected here.  Only DenormalizeTargets is defined here because it is a
# new inference-time wrapper not present in models.py.
from subhanga.models import WeatherResNetGAP   # noqa: E402  (after sys.path setup)


class DenormalizeTargets(nn.Module):
    """Wraps a normalized-output model and converts predictions to real units."""

    def __init__(self, base: nn.Module, y_mean: torch.Tensor, y_std: torch.Tensor) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("y_mean", y_mean.float().clone())
        self.register_buffer("y_std",  y_std.float().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) * self.y_std + self.y_mean


# ── Dataset helpers ───────────────────────────────────────────────────────────

def build_time_index(times: np.ndarray, lead_hours: int = 24) -> np.ndarray:
    t = np.arange(len(times), dtype=np.int64)
    return t[t + lead_hours < len(times)]


def year_mask(times: np.ndarray, years: set) -> np.ndarray:
    years_arr = times.astype("datetime64[Y]").astype(int) + 1970
    return np.isin(years_arr, np.array(sorted(years), dtype=years_arr.dtype))


def load_input(
    t_idx: int,
    times: np.ndarray,
    inputs_dir: Path,
    channel_mean: torch.Tensor | None,
    channel_std:  torch.Tensor | None,
) -> torch.Tensor:
    """Load, permute, clean, and optionally normalise a single input tensor."""
    dt   = pd.Timestamp(times[t_idx])
    path = inputs_dir / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
    x    = torch.load(path, weights_only=True).float()  # (H, W, C)
    x    = x.permute(2, 0, 1)                            # (C, H, W)
    if x[DSWRF_CH].isnan().any():
        x[DSWRF_CH] = torch.nan_to_num(x[DSWRF_CH], nan=0.0)
    if x.isnan().any():
        x = torch.nan_to_num(x, nan=0.0)
    if channel_mean is not None:
        x = (x - channel_mean[:, None, None]) / channel_std[:, None, None]
    return x  # (C, H, W)


# ── Saliency computation ──────────────────────────────────────────────────────

def saliency_one_sample(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    n_targets: int = 6,
) -> torch.Tensor:
    """
    Compute per-variable gradient saliency for a single (1, C, H, W) input.

    For variable v:
        sal_v(i,j) = max_c |∂ŷ_v / ∂X_{c,i,j}|

    Returns shape (n_targets, H, W).
    """
    x_in = x.unsqueeze(0).to(device)            # (1, C, H, W)
    x_in = x_in.requires_grad_(True)

    # Single forward pass; retain graph for multiple backward calls
    out = model(x_in)                            # (1, n_targets)

    sals = []
    for v in range(n_targets):
        grad = torch.autograd.grad(
            outputs=out[0, v],
            inputs=x_in,
            retain_graph=(v < n_targets - 1),
            create_graph=False,
        )[0]                                     # (1, C, H, W)
        sal_v = grad.squeeze(0).abs().max(dim=0)[0]   # (H, W)
        sals.append(sal_v.detach().cpu())

    return torch.stack(sals, dim=0)              # (n_targets, H, W)


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_projection():
    return ccrs.LambertConformal(
        central_longitude=262.5,
        central_latitude=38.5,
        standard_parallels=(38.5, 38.5),
        globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
    )


def plot_saliency(
    mean_sal: np.ndarray,   # (6, H, W)
    grid_x:   np.ndarray,   # (W,) Lambert-Conformal x in metres
    grid_y:   np.ndarray,   # (H,) Lambert-Conformal y in metres
    jumbo_x:  float,
    jumbo_y:  float,
    n_rainy:  int,
    n_dry:    int,
    out_path: Path,
) -> None:
    proj = make_projection()
    X, Y = np.meshgrid(grid_x, grid_y)          # (H, W) each

    fig, axes = plt.subplots(
        2, 3, figsize=(19, 12),
        subplot_kw={"projection": proj},
    )
    axes = axes.flatten()

    for v, ax in enumerate(axes):
        sal = mean_sal[v]                        # (H, W)

        # Robust colour limits: 1st–99th percentile
        vmin = float(np.percentile(sal,  1))
        vmax = float(np.percentile(sal, 99))
        if vmax <= vmin:
            vmax = vmin + 1e-12

        ax.set_extent(
            [grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
            crs=proj,
        )
        ax.add_feature(cfeature.OCEAN,     facecolor="#cce5ff", zorder=0)
        ax.add_feature(cfeature.LAND,      facecolor="#f5f5f5", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7,       zorder=2)
        ax.add_feature(cfeature.STATES,    linewidth=0.5,       zorder=2)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.5,       zorder=2)

        im = ax.pcolormesh(
            X, Y, sal,
            cmap="inferno",
            vmin=vmin, vmax=vmax,
            transform=proj,
            shading="auto",
            zorder=1,
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
        cbar.set_label("mean |∂ŷ/∂X| (max over channels)", fontsize=7)

        # Jumbo Statue of America (Tufts, Medford MA)
        ax.scatter(
            jumbo_x, jumbo_y,
            marker="*", s=180, color="cyan", edgecolors="black", linewidths=0.5,
            transform=proj, zorder=6, label="Jumbo",
        )
        ax.legend(loc="lower right", fontsize=7, framealpha=0.7)
        ax.set_title(TARGET_LABELS[v], fontsize=10, fontweight="bold", pad=4)

    fig.suptitle(
        f"Gradient Saliency Maps  |  WeatherResNetGAP  |  "
        f"2021 test set  ({n_rainy} rainy + {n_dry} dry)\n"
        r"$s_v(i,j) = \langle\, \max_c\,|\partial\hat{y}_v\,/\,\partial X_{c,i,j}|\,\rangle_t$",
        fontsize=12, fontweight="bold", y=1.01,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    print(f"Checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, weights_only=False, map_location=device)
    cfg  = ckpt["config"]
    base_ch      = cfg.get("base_channels", 16)
    y_mean       = ckpt["y_mean"].float()
    y_std        = ckpt["y_std"].float()
    channel_mean = ckpt.get("channel_mean")
    channel_std  = ckpt.get("channel_std")
    if channel_mean is not None:
        channel_mean = channel_mean.float()
        channel_std  = channel_std.float()

    base = WeatherResNetGAP(in_channels=42, base_channels=base_ch, out_dim=6)
    base.load_state_dict(ckpt["model_state_dict"])
    model = DenormalizeTargets(base, y_mean, y_std).to(device)
    model.eval()
    print(f"  base_channels={base_ch}, epoch={ckpt['epoch']}")
    print(f"  y_mean: {y_mean.tolist()}")
    print(f"  y_std:  {y_std.tolist()}")

    # ── Load metadata & targets ────────────────────────────────────────────
    print("Loading targets & metadata …")
    targets  = torch.load(DATASET_DIR / "targets.pt",  weights_only=False, map_location="cpu")
    metadata = torch.load(DATASET_DIR / "metadata.pt", weights_only=False, map_location="cpu")

    times    = np.asarray(targets["time"])      # (T,) datetime64
    y_binary = np.asarray(targets["binary_label"], dtype=bool)  # (T,)

    # Grid coordinates: fall back to data_spec values if not in metadata
    if "grid_x" in metadata:
        grid_x = np.asarray(metadata["grid_x"])
        grid_y = np.asarray(metadata["grid_y"])
    else:
        grid_x = 1352479.8574780696 + np.arange(449) * 3000.0
        grid_y =  212693.8474433364 + np.arange(450) * 3000.0
        print("  grid_x/grid_y not in metadata — using data_spec.py values")

    jumbo_x = float(grid_x[JUMBO_X_IDX])
    jumbo_y = float(grid_y[JUMBO_Y_IDX])
    print(f"  Jumbo coords in projection: ({jumbo_x:.0f} m, {jumbo_y:.0f} m)")

    # ── Stratified 2021 sample selection ──────────────────────────────────
    print(f"Building stratified {TEST_YEAR} sample index …")
    t_all     = build_time_index(times, lead_hours=LEAD_HOURS)
    mask_2021 = year_mask(times, {TEST_YEAR})
    t_2021    = t_all[mask_2021[t_all]]

    # y_binary is indexed at the *target* time (t + lead_hours)
    lead_binary = y_binary[t_2021 + LEAD_HOURS]
    rainy_pool  = t_2021[lead_binary]
    dry_pool    = t_2021[~lead_binary]

    rng = np.random.default_rng(SEED)
    sel_rainy = rng.choice(rainy_pool, size=min(N_RAINY, len(rainy_pool)), replace=False)
    sel_dry   = rng.choice(dry_pool,   size=min(N_DRY,   len(dry_pool)),   replace=False)
    sel_all   = np.concatenate([sel_rainy, sel_dry])
    rng.shuffle(sel_all)   # avoid bias from ordering
    n_rainy, n_dry = len(sel_rainy), len(sel_dry)
    print(f"  Rainy pool: {len(rainy_pool)}, sampled: {n_rainy}")
    print(f"  Dry pool:   {len(dry_pool)},   sampled: {n_dry}")
    print(f"  Total: {len(sel_all)}")

    # ── Accumulate saliency ────────────────────────────────────────────────
    H, W = 450, 449
    accum = torch.zeros(6, H, W)  # running sum over samples

    print(f"Computing saliency ({len(sel_all)} samples) …")
    for i, t_idx in enumerate(sel_all):
        x = load_input(int(t_idx), times, INPUTS_DIR, channel_mean, channel_std)
        sal = saliency_one_sample(model, x, device)  # (6, H, W)
        accum += sal
        if (i + 1) % 25 == 0 or (i + 1) == len(sel_all):
            print(f"  [{i+1:3d}/{len(sel_all)}] running max sal: {accum.max().item()/(i+1):.4e}")

    mean_sal = (accum / len(sel_all)).numpy()  # (6, H, W)

    # Quick summary per variable
    print("\nPer-variable saliency summary (mean map statistics):")
    for v, name in enumerate(TARGET_NAMES):
        s = mean_sal[v]
        # Find the spatial maximum
        flat_max = s.argmax()
        iy, ix = np.unravel_index(flat_max, s.shape)
        print(f"  [{v}] {name:<35s}  "
              f"mean={s.mean():.4e}  max={s.max():.4e}  "
              f"peak=(y={iy}, x={ix})")

    # ── Plot ───────────────────────────────────────────────────────────────
    print("\nPlotting …")
    out_path = FIGURES_DIR / "saliency_2021.png"
    plot_saliency(
        mean_sal, grid_x, grid_y,
        jumbo_x, jumbo_y,
        n_rainy, n_dry,
        out_path,
    )

    # Also save the raw saliency arrays for further analysis
    np.save(FIGURES_DIR / "mean_sal_2021.npy", mean_sal)
    print(f"Raw saliency arrays saved → {FIGURES_DIR / 'mean_sal_2021.npy'}")
    print("Done.")


if __name__ == "__main__":
    main()
