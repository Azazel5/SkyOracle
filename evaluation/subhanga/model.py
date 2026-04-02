"""
Subhanga model wrapper for evaluation/evaluate.py

This must expose:
    get_model(metadata: dict) -> torch.nn.Module

The returned model must accept input tensors shaped (B, 450, 449, c) float32
and return predictions shaped (B, 6) in real units, matching targets.pt order:
    [TMP@2m_above_ground, RH@2m_above_ground, UGRD@10m_above_ground,
     VGRD@10m_above_ground, GUST@surface, APCP_1hr_acc_fcst@surface]
"""

import os
from pathlib import Path

import torch
import torch.nn as nn

from subhanga.models import DenormalizeTargets, WeatherResNetGAP


class ChannelsLastToFirst(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # evaluator provides (B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.base(x)


def _load_checkpoint() -> dict:
    ckpt_env = os.environ.get("SKYORACLE_CHECKPOINT", "").strip()
    if ckpt_env:
        ckpt_path = Path(ckpt_env)
    else:
        ckpt_path = Path(__file__).resolve().parents[2] / "checkpoints" / "subhanga" / "best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Set environment variable SKYORACLE_CHECKPOINT to the checkpoint .pt path."
        )

    return torch.load(ckpt_path, weights_only=False, map_location="cpu")


def get_model(metadata: dict) -> nn.Module:
    ckpt = _load_checkpoint()

    base_channels = int(ckpt.get("config", {}).get("base_channels", 32))
    core = WeatherResNetGAP(in_channels=int(metadata["n_vars"]), base_channels=base_channels, out_dim=6)
    core.load_state_dict(ckpt["model_state_dict"])
    core.eval()

    y_mean = ckpt["y_mean"]
    y_std = ckpt["y_std"]
    model = ChannelsLastToFirst(DenormalizeTargets(core, y_mean=y_mean, y_std=y_std))
    model.eval()
    return model

