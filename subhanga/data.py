from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetPaths:
    dataset_dir: Path

    @property
    def inputs_dir(self) -> Path:
        return self.dataset_dir / "inputs"

    @property
    def targets_path(self) -> Path:
        return self.dataset_dir / "targets.pt"

    @property
    def metadata_path(self) -> Path:
        return self.dataset_dir / "metadata.pt"


def _timestamp_to_input_path(inputs_dir: Path, ts: np.datetime64) -> Path:
    dt = pd.Timestamp(ts)
    return inputs_dir / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"


def build_time_index(
    times: Iterable[np.datetime64],
    *,
    lead_hours: int = 24,
) -> np.ndarray:
    """
    Returns the list of valid t indices such that t+lead_hours is within bounds.
    """
    times = np.asarray(list(times))
    t = np.arange(len(times), dtype=np.int64)
    return t[t + lead_hours < len(times)]


def year_mask(times: np.ndarray, years: set[int]) -> np.ndarray:
    years_arr = times.astype("datetime64[Y]").astype(int) + 1970
    return np.isin(years_arr, np.array(sorted(list(years)), dtype=years_arr.dtype))


class WeatherForecastDataset(Dataset):
    """
    Loads X_t from dataset/inputs and returns (x, y_reg_norm, y_binary).

    - x is returned channels-first: (C, H, W)
    - y_reg is assumed already normalized outside (passed in)
    """

    def __init__(
        self,
        *,
        paths: DatasetPaths,
        times: np.ndarray,
        t_indices: np.ndarray,
        y_reg_norm: torch.Tensor,
        y_binary: torch.Tensor,
        lead_hours: int = 24,
        channel_mean: torch.Tensor | None = None,
        channel_std: torch.Tensor | None = None,
        dswrf_channel_idx: int = 5,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.times = times
        self.t_indices = t_indices.astype(np.int64)
        self.y_reg_norm = y_reg_norm
        self.y_binary = y_binary
        self.lead_hours = int(lead_hours)
        self.dswrf_channel_idx = int(dswrf_channel_idx)

        self.channel_mean = channel_mean
        self.channel_std = channel_std

    def __len__(self) -> int:
        return int(len(self.t_indices))

    def __getitem__(self, i: int):
        t_idx = int(self.t_indices[i])
        ts = self.times[t_idx]
        x_path = _timestamp_to_input_path(self.paths.inputs_dir, ts)

        x = torch.load(x_path, weights_only=True).float()  # (H, W, C)
        x = x.permute(2, 0, 1)  # (C, H, W)

        # If DSWRF still has NaNs (nighttime), fill with 0.0
        if x[self.dswrf_channel_idx].isnan().any():
            x[self.dswrf_channel_idx] = torch.nan_to_num(x[self.dswrf_channel_idx], nan=0.0)

<<<<<<< HEAD
        # Fill any remaining NaNs in other channels (e.g. files listed in bad_indices_x.txt)
        if x.isnan().any():
            x = torch.nan_to_num(x, nan=0.0)

=======
>>>>>>> CNN-train-subhanga
        if self.channel_mean is not None and self.channel_std is not None:
            x = (x - self.channel_mean[:, None, None]) / self.channel_std[:, None, None]

        y_idx = t_idx + self.lead_hours
        y_reg = self.y_reg_norm[y_idx]
        y_cls = self.y_binary[y_idx]
        return x, y_reg, y_cls

