"""
Stub / persistence baseline model.

Prediction strategy: use the current weather conditions at the Jumbo Statue
grid point as the forecast for 24 hours later.  No learning involved —
this serves as a lower-bound baseline.

Interface (shared by all models in this evaluation framework):
    get_model(metadata: dict) -> torch.nn.Module

    The model forward method accepts a float tensor of shape (B, 450, 449, c)
    and returns predictions of shape (B, 6) matching the target variable order:
        [TMP@2m_above_ground, RH@2m_above_ground, UGRD@10m_above_ground,
         VGRD@10m_above_ground, GUST@surface, APCP_1hr_acc_fcst@surface]
"""

import torch
import torch.nn as nn

import submodule  # To test whether submodules can be loaded

# Target variable order — must match targets.pt["variable_names"]
TASK2_TARGET_VARS = [
    "TMP@2m_above_ground",
    "RH@2m_above_ground",
    "UGRD@10m_above_ground",
    "VGRD@10m_above_ground",
    "GUST@surface",
    "APCP_1hr_acc_fcst@surface",
]


class StubModel(nn.Module):
    """
    Persistence baseline: predict t+24h by returning current values at Jumbo.

    Parameters
    ----------
    jumbo_y_idx : int
        Row index of the Jumbo Statue grid point in the spatial grid.
    jumbo_x_idx : int
        Column index of the Jumbo Statue grid point in the spatial grid.
    channel_indices : list[int]
        Indices into the c-dimension of the input tensor that correspond to
        the 6 target variables (in TASK2_TARGET_VARS order).
    """

    def __init__(self, jumbo_y_idx: int, jumbo_x_idx: int, channel_indices: list):
        super().__init__()
        self.iy = jumbo_y_idx
        self.ix = jumbo_x_idx
        # Stored as a buffer so it moves with .to(device) calls
        self.register_buffer("chan", torch.tensor(channel_indices, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 450, 449, c) — batch of spatial weather snapshots.

        Returns
        -------
        torch.Tensor
            Shape (B, 6) — predicted values for the 6 target variables.
        """
        point = x[:, self.iy, self.ix, :]   # (B, c)
        return point[:, self.chan]            # (B, 6)


def get_model(metadata: dict) -> StubModel:
    """
    Build and return a StubModel from dataset metadata.

    Parameters
    ----------
    metadata : dict
        Loaded from dataset/metadata.pt.  Must contain:
            - "jumbo_y_idx": int
            - "jumbo_x_idx": int
            - "variable_names": list of all input channel names (VAR_LEVELS order)
    """
    all_vars = list(metadata["variable_names"])
    channel_indices = [all_vars.index(v) for v in TASK2_TARGET_VARS]
    return StubModel(metadata["jumbo_y_idx"], metadata["jumbo_x_idx"], channel_indices)
