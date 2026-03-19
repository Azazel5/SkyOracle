import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda")
DATASET_DIR = '/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset'
metadata = torch.load(f'{DATASET_DIR}/metadata.pt', weights_only=False)
targets = torch.load(f'{DATASET_DIR}/targets.pt', weights_only=False)



# ── 1. Convolution Layer ──────────────────────────────────────────────────────
class ConvLayer(nn.Module):
    def __init__(self, in_channels=42, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1   # padding=1 keeps H/W identical with kernel_size=3
        )
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, 42, 450, 449) → (B, 64, 450, 449)
        return self.relu(self.bn(self.conv(x)))


# ── 2. MaxPool Layer ──────────────────────────────────────────────────────────
class MaxPoolLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # x: (B, 64, 450, 449) → (B, 64, 225, 224)
        return self.pool(x)

# Linear layer
class FCLayer(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc      = nn.LazyLinear(num_classes)  # ← infers in_features automatically

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# Get the data for a specific year
# Total number of files per year
# 2018: 4128
# 2019: 8760
# 2020: 8784
# 2021: 4608

n_2018 = 4128
n_2019 = 8760
n_2020 = 8784
n_2021 = 4608

start_date_2018 = '2018-07-13T00:00'
start_date_2019 = '2019-01-01T00:00'
start_date_2020 = '2020-01-01T00:00'
start_date_2021 = '2021-01-01T00:00'

def generate_datetime_strings(start_date: str, n: int) -> list[str]:
    """
    Generate YYYYMMDDHH strings for n dates starting from start_date,
    incrementing by 24 hours each step.
    
    Args:
        start_date: Date string in format 'YYYY-MM-DDTHH:MM'
        n: Number of dates to generate
    
    Returns:
        List of strings in YYYYMMDDHH format
    """
    dt = pd.Timestamp(start_date)
    return [("X_" + (dt + pd.Timedelta(hours=1 * i)).strftime('%Y%m%d%H') + ".pt") for i in range(n)]

tensor_names_2018 = generate_datetime_strings(start_date_2018, n_2018)
tensor_names_2019 = generate_datetime_strings(start_date_2019, n_2019)
tensor_names_2020 = generate_datetime_strings(start_date_2020, n_2020)
tensor_names_2021 = generate_datetime_strings(start_date_2021, n_2021)

file_names = tensor_names_2018 + tensor_names_2019 + tensor_names_2020 + tensor_names_2021
print(file_names[:10])
print(len(file_names))

class WeatherDataset(Dataset):
    def __init__(self, file_names, targets_dict, metadata, lead_time=24):
        self.file_names = file_names
        self.target_values = targets_dict['values']       # The 6 regression vars
        self.target_labels = targets_dict['binary_label'] # The 1 rain label
        self.lead_time = lead_time
        
        # Crop parameters (optional but recommended)
        self.iy = metadata['jumbo_y_idx']
        self.ix = metadata['jumbo_x_idx']

    def __len__(self):
        # We stop 24 hours early because we don't have "future" targets for the last day
        return len(self.file_names) - self.lead_time

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        year = fname.split('_')[1][:4]
        path = f"{DATASET_DIR}/inputs/{year}/{fname}"
        
        x = torch.load(path, weights_only=True).float()
        x = x.permute(2, 0, 1)  # (42, H, W)

        # DSWRF@surface (channel 5) is NaN at night — physically correct fill is 0.0
        if x[5].isnan().any():
            x[5] = torch.nan_to_num(x[5], nan=0.0)

        y_reg = self.target_values[idx + self.lead_time]
        y_cls = self.target_labels[idx + self.lead_time]
        return x, y_reg, y_cls

# Model architecture
class WeatherCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ConvLayer(42, 64)
        self.pool1  = MaxPoolLayer()
        self.layer2 = ConvLayer(64, 128)
        self.pool2  = MaxPoolLayer()
        # You can add more layers here to get deeper!
        
        self.head   = FCLayer() 

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        
        out = self.head(x) # Returns 7 values
        return out

# 1. Setup
print("Setup")
dataset = WeatherDataset(tensor_names_2018, targets, metadata)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# device = torch.device("cpu")

model = WeatherCNN().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
mse_loss_fn = nn.MSELoss()
bce_loss_fn = nn.BCEWithLogitsLoss()

print("Begining training")

# 2. Loop
for epoch in range(10):
    print(f"\n{'='*60}")
    print(f"Starting epoch {epoch}")
    
    nan_batches = 0
    total_batches = 0
    running_loss_reg = 0.0
    running_loss_cls = 0.0

    for batch_idx, (batch_x, batch_y_reg, batch_y_cls) in enumerate(train_loader):
        batch_x     = batch_x.to(device)
        batch_y_reg = batch_y_reg.to(device)
        batch_y_cls = batch_y_cls.to(device)
        total_batches += 1

        # ── DEBUG: Check inputs on first batch of first epoch ──────────
        if epoch == 0 and batch_idx == 0:
            print(f"\n[DEBUG] batch_x      shape={batch_x.shape}  "
                  f"min={batch_x.min():.4f}  max={batch_x.max():.4f}  "
                  f"has_nan={batch_x.isnan().any().item()}  "
                  f"has_inf={batch_x.isinf().any().item()}")
            print(f"[DEBUG] batch_y_reg  shape={batch_y_reg.shape}  "
                  f"min={batch_y_reg.min():.4f}  max={batch_y_reg.max():.4f}  "
                  f"has_nan={batch_y_reg.isnan().any().item()}")
            print(f"[DEBUG] batch_y_cls  shape={batch_y_cls.shape}  "
                  f"unique={batch_y_cls.unique().tolist()}  "
                  f"has_nan={batch_y_cls.isnan().any().item()}")

        # ── Forward ────────────────────────────────────────────────────
        preds = model(batch_x)

        # ── DEBUG: Check predictions on first batch of first epoch ─────
        if epoch == 0 and batch_idx == 0:
            print(f"[DEBUG] preds        shape={preds.shape}  "
                  f"min={preds.min():.4f}  max={preds.max():.4f}  "
                  f"has_nan={preds.isnan().any().item()}")

        # ── Loss ───────────────────────────────────────────────────────
        loss_reg   = mse_loss_fn(preds[:, :6], batch_y_reg)
        loss_cls   = bce_loss_fn(preds[:, 6],  batch_y_cls.float())
        total_loss = loss_reg + loss_cls

        # ── DEBUG: Detect NaN/Inf loss ─────────────────────────────────
        loss_reg_val = loss_reg.item()
        loss_cls_val = loss_cls.item()
        total_val    = total_loss.item()

        is_nan = (
            loss_reg.isnan().item() or
            loss_cls.isnan().item() or
            total_loss.isnan().item()
        )

        if is_nan or batch_idx % 50 == 0:
            print(f"  [Batch {batch_idx:04d}]  "
                  f"loss_reg={loss_reg_val:.4f}  "
                  f"loss_cls={loss_cls_val:.4f}  "
                  f"total={total_val:.4f}")

        if is_nan:
            nan_batches += 1
            print(f"  *** NaN DETECTED at batch {batch_idx} ***")

            # Find which samples and channels contain NaN
            nan_samples = batch_x.isnan().any(dim=(1,2,3)).nonzero(as_tuple=True)[0]
            if len(nan_samples):
                print(f"      NaN input samples (indices): {nan_samples.tolist()}")
                for s in nan_samples[:3]:  # show at most 3
                    nan_chans = batch_x[s].isnan().any(dim=(1,2)).nonzero(as_tuple=True)[0]
                    print(f"      Sample {s.item()} — NaN channels: {nan_chans.tolist()}")

            # Check if gradients were already NaN before this step
            grad_nans = {
                name: p.grad.isnan().any().item()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            nan_grad_layers = [k for k, v in grad_nans.items() if v]
            if nan_grad_layers:
                print(f"      NaN gradients in: {nan_grad_layers}")

            # Print prediction stats to see where things went wrong
            print(f"      preds stats: min={preds.min():.4f}  "
                  f"max={preds.max():.4f}  "
                  f"has_nan={preds.isnan().any().item()}")
            if nan_batches >= 3:
                print("  Too many NaN batches — stopping early for diagnosis.")
                break

        # ── Backward ───────────────────────────────────────────────────
        optimizer.zero_grad()
        total_loss.backward()

        # ── DEBUG: Gradient norm (helps spot explosions) ───────────────
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        if batch_idx % 50 == 0 or is_nan:
            print(f"           grad_norm={total_grad_norm:.4f}")

        optimizer.step()

        if not is_nan:
            running_loss_reg += loss_reg_val
            running_loss_cls += loss_cls_val

    good_batches = total_batches - nan_batches
    print(f"\nEpoch {epoch} summary:")
    print(f"  Total batches : {total_batches}  |  NaN batches: {nan_batches}")
    if good_batches > 0:
        print(f"  Avg loss_reg  : {running_loss_reg / good_batches:.4f}")
        print(f"  Avg loss_cls  : {running_loss_cls / good_batches:.4f}")