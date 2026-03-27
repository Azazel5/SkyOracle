import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset


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

import torch, os
from collections import defaultdict
from tqdm import tqdm

def scan_dataset_for_nans(file_names, dataset_dir, max_files=None):
    """
    Scans every input tensor and reports:
      - Which files contain NaNs
      - Which channels are dirty (nan count per channel across all files)
      - Which channels are dirty in targets
    """
    channel_nan_counts = defaultdict(int)   # channel_idx -> total nan pixels
    channel_nan_files  = defaultdict(set)   # channel_idx -> set of filenames
    dirty_files = []

    files_to_scan = file_names[:max_files] if max_files else file_names

    for fname in tqdm(files_to_scan, desc="Scanning"):
        year = fname.split('_')[1][:4]
        path = f"{dataset_dir}/inputs/{year}/{fname}"
        try:
            x = torch.load(path, weights_only=True).float()  # (H, W, 42)
        except Exception as e:
            print(f"  LOAD ERROR: {fname}: {e}")
            continue

        # Check per channel — x is (H, W, C) before permute
        for c in range(x.shape[2]):
            n_nan = x[:, :, c].isnan().sum().item()
            if n_nan > 0:
                channel_nan_counts[c] += n_nan
                channel_nan_files[c].add(fname)

        if x.isnan().any():
            dirty_files.append(fname)

    print(f"\n{'='*60}")
    print(f"Dirty files: {len(dirty_files)} / {len(files_to_scan)}")
    print(f"\nChannels with NaNs (sorted by total nan count):")
    for c, count in sorted(channel_nan_counts.items(), key=lambda x: -x[1]):
        print(f"  Channel {c:2d}: {count:>10,} NaN pixels across {len(channel_nan_files[c])} files")

    return channel_nan_counts, channel_nan_files, dirty_files

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

files = tensor_names_2018 + tensor_names_2019 + tensor_names_2020 + tensor_names_2021

import os

def delete_by_indices(lst: list, bad_indices: list) -> list:
    bad_set = set(bad_indices)
    return [val for i, val in enumerate(lst) if i not in bad_set]

# def filter_nan_files(file_names, dataset_dir, skip_channel=5, start_value=9000):
#     """
#     Scans all files and returns a cleaned list with NaN-containing files removed.
#     Channel `skip_channel` is ignored (e.g. DSWRF which is legitimately NaN at night).
#     """
#     bad_indices_x = [] # These are indeces of files from the training dataset
#     bad_indices_y = [] # These are indices from targets, basically 24 hours later, that must be removed

#     for i, fname in enumerate(file_names):
#         if i < start_value:
#             continue
#         year = fname.split('_')[1][:4]
#         path = f"{dataset_dir}/inputs/{year}/{fname}"

#         x = torch.load(path, weights_only=True).float()

#         # Zero out the exempt channel before checking
#         x_check = x.clone()
#         x_check[..., skip_channel] = 0.0  # shape is (H, W, 42) before permute

#         nan_channels = x_check.isnan().any(dim=0).any(dim=0)  # (42,)
#         if nan_channels.any():
#             bad_channels = nan_channels.nonzero(as_tuple=True)[0].tolist()
#             print(f"[{i+1}/{len(file_names)}] BAD  {fname}  ->  channels={bad_channels}")
#             if i + 24 < len(file_names):
#                 bad_indices_x.append(i)
#                 bad_indices_y.append(i+24)
#         if i % 1000 == 0:
#             print(i)
 
#     return bad_indices_x, bad_indices_y

# bad_indices_x, bad_indices_y = filter_nan_files(files, DATASET_DIR)

# # Save to disk
# with open("bad_indices_x.txt", "w") as f:
#     f.write("\n".join(str(i) for i in bad_indices_x))

# with open("bad_indices_y.txt", "w") as f:
#     f.write("\n".join(str(i) for i in bad_indices_y))

# assert(len(bad_indices_x) == len(bad_indices_x))
with open("bad_indices_x.txt") as f:
    bad_indices_x = [int(line.strip()) for line in f]

with open("bad_indices_y.txt") as f:
    bad_indices_y = [int(line.strip()) for line in f]
file_names = delete_by_indices(files, bad_indices_x)
y_reg = delete_by_indices(targets['values'], bad_indices_y)
y_cls = delete_by_indices(targets['binary_label'], bad_indices_y)




class WeatherDataset(Dataset):
    def __init__(self, file_names, metadata, y_reg_input, y_cls_input, lead_time=24):
        self.file_names = file_names
        self.target_values = y_reg_input       # The 6 regression vars
        self.target_labels = y_cls_input # The 1 rain label
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

            # # NaN inspection
            # nan_channels = x.isnan().any(dim=-1).any(dim=-1)  # shape (42,)
            # if nan_channels.any():
            #     print(f"[NaN] file={fname}")
            #     for ch in nan_channels.nonzero(as_tuple=True)[0].tolist():
            #         n = x[ch].isnan().sum().item()
            #         print(f"  channel {ch:02d}: {n} NaNs  "
            #             f"(min={x[ch].nanmean():.4f}, mean={torch.nanmean(x[ch]):.4f})")

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
# y indices still null
valid_indices = [
    i for i in range(len(file_names) - 24)
    if (not y_reg[i + 24].isnan().any() and not y_cls[i + 24].isnan().any())
]
print(f"Valid samples: {len(valid_indices)} / {len(file_names) - 24}")

dataset = Subset(WeatherDataset(file_names, metadata, y_reg, y_cls), valid_indices)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = WeatherCNN().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
mse_loss_fn = nn.MSELoss()
bce_loss_fn = nn.BCEWithLogitsLoss()

print("Begining training")

import os

# 2. Loop
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)
best_loss = float('inf')


for epoch in range(20):
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

        assert not batch_x.isnan().any(), "NaN survived into training!"
        assert not batch_y_reg.isnan().any(), "NaN survived into training!"
        assert not batch_y_cls.isnan().any(), "NaN survived into training!"

        # y_mean = batch_y_reg.mean(dim=0)
        # y_std  = batch_y_reg.std(dim=0)

        # Normalize targets
        # batch_y_reg_norm = (batch_y_reg - y_mean) / (y_std + 1e-8)

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

        preds = model(batch_x)

        if epoch == 0 and batch_idx == 0:
            print(f"[DEBUG] preds        shape={preds.shape}  "
                  f"min={preds.min():.4f}  max={preds.max():.4f}  "
                  f"has_nan={preds.isnan().any().item()}")

        loss_reg   = mse_loss_fn(preds[:, :6], batch_y_reg)
        loss_cls   = bce_loss_fn(preds[:, 6],  batch_y_cls.float())
        total_loss = loss_reg + loss_cls

        loss_reg_val = loss_reg.item()
        loss_cls_val = loss_cls.item()
        total_val    = total_loss.item()

        optimizer.zero_grad()
        total_loss.backward()
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f"  [Batch {batch_idx:04d}]  "
                  f"loss_reg={loss_reg_val:.4f}  "
                  f"loss_cls={loss_cls_val:.4f}  "
                  f"total={total_val:.4f}")
            print(f"           grad_norm={total_grad_norm:.4f}")

        

        running_loss_reg += loss_reg_val
        running_loss_cls += loss_cls_val

    good_batches = total_batches - nan_batches
    avg_loss_reg = running_loss_reg / good_batches if good_batches > 0 else float('nan')
    avg_loss_cls = running_loss_cls / good_batches if good_batches > 0 else float('nan')
    avg_total    = avg_loss_reg + avg_loss_cls

    print(f"\nEpoch {epoch} summary:")
    print(f"  Total batches : {total_batches}  |  NaN batches: {nan_batches}")
    if good_batches > 0:
        print(f"  Avg loss_reg  : {avg_loss_reg:.4f}")
        print(f"  Avg loss_cls  : {avg_loss_cls:.4f}")

       # ── Save best model separately ─────────────────────────────────────
    if good_batches > 0 and avg_total < best_loss:
        best_loss = avg_total
        best_path = os.path.join(save_dir, "best_model.pt")
        torch.save(model.state_dict(), best_path)
        print(f"  *** New best model saved → {best_path}  (loss={best_loss:.4f}) ***")