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
        # Need to make sure these keys exist!!!!!!!!!!!!!!!
        self.target_values = targets_dict['values']       # The 5 regression vars
        self.target_labels = targets_dict['binary_label'] # The 1 rain label
        self.lead_time = lead_time
        
        # Crop parameters (optional but recommended)
        self.iy = metadata['jumbo_y_idx']
        self.ix = metadata['jumbo_x_idx']

    def __len__(self):
        # We stop 24 hours early because we don't have "future" targets for the last day
        return len(self.file_names) - self.lead_time

    def __getitem__(self, idx):
        # 1. Load Input (Current Time)
        fname = self.file_names[idx]
        # Extract year from filename to find the right folder
        year = fname.split('_')[1][:4] 
        path = f"{DATASET_DIR}/inputs/{year}/{fname}"
        
        x = torch.load(path, weights_only=True).float() # (450, 449, 42)
        x = x.permute(2, 0, 1) # PyTorch expects (Channels, H, W)
        
        # 2. Load Target (Current Time + 24 Hours)
        target_idx = idx + self.lead_time
        y_reg = self.target_values[target_idx]
        y_cls = self.target_labels[target_idx]
        
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
    print("Starting epoch", epoch)
    for batch_x, batch_y_reg, batch_y_cls in train_loader:
        batch_x = batch_x.to(device)
        batch_y_reg = batch_y_reg.to(device)
        batch_y_cls = batch_y_cls.to(device)
        
        # Forward
        preds = model(batch_x)
        
        # Calculate Loss
        loss_reg = mse_loss_fn(preds[:, :6], batch_y_reg)
        loss_cls = bce_loss_fn(preds[:, 6], batch_y_cls.float())
        total_loss = loss_reg + loss_cls
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch} complete. Loss: {total_loss.item()}")