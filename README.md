# Weather Forecasting Model: Architecture and Training Report

This report outlines the technical implementation of a multi-task Convolutional Neural Network (CNN) designed for 24-hour atmospheric forecasting using multi-year meteorological datasets (2018–2021).

---

## 1. Model Architecture

The model, `WeatherCNN`, uses a deep spatial feature extractor coupled with a multi-task prediction head.

### Feature Extraction (Convolutional Layers)
The backbone consists of sequential blocks designed to downsample spatial resolution while increasing feature depth:
* **Input Stage:** Accepts a 42-channel tensor representing various atmospheric variables (e.g., temperature, pressure, wind).
* **Layer 1:** `ConvLayer` (42 $\rightarrow$ 64 filters, $3 \times 3$ kernel) followed by **BatchNorm2d** and **ReLU**.
* **Pooling:** `MaxPool2d` ($2 \times 2$ kernel, stride 2) reduces the spatial dimensions.
* **Layer 2:** `ConvLayer` (64 $\rightarrow$ 128 filters, $3 \times 3$ kernel) followed by **BatchNorm2d** and **ReLU**.
* **Global Pooling:** A second `MaxPool2d` layer further abstracts the spatial features.

### Prediction Head (Fully Connected)
The model utilizes a **LazyLinear** layer in the `FCLayer` class. This allows the model to automatically infer the flattened input size from the convolutional backbone, outputting a 7-dimensional vector:
* **Indices 0–5:** Continuous variables (Regression).
* **Index 6:** Event probability (Classification).

---

## 2. Data Pipeline & Preprocessing

### NaN Sanitization
Meteorological data often contains missing values that can destabilize training. This pipeline employs a three-tier cleaning strategy:
1. **Scanning:** A utility identifies "dirty" files containing NaNs across 42 channels.
2. **Specific Logic:** Channel 5 (**DSWRF** at surface) contains legitimate NaNs during nighttime. These are explicitly imputed with `0.0` within the `WeatherDataset` class.
3. **Filtering:** Any file containing NaNs in non-exempt channels is excluded from the training indices.

### Normalization
To ensure stable gradient descent, both inputs and targets are scaled:
* **Inputs:** Per-channel Z-score normalization using pre-computed means and standard deviations from the training set.
* **Targets:** Regression targets are normalized. During training, the model predicts in the normalized space, while "real-unit" MSE and RMSE are tracked for interpretability.

---

## 3. Training Protocol

The model was trained on a single NVIDIA GPU using a multi-task loss approach.

| Component | Specification |
| :--- | :--- |
| **Optimizer** | AdamW |
| **Learning Rate** | $1 \times 10^{-5}$ |
| **Batch Size** | 128 |
| **Gradient Clipping** | $\text{max\_norm} = 1.0$ |
| **Lead Time** | 24 Hours |

### Loss Function
The training objective minimizes a combined loss:
$$L_{total} = L_{MSE}(\text{Regression}) + L_{BCEWithLogits}(\text{Classification})$$

---

## 4. Training Results (Logs Summary)

Training logs below combine work from two setups: the original **42-channel** report (Darius) and **Subhanga-6 feature CNN** training on the Tufts CS 137 cluster.

---

### Darius — 42-channel baseline

The model showed strong convergence within the first three epochs, particularly in the classification task.

| Epoch | Avg Reg Loss (Norm) | Avg Cls Loss |
| :--- | :--- | :--- |
| 0 | 1.1367 | 0.1628 |
| 1 | 0.6316 | 0.0801 |
| 2 | 0.4956 | 0.0589 |
| 3 | 0.6304 | 0.0496 |

**Analysis:**

* **Rapid convergence:** The model effectively learned the primary variance of the weather patterns within two epochs, cutting the real-unit RMSE by more than half.
* **Stability:** BatchNorm and AdamW prevented gradient spikes despite the high dynamic range of the input weather data.
* **Best model:** The lowest total loss ($0.5546$) was achieved at Epoch 2, after which the model was saved as the production checkpoint.

Checkpoint: `checkpoints/best_model.pt`.

---

### Subhanga-6 feature CNN (Tufts HPC / CS 137)

#### Cluster layout (homework)

| | Path |
| :--- | :--- |
| **Repo (on cluster)** | `/cluster/tufts/c26sp1cs0137/supadh03/SkyOracle/` |
| **Dataset** | `/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset` |
| **Checkpoints (general)** | `/cluster/tufts/c26sp1cs0137/supadh03/SkyOracle/checkpoints` |
| **This run (Subhanga)** | `/cluster/tufts/c26sp1cs0137/supadh03/SkyOracle/checkpoints/subhanga` |

#### Hardware and Python stack

* **GPU:** NVIDIA A100-PCIE-40GB, **CUDA 12.9**
* **PyTorch (module):** `module load pytorch/2.8.0-cuda12.9-cudnn9`
* **Environment:** `conda activate` base on the HPC; training driven from that stack instead of a one-off venv (avoids missing-package issues on compute nodes).

#### Iteration notes (representative job timeline)

Training started **~9:13 AM** targeting the A100 + CUDA 12.9 stack. Notable batch attempts:

1. **Job 386265** (~9:27 AM) — virtual environment not installed on the node.
2. **Node 386279** (~10:02 AM) — switched to **conda** after fixes; **job 386281** (~10:08 AM).
3. **Job 386303** (~10:37 AM).
4. **`torch.OutOfMemoryError`** on GPU (requested ~1.54 GiB; ~44 GiB already in use by PyTorch). **Node 386627**, **job 386631** — led to batch/memory tuning (PyTorch docs suggest `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when fragmentation is an issue).
5. **Timeout** on **node 387630**.
6. **Job 387680** — further retry.

Final successful log excerpt below matches a completed run (e.g. `subhanga/logs/train_388842.out`); best weights written to `checkpoints/subhanga/best.pt`.

#### Log summary — data split and setup

```
Python 3.11.13
torch 2.8.0+cu129
cuda available True
DATASET_DIR=/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset
CHECKPOINT_DIR=/cluster/tufts/c26sp1cs0137/supadh03/SkyOracle/checkpoints/subhanga
Device: cuda
Total time steps: 47952
Total usable inputs (t with t+24 in range): 47928
Train inputs: 21672  years=(2018, 2019, 2020)
Val inputs  : 8760  years=(2021,)
Targets shape: (47952, 6)
Binary label positives: 988/47952
Train inputs after NaN-target filter: 21144
Val inputs after NaN-target filter  : 8760
Loaded channel normalization stats from channel_stats.pt
Mixed precision (autocast): enabled, dtype=torch.bfloat16
```

#### Log summary — epochs (normalized RMSE / best MSE)

| Epoch | train_rmse (norm) | val_rmse (norm) | Best checkpoint (val MSE) |
| :--- | :--- | :--- | :--- |
| 0 | 0.7985 | 0.8810 | 0.776150 |
| 1 | 0.7631 | 0.8706 | 0.757902 |
| 2 | 0.7515 | 0.8697 | 0.756311 |
| 3 | 0.7409 | 0.8527 | 0.727078 |
| 4 | 0.7316 | 0.8559 | — |
| 5 | 0.7255 | 0.8477 | 0.718675 |
| 6 | 0.7195 | 0.8571 | — |
| 7 | 0.7135 | 0.8681 | — |
| 8 | 0.7096 | 0.8628 | — |
| 9 | 0.7024 | 0.8671 | — |

Training completed with **Done.** **Job finished: Thu Apr 2 02:58:42 PM EDT 2026.**

**Takeaways:** Mixed precision (**bfloat16** autocast), filtered NaN targets (21144 train / 8760 val), and channel stats from `channel_stats.pt`. Best validation MSE in this run was **0.718675** at epoch 5 (`checkpoints/subhanga/best.pt`).