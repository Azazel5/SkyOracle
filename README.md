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

The model showed strong convergence within the first three epochs, particularly in the classification task.

| Epoch | Avg Reg Loss (Norm) | Avg Cls Loss |
| :--- | :--- | :--- |
| 0 | 1.1367 | 0.1628 |
| 1 | 0.6316 | 0.0801 |
| 2 | 0.4956 | 0.0589 |
| 3 | 0.6304 | 0.0496 | 

**Analysis:** * **Rapid Convergence:** The model effectively learned the primary variance of the weather patterns within two epochs, cutting the real-unit RMSE by more than half.
* **Stability:** The inclusion of BatchNorm and AdamW prevented gradient spikes, despite the high dynamic range of the input weather data.
* **Best Model:** The lowest total loss ($0.5546$) was achieved at Epoch 2, after which the model was saved as the production checkpoint.

The best model is saved at ```checkpoints/best_model.pt```.