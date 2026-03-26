# WeatherCNN: Model Architecture Document

## 1. Overview

**WeatherCNN** is a multi-task deep learning model designed for short-term (24h) meteorological forecasting. It processes high-resolution geospatial grids to simultaneously perform multi-variable regression (e.g., temperature, pressure) and binary classification (precipitation detection).

The architecture follows a modular Convolutional Neural Network (CNN) design, prioritizing spatial feature extraction through stacked convolutional blocks and downsampling layers.

---

## 2. Structural Components

### A. Convolutional Block (`ConvLayer`)

Each convolutional block is designed to extract local spatial patterns while maintaining the feature map resolution.

* **Convolution:** 3x3 kernel, stride 1, padding 1.
* **Normalization:** `BatchNorm2d` is applied to stabilize the distribution of activations, allowing for faster convergence and higher learning rates.
* **Activation:** `ReLU` (In-place) introduces non-linearity.

### B. Pooling Block (`MaxPoolLayer`)

* **Mechanism:** 2x2 Max Pooling with a stride of 2.
* **Purpose:** Reduces spatial dimensions by 50% per axis, increasing the "receptive field" of subsequent layers and reducing the total parameter count for the dense head.

### C. Dense Head (`FCLayer`)

* **Flattening:** Converts the final 3D feature maps into a 1D feature vector.
* **Inference:** Utilizes `nn.LazyLinear`, which automatically calculates the input features based on the preceding layer's output shape.
* **Output:** A linear projection to 7 dimensions.

---

## 3. Layer-by-Layer Data Flow

| Layer | Type | Input Channels | Output Channels | Output Spatial Dim |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Tensor | 42 | - | 450 x 449 |
| **Layer 1** | ConvBlock | 42 | 64 | 450 x 449 |
| **Pool 1** | MaxPool | 64 | 64 | 225 x 224 |
| **Layer 2** | ConvBlock | 64 | 128 | 225 x 224 |
| **Pool 2** | MaxPool | 128 | 128 | 112 x 112 |
| **Head** | Linear | 1,605,632* | 7 | 1 x 7 |

*\*Calculated as $128 \times 112 \times 112$*

---

## 4. Multi-Task Training Strategy

The model produces a single output vector $\hat{y} \in \mathbb{R}^7$, which is split for loss calculation:

1. **Regression Task ($\hat{y}_{0:5}$):** Predicts 6 continuous atmospheric variables.
    * **Loss Function:** `MSELoss` (Mean Squared Error).
2. **Classification Task ($\hat{y}_{6}$):** Predicts the likelihood of rain.
    * **Loss Function:** `BCEWithLogitsLoss` (Binary Cross Entropy with built-in Sigmoid).

### Data Preprocessing

* **Lead Time:** The targets are offset by 24 hours relative to the input, creating a 24-hour predictive horizon.
* **Channel 5 Handling:** Shortwave radiation (DSWRF) values containing `NaN` (due to nighttime) are numerically imputed with `0.0`.
* **Optimization:** The model uses `AdamW` optimizer with a learning rate of $1e-3$ and gradient norm clipping to prevent numerical instability.

---

## 5. Summary of Hyperparameters

* **Input Channels:** 42
* **Base Filter Count:** 64
* **Kernel Size:** 3x3
* **Optimizer:** AdamW
* **Batch Size:** 32
