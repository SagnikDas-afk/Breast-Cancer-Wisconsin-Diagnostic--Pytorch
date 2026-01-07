# Breast Cancer Wisconsin Diagnostic – PyTorch

This project implements a **Neural Network using PyTorch** to classify breast cancer tumors as **Malignant (M)** or **Benign (B)** using the **Breast Cancer Wisconsin (Diagnostic) Dataset** from the UCI Machine Learning Repository.

#### Offical URL: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
---

<img width="388" height="280" alt="image" src="https://github.com/user-attachments/assets/b357b495-090a-4a54-bdbf-d342a01375f1" />

##### Above pic. is representation of how the neural network looks.

## 📊 Dataset

- **Source:** UCI Machine Learning Repository  
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic)
- **Total Samples:** 569
- **Features:** 30 numerical features computed from digitized images of fine needle aspirates (FNA) of breast masses
- **Target Variable:** `diagnosis`
  - `M` → Malignant (encoded as `0`)
  - `B` → Benign (encoded as `1`)

---
<img width="720" height="386" alt="image" src="https://github.com/user-attachments/assets/e7117d01-f6df-4ce5-aa28-223df58e8482" />



## 🧠 Model Architecture

A **fully connected feed-forward neural network** is implemented using PyTorch:

- Input Layer: 30 features
- Hidden Layer 1: ReLU activation
- Hidden Layer 2: ReLU activation
- Output Layer: 2 neurons (binary classification)
- Loss Function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Learning Rate: `0.01`
- Epochs: `100`

---

## 🏗️ Workflow

1. Import required libraries
2. Load and preprocess the dataset
3. Encode target labels (`M` → 0, `B` → 1)
4. Split data into training and testing sets
5. Convert data into PyTorch tensors
6. Define neural network model
7. Train the model for 100 epochs
8. Track and visualize training loss

---

## 📈 Training Visualization

The training loss is plotted against epochs to visually analyze model learning and convergence.
