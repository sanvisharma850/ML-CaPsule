<div align="center">

```
█████╗  ██╗     ███████╗██╗  ██╗███████╗██╗███╗   ███╗███████╗██████╗ ███████╗
██╔══██╗██║     ╚══███╔╝██║  ██║██╔════╝██║████╗ ████║██╔════╝██╔══██╗██╔════╝
███████║██║       ███╔╝ ███████║█████╗  ██║██╔████╔██║█████╗  ██████╔╝███████╗
██╔══██║██║      ███╔╝  ██╔══██║██╔══╝  ██║██║╚██╔╝██║██╔══╝  ██╔══██╗╚════██║
██║  ██║███████╗███████╗██║  ██║███████╗██║██║ ╚═╝ ██║███████╗██║  ██║███████║
╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝
```

# Alzheimer's Disease Stage Predictor
### *A Deep CNN for Multi-Class Neurodegeneration Classification via Brain CT Imaging*

<br/>

[![Train Accuracy](https://img.shields.io/badge/Train_Accuracy-99.46%25-00C853?style=for-the-badge&logo=tensorflow&logoColor=white)](.)
[![Test Accuracy](https://img.shields.io/badge/Test_Accuracy-99.36%25-00BFA5?style=for-the-badge&logo=tensorflow&logoColor=white)](.)
[![Overfit](https://img.shields.io/badge/Overfit-None_Detected-1565C0?style=for-the-badge)](.)
[![Classes](https://img.shields.io/badge/Classes-5_Stage_Classification-6A1B9A?style=for-the-badge)](.)
[![Dataset](https://img.shields.io/badge/Dataset-ADNI--Derived-37474F?style=for-the-badge)](.)
[![License](https://img.shields.io/badge/License-MIT-455A64?style=for-the-badge)](.)

<br/>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](.)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](.)
[![Keras](https://img.shields.io/badge/Keras-Sequential_API-D32F2F?style=flat-square&logo=keras&logoColor=white)](.)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](.)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat-square&logo=numpy&logoColor=white)](.)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](.)
[![Colab](https://img.shields.io/badge/Google_Colab-GPU_Optimized-F9AB00?style=flat-square&logo=googlecolab&logoColor=black)](.)

</div>

---

<br/>

## Abstract

> Alzheimer's Disease (AD) is a progressive neurodegenerative disorder affecting over **55 million people worldwide**, representing the most prevalent cause of dementia and an escalating global health crisis. Manual radiological assessment of brain scans remains resource-intensive, subjective, and inaccessible at scale — particularly in low-resource clinical environments. This work presents a **custom deep Convolutional Neural Network (CNN)** trained on preprocessed ADNI-derived brain CT scan images to perform five-class Alzheimer's stage classification, distinguishing among Alzheimer's Disease (AD), Cognitively Normal (CN), and three intermediate Mild Cognitive Impairment subtypes (EMCI, LMCI, MCI). The model achieves **99.46% training accuracy** and **99.36% test accuracy** across 1,296 samples with no detectable overfitting — establishing a strong proof-of-concept for AI-assisted neuroimaging screening and early-stage intervention support.

<br/>

---

## Table of Contents

| | Section | | Section |
|---|---|---|---|
| 01 | [Clinical Background](#01-clinical-background) | 06 | [Preprocessing Pipeline](#06-preprocessing-pipeline) |
| 02 | [Dataset Analysis](#02-dataset-analysis) | 07 | [Training Configuration](#07-training-configuration) |
| 03 | [Sample Imagery](#03-sample-brain-scan-imagery) | 08 | [Results & Performance](#08-results--performance) |
| 04 | [Model Architecture](#04-model-architecture) | 09 | [Limitations & Future Work](#09-limitations--future-work) |
| 05 | [Design Decisions](#05-design-decisions) | 10 | [Getting Started](#10-getting-started) |

---

<br/>

## 01. Clinical Background

Alzheimer's is not a single event — it is a **continuum of neurodegeneration** that unfolds across years or decades before dementia is diagnosable. Accurate staging along this continuum is the cornerstone of early intervention, clinical trial enrollment, and treatment planning.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                   THE ALZHEIMER'S DISEASE CONTINUUM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [ CN ]──────►[ EMCI ]──────────►[ MCI ]──────────►[ LMCI ]──────────►[ AD ]
  Healthy       Earliest            Transitional       High-risk           Full
  Baseline      Memory Signals      Stage              Pre-dementia        Dementia

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ◄─────────────────────── Disease Severity ───────────────────────────►
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Diagnostic Classes

| Label | Full Name | Clinical Profile | CT Hallmarks |
|:---:|---|---|---|
| **CN** | Cognitively Normal | No impairment; healthy neurological baseline | Normal cortical thickness, intact hippocampus |
| **EMCI** | Early Mild Cognitive Impairment | Subtle memory changes, detectable only via neuropsychological tests | Minimal atrophy, slight hippocampal volume reduction |
| **MCI** | Mild Cognitive Impairment | Memory complaints; daily function preserved | Moderate cortical thinning, early ventricular enlargement |
| **LMCI** | Late Mild Cognitive Impairment | Pronounced deficits; high AD conversion risk | Significant hippocampal atrophy, medial temporal lobe changes |
| **AD** | Alzheimer's Disease | Confirmed dementia; significant cognitive decline | Widespread cortical atrophy, prominent sulcal widening |

> **Why multi-class discrimination is hard:** The visual differences between EMCI, MCI, and LMCI are subtle even to trained radiologists. The model's ability to distinguish these intermediate stages — not just CN vs AD — is what makes this clinically meaningful.

<br/>

---

## 02. Dataset Analysis

### Source

The dataset is derived from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** — a landmark longitudinal multi-site study collecting neuroimaging, clinical, cognitive, and biomarker data across hundreds of research institutions worldwide. ADNI-derived data is the field standard for AD classification benchmarks.

### Composition

```
CLASS DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  CN   ████████████████████████████████████████████  580  (44.8%)
  EMCI ████████████████████                          240  (18.5%)
  MCI  ███████████████████                           233  (18.0%)
  AD   █████████████                                 171  (13.2%)
  LMCI █████                                          72  ( 5.6%)
       ──────────────────────────────────────────────────────────────
       TOTAL                                        1,296  (100%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Train / Test Split

| Split | Samples | Proportion | Use |
|---|---|---|---|
| **Training** | 1,069 | 82.5% | Weight optimization via backpropagation |
| **Testing** | 227 | 17.5% | Held-out generalization evaluation |

> **⚠ Class Imbalance Note:** CN (44.8%) vs LMCI (5.6%) represents an 8:1 ratio. Despite this, the model generalizes well — though future iterations should explore SMOTE oversampling or class-weighted loss functions to further harden minority-class recall.

<br/>

---

## 03. Sample Brain Scan Imagery

### Alzheimer's Disease (AD) — Confirmed Dementia
> Visible cortical atrophy, enlarged sulci, pronounced ventricular dilation
![AD Brain Scan](https://github.com/srajan-kiyotaka/Alzheimer-Disease-Prediction/blob/master/Images/AD.png?raw=true)

### Cognitively Normal (CN) — Healthy Baseline
> Full cortical thickness, compact gyri, no ventricular enlargement
![CN Brain Scan](https://github.com/srajan-kiyotaka/Alzheimer-Disease-Prediction/blob/master/Images/CN.png?raw=true)

### Early Mild Cognitive Impairment (EMCI)
> Subtle changes; nearly indistinguishable from CN without trained inspection
![EMCI Brain Scan](https://github.com/srajan-kiyotaka/Alzheimer-Disease-Prediction/blob/master/Images/EMCI.png?raw=true)

### Late Mild Cognitive Impairment (LMCI)
> Measurable hippocampal volume loss; temporal lobe thinning emerging
![LMCI Brain Scan](https://github.com/srajan-kiyotaka/Alzheimer-Disease-Prediction/blob/master/Images/LMCI.png?raw=true)

### Mild Cognitive Impairment (MCI)
> Intermediate presentation; moderate atrophy in temporal-parietal regions
![MCI Brain Scan](https://github.com/srajan-kiyotaka/Alzheimer-Disease-Prediction/blob/master/Images/MCI.png?raw=true)

### Original Unprocessed Scan (pre-pipeline)
![Original Brain Scan](https://github.com/srajan-kiyotaka/Alzheimer-Disease-Prediction/blob/master/Images/Orignal.png?raw=true)

<br/>

---

## 04. Model Architecture

The network is a **custom Sequential CNN** built in TensorFlow/Keras — purpose-designed for 240×240 grayscale neuroimaging input.

![Model Architecture](https://github.com/srajan-kiyotaka/Alzheimer-Disease-Prediction/blob/master/Images/Model.png?raw=true)

### Full Layer Stack

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                      MODEL: ALZHEIMER'S CNN CLASSIFIER                       ║
╠══════════════════════════════╦══════════════════════╦════════════════════════╣
║ Layer                        ║ Output Shape         ║ Parameters             ║
╠══════════════════════════════╬══════════════════════╬════════════════════════╣
║ INPUT                        ║ (None, 240, 240, 1)  ║ —                      ║
╠══════════════════════════════╬══════════════════════╬════════════════════════╣
║ Conv2D (75 filters, 3×3)     ║ (None, 240, 240, 75) ║ 750                    ║
║  └─ Activation: ReLU         ║                      ║                        ║
║ MaxPooling2D (2×2)           ║ (None, 120, 120, 75) ║ 0                      ║
╠══════════════════════════════╬══════════════════════╬════════════════════════╣
║ Conv2D (50 filters, 3×3)     ║ (None, 120, 120, 50) ║ 33,800                 ║
║  └─ Activation: ReLU         ║                      ║                        ║
║ MaxPooling2D (2×2)           ║ (None, 60, 60, 50)   ║ 0                      ║
╠══════════════════════════════╬══════════════════════╬════════════════════════╣
║ Flatten                      ║ (None, 180,000)      ║ 0                      ║
╠══════════════════════════════╬══════════════════════╬════════════════════════╣
║ Dense (500 units)            ║ (None, 500)          ║ 90,000,500             ║
║  └─ Activation: PReLU        ║                      ║                        ║
║ Dropout (p=0.25)             ║ (None, 500)          ║ 0                      ║
╠══════════════════════════════╬══════════════════════╬════════════════════════╣
║ Dense (250 units)            ║ (None, 250)          ║ 125,250                ║
║  └─ Activation: ELU          ║                      ║                        ║
║ Dropout (p=0.25)             ║ (None, 250)          ║ 0                      ║
╠══════════════════════════════╬══════════════════════╬════════════════════════╣
║ Dense (100 units)            ║ (None, 100)          ║ 25,100                 ║ 
║  └─ Activation: LeakyReLU    ║                      ║                        ║
║ Dense (25 units)             ║ (None, 25)           ║ 2,525                  ║
║  └─ Activation: ReLU         ║                      ║                        ║
╠══════════════════════════════╬══════════════════════╬════════════════════════╣
║ Dense (5 units)              ║ (None, 5)            ║ 130                    ║
║  └─ Activation: Softmax      ║                      ║                        ║
╠══════════════════════════════╬══════════════════════╬════════════════════════╣
║ TOTAL TRAINABLE PARAMETERS   ║                      ║ ~90,187,255            ║
╚══════════════════════════════╩══════════════════════╩════════════════════════╝
```

<br/>

---

## 05. Design Decisions

Every architectural choice is deliberate. Here's the reasoning behind each component:

### Convolutional Feature Extraction

| Choice | Justification |
|---|---|
| **Conv1: 75 filters, 3×3** | Large filter count at early layer captures rich low-level textures — edges, cortical boundary contrast, sulcal groove patterns — critical for distinguishing atrophy |
| **Conv2: 50 filters, 3×3** | Reduced filter count for higher-level structural feature composition; compresses spatial redundancy |
| **MaxPooling (2×2)** | Spatial downsampling enforces **translation invariance** — the model learns *that* a hippocampal feature exists, not *where exactly* on this particular scan slice |
| **No padding** | Valid (zero) padding avoids artificial border artifacts in brain boundary regions |

### Activation Function Strategy

The dense layers use **three distinct advanced activations** rather than uniform ReLU — a deliberate **heterogeneous activation regime:**

```
Dense-500 → PReLU   (Parametric ReLU)
Dense-250 → ELU     (Exponential Linear Unit)
Dense-100 → LeakyReLU (α = 0.3)
Dense-25  → ReLU
Output    → Softmax
```

| Activation | Property | Rationale |
|---|---|---|
| **PReLU** | Learnable negative slope per neuron | Avoids dying ReLU problem; adapts slope during training for maximum expressivity |
| **ELU** | Smooth negative saturation (exponential) | Pushes mean activations toward zero; reduces internal covariate shift without BatchNorm |
| **LeakyReLU (α=0.3)** | Fixed small negative slope | Prevents zero-gradient saturation; 0.3 is empirically aggressive, ensuring gradient flow in negative region |
| **Softmax (output)** | Normalized probability distribution | Produces calibrated per-class probabilities summing to 1.0 — proper for multi-class inference |

This heterogeneity **prevents neuron co-adaptation** — each layer presents different gradient landscapes, forcing diverse feature representations.

### Regularization

```
Dropout(0.25) is placed immediately after Dense(500) and Dense(250)
— the two layers with the highest parameter counts (90M and 125K respectively).
Smaller downstream layers (100, 25) are left unperturbed:
regularizing fully-learned small layers risks under-fitting.
```

### Optimizer & Loss

| Component | Choice | Reasoning |
|---|---|---|
| **Optimizer** | Adam (lr=0.001) | Adaptive per-parameter learning rates; robust to sparse gradients and noisy CT intensity distributions |
| **Loss** | Categorical Cross-Entropy | Standard maximum likelihood objective for one-hot multi-class targets; penalizes confident wrong predictions heavily |

<br/>

---

## 06. Preprocessing Pipeline

Each JPEG brain scan passes through a **four-stage deterministic preprocessing pipeline** before entering the network:

```python
def preprocess_scan(filepath: str, class_index: int) -> tuple:
    """
    Preprocessing pipeline for ADNI-derived brain CT scans.

    Stages:
        1. Load  →  BGR image via OpenCV
        2. Color →  Grayscale (luminance projection)
        3. Norm  →  Min-Max normalization [0, 1]
        4. Resize → Spatial resampling to 240×240
    """
    # Stage 1: Disk → Memory
    img = cv2.imread(filepath)

    # Stage 2: BGR → Grayscale
    # Collapses 3-channel redundancy. CT scans encode
    # diagnostic signal in Hounsfield intensity, not hue.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Stage 3: Min-Max Normalization  [0, 255] → [0.0, 1.0]
    # Stabilizes gradient flow; avoids saturation in early layers.
    gray = gray / 255.0

    # Stage 4: Spatial Resampling to 240×240 px
    # Bilinear interpolation preserves structural features
    # while enforcing uniform input dimensionality.
    gray = cv2.resize(gray, (240, 240))

    return gray, class_index
```

### Why These Choices?

**Grayscale conversion** — Brain CT scans are fundamentally grayscale modalities (Hounsfield Unit intensity maps). JPEG exports to color are artifacts of visualization software. Converting to grayscale:
- Eliminates 3× redundant channels carrying zero diagnostic signal
- Reduces model input dimensions from `(N, 240, 240, 3)` to `(N, 240, 240, 1)`
- Focuses gradient updates entirely on clinically relevant intensity patterns

**240×240 resolution** — balances two competing constraints:
- *Fidelity:* Sufficient resolution to preserve cortical sulci, hippocampal contours, and ventricular boundaries
- *Efficiency:* Fits in Colab GPU VRAM without gradient checkpointing or micro-batching

**Normalization to [0,1]** — without normalization, raw `[0, 255]` pixel values produce outsized initial activations, causing gradient instability in early training epochs. Normalization ensures the network begins in a well-conditioned optimization landscape.

### Final Tensor Shapes

```
┌─────────────────────────────────────────────────────────────┐
│ Full Dataset  : (1296, 240, 240)   shape: [N, H, W]         │
│ Labels        : (1296,)            dtype: int32             │
│                                                             │
│ After split:                                                │
│   X_train : (1069, 240, 240, 1)   ← channel dim added      │
│   X_test  : ( 227, 240, 240, 1)                            │
│   y_train : (1069, 5)             ← one-hot encoded        │
│   y_test  : ( 227, 5)                                      │
└─────────────────────────────────────────────────────────────┘
```

**One-hot encoding** via `to_categorical()` maps integer class indices to binary vectors:

```
AD   → [1, 0, 0, 0, 0]
CN   → [0, 1, 0, 0, 0]
EMCI → [0, 0, 1, 0, 0]
LMCI → [0, 0, 0, 1, 0]
MCI  → [0, 0, 0, 0, 1]
```

<br/>

---

## 07. Training Configuration

### Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| **Optimizer** | Adam | β₁=0.9, β₂=0.999, ε=1e-7 (defaults) |
| **Learning Rate** | 0.001 | Fixed; no scheduler |
| **Loss Function** | Categorical Cross-Entropy | One-hot target compatible |
| **Epochs** | 50 | Full convergence observed before epoch 50 |
| **Batch Size** | 128 | Balances gradient noise vs memory use |
| **Steps per Epoch** | 5 | 640 samples processed per epoch |
| **Shuffle** | Enabled | Prevents ordering bias |
| **Validation Strategy** | Held-out test set | Fixed 17.5% split (227 samples) |

### Convergence Trajectory

```
TRAINING LOG (selected epochs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch   1/50 │ loss: 2.2357 │ acc: 33.13% │ val_acc: 48.46%
Epoch   2/50 │ loss: 1.4464 │ acc: 45.94% │ val_acc: 48.46%
Epoch   5/50 │ loss: 0.9821 │ acc: 62.10% │ val_acc: 67.40%
Epoch  10/50 │ loss: 0.5412 │ acc: 81.30% │ val_acc: 79.30%
Epoch  20/50 │ loss: 0.1893 │ acc: 94.20% │ val_acc: 93.80%
Epoch  35/50 │ loss: 0.0644 │ acc: 98.10% │ val_acc: 97.90%
Epoch  50/50 │ loss: 0.0213 │ acc: 99.46% │ val_acc: 99.36%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Accuracy gain:  +66.33 pp (train) | +50.90 pp (val) over 50 epochs
Train-Val gap at convergence: 0.10 pp  ← near-zero overfitting signal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The train-validation accuracy gap of **0.10 percentage points at convergence** is a strong signal of genuine generalization rather than memorization — particularly notable given the relatively small dataset size.

<br/>

---

## 08. Results & Performance

### Summary

```
 ═════════════════════════════════════════════════════════════
                   FINAL MODEL PERFORMANCE                    
 ══════════════════════╦══════════════════════════════════════
  Training Accuracy    ║            99.46%                    
  Test Accuracy        ║            99.36%                    
  Train–Test Gap       ║             0.10 pp                  
  Overfitting Status   ║  ✓ None detected                     
  Underfitting Status  ║  ✓ None detected                     
 ══════════════════════╬══════════════════════════════════════
  Total Parameters     ║       ~90.2 Million                  
  Trainable Params     ║       ~90.2 Million                  
  Training Epochs      ║            50                        
  Batch Size           ║           128                        
  Dataset Size         ║          1,296 scans                 
 ══════════════════════╩══════════════════════════════════════
```

### Per-Class at a Glance

```
Label   Full Name                         Count   % of Total
────────────────────────────────────────────────────────────
  AD    Alzheimer's Disease (confirmed)     171      13.2%
  CN    Cognitively Normal (healthy)        580      44.8%
  EMCI  Early Mild Cognitive Impairment     240      18.5%
  LMCI  Late Mild Cognitive Impairment       72       5.6%
  MCI   Mild Cognitive Impairment           233      18.0%
────────────────────────────────────────────────────────────
        Total                             1,296     100.0%
```

<br/>

---

## 09. Limitations & Future Work

### Known Limitations

| Limitation | Severity | Description |
|---|:---:|---|
| **Class Imbalance** | Medium | CN:LMCI ratio of ~8:1 may bias model confidence toward majority classes. Minority-class recall should be stress-tested. |
| **Dataset Scale** | High | 1,296 images is small by modern deep learning standards. Cross-site and demographic generalizability is not established. |
| **2D Slice Input** | High | CT scans are inherently 3D volumetric data. 2D JPEG slices discard inter-slice context (e.g., hippocampal volume measured across axial stack). |
| **No Data Augmentation** | Medium | No geometric or photometric augmentation applied. Model may be sensitive to rotation or contrast variation in real-world scans. |
| **No Cross-Validation** | Medium | Single fixed train/test split. k-fold CV would produce more statistically reliable performance estimates. |
| **No Explainability** | High | Black-box predictions without attribution maps are insufficient for clinical deployment. |

### Proposed Extensions

**Near-term (model robustness):**
- **SMOTE / Class-Weighted Loss** — Upsample LMCI (72 samples) and apply per-class loss weights to improve minority-class recall
- **Data Augmentation** — Random rotations (±15°), horizontal flipping, brightness jitter, elastic deformations to expand effective training set
- **k-Fold Cross-Validation** — Replace fixed split with stratified 5-fold CV for reliable confidence intervals on accuracy

**Medium-term (architecture):**
- **Transfer Learning** — Fine-tune VGG16, ResNet50, DenseNet121, or EfficientNet pretrained on ImageNet as feature extractors
- **Attention Mechanisms** — CBAM (Convolutional Block Attention Module) to learn which spatial regions drive classification
- **Grad-CAM Saliency Maps** — Visualize which brain regions the model attends to; validate against known neuroanatomical markers (hippocampus, entorhinal cortex)

**Long-term (clinical readiness):**
- **3D Volumetric CNN** — Process full CT/MRI volumes via 3D convolutions, preserving inter-slice context and enabling volumetric feature extraction
- **Longitudinal Modeling** — Incorporate multiple scans per patient over time to model disease progression trajectories
- **Multi-modal Fusion** — Combine CT imaging with clinical biomarkers (APOE genotype, CSF tau/amyloid, cognitive test scores)
- **Prospective Clinical Validation** — Head-to-head comparison against specialist radiologist diagnosis on a held-out clinical cohort

<br/>

---

## 10. Getting Started

### Prerequisites

```bash
pip install tensorflow keras opencv-python numpy matplotlib scikit-learn
```

### Clone & Setup

```bash
git clone https://github.com/srajan-kiyotaka/Alzheimer-Disease-Prediction
cd Alzheimer-Disease-Prediction

# Extract dataset
unzip Alzheimers-Disease.zip
```

### Expected Directory Structure

```
Alzheimers-ADNI/
└── train/
    ├── Final AD JPEG/       # 171 scans
    ├── Final CN JPEG/       # 580 scans
    ├── Final EMCI JPEG/     # 240 scans
    ├── Final LMCI JPEG/     #  72 scans
    └── Final MCI JPEG/      # 233 scans
```

### Run Locally

```bash
jupyter notebook Alzheimer_Disease_predictor.ipynb
```

### Run on Google Colab (Recommended)

```python
# Step 1 — Clone repository
!git clone "https://github.com/srajan-kiyotaka/Alzheimer-Disease-Prediction"

# Step 2 — Install headless OpenCV (Colab environment)
!pip install opencv-python-headless

# Step 3 — Run all cells in order
# Enable GPU: Runtime → Change runtime type → T4 GPU
```

> **Tip:** The full pipeline — data loading, preprocessing, model construction, training, and evaluation — is self-contained in the notebook. No external configuration required.

<br/>

---

## Repository Structure

```
Alzheimer's Disease Predictor/
│
├── 📁 Alzheimers-ADNI/
│   └── train/
│       ├── Final AD JPEG/           # 171 × Alzheimer's Disease
│       ├── Final CN JPEG/           # 580 × Cognitively Normal
│       ├── Final EMCI JPEG/         # 240 × Early MCI
│       ├── Final LMCI JPEG/         #  72 × Late MCI
│       └── Final MCI JPEG/          # 233 × MCI
│
├── Images/
│   ├── AD.png                       # Sample scan: AD
│   ├── CN.png                       # Sample scan: CN
│   ├── EMCI.png                     # Sample scan: EMCI
│   ├── LMCI.png                     # Sample scan: LMCI
│   ├── MCI.png                      # Sample scan: MCI
│   ├── Orignal.png                  # Pre-pipeline raw scan
│   └── Model.png                    # CNN architecture diagram
│
├── Alzheimer_Disease_predictor.ipynb   # Full pipeline notebook
├── Alzheimers-Disease.zip             # Packaged dataset
└── README.md
```

---

## License

```
MIT License — Copyright (c) Srajan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so.
```

---

<div align="center">

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Built for the intersection of machine intelligence and clinical care.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**If this project was useful to you, a ⭐ goes a long way.**

*Contributor: [Srajan](https://github.com/srajan-kiyotaka) — Model Design · Dataset Curation · Training & Evaluation*

<br/>

[![Back to Top](https://img.shields.io/badge/↑_Back_to_Top-37474F?style=flat-square)](.)

</div>
