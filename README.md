# EEG Seizure Detection using Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A comprehensive implementation of automated seizure detection from scalp EEG recordings using 1D Convolutional Neural Networks. This project demonstrates proper machine learning practices for highly imbalanced medical time-series data.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Results](#results)
- [Clinical Considerations](#clinical-considerations)
- [Future Work](#future-work)
- [License](#license)

## 🎯 Overview

This repository contains a complete end-to-end pipeline for EEG seizure detection, including:

- Data preprocessing and visualization
- Patient-aware cross-validation
- Custom CNN architecture optimized for EEG signals
- Comprehensive evaluation metrics
- Production-ready code structure

**Key Metrics**:
- ROC-AUC: 0.74
- Specificity: 99.19%
- Recall: 8.61% (requires improvement for clinical use)

## ✨ Features

### Technical Highlights

✅ **Patient-Safe Cross-Validation**
- GroupKFold prevents data leakage
- No patient appears in both train/validation sets

✅ **Memory-Efficient Design**
- Streaming dataset loads epochs on-demand
- Processes full dataset on single GPU

✅ **Class Imbalance Solutions**
- Balanced sampling (20% positive rate)
- Weighted loss function
- Threshold optimization

✅ **Comprehensive Evaluation**
- ROC and Precision-Recall curves
- Confusion matrix analysis
- Learning curve visualization

## 📊 Dataset

**CHB-MIT Scalp EEG Database**

| Property | Value |
|----------|-------|
| Source | PhysioNet |
| Patients | 14 (pediatric epilepsy) |
| Recordings | 421 EDF files |
| Channels | 23 bipolar montage |
| Sampling Rate | 256 Hz |
| Duration | ~916 hours total |
| Seizure Events | 152 annotated |
| Class Imbalance | 0.72% seizure epochs |

**Download Instructions**:
```bash
# Create data directory
mkdir -p ~/Desktop/EEG

# Download from PhysioNet (requires registration)
# https://physionet.org/content/chbmit/1.0.0/

# Expected structure:
# ~/Desktop/EEG/
# ├── chb01/
# │   ├── chb01_01.edf
# │   └── chb01-summary.txt
# ├── chb02/
# └── ...
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Quick Install
```bash
# Clone repository
git clone https://github.com/yourusername/eeg-seizure-detection.git
cd eeg-seizure-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

Create `requirements.txt`:
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
mne>=1.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
jupyter>=1.0.0
```

## 🏁 Quick Start

### 1. Download Data

Visit [CHB-MIT Database](https://physionet.org/content/chbmit/1.0.0/) and download to `~/Desktop/EEG/`

### 2. Run Notebook
```bash
jupyter notebook v2.ipynb
```

### 3. Execute Cells

Run sections sequentially:
1. **Setup** (Sections 1-2): ~2 minutes
2. **EDA** (Section 3): ~5 minutes
3. **Training** (Section 4): ~45 minutes (GPU) / ~12 hours (CPU)
4. **Evaluation** (Sections 5-6): ~3 minutes

### 4. Expected Output
```
============================================================
FINAL MODEL PERFORMANCE (Out-of-Fold)
============================================================
ROC-AUC:        0.7362
PR-AUC:         0.0330
F1 Score:       0.0804
Precision:      0.0754
Recall:         0.0861
Threshold used: 0.980
============================================================
```

## 🏗️ Model Architecture

### CNN Design
```
Input: (batch, 23 channels, 6400 timesteps)

Layer                Output Shape        Parameters
──────────────────────────────────────────────────────
Conv1d (k=7)         (B, 64, ~3200)      10,432
BatchNorm1d          (B, 64, ~3200)      128
ReLU + MaxPool       (B, 64, ~1600)      -
──────────────────────────────────────────────────────
Conv1d (k=5)         (B, 128, ~800)      40,960
BatchNorm1d          (B, 128, ~800)      256
ReLU + MaxPool       (B, 128, ~400)      -
──────────────────────────────────────────────────────
Conv1d (k=3)         (B, 256, ~400)      98,560
BatchNorm1d          (B, 256, ~400)      512
ReLU + AdaptivePool  (B, 256, 1)         -
──────────────────────────────────────────────────────
Flatten + Dropout    (B, 256)            -
Linear               (B, 1)              257
──────────────────────────────────────────────────────
Total: 151,169 parameters
```

**Design Rationale**:
- Progressive channel expansion captures hierarchical features
- Adaptive pooling handles variable sequence lengths
- Dropout prevents overfitting on rare seizure class

## 📈 Performance

### Metrics Summary

| Metric | Score | Clinical Target |
|--------|-------|-----------------|
| ROC-AUC | 0.7362 | >0.90 |
| PR-AUC | 0.0330 | >0.50 |
| Sensitivity | 8.61% | **>95%** |
| Specificity | 99.19% | >85% |
| Precision | 7.54% | >20% |
| F1 Score | 8.04% | - |

### Confusion Matrix
```
                 Predicted
                 Negative  Positive
Actual Negative   58,339      478
Actual Positive      414       39
```

**Interpretation**:
- ✅ Excellent at ruling out non-seizures (99.19% specificity)
- ❌ Poor at detecting actual seizures (8.61% recall)
- ⚠️ **Not clinically viable** - missing 91% of seizures

## 🔬 Results

### Learning Curves

The model shows:
- Steady training loss decrease
- Moderate generalization gap
- Early stopping at epoch 2-3 per fold

### Threshold Optimization

- Optimal F1 at threshold = 0.98
- Very conservative decision boundary
- Prioritizes precision over recall

### Class Imbalance Impact

- Only 453 seizure epochs out of 63,336 total
- PR-AUC (0.03) reflects difficulty with rare class
- ROC-AUC (0.74) shows reasonable discrimination

## 🏥 Clinical Considerations

### Critical Limitations

⚠️ **This model is NOT ready for clinical use**

**Why?**
1. **Low Recall (8.61%)**: Misses 91% of seizures
2. **Clinical Requirement**: Need >95% sensitivity
3. **Safety Critical**: False negatives are unacceptable

### Clinical Requirements

For medical deployment:

- **Sensitivity**: >95% (currently 8.61%)
- **Specificity**: >85% (currently 99.19% ✓)
- **Latency**: <1 second real-time
- **Regulatory**: FDA/CE approval required
- **Validation**: Clinical trial needed

### Path to Clinical Viability

1. **Increase Recall**:
   - Lower threshold (0.5 instead of 0.98)
   - More aggressive minority oversampling
   - Focal loss for hard examples

2. **Architecture Upgrades**:
   - Attention mechanisms
   - Transformer encoders
   - Multi-scale features

3. **Clinical Integration**:
   - Patient-specific calibration
   - Real-time optimization
   - Regulatory compliance

## 🔧 Future Work

### Immediate Priorities

1. **Boost Recall**:
```python
# Proposed changes
- Increase training positive rate to 50%
- Implement focal loss (γ=2)
- Add SMOTE-style augmentation
- Ensemble multiple models
```

2. **Data Augmentation**:
```python
# Time-series specific
- Temporal warping
- Amplitude jittering
- Frequency perturbation
- Channel dropout
```

3. **Advanced Architecture**:
```python
# Model improvements
- Self-attention layers
- Multi-head attention
- Dilated convolutions
- Wavelet features
```

### Long-term Goals

- Patient-specific fine-tuning
- Real-time inference pipeline
- Model quantization (INT8)
- Regulatory approval
- Clinical validation study

## 📚 Citation
```bibtex
@software{eeg_seizure_detection,
  title={EEG Seizure Detection using Deep Learning},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/eeg-seizure-detection}
}
```

**Dataset**:
```bibtex
@article{shoeb2009,
  title={Application of machine learning to epileptic seizure detection},
  author={Shoeb, Ali H},
  year={2009},
  school={MIT}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**For research and educational purposes only.**

- NOT approved for clinical/medical use
- Insufficient recall for patient safety
- Requires extensive validation

## 🙏 Acknowledgments

- CHB-MIT Dataset: Children's Hospital Boston & MIT
- PhysioNet: Physiological signal research platform
- MNE-Python: EEG processing tools
- PyTorch: Deep learning framework

---

**⭐ Star this repo if you find it useful!**

---
**Last Updated**: January 2025
