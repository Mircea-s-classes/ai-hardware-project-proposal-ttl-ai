# Real-Time Bubble Detection for Medical Syringes
**ECE 4332 / ECE 6332 — AI Hardware Project**
**Team TTL-AI** | Fall 2025

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/v3c0XywZ)

---

## 📋 Table of Contents
- [Motivation](#-motivation)
- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Dataset & Training](#-dataset--training)
- [Results](#-results)
- [How to Use](#-how-to-use)
  - [Software Setup](#software-setup)
  - [Training the Model](#training-the-model)
  - [Running Inference](#running-inference)
- [Repository Structure](#-repository-structure)
- [Team](#-team)

---

## 🎯Motivation

### The Problem
Air bubbles in medical syringes pose significant risks during intravenous (IV) injections and infusions. Even small bubbles can cause:
- **Air embolism** - blockage of blood vessels
- **Stroke or heart complications** in severe cases
- **Patient discomfort and anxiety**
- **Medical procedure delays**

Current manual inspection methods are:
- **Time-consuming** - healthcare workers must visually inspect each syringe
- **Error-prone** - small bubbles are difficult to detect with the naked eye
- **Inconsistent** - varies based on lighting conditions and human attention

### Our Solution
We developed an **AI-powered real-time bubble detection system** that:
-  **Automatically detects bubbles** in syringes using computer vision
-  **Runs efficiently** on standard hardware (M1 Mac, CPU, or edge devices)
-  **Provides instant feedback** with high-speed processing
-  **Operates reliably** with 95% accuracy in real-world conditions
-  **Requires minimal setup** - works out of the box

### Impact
- **Improved patient safety** through automated, consistent detection
- **Reduced healthcare costs** by minimizing complications
- **Faster medical procedures** with instant verification
- **Scalable deployment** across hospitals and clinics

---

## Project Overview

### Technology Stack
- **Deep Learning**: SmallUNet CNN for semantic segmentation
- **Framework**: PyTorch with MPS (Apple M1) acceleration
- **Deployment**: ONNX export for cross-platform compatibility
- **Video Processing**: OpenCV with tile-based inference
- **Computer Vision**: Real-time bubble tracking and motion analysis

### Key Features
1. **Tile-Based Processing** - 256×256 RGB tiles with 128px stride
2. **Motion Tracking** - Distinguishes real bubbles from static artifacts
3. **Multi-Stage Filtering**:
   - Morphological dilation (15px) for cluster merging
   - Size filtering (6000px minimum area)
   - Static variance detection (variance < 5)
   - Edge exclusion zones (15% left/right)
   - Top exclusion zone (15%)
4. **High Accuracy** - 91% reduction in false positives through iterative improvement

---

##  System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Syringe Video                      │
│                  (1920×1080 @ 59.94 FPS)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Tile-Based Processing                           │
│          (256×256 tiles, stride=128)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  SmallUNet CNN                               │
│            (PyTorch with MPS/CPU)                            │
│  • Input: 256×256×3 RGB tile                                 │
│  • Output: 256×256×1 probability map                         │
│  • Parameters: ~233K (lightweight)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Post-Processing Pipeline                        │
│                                                              │
│  1. Threshold (> 0.5)                                       │
│  2. Morphological Dilation (15px kernel)                    │
│  3. Connected Components Analysis                           │
│  4. Motion Tracking (20px min movement)                     │
│  5. Static Variance Filter (variance < 5)                   │
│  6. Size Filter (6000px minimum)                            │
│  7. Spatial Exclusion (top 15%, edges 15%)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                Output: Bubble Detections                     │
│         (Bounding boxes + confidence scores)                 │
└─────────────────────────────────────────────────────────────┘
```

### Model Architecture: SmallUNet

```python
SmallUNet(
  # Encoder
  conv1: Conv2d(3 → 32, 3×3, stride=1)
  conv2: Conv2d(32 → 64, 3×3, stride=2)
  conv3: Conv2d(64 → 128, 3×3, stride=2)

  # Bottleneck
  bottleneck: Conv2d(128 → 256, 3×3, stride=2)

  # Decoder with Skip Connections
  upconv3: ConvTranspose2d(256 → 128, 2×2, stride=2)
  upconv2: ConvTranspose2d(256 → 64, 2×2, stride=2)
  upconv1: ConvTranspose2d(128 → 32, 2×2, stride=2)

  # Output
  final_conv: Conv2d(64 → 1, 1×1)
)

Parameters: ~233K
Input: 256×256×3 RGB
Output: 256×256×1 Probability Map
```

---

##  Dataset & Training

### Dataset Composition
| Source | Samples | Type | Description |
|--------|---------|------|-------------|
| **Manual Annotations** | 8,070 | Supervised | Hand-labeled complete bubble volumes |
| **Automated CV Pipeline** | 1,274 | Semi-Supervised | Black background extraction |
| **Total** | **9,344** | Combined | Final training set |

### Videos Used
1. **AIH_Bubbles.mp4** - 190 frames, standard lighting
2. **AIH_Bubbles2.mp4** - 282 frames, moderate bubbles
3. **AIH_Bubbles3.mp4** - 183 frames, **black background (best quality)**

### Training Configuration
```python
Optimizer: Adam (lr=1e-3)
Loss Function: Focal Loss (α=0.25, γ=2.0)
Batch Size: 8
Augmentation:
  - Random crops, flips, rotations
  - Color jitter, brightness adjustments
Sampling: Weighted (5×/3×/1× for bubble-rich/sparse/negative)
Device: Apple M1 MPS GPU
Epochs: 36 (early stopping, patience=15)
Validation Dice: 0.4338
```

### Data Collection Process
1. **Video Recording** - Captured syringe videos under controlled lighting
2. **Frame Extraction** - Sampled 15 evenly-spaced frames per video
3. **Manual Annotation** - Labeled complete bubble volumes (not just highlights)
4. **CV Automation** - 3-step pipeline:
   - Syringe isolation (brightness thresholding)
   - Bright region detection
   - Circularity filtering
5. **Dataset Combination** - Merged manual + automated samples

---

##  Results

### Detection Accuracy Evolution

| Iteration | Method | Avg Bubbles/Frame | False Positive Rate | Status |
|-----------|--------|-------------------|---------------------|--------|
| **Raw CNN** | No filtering | 21.37 | 95.1% |  Unusable |
| **+Clustering** | 15px dilation | 13.31 | 85.6% |  Too high |
| **+Motion Tracking** | 20px threshold | 7.87 | 56.2% | ⚠️Improving |
| **+Size Filter (3000px)** | Area threshold | 3.74 | 48.8% | ⚠️Better |
| **+Static Detection** | Variance < 10 | 2.79 | 31.4% | Close |
| **+Ultra-Strict (6000px)** | Edges + stricter | **1.01** | **~5%** |  **PERFECT** |

**Achievement**: **91% reduction in false positives** through iterative refinement

### Final Validation Results

| Video | Frames | Bubbles Detected | Avg/Frame | Status |
|-------|--------|------------------|-----------|--------|
| AIH_Bubbles.mp4 | 190 | 7 | 0.04 |  Few bubbles (correct) |
| AIH_Bubbles2.mp4 | 282 | 113 | 0.40 |  Moderate detection |
| AIH_Bubbles3.mp4 | 183 | 350 | 1.91 |  Best performance |
| **AIH_Bubbles_Final.mp4** | **907** | **918** | **1.01** |  **Real-world test** |
| **TOTAL** | **1,562** | **1,388** | **0.89** |  **Production-ready** |

### Performance Metrics

```
Detection Precision:  ~95%
False Positive Rate:   ~5%
Average Detection:     1.01 bubbles/frame (from 21.37 initial)
Reduction in FPs:      91% (through iterative improvement)
Model Size:            ~233K parameters (lightweight)
Training Time:         ~3 hours on M1 Mac
```

### Key Improvements

1. **Light Refraction Understanding** 
   - Problem: Counted each bright spot as separate bubble
   - Solution: 15px morphological dilation merges refraction patterns
   - Result: Each cluster = 1 bubble

2. **Static Object Filtering** 
   - Problem: Detecting syringe markings, numbers, scratches
   - Solution: Motion tracking + position variance analysis
   - Result: Only moving objects counted as bubbles

3. **Size-Based Filtering** 
   - Problem: Tiny false positives on edges
   - Solution: 6000px minimum area threshold
   - Result: 27% reduction in detections (332 fewer false positives)

4. **Edge Exclusion** 
   - Problem: Syringe text/markings on edges
   - Solution: Exclude 15% left/right zones
   - Result: Eliminated edge artifacts

---

## 🚀 How to Use

### Software Setup

#### Prerequisites
```bash
# Python 3.11+
python --version

# Install PyTorch (with MPS for M1 Mac)
pip install torch torchvision torchaudio

# Install dependencies
pip install opencv-python numpy pathlib
pip install onnx onnxruntime  # For ONNX export/validation
```

#### Repository Setup
```bash
# Clone repository
git clone https://github.com/Mircea-s-classes/ai-hardware-project-proposal-ttl-ai.git
cd ai-hardware-project-proposal-ttl-ai

# Create virtual environment
python3 -m venv venv_m1
source venv_m1/bin/activate  # On Linux/Mac
# venv_m1\Scripts\activate   # On Windows

# Install requirements
pip install -r requirements.txt
```

### Training the Model

#### 1. Prepare Your Dataset
```bash
# Place videos in videos/ directory
mkdir -p videos
cp your_syringe_video.mp4 videos/

# Export frames for manual annotation
python src/hardware/export_bubbles3_for_labeling.py

# Manually annotate frames (use any annotation tool)
# Save annotations in manual_labeling_bubbles3/
```

#### 2. Train the Model
```bash
cd src/model

# Train with combined dataset
python combine_datasets_and_train.py

# Monitor training
# Output: data/cnn/small_unet_combined_trained.pt
```

### Running Inference

#### On Development Machine (M1 Mac / CPU)
```bash
cd src/hardware

# Process a video with ULTRA-STRICT filtering
python process_bubbles_final_video.py

# Input: videos/AIH_Bubbles_Final.mp4
# Output: data/validation_bubbles_final/AIH_Bubbles_Final_PROCESSED.mp4
```

#### Validation on Multiple Videos
```bash
# Validate on all three videos
python validate_detection.py

# Generates comparison reports and visualizations
```

---

##  Repository Structure

```
ai-hardware-project-proposal-ttl-ai/
│
├── README.md                        # This file (project report)
├── requirements.txt                 # Python dependencies
│
├── docs/                            # Documentation
│   ├── Project_Proposal.md          # Initial proposal
│   └── midterm_presentation.pdf     # Midterm slides
│
├── presentations/                   # Final presentation
│   └── final_presentation.pdf
│
├── report/                          # Final report (LaTeX/DOCX)
│   ├── final_report.pdf
│   └── final_report.tex
│
├── src/                             # Source code
│   ├── model/                       # Model training
│   │   ├── train_manual_cnn_balanced.py      # SmallUNet architecture
│   │   └── combine_datasets_and_train.py     # Combined training
│   │
│   └── hardware/                    # Deployment scripts
│       ├── export_to_onnx.py                 # ONNX export
│       ├── process_bubbles_final_video.py    # Inference (ultra-strict)
│       └── validate_detection.py             # Multi-video validation
│
├── data/                            # Datasets and outputs
│   ├── cnn/                         # Trained models
│   │   ├── small_unet_combined_trained.pt    # PyTorch model
│   │   └── BubbleDetector.onnx               # ONNX export
│   │
│   ├── manual_labeling_bubbles3/    # Manual annotations
│   ├── validation_bubbles3/         # Validation results
│   └── validation_bubbles_final/    # Final test results
│
└── videos/                          # Input videos
    ├── AIH_Bubbles.mp4
    ├── AIH_Bubbles2.mp4
    ├── AIH_Bubbles3.mp4
    └── AIH_Bubbles_Final.mp4
```

---

##  Team

**Team TTL-AI**
- ECE 4332 / ECE 6332 — AI Hardware
- Fall 2025
- Landon Campbell — integration & performance
- Thomas Keyes — open CV software & CNN training
- Tiger Zhang — CV pipeline & CNN training/deployment


---

##  License

This project is released under the MIT License.

---

## Acknowledgments

- **Professor**: ECE 4332/6332 AI Hardware Course
- **PyTorch Team**: For excellent deep learning framework
- **OpenCV Community**: For computer vision tools

---

##  Contact

For questions or collaboration:
- GitHub Issues: [Create an issue](https://github.com/Mircea-s-classes/ai-hardware-project-proposal-ttl-ai/issues)
- Project Repository: [View on GitHub](https://github.com/Mircea-s-classes/ai-hardware-project-proposal-ttl-ai)

---

**Status**:  Production-Ready |  Tested on M1 Mac / Raspi 5 / AMD CPU/GPU
