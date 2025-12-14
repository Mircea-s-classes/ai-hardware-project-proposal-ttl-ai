# CNN Bubble Detection Training Results

## Summary

Trained CNN on real bubble data from AIH_Bubbles.mp4 using M1 Neural Engine (as Hailo-8L substitute). After fixing critical CV model bugs, discovered insufficient training data for deep learning approach.

## Current Status

### CV Model Performance (Tuned)
- **Detection rate**: 6-20 bubbles per frame
- **Bubble sizes**: 9-21px diameter
- **Filters applied**:
  - Size: 10-300px
  - Circularity: >0.3
  - Aspect ratio: 0.5-2.0
  - Edge margin: 5px

### CNN Model Performance
- **Training samples**: 27 (after proper filtering)
- **Best Dice score**: 0.0473 (essentially random)
- **Detection rate**: 0 bubbles
- **Model size**: 252KB (FP16 CoreML)
- **Status**: Did not learn - insufficient training data

## Critical Bugs Fixed

### Bug 1: Mask Polarity Inversion
**Problem**: Otsu threshold made bright liquid regions white, dark bubbles black

**Fix**: Added threshold inversion in `bubble_cv_model_tuned.py:22`
```python
th = cv2.bitwise_not(th)
```

**Result**: White pixel ratio changed from 65.22% to 32.76%

### Bug 2: Wrong Mask Scope
**Problem**: CV model returned full threshold mask instead of filtered bubbles only

**Fix**: Modified `bubble_cv_model_tuned.py:30-69` to create blank mask and only fill pixels for bubbles passing ALL filters

**Result**: White pixel ratio reduced to 0.08% (only actual bubbles)

## Training Data Evolution

| Generation | Samples | Issue | Dice Score |
|------------|---------|-------|------------|
| 1st        | 182     | Inverted masks (bubbles=black) | 0.9013 (learned wrong) |
| 2nd        | 132     | Full threshold (includes walls) | 0.6881 |
| 3rd        | **27**  | Properly filtered bubbles | **0.0473** (failed) |

## Root Cause Analysis

### Why Only 27 Samples?

Bubbles are sparse in AIH_Bubbles.mp4:
- Only 0.08% of frame pixels are bubbles
- Most 256x256 crops contain zero bubbles
- After proper filtering: only 27/1000 crops have any bubbles

### Why CNN Failed to Learn?

Deep learning requires hundreds/thousands of samples:
- 27 samples << minimum needed
- Model outputs all-black masks (0 detections)
- Essentially random predictions (Dice 0.0473)

## Manual Labeling for Supervised Learning

### Exported Frames
Location: `manual_labeling_samples/`

10 frames exported with:
- `frame_XXX_for_labeling.png` - Shows CV (green) vs CNN (blue) predictions
- `frame_XXX_original.png` - Clean frame for manual annotation

### Frame Details
| Frame | CV Detections | CNN Detections |
|-------|---------------|----------------|
| 015   | 6 bubbles     | 0 bubbles      |
| 030   | 8 bubbles     | 0 bubbles      |
| 045   | 12 bubbles    | 0 bubbles      |
| 060   | 15 bubbles    | 0 bubbles      |
| 075   | 11 bubbles    | 0 bubbles      |
| 090   | 9 bubbles     | 0 bubbles      |
| 105   | 14 bubbles    | 0 bubbles      |
| 120   | 20 bubbles    | 0 bubbles      |
| 150   | 18 bubbles    | 0 bubbles      |
| 180   | 16 bubbles    | 0 bubbles      |

## Next Steps

### Option 1: Manual Supervised Labeling (Recommended)
1. Manually annotate bubbles in exported frames
2. Generate ground truth masks
3. Retrain CNN on manually labeled data
4. Requires annotation tool (e.g., LabelMe, CVAT)

### Option 2: Collect More Bubble Videos
1. Record additional AIH bubble videos
2. Generate more training samples from diverse footage
3. Aim for 500+ samples minimum

### Option 3: Use CV Model Only
1. CV model already detects 6-20 bubbles/frame
2. No CNN inference overhead
3. Accept CV model limitations

### Option 4: Hybrid Approach
1. Use CV for initial detection
2. Train CNN as refinement filter
3. Requires fewer training samples

## Performance Metrics

### M1 Neural Engine (Current)
- **Model**: Small U-Net (252KB FP16)
- **Backend**: CoreML with Neural Engine
- **Status**: Not functional (0 detections)

### CV Model (Fallback)
- **Detection**: 6-20 bubbles/frame
- **Speed**: Real-time capable
- **Accuracy**: Unknown (needs ground truth)

## Files Generated

### Models
- `data/cnn/small_unet_real_trained.pt` - PyTorch checkpoint (Dice 0.0473)
- `data/cnn/small_unet_real_fp16.mlpackage` - CoreML FP16 (252KB)
- `data/cnn/small_unet_real_history.json` - Training history

### Training Data
- `data/cnn_real/` - 27 samples (256x256 crops with masks)

### Visualizations
- `manual_labeling_samples/` - 10 frames for annotation
- `visualization_output/` - Detection visualizations

### Scripts
- `src/hardware/export_for_labeling.py` - Export frames for manual labeling
- `src/hardware/visualize_results.py` - Visualize model predictions
- `src/model/generate_real_data.py` - Generate training samples from video
- `src/model/train_real_cnn.py` - Train CNN on real data
- `src/model/convert_to_coreml.py` - Convert PyTorch to CoreML

## Conclusion

The properly filtered CV model produces sparse but realistic bubble detections. However, this sparsity means insufficient training data for CNN deep learning (27 samples). **Manual supervised labeling is required** to create a larger, ground-truth dataset for successful CNN training.

Current recommendation: Use CV model for production until sufficient manually labeled training data is collected.
