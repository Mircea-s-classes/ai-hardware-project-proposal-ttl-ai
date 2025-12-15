# Manual Bubble Labeling Instructions

## Overview
This folder contains 10 frames (5 from each video) for you to manually annotate with ground truth bubble locations.

## Files
- **AIH_Bubbles_frame_XXX.png** - 5 frames from AIH_Bubbles.mp4
- **AIH_Bubbles2_frame_XXX.png** - 5 frames from AIH_Bubbles2.mp4

## How to Annotate

### 1. Open Each Frame
Use any image editor that supports drawing circles (Preview, Paint, Photoshop, GIMP, etc.)

### 2. Draw RED Circles Around EVERY Bubble
- **Color**: Use pure RED (RGB: 255, 0, 0) or close to red
- **Shape**: Draw circles around each bubble you see
- **Size**: Circle should encompass the entire bubble
- **Coverage**: Mark EVERY bubble, even small or faint ones

### 3. Save Annotated Images
Save each annotated image with `_annotated` suffix:

**Examples**:
- `AIH_Bubbles_frame_030.png` → `AIH_Bubbles_frame_030_annotated.png`
- `AIH_Bubbles2_frame_120.png` → `AIH_Bubbles2_frame_120_annotated.png`

**Important**: Save in the SAME folder as the original frames

### 4. What Counts as a Bubble?
Based on your feedback:
- **AIH_Bubbles.mp4**: Typically 2-3 bubbles per frame from syringe to waterline
- **AIH_Bubbles2.mp4**: Typically 5-10 bubbles per frame

Mark any air pockets or bubbles you see, regardless of size.

## After Annotation

Once you've annotated all 10 frames, run:

```bash
cd src/hardware
python convert_annotations_to_masks.py
```

This will:
1. Detect your red circles
2. Convert them to binary masks
3. Create training samples in `data/cnn_manual/`

## Then Retrain

```bash
cd src/model
python train_manual_cnn.py
```

This will train the CNN on your ground truth annotations, achieving much better accuracy!

## Need Help?
- Red circles should be clearly visible
- Don't worry about perfect circles - approximate shape is fine
- If unsure about a bubble, include it (better to over-annotate than under-annotate)
