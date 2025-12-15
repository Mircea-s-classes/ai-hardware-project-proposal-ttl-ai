#!/usr/bin/env python3
"""
Convert trained PyTorch model to CoreML FP16 for iOS deployment
"""

import torch
import coremltools as ct
from pathlib import Path
import sys

# Add model directory to path
sys.path.append(str(Path(__file__).parent.parent / "model"))
from train_manual_cnn_balanced import SmallUNet

def convert_to_coreml():
    """Convert SmallUNet PyTorch model to CoreML FP16"""

    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "data" / "cnn" / "small_unet_combined_trained.pt"
    output_path = base_dir / "data" / "cnn" / "BubbleDetector.mlpackage"

    print("="*60)
    print("CONVERTING PYTORCH MODEL TO COREML FP16")
    print("="*60)
    print(f"\nInput: {model_path}")
    print(f"Output: {output_path}")

    # Load PyTorch model
    print("\n[1/4] Loading PyTorch model...")
    device = torch.device("cpu")  # Use CPU for conversion
    model = SmallUNet()

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  ✓ Model loaded")
    print(f"  ✓ Model architecture: SmallUNet")
    print(f"  ✓ Training Dice score: {checkpoint.get('best_val_dice', 'N/A')}")

    # Create example input (256x256 RGB tile)
    print("\n[2/4] Creating example input...")
    example_input = torch.rand(1, 3, 256, 256)
    print(f"  ✓ Input shape: (1, 3, 256, 256)")
    print(f"  ✓ Input format: RGB image tile")

    # Trace the model
    print("\n[3/4] Tracing PyTorch model...")
    traced_model = torch.jit.trace(model, example_input)
    print(f"  ✓ Model traced successfully")

    # Convert to CoreML
    print("\n[4/4] Converting to CoreML FP16...")

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(
            name="input",
            shape=example_input.shape,
            dtype=float
        )],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS15
    )

    # Add metadata
    mlmodel.short_description = "Bubble detection model for syringe imaging"
    mlmodel.author = "AI Hardware Project"
    mlmodel.license = "Proprietary"
    mlmodel.version = "1.0"

    mlmodel.input_description["input"] = "256x256 RGB image tile from syringe video"
    mlmodel.output_description["output"] = "256x256 bubble segmentation mask"

    print(f"  ✓ CoreML conversion complete")
    print(f"  ✓ Precision: FP16 (half precision)")
    print(f"  ✓ Deployment target: iOS 15+")

    # Save model
    mlmodel.save(str(output_path))

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"CONVERSION SUCCESSFUL!")
    print(f"{'='*60}")
    print(f"\nModel saved to: {output_path}")
    print(f"File size: {file_size:.2f} MB")
    print(f"\nModel details:")
    print(f"  - Input: 256x256x3 RGB image")
    print(f"  - Output: 256x256x1 mask (0-1 float)")
    print(f"  - Precision: FP16")
    print(f"  - iOS: 15+")
    print(f"\nUsage in iOS:")
    print(f"  1. Add BubbleDetector.mlpackage to Xcode project")
    print(f"  2. Process video frames in 256x256 tiles")
    print(f"  3. Apply threshold (>0.5) to get binary mask")
    print(f"  4. Use post-processing:")
    print(f"     - Morphological dilation (15px)")
    print(f"     - Minimum bubble size: 3000px")
    print(f"     - Motion tracking (20px movement)")
    print(f"     - Static filter (variance < 10)")
    print(f"\n{'='*60}\n")

    return output_path

if __name__ == "__main__":
    try:
        convert_to_coreml()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
