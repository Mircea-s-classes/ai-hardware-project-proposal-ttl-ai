import torch
import coremltools as ct
from pathlib import Path
import sys

# Add model directory to path
sys.path.insert(0, str(Path(__file__).parent))
from small_unet import SmallUNet

def convert_model():
    print("=" * 70)
    print(" CoreML Model Conversion (Phase 2)")
    print("=" * 70)

    # Paths
    ckpt_path = Path(__file__).resolve().parents[2] / "data" / "cnn" / "small_unet_real_trained.pt"
    output_dir = ckpt_path.parent

    # Load PyTorch model
    print(f"\n[1/5] Loading PyTorch model: {ckpt_path}")
    model = SmallUNet()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print("✓ Model loaded")

    # Create example input
    example_input = torch.randn(1, 3, 256, 256)

    # Trace model (preferred for CoreML conversion)
    print("\n[2/5] Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    # Validate traced model
    original_out = model(example_input)
    traced_out = traced_model(example_input)
    max_diff = torch.abs(original_out - traced_out).max().item()
    print(f"✓ Tracing validation - max diff: {max_diff:.6f}")
    assert max_diff < 1e-5, "Traced model doesn't match original!"

    # Convert to CoreML with FP16 (Neural Engine optimized)
    print("\n[3/5] Converting to CoreML FP16 (Neural Engine optimized)...")
    print("  This may take 1-2 minutes...")

    mlmodel_fp16 = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=(1, 3, 256, 256))],
        outputs=[ct.TensorType(name="logits")],
        compute_units=ct.ComputeUnit.ALL,  # Neural Engine + GPU + CPU
        minimum_deployment_target=ct.target.macOS13,  # M1 optimization
        convert_to="mlprogram",  # Modern format
        compute_precision=ct.precision.FLOAT16,  # FP16 for Neural Engine
    )

    # Add metadata
    mlmodel_fp16.author = "Bubble Detection Team"
    mlmodel_fp16.short_description = "SmallUNet for bubble segmentation (FP16, Neural Engine optimized)"
    mlmodel_fp16.input_description["input"] = "RGB image normalized [0,1], shape (1,3,256,256)"
    mlmodel_fp16.output_description["logits"] = "Segmentation logits, shape (1,1,256,256)"

    # Save FP16 model
    fp16_path = output_dir / "small_unet_real_fp16.mlpackage"
    mlmodel_fp16.save(str(fp16_path))
    print(f"✓ Saved FP16 model: {fp16_path}")

    # Also create FP32 version for comparison
    print("\n[4/5] Converting to CoreML FP32 (baseline)...")
    mlmodel_fp32 = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=(1, 3, 256, 256))],
        outputs=[ct.TensorType(name="logits")],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13,
        convert_to="mlprogram",
    )

    mlmodel_fp32.author = "Bubble Detection Team"
    mlmodel_fp32.short_description = "SmallUNet for bubble segmentation (FP32, baseline)"

    fp32_path = output_dir / "small_unet_real_fp32.mlpackage"
    mlmodel_fp32.save(str(fp32_path))
    print(f"✓ Saved FP32 model: {fp32_path}")

    # Get file sizes
    import os
    def get_dir_size(path):
        total = 0
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
        return total

    fp16_size = get_dir_size(fp16_path) / 1024  # KB
    fp32_size = get_dir_size(fp32_path) / 1024  # KB

    print("\n[5/5] Conversion Complete!")
    print("=" * 70)
    print(" CoreML Models Summary")
    print("=" * 70)
    print(f"FP16 (Neural Engine): {fp16_path.name}")
    print(f"  Size: {fp16_size:.1f} KB")
    print(f"  Use: Maximum performance on M1")
    print()
    print(f"FP32 (Baseline):      {fp32_path.name}")
    print(f"  Size: {fp32_size:.1f} KB")
    print(f"  Use: Comparison/debugging")
    print()
    print(f"Saved to: {output_dir}")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run bubble_coreml_model.py to test inference")
    print("  2. Run run_aih_bubbles.py --backend coreml to process video")
    print("  3. Compare performance vs PyTorch MPS")
    print("=" * 70)

if __name__ == "__main__":
    convert_model()
