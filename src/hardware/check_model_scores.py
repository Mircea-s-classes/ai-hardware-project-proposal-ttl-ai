#!/usr/bin/env python3
"""
Check dice scores and metadata for all trained models
"""

import torch
from pathlib import Path

def check_models():
    """Check all model files and display their metadata"""

    base_dir = Path(__file__).parent.parent.parent
    model_dir = base_dir / "data" / "cnn"

    model_files = sorted(model_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)

    print("\n" + "="*80)
    print("TRAINED MODELS SUMMARY")
    print("="*80 + "\n")

    best_dice = 0.0
    best_model = None

    for i, model_path in enumerate(model_files, 1):
        print(f"{i}. {model_path.name}")
        print(f"   File size: {model_path.stat().st_size / 1024:.1f} KB")
        print(f"   Modified: {model_path.stat().st_mtime}")

        try:
            checkpoint = torch.load(model_path, map_location='cpu')

            # Extract metadata
            epoch = checkpoint.get('epoch', 'N/A')
            best_val_dice = checkpoint.get('best_val_dice', None)
            train_loss = checkpoint.get('train_loss', None)
            val_loss = checkpoint.get('val_loss', None)

            print(f"   Epoch: {epoch}")

            if best_val_dice is not None:
                print(f"   Best Validation Dice: {best_val_dice:.4f} ({best_val_dice*100:.2f}%)")
                if best_val_dice > best_dice:
                    best_dice = best_val_dice
                    best_model = model_path
            else:
                print(f"   Best Validation Dice: N/A")

            if train_loss is not None:
                print(f"   Training Loss: {train_loss:.6f}")
            if val_loss is not None:
                print(f"   Validation Loss: {val_loss:.6f}")

            # Check for additional metadata
            if 'optimizer_state_dict' in checkpoint:
                print(f"   Contains optimizer state: Yes")
            if 'scheduler_state_dict' in checkpoint:
                print(f"   Contains scheduler state: Yes")

        except Exception as e:
            print(f"   Error loading: {e}")

        print()

    print("="*80)
    if best_model:
        print(f"\nBEST MODEL (by Dice score):")
        print(f"  {best_model.name}")
        print(f"  Dice Score: {best_dice:.4f} ({best_dice*100:.2f}%)")
    print("\nMOST RECENT MODEL (trained on AIH_Bubbles3):")
    if model_files:
        print(f"  {model_files[0].name}")
    print("="*80 + "\n")

if __name__ == "__main__":
    check_models()
