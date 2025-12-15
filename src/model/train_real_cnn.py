#!/usr/bin/env python3
"""
Retrain SmallUNet on real bubble data from AIH_Bubbles.mp4.
This creates a model optimized for real bubble detection (not synthetic).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from small_unet import SmallUNet

class RealBubbleDataset(Dataset):
    """Dataset for real bubble training data"""
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"

        self.image_files = sorted(list(self.img_dir.glob("*.png")))

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        print(f"Loaded {len(self.image_files)} training samples from {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name

        # Load image and mask
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Normalize
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)  # Binary mask

        # Convert to tensors (CHW format)
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dim

        return img, mask

def dice_coefficient(pred, target, smooth=1.0):
    """Dice score for evaluation"""
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def train_on_real_data(
    data_dir,
    output_path,
    batch_size=8,
    epochs=30,  # More epochs for real data
    lr=1e-3,
    val_split=0.2,
    device=None
):
    """
    Train SmallUNet on real bubble data

    Args:
        data_dir: Path to cnn_real/ directory
        output_path: Where to save trained model
        batch_size: Training batch size
        epochs: Number of training epochs
        lr: Learning rate
        val_split: Validation split ratio
        device: Training device (auto-detect if None)
    """
    print("=" * 80)
    print(" Retraining SmallUNet on Real Bubble Data")
    print(" (Mimicking Hailo-8L Edge Training Workflow)")
    print("=" * 80)

    # Auto-detect device (prioritize MPS for M1)
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    print(f"\nTraining device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")

    # Load dataset
    dataset = RealBubbleDataset(data_dir)

    # Split train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    print("\nInitializing SmallUNet...")
    model = SmallUNet().to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training loop
    best_dice = 0.0
    train_history = []

    print("\n" + "=" * 80)
    print(" Training Progress")
    print("=" * 80)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            train_dice += dice_coefficient(outputs, masks).item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # Update scheduler
        scheduler.step(val_loss)

        # Log
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

        train_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice
        })

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), output_path)
            print(f"  ✓ Saved best model (Dice: {val_dice:.4f})")

    print("\n" + "=" * 80)
    print(" Training Complete")
    print("=" * 80)
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Model saved to: {output_path}")

    # Save training history
    import json
    history_path = Path(output_path).parent / f"{Path(output_path).stem}_history.json"
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    print("\n" + "=" * 80)
    print(" Next Steps")
    print("=" * 80)
    print("  1. Convert to CoreML (mimicking Hailo ONNX→INT8 workflow)")
    print("  2. Validate on AIH_Bubbles.mp4")
    print("  3. Benchmark performance")
    print("=" * 80)

    return model, best_dice

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data" / "cnn_real"
    output_path = Path(__file__).resolve().parents[2] / "data" / "cnn" / "small_unet_real_trained.pt"

    model, best_dice = train_on_real_data(
        data_dir=data_dir,
        output_path=output_path,
        batch_size=16,  # Larger batch for M1
        epochs=30,
        lr=1e-3,
        val_split=0.2
    )

    print(f"\n✓ Real-data model ready! (Dice: {best_dice:.4f})")
