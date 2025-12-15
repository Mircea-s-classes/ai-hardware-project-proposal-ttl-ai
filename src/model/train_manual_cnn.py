#!/usr/bin/env python3
"""
Train CNN on manually labeled ground truth data.
This uses the user's manual annotations to learn real bubble characteristics.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from small_unet import SmallUNet

class ManualBubbleDataset(Dataset):
    """Dataset from manually annotated frames"""
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"

        self.samples = sorted(list(self.img_dir.glob("*.png")))

        if len(self.samples) == 0:
            raise ValueError(f"No training samples found in {self.img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        mask_path = self.mask_dir / img_path.name

        # Load image and mask
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        # Convert to tensors (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask

def dice_coefficient(pred, target):
    """Dice coefficient for binary segmentation"""
    smooth = 1e-5
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train_on_manual_labels():
    print("=" * 80)
    print(" Training CNN on Manual Ground Truth Labels")
    print(" (Supervised Learning with Human Annotations)")
    print("=" * 80)

    # Configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = Path(__file__).resolve().parents[2] / "data" / "cnn_manual"
    output_dir = Path(__file__).resolve().parents[2] / "data" / "cnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 16
    epochs = 50  # More epochs since we have ground truth
    learning_rate = 0.001

    print(f"\nTraining device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")

    # Load dataset
    try:
        dataset = ManualBubbleDataset(data_dir)
        print(f"Loaded {len(dataset)} manually labeled samples from {data_dir}")
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run these steps first:")
        print("  1. export_frames_for_manual_labeling.py")
        print("  2. Manually annotate frames with red circles")
        print("  3. convert_annotations_to_masks.py")
        return

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nDataset split:")
    print(f"  Training: {train_size} samples")
    print(f"  Validation: {val_size} samples")

    # Initialize model
    print("\nInitializing SmallUNet...")
    model = SmallUNet(in_channels=3, out_channels=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("\n" + "=" * 80)
    print(" Training Progress")
    print("=" * 80)

    best_dice = 0.0
    history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_coefficient(torch.sigmoid(outputs), masks).item()

            pbar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # Validation
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
                val_dice += dice_coefficient(torch.sigmoid(outputs), masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        # Print epoch summary
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': val_dice,
            }, output_dir / "small_unet_manual_trained.pt")
            print(f"  ✓ Saved best model (Dice: {val_dice:.4f})")

    # Save training history
    history_path = output_dir / "small_unet_manual_trained_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print(" Training Complete")
    print("=" * 80)
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Model saved to: {output_dir / 'small_unet_manual_trained.pt'}")
    print(f"Training history saved to: {history_path}")

    print("\n" + "=" * 80)
    print(" Next Steps")
    print("=" * 80)
    print("  1. Convert to CoreML: python convert_to_coreml.py")
    print("  2. Validate on videos: python validate_retrained.py")
    print("  3. Compare with CV model")
    print("=" * 80)

    print(f"\n✓ Manual-labeled model ready! (Dice: {best_dice:.4f})")

if __name__ == "__main__":
    train_on_manual_labels()
