#!/usr/bin/env python3
"""
Train CNN on manually labeled data with proper class imbalance handling.

Implements:
- Focal Loss (focuses on hard examples)
- Weighted sampling (oversamples positive crops)
- Data augmentation
- Balanced batch construction
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from small_unet import SmallUNet

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard examples, down-weights easy negatives.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )

        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()

class ManualBubbleDataset(Dataset):
    """Dataset from manually annotated frames with augmentation"""

    def __init__(self, data_dir, augment=True):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"
        self.augment = augment

        self.samples = sorted(list(self.img_dir.glob("*.png")))

        if len(self.samples) == 0:
            raise ValueError(f"No training samples found in {self.img_dir}")

        # Calculate sample weights for balanced sampling
        self.weights = []
        for img_path in self.samples:
            mask_path = self.mask_dir / img_path.name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # Weight samples by bubble content
            bubble_pixels = np.sum(mask > 127)
            if bubble_pixels > 100:  # Has significant bubbles
                weight = 5.0  # Oversample positive samples
            elif bubble_pixels > 0:  # Has few bubbles
                weight = 3.0
            else:  # Negative sample (no bubbles)
                weight = 1.0

            self.weights.append(weight)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        mask_path = self.mask_dir / img_path.name

        # Load image and mask
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Data augmentation
        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

            # Rotation
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h))
                mask = cv2.warpAffine(mask, M, (w, h))

            # Brightness/contrast
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)  # contrast
                beta = np.random.uniform(-20, 20)    # brightness
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

            # Gaussian noise
            if np.random.rand() > 0.7:
                noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)

        # Normalize
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # Convert to tensors
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask

def dice_coefficient(pred, target):
    """Dice coefficient for binary segmentation"""
    smooth = 1e-5
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train_with_imbalance_handling():
    print("=" * 80)
    print(" Training CNN on Manual Annotations")
    print(" (With Class Imbalance Handling)")
    print("=" * 80)

    # Configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = Path(__file__).resolve().parents[2] / "data" / "cnn_manual"
    output_dir = Path(__file__).resolve().parents[2] / "data" / "cnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 16
    epochs = 100  # More epochs with early stopping
    learning_rate = 0.0005
    patience = 15  # Early stopping patience

    print(f"\nTraining device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs} (with early stopping)")
    print(f"Learning rate: {learning_rate}")
    print(f"Using Focal Loss for class imbalance")

    # Load dataset
    try:
        dataset = ManualBubbleDataset(data_dir, augment=True)
        print(f"Loaded {len(dataset)} manually labeled samples")
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run: python detect_black_box_annotations.py")
        return

    # Create weighted sampler for balanced batches
    sampler = WeightedRandomSampler(
        weights=dataset.weights,
        num_samples=len(dataset),
        replacement=True
    )

    # Data loader with weighted sampling
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0
    )

    # Small validation set (20% random split)
    val_size = max(int(0.2 * len(dataset)), 1)
    train_size = len(dataset) - val_size
    indices = torch.randperm(len(dataset))
    val_indices = indices[:val_size]

    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nDataset split:")
    print(f"  Training: {train_size} samples (weighted sampling)")
    print(f"  Validation: {val_size} samples")

    # Initialize model
    print("\nInitializing SmallUNet...")
    model = SmallUNet().to(device)

    # Use Focal Loss instead of BCE
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training loop
    print("\n" + "=" * 80)
    print(" Training Progress")
    print("=" * 80)

    best_dice = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        batch_count = 0

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
            batch_count += 1

            pbar.set_postfix({"loss": loss.item()})

        train_loss /= batch_count
        train_dice /= batch_count

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
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

        # Learning rate scheduling
        scheduler.step(val_dice)

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': val_dice,
            }, output_dir / "small_unet_manual_trained.pt")
            print(f"  ✓ Saved best model (Dice: {val_dice:.4f})")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {patience} epochs)")
            break

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
    print("  2. Validate: python validate_manual_model.py")
    print("=" * 80)

    print(f"\n✓ Manual-labeled model ready! (Dice: {best_dice:.4f})")

if __name__ == "__main__":
    train_with_imbalance_handling()
