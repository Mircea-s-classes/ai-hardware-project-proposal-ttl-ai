#!/usr/bin/env python3
"""
Combine all training datasets and retrain CNN with improved data.
Merges manual annotations + Bubbles3 automated masks.
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
import shutil

sys.path.insert(0, str(Path(__file__).parent))
from small_unet import SmallUNet

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
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

class CombinedBubbleDataset(Dataset):
    """Combined dataset from multiple sources"""

    def __init__(self, data_dirs, augment=True):
        self.augment = augment
        self.samples = []
        self.weights = []

        # Collect samples from all directories
        for data_dir in data_dirs:
            img_dir = Path(data_dir) / "images"
            mask_dir = Path(data_dir) / "masks"

            if not img_dir.exists():
                print(f"Warning: {img_dir} does not exist, skipping")
                continue

            samples = sorted(list(img_dir.glob("*.png")))

            for img_path in samples:
                mask_path = mask_dir / img_path.name

                if not mask_path.exists():
                    continue

                # Calculate weight
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                bubble_pixels = np.sum(mask > 127)

                if bubble_pixels > 100:
                    weight = 5.0  # High bubble content
                elif bubble_pixels > 0:
                    weight = 3.0  # Some bubbles
                else:
                    weight = 1.0  # Negative sample

                self.samples.append((img_path, mask_path))
                self.weights.append(weight)

        if len(self.samples) == 0:
            raise ValueError("No training samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load image and mask
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Data augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h))
                mask = cv2.warpAffine(mask, M, (w, h))

            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.uniform(-20, 20)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

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
    """Dice coefficient"""
    smooth = 1e-5
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train_combined():
    print("=" * 80)
    print(" Training CNN on Combined Dataset")
    print(" Manual Annotations + AIH_Bubbles3 Automated Masks")
    print("=" * 80)

    # Configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parents[2]

    # Data directories
    data_dirs = [
        base_dir / "data" / "cnn_manual",    # 6,600 samples (manual annotations)
        base_dir / "data" / "cnn_bubbles3"   # 1,274 samples (black bg video)
    ]

    output_dir = base_dir / "data" / "cnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 16
    epochs = 100
    learning_rate = 0.0005
    patience = 15

    print(f"\nTraining device: {device}")
    print(f"Data sources:")
    for d in data_dirs:
        if d.exists():
            img_count = len(list((d / "images").glob("*.png")))
            print(f"  {d.name}: {img_count} samples")

    # Load combined dataset
    print("\nLoading combined dataset...")
    dataset = CombinedBubbleDataset(data_dirs, augment=True)
    print(f"Total samples: {len(dataset)}")

    # Weighted sampling
    sampler = WeightedRandomSampler(
        weights=dataset.weights,
        num_samples=len(dataset),
        replacement=True
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0
    )

    # Validation split (20%)
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

    # Use Focal Loss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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

        # Print summary
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

        # LR scheduling
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
            }, output_dir / "small_unet_combined_trained.pt")
            print(f"  ✓ Saved best model (Dice: {val_dice:.4f})")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {patience} epochs)")
            break

    # Save history
    history_path = output_dir / "small_unet_combined_trained_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print(" Training Complete")
    print("=" * 80)
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Model saved to: {output_dir / 'small_unet_combined_trained.pt'}")
    print(f"Training history: {history_path}")
    print(f"\nDataset summary:")
    print(f"  Manual annotations: 6,600 samples")
    print(f"  AIH_Bubbles3 (auto): 1,274 samples")
    print(f"  Total: {len(dataset)} samples")
    print("=" * 80)

    return best_dice

if __name__ == "__main__":
    train_combined()
