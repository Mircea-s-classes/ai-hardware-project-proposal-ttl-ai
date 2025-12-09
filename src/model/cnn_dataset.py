from pathlib import Path
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset

class BubbleSegDataset(Dataset):
    def __init__(self, root, split="train"):
        root = Path(root)
        self.img_dir = root / "images"
        self.msk_dir = root / "masks"

        all_imgs = sorted(self.img_dir.glob("*.png"))
        n = len(all_imgs)
        split_idx = int(0.8 * n)  # 80/20 train/val

        if split == "train":
            self.img_paths = all_imgs[:split_idx]
        else:
            self.img_paths = all_imgs[split_idx:]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        msk_path = self.msk_dir / img_path.name

        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path).convert("L")

        img = torch.from_numpy(np.array(img)).float() / 255.0   # HWC 0–1
        msk = torch.from_numpy(np.array(msk)).float() / 255.0   # HWC 0–1

        # HWC → CHW
        img = img.permute(2, 0, 1)
        msk = msk.unsqueeze(0)  # [1, H, W]

        return img, msk
