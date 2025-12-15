import cv2
import numpy as np
import torch
from pathlib import Path
from small_unet import SmallUNet

class BubbleCNNModel:
    def __init__(self, ckpt_path, min_diam_px=5, device=None):
        self.min_diam_px = min_diam_px
        # Prioritize: MPS (M1) > CUDA (NVIDIA) > CPU
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # load model
        self.model = SmallUNet().to(self.device)
        state = torch.load(Path(ckpt_path), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, frame_bgr):
        """
        frame_bgr: HxWx3 uint8 BGR
        returns:
          mask      - uint8 0/255 mask in original resolution
          bubbles   - list of dicts like BubbleCVModel
        """
        h, w, _ = frame_bgr.shape

        # BGR->RGB, resize to 256x256 like training
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_small = cv2.resize(frame_rgb, (256, 256))

        x = frame_small.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)[0,0].cpu().numpy()

        pred_small = (probs > 0.5).astype(np.uint8) * 255

        # back to original size
        mask = cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)

        # connected components same as BubbleCVModel
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        bubbles = []
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            diam_px = max(bw, bh)
            if diam_px < self.min_diam_px:
                continue

            bubbles.append({
                "x": int(x), "y": int(y),
                "w": int(bw), "h": int(bh),
                "diam_px": float(diam_px),
            })

        return mask, bubbles
