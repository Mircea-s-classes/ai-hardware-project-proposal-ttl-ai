import cv2
import numpy as np
import coremltools as ct
from pathlib import Path

class BubbleCoreMLModel:
    """
    CoreML-based bubble detection with Neural Engine acceleration.
    Drop-in replacement for BubbleCNNModel with identical API.
    """

    def __init__(self, mlpackage_path, min_diam_px=5):
        """
        Args:
            mlpackage_path: Path to .mlpackage file (FP16 recommended)
            min_diam_px: Minimum bubble diameter to detect
        """
        self.min_diam_px = min_diam_px

        # Load CoreML model
        mlpackage_path = Path(mlpackage_path)
        if not mlpackage_path.exists():
            raise FileNotFoundError(f"CoreML model not found: {mlpackage_path}")

        print(f"Loading CoreML model: {mlpackage_path.name}")
        self.model = ct.models.MLModel(str(mlpackage_path))

        # Warm up Neural Engine (first inference triggers compilation)
        print("Warming up Neural Engine...")
        dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
        _ = self.model.predict({"input": dummy})
        print("✓ CoreML model ready!")

        # For compatibility with BubbleCNNModel API
        self.device = "Neural Engine (CoreML)"

    def predict(self, frame_bgr):
        """
        Predict bubble mask from BGR frame.
        Same API as BubbleCNNModel.predict()

        Args:
            frame_bgr: HxWx3 uint8 BGR image (OpenCV format)

        Returns:
            mask: HxW uint8 binary mask (0/255), original resolution
            bubbles: list of dict with keys: x, y, w, h, diam_px
        """
        h, w, _ = frame_bgr.shape

        # Preprocess (same as BubbleCNNModel lines 27-31)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_small = cv2.resize(frame_rgb, (256, 256))

        # Normalize to [0, 1] and convert to CHW format
        x = frame_small.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # HWC → CHW
        x = np.expand_dims(x, 0)  # Add batch: (1, 3, 256, 256)

        # CoreML inference (Neural Engine accelerated)
        output = self.model.predict({"input": x})

        # Get logits
        output_name = list(output.keys())[0]
        logits = output[output_name][0, 0]  # Shape: (256, 256)

        # Apply sigmoid (same as BubbleCNNModel line 36)
        probs = 1.0 / (1.0 + np.exp(-logits))

        # Threshold at 0.5 (same as line 38)
        pred_small = (probs > 0.5).astype(np.uint8) * 255

        # Resize back to original (same as line 41)
        mask = cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Connected components (same as BubbleCNNModel lines 44-60)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        bubbles = []
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, bw, bh, area = stats[i]
            diam_px = max(bw, bh)

            if diam_px < self.min_diam_px:
                continue

            bubbles.append({
                "x": int(x),
                "y": int(y),
                "w": int(bw),
                "h": int(bh),
                "diam_px": float(diam_px),
            })

        return mask, bubbles
