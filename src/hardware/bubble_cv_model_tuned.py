# Auto-generated tuned CV model
from pathlib import Path
import cv2
import numpy as np

class BubbleCVModel:
    """Tuned CV model for real bubble detection"""
    def __init__(self, min_diam_px=15, max_diam_px=250, min_circularity=0.4):
        self.min_diam_px = min_diam_px
        self.max_diam_px = max_diam_px
        self.min_circularity = min_circularity

    def predict(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert: bubbles should be white (255), background should be black (0)
        th = cv2.bitwise_not(th)

        kernel = np.ones((5, 5), np.uint8)
        th_clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
        th_clean = cv2.morphologyEx(th_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th_clean, connectivity=8)

        # Create output mask with ONLY filtered bubbles (not full threshold)
        h_frame, w_frame = frame_bgr.shape[:2]
        output_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)

        bubbles = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            diam_px = max(w, h)

            if diam_px < self.min_diam_px or diam_px > self.max_diam_px:
                continue

            aspect_ratio = float(w) / float(h) if h > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            mask_i = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < self.min_circularity:
                        continue

            margin = 5
            if x < margin or y < margin or (x + w) > (w_frame - margin) or (y + h) > (h_frame - margin):
                continue

            # Add this bubble to the output mask
            output_mask[labels == i] = 255

            bubbles.append({
                "x": int(x), "y": int(y),
                "w": int(w), "h": int(h),
                "diam_px": float(diam_px),
            })

        return output_mask, bubbles
