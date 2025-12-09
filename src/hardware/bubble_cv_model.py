import cv2
import numpy as np

class BubbleCVModel:
    def __init__(self, min_diam_px=5):
        self.min_diam_px = min_diam_px

    def predict(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        _, th = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = np.ones((3, 3), np.uint8)
        th_clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            th_clean, connectivity=8
        )

        bubbles = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            diam_px = max(w, h)
            if diam_px < self.min_diam_px:
                continue

            bubbles.append({
                "x": int(x), "y": int(y),
                "w": int(w), "h": int(h),
                "diam_px": float(diam_px),
            })

        return th_clean, bubbles
