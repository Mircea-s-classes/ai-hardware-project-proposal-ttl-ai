from pathlib import Path
import cv2
from bubble_cv_model import BubbleCVModel

# path relative to repo root
VIDEO_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "bubbles_synth_01.mp4"


def main():
    print("Video path:", VIDEO_PATH)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    model = BubbleCVModel(min_diam_px=5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask, bubbles = model.predict(frame)

        vis = frame.copy()
        for b in bubbles:
            x, y, w, h = b["x"], b["y"], b["w"], b["h"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)

        cv2.imshow("bubbles_vis", vis)
        cv2.imshow("mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
