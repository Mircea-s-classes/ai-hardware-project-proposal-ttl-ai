import cv2
import numpy as np
import math

def fill_holes(bin_u8: np.ndarray) -> np.ndarray:
    h, w = bin_u8.shape
    ff = bin_u8.copy()
    tmp = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, tmp, (0, 0), 255)
    inv = cv2.bitwise_not(ff)
    return bin_u8 | inv

def postprocess_bubbles(
    probs: np.ndarray,
    roi_u8: np.ndarray,
    thr: float = 0.20,

    # junk suppression
    static_mask_u8: np.ndarray | None = None,
    min_dist_to_edge: int = 0,

    # component filters (pixel count of thresholded component)
    min_area: int = 30,
    max_area: int = 6000,
    min_solidity: float = 0.75,
    min_ellipticity: float = 0.25,

    # shape filters for text/ticks
    min_circularity: float = 0.25,
    max_aspect_ratio: float = 6.0,

    # NEW: ellipse size filters (much better for “small printed circles”)
    min_major_px: float = 0.0,          # reject if major axis < this
    min_minor_px: float = 0.0,          # reject if minor axis < this
    min_ellipse_area: float = 0.0,      # reject if (pi*(MA/2)*(ma/2)) < this

    # morphology
    close_iters: int = 2,
    ellipse_refill: bool = True,
):
    """
    probs: HxW float32 in [0,1]
    roi_u8: HxW uint8 0/255 (syringe ROI)
    returns: HxW uint8 0/255 bubble mask
    """
    roi = (roi_u8 > 127).astype(np.uint8) * 255

    # low threshold for recall
    raw = (probs > thr).astype(np.uint8) * 255
    raw = cv2.bitwise_and(raw, roi)

    # subtract static junk
    if static_mask_u8 is not None:
        sj = (static_mask_u8 > 127).astype(np.uint8) * 255
        raw = cv2.bitwise_and(raw, cv2.bitwise_not(sj))

    # distance-to-edge gating
    if min_dist_to_edge > 0:
        dist = cv2.distanceTransform((roi > 0).astype(np.uint8), cv2.DIST_L2, 3)
        raw[dist < float(min_dist_to_edge)] = 0

    # merge splits + fill
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k5, iterations=close_iters)
    raw = fill_holes(raw)

    m01 = (raw > 127).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m01, connectivity=8)
    out = np.zeros_like(m01, dtype=np.uint8)

    for k in range(1, num):
        x, y, w, h, area = stats[k]
        if area < min_area or area > max_area:
            continue

        # reject very elongated components
        aspect = (max(w, h) / max(1, min(w, h)))
        if aspect > max_aspect_ratio:
            continue

        comp = (labels == k).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts or len(cnts[0]) < 20:
            continue
        cnt = cnts[0]

        # solidity
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull)) + 1e-6
        solidity = float(area) / hull_area
        if solidity < min_solidity:
            continue

        # circularity
        perim = float(cv2.arcLength(cnt, True)) + 1e-6
        circularity = (4.0 * math.pi * float(area)) / (perim * perim)
        if circularity < min_circularity:
            continue

        # ellipse fit + oval-ish
        try:
            (cx, cy), (MA, ma), ang = cv2.fitEllipse(cnt)
        except cv2.error:
            continue

        major = float(max(MA, ma))
        minor = float(min(MA, ma))
        if major <= 1:
            continue

        ellipticity = minor / major
        if ellipticity < min_ellipticity:
            continue

        # NEW: minimum ellipse axis sizes
        if min_major_px > 0 and major < min_major_px:
            continue
        if min_minor_px > 0 and minor < min_minor_px:
            continue

        # NEW: minimum ellipse area (more intuitive than component pixel area)
        if min_ellipse_area > 0:
            ellipse_area = math.pi * (major / 2.0) * (minor / 2.0)
            if ellipse_area < min_ellipse_area:
                continue

        if ellipse_refill:
            cv2.ellipse(out, ((cx, cy), (MA, ma), ang), 255, thickness=-1)
        else:
            out[labels == k] = 255

    out_u8 = (out > 0).astype(np.uint8) * 255
    out_u8 = cv2.morphologyEx(out_u8, cv2.MORPH_CLOSE, k5, iterations=1)
    out_u8 = fill_holes(out_u8)
    return out_u8
