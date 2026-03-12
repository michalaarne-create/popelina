# -*- coding: utf-8 -*-
"""
Standalone tester – nie importuje rating.py, sprawdza TYLKO logikę detekcji.
"""
import sys, math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def detect(img_path: str) -> dict:
    out = {"marker_shape": "none", "marker_kind": "none", "marker_conf": 0.0}
    pil = Image.open(img_path).convert("RGB")
    roi = np.array(pil)
    rh, rw = roi.shape[:2]
    min_dim = min(rh, rw)
    print(f"ROI size: {rw}x{rh}, min_dim={min_dim}")

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive blockSize — kluczowa poprawka!
    block_size = max(7, min(21, (min_dim // 4) | 1))
    print(f"blockSize użyty w adaptiveThreshold: {block_size}")

    # Hough
    hough_circle_found = False
    hough_conf = 0.65
    min_r = max(2, int(min_dim * 0.08))
    max_r = max(min_r + 2, int(min_dim * 0.50))
    min_dist = max(4, int(min_dim * 0.15))
    for p2 in (10, 12, 15, 20):
        c = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                             minDist=min_dist, param1=80, param2=p2,
                             minRadius=min_r, maxRadius=max_r)
        if c is not None and len(c) > 0:
            hough_circle_found = True
            hough_conf = {10: 0.72, 12: 0.75, 15: 0.80, 20: 0.85}[p2]
            print(f"  Hough WYKRYTO kółko p2={p2}, conf={hough_conf}")
            break
    if not hough_circle_found:
        print("  Hough: brak wykrytych kółek")

    # Kontury
    bin_inv = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block_size, 4)
    bin_nrm = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, block_size, 4)

    best = None
    best_score = 0.0
    for candidate in (bin_inv, bin_nrm):
        contours, _ = cv2.findContours(candidate, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        roi_area = float(rh * rw)
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 8.0 or area > roi_area * 0.85: continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 4 or h < 4: continue
            ratio = float(w) / max(1.0, float(h))
            if ratio < 0.3 or ratio > 3.0: continue
            peri = float(cv2.arcLength(c, True))
            if peri <= 0.0: continue
            circularity = float((4.0 * math.pi * area) / (peri * peri + 1e-6))
            extent = float(area / (w * h + 1e-6))
            circle_score = max(0.0, min(1.0, (circularity - 0.35) / 0.5))
            square_score = max(0.0, min(1.0, 1.1 - abs(ratio - 1.0) / 0.5))
            square_score = min(square_score, max(0.0, min(1.0, (extent - 0.5) / 0.4)))
            if hough_circle_found:
                circle_score = max(circle_score, 0.85)
            m_score = max(circle_score, square_score)
            if m_score > best_score:
                best_score = m_score
                best = {"circle_like": circle_score, "square_like": square_score, "score": m_score}

    if not best:
        if hough_circle_found:
            out.update({"marker_shape": "circle", "marker_kind": "radio",
                        "marker_conf": round(hough_conf, 4)})
        return out

    conf = float(best["score"])
    if conf < 0.30 and not hough_circle_found:
        return out

    if hough_circle_found or best["circle_like"] >= best["square_like"]:
        out = {"marker_shape": "circle", "marker_kind": "radio",
               "marker_conf": round(max(conf, hough_conf if hough_circle_found else 0.0), 4)}
    else:
        out = {"marker_shape": "square", "marker_kind": "checkbox",
               "marker_conf": round(conf, 4)}
    return out


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else str(
        Path("C:/Users/user/.gemini/antigravity/brain/ec6d7192-2bec-46de-b143-36bf11e6dee3/uploaded_image_1773329224075.png")
    )
    print(f"\nTestowanie: {img}")
    result = detect(img)
    print(f"\n=== WYNIK ===")
    print(f"  marker_shape : {result['marker_shape']}")
    print(f"  marker_kind  : {result['marker_kind']}")
    print(f"  marker_conf  : {result['marker_conf']}")
    expected_ok = result['marker_kind'] == 'radio'
    print(f"\n  {'✓ POPRAWNIE wykryto kółko (radio)!' if expected_ok else '✗ BŁĄD – nie wykryto radio!'}")
