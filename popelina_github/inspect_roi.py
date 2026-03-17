# -*- coding: utf-8 -*-
import sys, math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def analyze_roi(img_path: str):
    pil = Image.open(img_path).convert("RGB")
    roi = np.array(pil)
    rh, rw = roi.shape[:2]
    min_dim = min(rh, rw)
    print(f"ROI size: {rw}x{rh}, min_dim={min_dim}")

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive blockSize
    block_size = max(7, min(21, (min_dim // 4) | 1))
    print(f"blockSize: {block_size}")

    # Hough (simulated)
    hough_circle_found = False
    hough_conf = 0.72
    for p2 in (10, 12, 15, 20):
        c = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                             minDist=max(4, int(min_dim * 0.15)), param1=80, param2=p2,
                             minRadius=max(2, int(min_dim * 0.08)), maxRadius=max(8, int(min_dim * 0.50)))
        if c is not None and len(c) > 0:
            hough_circle_found = True
            break

    # Thresholds
    bin_inv = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block_size, 4)
    bin_nrm = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, block_size, 4)

    best_c_overall = 0.0
    best_s_overall = 0.0

    for name, candidate in [("INV", bin_inv), ("NRM", bin_nrm)]:
        print(f"\n--- Analysis of {name} threshold ---")
        contours, _ = cv2.findContours(candidate, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = float(rh * rw)
        for i, c in enumerate(contours):
            area = float(cv2.contourArea(c))
            if area < 8.0 or area > roi_area * 0.95: continue
            
            x, y, w, h = cv2.boundingRect(c)
            if w < 4 or h < 4: continue
            
            ratio = float(w) / max(1.0, float(h))
            peri = float(cv2.arcLength(c, True))
            if peri <= 0.0: continue
            
            circularity = float((12.566 * area) / (peri * peri + 1e-6))
            extent = float(area / (w * h + 1e-6))
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            vertices = len(approx)

            # LATEST LOGIC from rating.py
            c_score = max(0.0, min(1.0, (circularity - 0.65) / 0.30))
            s_score = max(0.0, min(1.0, 1.1 - abs(ratio - 1.0) / 0.5))
            s_score = min(s_score, max(0.0, min(1.0, (extent - 0.40) / 0.50)))
            
            if extent > 0.81: c_score *= 0.4
            if vertices == 4:
                s_score = max(s_score, 0.90)
                c_score *= 0.3
            
            if hough_circle_found and circularity > 0.6:
                h_boost = 0.25 + (hough_conf - 0.7) * 0.5
                c_score = max(c_score, min(0.95, c_score + h_boost))

            best_c_overall = max(best_c_overall, c_score)
            best_s_overall = max(best_s_overall, s_score)

            print(f"Contour #{i}: vertices={vertices}, circ={circularity:.3f}, ext={extent:.3f} -> C:{c_score:.2f} S:{s_score:.2f}")

    # Final decision logic (Bias on square)
    if best_s_overall > 0.70 and best_s_overall >= (best_c_overall - 0.20):
        best_final = "SQUARE"
    elif best_c_overall > best_s_overall:
        best_final = "CIRCLE"
    else:
        best_final = "SQUARE"

    print(f"\nFINAL DECISION: {best_final} (S:{best_s_overall:.2f} C:{best_c_overall:.2f})")

if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "e:/BOT ANK/bot/moje_AI/yolov8/FULL BOT_BACKUP_2026-02-05_000842/popelina_github/data/screen/ROI_current/05_radio_circle_0.96.png"
    analyze_roi(img)
