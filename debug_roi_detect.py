# -*- coding: utf-8 -*-
"""
debug_roi_detect.py
-------------------
Skrypt diagnostyczny: testuje detekcję markerów (kółko/kwadrat) krok po kroku
na obrazku ROI podanym jako argument lub na ostatnim pliku w data/screen/ROI/.

Użycie:
    python debug_roi_detect.py [ścieżka_do_obrazka_roi.png]
"""

import sys
import os
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ── output dir ──────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "debug_roi_steps"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(name: str, img):
    path = OUT_DIR / name
    cv2.imwrite(str(path), img)
    print(f"  [SAVE] {path}")


def analyze(roi_img_path: str):
    print(f"\n{'='*60}")
    print(f"Analiza: {roi_img_path}")
    print('='*60)

    # Wczytaj obrazek ROI jako RGB (PIL) → numpy
    pil = Image.open(roi_img_path).convert("RGB")
    roi = np.array(pil)
    h, w = roi.shape[:2]
    print(f"ROI size: {w}x{h} px")

    # Zapisz oryginał
    save("00_original.png", cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

    # Gray + Blur
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    save("01_gray.png", gray)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    save("02_blur.png", blur)

    # Adaptive threshold (wersja INV i normalna)
    for C_val in [2, 4, 7, 10]:
        bin_inv = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, C_val
        )
        bin_nrm = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, C_val
        )
        save(f"03_thresh_inv_C{C_val}.png", bin_inv)
        save(f"03_thresh_nrm_C{C_val}.png", bin_nrm)

    # Hough Circles – wiele wariantów parametrów
    print("\n--- Hough Circles ---")
    hough_vis = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    min_dim = min(h, w)
    found_any = False
    for dp in [1.0, 1.2, 1.5]:
        for p2 in [8, 10, 12, 15, 20]:
            for minR in [2, 3, 4]:
                maxR = max(8, int(min_dim * 0.50))
                circles = cv2.HoughCircles(
                    blur,
                    cv2.HOUGH_GRADIENT,
                    dp=dp,
                    minDist=max(4, int(min_dim * 0.2)),
                    param1=80,
                    param2=p2,
                    minRadius=minR,
                    maxRadius=maxR,
                )
                if circles is not None and len(circles) > 0:
                    print(f"  WYKRYTO kółka! dp={dp}, p2={p2}, minR={minR}: {circles[0][:3]}")
                    found_any = True
                    vis = cv2.cvtColor(blur.copy(), cv2.COLOR_GRAY2BGR)
                    for cx, cy, r in np.uint16(np.around(circles[0])):
                        cv2.circle(vis, (cx, cy), r, (0, 255, 0), 1)
                        cv2.circle(vis, (cx, cy), 2, (0, 0, 255), 2)
                    save(f"04_hough_dp{str(dp).replace('.','')}_p2{p2}_minR{minR}.png", vis)
    if not found_any:
        print("  BRAK wykrytych kółek w żadnej konfiguracji Hougha!")

    # Kontury na wszystkich wariantach progowania
    print("\n--- Kontury ---")
    for C_val in [2, 4, 7]:
        for inv in [True, False]:
            thr_type = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
            thr_name = "inv" if inv else "nrm"
            bw = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thr_type, 11, C_val
            )
            contours, _ = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            vis = cv2.cvtColor(bw.copy(), cv2.COLOR_GRAY2BGR)
            roi_area = float(h * w)
            n_good = 0
            for c in contours:
                area = float(cv2.contourArea(c))
                if area < 8 or area > roi_area * 0.9:
                    continue
                cw, ch = cv2.boundingRect(c)[2], cv2.boundingRect(c)[3]
                if cw < 4 or ch < 4:
                    continue
                ratio = cw / max(1.0, ch)
                if ratio < 0.25 or ratio > 3.5:
                    continue
                peri = float(cv2.arcLength(c, True))
                if peri <= 0:
                    continue
                circularity = (4.0 * math.pi * area) / (peri * peri + 1e-6)
                extent = area / (cw * ch + 1e-6)
                circle_score = max(0.0, min(1.0, (circularity - 0.35) / 0.5))
                square_score = min(
                    max(0.0, min(1.0, 1.1 - abs(ratio - 1.0) / 0.5)),
                    max(0.0, min(1.0, (extent - 0.5) / 0.4))
                )
                ms = max(circle_score, square_score)
                color = (0, 255, 0) if ms > 0.5 else (0, 128, 255)
                cv2.drawContours(vis, [c], -1, color, 1)
                x, y = cv2.boundingRect(c)[:2]
                label = f"c{circularity:.2f} s{square_score:.2f}"
                cv2.putText(vis, label, (x, max(0, y-2)), cv2.FONT_HERSHEY_PLAIN, 0.6, color, 1)
                n_good += 1
                if ms > 0.3:
                    kind = "CIRCLE" if circle_score >= square_score else "SQUARE"
                    print(f"  C={C_val} {thr_name}: {kind} area={area:.0f} circ={circularity:.3f} ext={extent:.3f} ratio={ratio:.2f} score={ms:.3f}")
            save(f"05_contours_C{C_val}_{thr_name}.png", vis)

    print(f"\nWszystkie etapy zapisane w: {OUT_DIR}")
    print("Otwórz folder i sprawdź obrazki, aby zobaczyć co algorytm 'widzi'.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Automatycznie użyj ostatniego pliku ROI
        roi_dir = Path(__file__).parent / "data" / "screen" / "ROI"
        candidates = sorted(roi_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        # Jeśli user wgrał obrazek do .gemini, użyj go
        gemini_img = Path(r"C:/Users/user/.gemini/antigravity/brain/ec6d7192-2bec-46de-b143-36bf11e6dee3/uploaded_image_1773329224075.png")
        if gemini_img.exists():
            img_path = str(gemini_img)
            print(f"[INFO] Używam wgranego obrazka: {img_path}")
        elif candidates:
            img_path = str(candidates[0])
            print(f"[INFO] Używam ostatniego ROI: {img_path}")
        else:
            print("[ERROR] Podaj ścieżkę do obrazka jako argument!")
            sys.exit(1)

    analyze(img_path)
