"""Simple OCR test: draw red rectangular boxes around detected text.

Usage:
    python scripts/test/test_ocr_boxes.py [--image PATH] [--lang CODE] [--show]

Notes:
    - Uses EasyOCR via utils/ocr_features helpers.
    - Saves an annotated image next to the input with "_ocr" suffix.
    - If --show is passed, opens a preview window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np


# Resolve project root so we can import utils.ocr_features
PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "envs").exists()),
    Path(__file__).resolve().parents[2],  # fallback: repo root
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.ocr_features import (  # noqa: E402
    OCRNotAvailableError,
    create_easyocr_reader,
    compute_ocr_state,
)


def _auto_find_image(default_dir: Path) -> Path | None:
    """Pick an image file near this script if none provided."""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    candidates = sorted([p for p in default_dir.iterdir() if p.suffix.lower() in exts])
    return candidates[0] if candidates else None


def draw_rectangles(frame: np.ndarray, boxes: List[Dict[str, Any]]) -> np.ndarray:
    """Draw red rectangular bounding boxes from EasyOCR polygon boxes."""
    out = frame.copy()
    for item in boxes:
        poly = np.array(item.get("box", []), dtype=np.float32).reshape(-1, 2)
        if poly.size == 0:
            continue
        xs = np.clip(poly[:, 0], 0, frame.shape[1] - 1)
        ys = np.clip(poly[:, 1], 0, frame.shape[0] - 1)
        x_min = int(xs.min())
        y_min = int(ys.min())
        x_max = int(xs.max())
        y_max = int(ys.max())
        if x_max <= x_min or y_max <= y_min:
            continue
        # Red rectangle (BGR: (0,0,255))
        cv2.rectangle(out, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple OCR box visualizer")
    ap.add_argument("--image", type=str, default=None, help="Input image path (default: auto-detect near script)")
    ap.add_argument("--lang", type=str, default="en", help="OCR language code (default: en)")
    ap.add_argument("--show", action="store_true", help="Show preview window of the annotated image")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    img_path = Path(args.image) if args.image else _auto_find_image(script_dir)
    if img_path is None or not img_path.exists():
        raise FileNotFoundError(
            "No input image found. Place a screenshot next to this script or pass --image PATH."
        )

    frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    try:
        reader = create_easyocr_reader(lang=args.lang)
    except OCRNotAvailableError as exc:
        raise ImportError(
            "easyocr is required. Install with `pip install easyocr` (models cache at models/easyocr_cache)."
        ) from exc

    # Run OCR and get polygons
    _, boxes = compute_ocr_state(reader, frame, return_boxes=True)
    print(f"Detected {len(boxes)} text regions.")

    annotated = draw_rectangles(frame, boxes)

    out_path = img_path.with_name(img_path.stem + "_ocr" + img_path.suffix)
    if not cv2.imwrite(str(out_path), annotated):
        raise RuntimeError(f"Failed to write output image: {out_path}")
    print(f"Saved annotated image to: {out_path}")

    if args.show:
        cv2.imshow("OCR Boxes", annotated)
        print("Press any key in the window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
