#!/usr/bin/env python3
"""
Lightweight wrapper for hover_bot.process_image to process a single screenshot.

This allows the main orchestrator to feed a cropped screenshot into the hover-bot
pipeline without batching through folders.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from scripts.hard_bot.hover_bot import (
        OCRNotAvailableError,
        create_paddleocr_reader,
        process_image,
        DotSequence,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Cannot import from scripts.hard_bot.hover_bot. Ensure PYTHONPATH includes project root."
    ) from exc


def save_sequences_json(sequences: list[DotSequence], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [seq.__dict__ for seq in sequences]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fallback_sequences(image_path: Path) -> list[DotSequence]:
    """Return a simple path across the image when OCR is unavailable."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError("cv2.imread returned None")
        h, w = img.shape[:2]
    except Exception:
        w, h = 1920, 1080
    cx, cy = int(w * 0.5), int(h * 0.5)
    dots = [(cx - 40, cy), (cx - 20, cy + 6), (cx, cy), (cx + 20, cy - 6), (cx + 40, cy)]
    box = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    return [DotSequence(index=0, text="fallback", confidence=1.0, box=box, dots=dots)]


def _render_annot(image_path: Path, dots: list[tuple[float, float]]):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    for x, y in dots:
        cv2.circle(img, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hover_bot on a single image.")
    parser.add_argument("--image", required=True, help="Ścieżka do obrazu wejściowego.")
    parser.add_argument(
        "--json-out",
        required=True,
        help="Ścieżka wyjściowa dla JSON (punkty i boxy).",
    )
    parser.add_argument(
        "--annot-out",
        default=None,
        help="Opcjonalna ścieżka do zapisu obrazu z adnotacjami.",
    )
    parser.add_argument("--lang", default="en", help="Język PaddleOCR (domyślnie: en).")

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Brak pliku: {image_path}")

    json_out = Path(args.json_out)
    annot_out: Optional[Path] = Path(args.annot_out) if args.annot_out else None

    fallback = False
    annotated = None
    try:
        reader = create_paddleocr_reader(lang=args.lang)
    except OCRNotAvailableError as exc:
        print(f"[WARN] PaddleOCR reader niedostępny: {exc}")
        sequences = _fallback_sequences(image_path)
        fallback = True
    else:
        try:
            sequences, annotated = process_image(image_path, reader=reader)
        except Exception as exc:
            print(f"[WARN] hover_single failed: {exc}")
            sequences = _fallback_sequences(image_path)
            fallback = True

    if not sequences:
        sequences = _fallback_sequences(image_path)
        fallback = True

    save_sequences_json(sequences, json_out)

    if annot_out:
        annot_out.parent.mkdir(parents=True, exist_ok=True)
        img_to_save = None
        if fallback:
            dots = sequences[0].dots if sequences else []
            img_to_save = _render_annot(image_path, dots)
        else:
            img_to_save = annotated
        if img_to_save is not None:
            cv2.imwrite(str(annot_out), img_to_save)


if __name__ == "__main__":
    main()
