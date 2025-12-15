#!/usr/bin/env python3
from __future__ import annotations
"""
Dropdown UI Generator (Python/OpenCV)
Creates a canvas with 40 visually diverse dropdowns and exports full metadata.

This version delegates drawing + validation to utils/dropdown_ui_shared.py
so that generation matches the validation schema and visuals across the codebase.

Usage:
  python scripts/ui/dropdown_ui_generator.py --out output --validate
"""
import argparse
import json
from pathlib import Path
import cv2
from utils.dropdown_ui_shared import render_grid, validate_meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Dropdown UI Generator")
    ap.add_argument("--out", type=str, default="output")
    ap.add_argument("--canvas-w", type=int, default=1600)
    ap.add_argument("--canvas-h", type=int, default=1000)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--expanded-prob", type=float, default=0.6)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    canvas, meta_payload = render_grid(
        canvas_w=args.canvas_w,
        canvas_h=args.canvas_h,
        cols=args.cols,
        rows=args.rows,
        seed=args.seed,
        expanded_prob=args.expanded_prob,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_path = out / "dropdowns.png"
    cv2.imwrite(str(img_path), canvas)
    meta_path = out / "dropdowns_meta.json"
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    print(f"Saved: {img_path}\nSaved: {meta_path}")
    if args.validate:
        validate_meta(meta_path)


if __name__ == "__main__":
    main()

