#!/usr/bin/env python3
from __future__ import annotations
"""
Save Ideal Dropdown Transitions (one command)

Generates an idealized sequence of dropdown states using the same visuals
and metadata schema as validation, then saves all frames and a single meta JSON.

Usage:
  python scripts/ui/save_dropdown_transitions.py --out debug/transitions --seed 123
"""
import argparse
import os
import sys
import json
from pathlib import Path
import cv2
from utils.dropdown_ui_shared import render_transition_sequence


def main() -> None:
    # Ensure repo root on sys.path for absolute imports when run from any CWD
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser(description="Save ideal dropdown transition sequence")
    ap.add_argument("--out", type=str, default="debug/transitions", help="Output directory")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--canvas-w", type=int, default=600)
    ap.add_argument("--canvas-h", type=int, default=300)
    ap.add_argument("--total", type=int, default=15)
    ap.add_argument("--visible", type=int, default=5)
    ap.add_argument("--row-h", type=int, default=28)
    ap.add_argument("--prefix", type=str, default="step")
    args = ap.parse_args()

    frames, metas = render_transition_sequence(
        canvas_w=args.canvas_w,
        canvas_h=args.canvas_h,
        seed=args.seed,
        total=args.total,
        visible=args.visible,
        row_h=args.row_h,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Save frames and collect file refs in meta
    frame_files = []
    for i, img in enumerate(frames):
        fp = out / f"{args.prefix}_{i:03d}.png"
        cv2.imwrite(str(fp), img)
        frame_files.append(fp.name)

    # Save one meta file with references to frame files
    meta_payload = {
        "frames": frame_files,
        "sequences": metas,
        "canvas_w": args.canvas_w,
        "canvas_h": args.canvas_h,
        "seed": args.seed,
        "total": args.total,
        "visible": args.visible,
        "row_h": args.row_h,
    }
    meta_path = out / "transitions_meta.json"
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    print(f"Saved {len(frames)} frames to: {out}")
    print(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
