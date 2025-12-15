#!/usr/bin/env python3
from __future__ import annotations
"""
Generate Dropdown Dataset (no validation)

One command to generate dropdown visuals and metadata in a consistent format.

Modes:
 - grid: renders up to 40 diverse dropdowns on a canvas and saves PNG + JSON
 - transitions: renders an ideal dropdown interaction sequence and saves frames + JSON

Examples:
  python scripts/ui/generate_dropdown_dataset.py grid --out output/grid --seed 42
  python scripts/ui/generate_dropdown_dataset.py transitions --out output/transitions --seed 123
"""
import argparse
import os
import sys
import json
from pathlib import Path
import cv2
from utils.dropdown_ui_shared import render_grid, render_transition_sequence, Style, State, draw_dropdown


def run_grid(args: argparse.Namespace) -> None:
    canvas, payload = render_grid(
        canvas_w=args.canvas_w,
        canvas_h=args.canvas_h,
        cols=args.cols,
        rows=args.rows,
        seed=args.seed,
        expanded_prob=args.expanded_prob,
        limit=40,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_path = out / "dropdowns.png"
    json_path = out / "dropdowns_meta.json"
    cv2.imwrite(str(img_path), canvas)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {img_path}\nSaved: {json_path}")


def run_transitions(args: argparse.Namespace) -> None:
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
    frame_files = []
    for i, img in enumerate(frames):
        fp = out / f"{args.prefix}_{i:03d}.png"
        cv2.imwrite(str(fp), img)
        frame_files.append(fp.name)
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
    print(f"Saved {len(frames)} frames to: {out}\nSaved: {meta_path}")


def run_batch(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = int(max(1, args.count))
    for i in range(total):
        seed_i = (args.seed or 0) + i
        canvas, payload = render_grid(
            canvas_w=args.canvas_w,
            canvas_h=args.canvas_h,
            cols=args.cols,
            rows=args.rows,
            seed=seed_i,
            expanded_prob=args.expanded_prob,
            limit=args.limit,
        )
        img_path = out / f"dropdowns_{i:04d}.png"
        json_path = out / f"dropdowns_{i:04d}.json"
        cv2.imwrite(str(img_path), canvas)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {total} images to: {out}")


def _rand_placement(canvas_w: int, canvas_h: int, rng, min_w: int = 240, max_w: int = 420) -> tuple[int,int,int,int]:
    w = int(rng.integers(min_w, max_w + 1))
    h = int(rng.integers(30, 42))
    margin = 20
    x = int(rng.integers(margin, max(margin + 1, canvas_w - w - margin)))
    y = int(rng.integers(margin, max(margin + 1, canvas_h - (h + 6 * 34) - margin)))
    return x, y, w, h


def run_single4(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = __import__('numpy').random.default_rng(args.seed)
    count = int(max(1, args.count))
    for i in range(count):
        seed_i = (args.seed or 0) + i
        # Random placement and state
        x, y, w, h = _rand_placement(args.canvas_w, args.canvas_h, rng)
        style = Style(i)
        st = State(1.0)  # start as open for convenience
        st.total = int(rng.integers(args.total_min, args.total_max + 1))
        st.visible = int(rng.integers(args.visible_min, min(args.visible_max, st.total) + 1))
        st.row_h = int(rng.integers(args.row_h_min, args.row_h_max + 1))
        st.scroll = 0
        # ensure texts length matches total to avoid IndexError in draw_dropdown
        st.texts = [f"Item {k}" for k in range(st.total)]

        # Build four frames
        # 1) closed
        closed = __import__('numpy').ones((args.canvas_h, args.canvas_w, 3), dtype=__import__('numpy').uint8) * 245
        st_closed = State(0.0)
        st_closed.header_text = st.header_text
        meta_closed = draw_dropdown(closed, style, st_closed, x, y, w, h)
        # ideal click = header center
        hb = meta_closed['headerBox']; click_open = {"x": hb['x'] + hb['w']//2, "y": hb['y'] + hb['h']//2}

        # 2) open_top
        open_img = __import__('numpy').ones((args.canvas_h, args.canvas_w, 3), dtype=__import__('numpy').uint8) * 245
        st.scroll = 0
        meta_open = draw_dropdown(open_img, style, st, x, y, w, h)
        pb = meta_open['panelBox']
        if pb:
            # ideal scroll click: 80% panel height center
            click_scroll_top = {"x": pb['x'] + pb['w']//2, "y": int(pb['y'] + pb['h']*0.8)}
        else:
            click_scroll_top = click_open

        # 3) scrolled (mid)
        scrolled_img = __import__('numpy').ones((args.canvas_h, args.canvas_w, 3), dtype=__import__('numpy').uint8) * 245
        max_scroll = max(0, st.total - st.visible)
        mid_scroll = max(1, max_scroll//2) if max_scroll > 0 else 0
        st.scroll = mid_scroll
        meta_scrolled = draw_dropdown(scrolled_img, style, st, x, y, w, h)
        pb_s = meta_scrolled['panelBox']
        if pb_s:
            click_scroll_mid = {"x": pb_s['x'] + pb_s['w']//2, "y": int(pb_s['y'] + pb_s['h']*0.8)}
        else:
            click_scroll_mid = click_open

        # 4) selected (bottom)
        selected_img = __import__('numpy').ones((args.canvas_h, args.canvas_w, 3), dtype=__import__('numpy').uint8) * 245
        st.scroll = max_scroll
        # select bottom visible option
        st.selected = min(st.total - 1, st.scroll + st.visible - 1)
        meta_selected = draw_dropdown(selected_img, style, st, x, y, w, h)
        # find selected option center
        select_center = click_open
        for opt in meta_selected.get('options', []):
            if opt.get('index') == st.selected and opt.get('center'):
                select_center = opt['center']
                break

        # Save frames
        base = f"dd_{i:04d}"
        paths = {
            'closed': out / f"{base}_closed.png",
            'open': out / f"{base}_open.png",
            'scrolled': out / f"{base}_scrolled.png",
            'selected': out / f"{base}_selected.png",
        }
        cv2.imwrite(str(paths['closed']), closed)
        cv2.imwrite(str(paths['open']), open_img)
        cv2.imwrite(str(paths['scrolled']), scrolled_img)
        cv2.imwrite(str(paths['selected']), selected_img)

        # Helper to dump per-frame json
        def dump_json(suffix: str, meta: dict, action: str, click: dict, scroll_idx: int | None, selected_idx: int | None):
            payload = {
                'action': action,
                'ideal_click': click,
                'scrollIndex': scroll_idx,
                'selectedIndex': selected_idx,
                'headerBox': meta.get('headerBox'),
                'panelBox': meta.get('panelBox'),
                'options': meta.get('options'),
            }
            (out / f"{base}_{suffix}.json").write_text(json.dumps(payload, indent=2), encoding='utf-8')

        dump_json('closed', meta_closed, 'open', click_open, 0, None)
        dump_json('open', meta_open, 'scroll', click_scroll_top, 0, None)
        dump_json('scrolled', meta_scrolled, 'scroll', click_scroll_mid, mid_scroll, None)
        dump_json('selected', meta_selected, 'select', select_center, max_scroll, st.selected)

    print(f"Saved {count*4} images with per-frame JSON to: {out}")


def main() -> None:
    # Ensure repo root on sys.path for absolute imports when run from any CWD
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser(description="Generate dropdown dataset (no validation)")
    sub = ap.add_subparsers(dest="mode", required=True)

    ap_grid = sub.add_parser("grid", help="Render 40 dropdowns on a canvas")
    ap_grid.add_argument("--out", type=str, default="output/grid")
    ap_grid.add_argument("--canvas-w", type=int, default=1600)
    ap_grid.add_argument("--canvas-h", type=int, default=1000)
    ap_grid.add_argument("--cols", type=int, default=5)
    ap_grid.add_argument("--rows", type=int, default=8)
    ap_grid.add_argument("--seed", type=int, default=42)
    ap_grid.add_argument("--expanded-prob", type=float, default=0.6)
    ap_grid.set_defaults(func=run_grid)

    ap_tr = sub.add_parser("transitions", help="Render an ideal sequence of dropdown states")
    ap_tr.add_argument("--out", type=str, default="output/transitions")
    ap_tr.add_argument("--canvas-w", type=int, default=600)
    ap_tr.add_argument("--canvas-h", type=int, default=300)
    ap_tr.add_argument("--seed", type=int, default=123)
    ap_tr.add_argument("--total", type=int, default=15)
    ap_tr.add_argument("--visible", type=int, default=5)
    ap_tr.add_argument("--row-h", type=int, default=28)
    ap_tr.add_argument("--prefix", type=str, default="step")
    ap_tr.set_defaults(func=run_transitions)

    ap_batch = sub.add_parser("batch", help="Generate many grid canvases (bulk)")
    ap_batch.add_argument("--out", type=str, default="output/bulk")
    ap_batch.add_argument("--count", type=int, default=50, help="How many images to generate")
    ap_batch.add_argument("--canvas-w", type=int, default=1600)
    ap_batch.add_argument("--canvas-h", type=int, default=1000)
    ap_batch.add_argument("--cols", type=int, default=5)
    ap_batch.add_argument("--rows", type=int, default=8)
    ap_batch.add_argument("--limit", type=int, default=40, help="Max dropdowns per canvas")
    ap_batch.add_argument("--seed", type=int, default=42)
    ap_batch.add_argument("--expanded-prob", type=float, default=0.6)
    ap_batch.set_defaults(func=run_batch)

    ap_s4 = sub.add_parser("single4", help="Per dropdown: 4 frames (closed/open/scrolled/selected) each with own JSON")
    ap_s4.add_argument("--out", type=str, default="output/single4")
    ap_s4.add_argument("--count", type=int, default=500, help="How many dropdowns (x4 frames)")
    ap_s4.add_argument("--seed", type=int, default=42)
    ap_s4.add_argument("--canvas-w", type=int, default=1600)
    ap_s4.add_argument("--canvas-h", type=int, default=1000)
    ap_s4.add_argument("--visible-min", type=int, default=3)
    ap_s4.add_argument("--visible-max", type=int, default=6)
    ap_s4.add_argument("--total-min", type=int, default=10)
    ap_s4.add_argument("--total-max", type=int, default=30)
    ap_s4.add_argument("--row-h-min", type=int, default=24)
    ap_s4.add_argument("--row-h-max", type=int, default=34)
    ap_s4.set_defaults(func=run_single4)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
