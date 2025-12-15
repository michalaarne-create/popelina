from __future__ import annotations

"""
Generate synthetic dropdown interaction trajectories.

For each sample:
  - images (640x640): closed.png (header only), bottom.png (panel open at bottom)
  - metadata.json: geometry + actions timeline (click -> move -> scroll -> click)
Default count: 2000 samples.

Modes:
  - single   : closed.png + bottom.png + metadata.json (existing)
  - sequence : frames/frame_###.png + sequence.json (multi-frame: closed -> open -> scroll steps -> bottom)
"""

import argparse
from pathlib import Path
import json
import random
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np


IMG_W = 640
IMG_H = 640


def _clip_box(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x = max(0, min(IMG_W - 1, x))
    y = max(0, min(IMG_H - 1, y))
    w = max(1, min(IMG_W - x, w))
    h = max(1, min(IMG_H - y, h))
    return x, y, w, h


def _draw_rect(img: np.ndarray, x: int, y: int, w: int, h: int,
               color=(64, 64, 64), fill=(240, 240, 240), thick: int = 2):
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    if thick > 0:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)


def _rand_dropdown_geom(rng: random.Random) -> Dict[str, int]:
    hdr_w = rng.randint(220, 360)
    hdr_h = rng.randint(30, 40)
    hdr_x = rng.randint(20, IMG_W - hdr_w - 20)
    hdr_y = rng.randint(20, IMG_H - 240)
    return {"x": hdr_x, "y": hdr_y, "w": hdr_w, "h": hdr_h}


def _rand_panel_spec(rng: random.Random) -> Tuple[int, int]:
    opt_h = rng.randint(26, 32)
    vis_rows = rng.randint(4, 6)
    return opt_h, vis_rows


def _path_points(start: Tuple[int, int], end: Tuple[int, int], steps: int = 20,
                 jitter: int = 2) -> List[Tuple[int, int, float]]:
    sx, sy = start
    ex, ey = end
    pts: List[Tuple[int, int, float]] = []
    for i in range(steps + 1):
        t = i / max(1, steps)
        x = int(round(sx + (ex - sx) * t + np.random.randint(-jitter, jitter + 1)))
        y = int(round(sy + (ey - sy) * t + np.random.randint(-jitter, jitter + 1)))
        pts.append((int(np.clip(x, 0, IMG_W - 1)), int(np.clip(y, 0, IMG_H - 1)), float(t)))
    return pts


def _make_closed_image(hdr: Dict[str, int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    img = np.ones((IMG_H, IMG_W, 3), dtype=np.uint8) * 255
    _draw_rect(img, hdr["x"], hdr["y"], hdr["w"], hdr["h"], color=(48, 48, 48), fill=(235, 235, 235))
    meta = {
        "header_box": {"x": hdr["x"], "y": hdr["y"], "w": hdr["w"], "h": hdr["h"]},
    }
    return img, meta


def _make_bottom_image(hdr: Dict[str, int], opt_h: int, vis_rows: int,
                       total_rows: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    img = np.ones((IMG_H, IMG_W, 3), dtype=np.uint8) * 255
    _draw_rect(img, hdr["x"], hdr["y"], hdr["w"], hdr["h"], color=(48, 48, 48), fill=(235, 235, 235))
    px, py = hdr["x"], hdr["y"] + hdr["h"]
    pw, ph = hdr["w"], vis_rows * opt_h
    px, py, pw, ph = _clip_box(px, py, pw, ph)
    _draw_rect(img, px, py, pw, ph, color=(36, 36, 36), fill=(250, 250, 250), thick=1)
    option_boxes: List[Dict[str, int]] = []
    for i in range(vis_rows):
        oy = py + i * opt_h
        row_fill = (224, 224, 224) if i == vis_rows - 1 else (255, 255, 255)
        _draw_rect(img, px, oy, pw, opt_h, color=(160, 160, 160), fill=row_fill, thick=1)
        option_boxes.append({"x": px, "y": oy, "w": pw, "h": opt_h})
    fh = max(2, int(0.08 * opt_h))
    fx, fy, fw, fh = _clip_box(px + 2, py + ph - fh - 2, pw - 4, fh)
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (128, 64, 0), -1)

    meta = {
        "header_box": {"x": hdr["x"], "y": hdr["y"], "w": hdr["w"], "h": hdr["h"]},
        "panel_box": {"x": px, "y": py, "w": pw, "h": ph},
        "option_boxes": option_boxes,
        "target_index": vis_rows - 1,
        "target_box": option_boxes[-1],
        "total_rows": int(total_rows),
        "visible_rows": int(vis_rows),
        "opt_h": int(opt_h),
    }
    return img, meta


def _make_open_image(hdr: Dict[str, int], opt_h: int, vis_rows: int,
                     total_rows: int, scroll_idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Render open panel at given scroll_idx; highlight last row only if at bottom."""
    img = np.ones((IMG_H, IMG_W, 3), dtype=np.uint8) * 255
    _draw_rect(img, hdr["x"], hdr["y"], hdr["w"], hdr["h"], color=(48, 48, 48), fill=(235, 235, 235))
    px, py = hdr["x"], hdr["y"] + hdr["h"]
    pw, ph = hdr["w"], vis_rows * opt_h
    px, py, pw, ph = _clip_box(px, py, pw, ph)
    _draw_rect(img, px, py, pw, ph, color=(36, 36, 36), fill=(250, 250, 250), thick=1)

    option_boxes: List[Dict[str, int]] = []
    at_bottom = scroll_idx >= max(0, total_rows - vis_rows)
    for i in range(vis_rows):
        oy = py + i * opt_h
        row_fill = (224, 224, 224) if (at_bottom and i == vis_rows - 1) else (255, 255, 255)
        _draw_rect(img, px, oy, pw, opt_h, color=(160, 160, 160), fill=row_fill, thick=1)
        option_boxes.append({"x": px, "y": oy, "w": pw, "h": opt_h})
    if at_bottom:
        fh = max(2, int(0.08 * opt_h))
        fx, fy, fw, fh = _clip_box(px + 2, py + ph - fh - 2, pw - 4, fh)
        cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (128, 64, 0), -1)

    meta = {
        "header_box": {"x": hdr["x"], "y": hdr["y"], "w": hdr["w"], "h": hdr["h"]},
        "panel_box": {"x": px, "y": py, "w": pw, "h": ph},
        "option_boxes": option_boxes,
        "at_bottom": bool(at_bottom),
        "target_index": (vis_rows - 1) if at_bottom else None,
        "target_box": option_boxes[-1] if at_bottom else None,
        "total_rows": int(total_rows),
        "visible_rows": int(vis_rows),
        "opt_h": int(opt_h),
        "scroll_idx": int(scroll_idx),
    }
    return img, meta


def generate_dataset(out_dir: Path, n: int, seed: int = 123, *, mode: str = "single") -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        sample_dir = out_dir / f"sample_{i:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        hdr = _rand_dropdown_geom(rng)
        opt_h, vis_rows = _rand_panel_spec(rng)
        total_rows = rng.randint(vis_rows + 3, vis_rows + 12)

        if mode == "single":
            img_closed, meta_closed = _make_closed_image(hdr)
            img_bottom, meta_bottom = _make_bottom_image(hdr, opt_h, vis_rows, total_rows)
            cv2.imwrite(str(sample_dir / "closed.png"), img_closed)
            cv2.imwrite(str(sample_dir / "bottom.png"), img_bottom)

            actions: List[Dict[str, Any]] = []
            hx = hdr["x"] + hdr["w"] // 2
            hy = hdr["y"] + hdr["h"] // 2
            t = 0.0
            actions.append({"t": t, "type": "click", "x": hx, "y": hy})
            t += 0.05
            px, py = meta_bottom["panel_box"]["x"], meta_bottom["panel_box"]["y"]
            pw, ph = meta_bottom["panel_box"]["w"], meta_bottom["panel_box"]["h"]
            mx, my = px + int(0.8 * pw), py + ph // 2
            path = _path_points((hx, hy), (mx, my), steps=24, jitter=2)
            actions.append({"t": t, "type": "move", "points": path})
            t += 0.35
            scroll_steps = max(1, total_rows - vis_rows)
            actions.append({"t": t, "type": "scroll", "amount": scroll_steps})
            t += 0.20 + 0.02 * scroll_steps
            tb = meta_bottom["target_box"]
            tx = tb["x"] + tb["w"] // 2
            ty = tb["y"] + tb["h"] // 2
            actions.append({"t": t, "type": "click", "x": tx, "y": ty})

            label = {
                "images": {
                    "closed": str((sample_dir / "closed.png").as_posix()),
                    "bottom": str((sample_dir / "bottom.png").as_posix()),
                },
                "metadata": {**meta_bottom},
                "actions": actions,
            }
            (sample_dir / "metadata.json").write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            frames_dir = sample_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            frames: List[Dict[str, Any]] = []

            img_closed, _ = _make_closed_image(hdr)
            f0 = frames_dir / "frame_000.png"
            cv2.imwrite(str(f0), img_closed)
            frames.append({"path": f0.name, "state": {"open": False}})

            img_open0, meta0 = _make_open_image(hdr, opt_h, vis_rows, total_rows, scroll_idx=0)
            f1 = frames_dir / "frame_001.png"
            cv2.imwrite(str(f1), img_open0)
            frames.append({"path": f1.name, "state": {"open": True, "scroll_idx": 0}})

            max_scroll = max(0, total_rows - vis_rows)
            idx = 2
            last_meta = meta0
            for s in range(1, max_scroll + 1):
                img_s, metas = _make_open_image(hdr, opt_h, vis_rows, total_rows, scroll_idx=s)
                fp = frames_dir / f"frame_{idx:03d}.png"
                cv2.imwrite(str(fp), img_s)
                frames.append({"path": fp.name, "state": {"open": True, "scroll_idx": s}})
                last_meta = metas
                idx += 1

            hx = hdr["x"] + hdr["w"] // 2
            hy = hdr["y"] + hdr["h"] // 2
            actions: List[Dict[str, Any]] = []
            actions.append({"frame": 0, "type": "click", "x": hx, "y": hy})

            px, py = meta0["panel_box"]["x"], meta0["panel_box"]["y"]
            pw, ph = meta0["panel_box"]["w"], meta0["panel_box"]["h"]
            mx, my = px + int(0.8 * pw), py + ph // 2
            path = _path_points((hx, hy), (mx, my), steps=24, jitter=2)
            actions.append({"frame": 1, "type": "move", "points": path})

            for s in range(1, max_scroll + 1):
                actions.append({"frame": 1 + s, "type": "scroll", "amount": 1})

            tb = last_meta["target_box"] or {"x": px, "y": py + (vis_rows - 1) * opt_h, "w": pw, "h": opt_h}
            tx = tb["x"] + tb["w"] // 2
            ty = tb["y"] + tb["h"] // 2
            actions.append({"frame": len(frames) - 1, "type": "click", "x": tx, "y": ty})

            seq = {"frames": frames, "actions": actions, "meta": {"opt_h": opt_h, "vis_rows": vis_rows, "total_rows": total_rows}}
            (sample_dir / "sequence.json").write_text(json.dumps(seq, ensure_ascii=False, indent=2), encoding="utf-8")

    index = {"count": n, "samples": [f"sample_{i:06d}" for i in range(n)]}
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/il_dropdown")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--mode", type=str, default="sequence", choices=["single", "sequence"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    generate_dataset(out_dir, n=args.n, seed=args.seed, mode=args.mode)
    print(f"Generated {args.n} samples at {out_dir}")


if __name__ == "__main__":
    main()
