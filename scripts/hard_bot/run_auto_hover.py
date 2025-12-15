from __future__ import annotations

"""
End-to-end hover pipeline:

1) Start ControlAgent (UDP) in a background process
2) Wait 2s
3) Load screenshot captured by the main pipeline (no self-capture)
4) Run PaddleOCR + hover dots
5) Build trajectory from dots and send to agent (cmd=path)
6) Agent executes with timing based on trajectory.json/transforms logic
7) No clicking – just cursor painting
8) ESC stops everything (agent subprocess terminated)

Usage:
  python scripts/hard_bot/run_auto_hover.py [--port 8765] [--speed-factor 1.0]
                                            [--gap-boost 3.0] [--line-jump-boost 1.5]
                                            [--min-dt 0.004] [--show]
"""

import atexit
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

try:
    from pynput import keyboard
except Exception as e:  # pragma: no cover - runtime dep
    keyboard = None  # type: ignore

# Make project modules importable
PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "envs").exists()),
    Path(__file__).resolve().parents[2],
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATA_SCREEN_DIR = PROJECT_ROOT / "data" / "screen"
CURRENT_RUN_DIR = DATA_SCREEN_DIR / "current_run"
DEFAULT_SCREENSHOT = CURRENT_RUN_DIR / "screenshot.png"

from utils.ocr_features import OCRNotAvailableError
from scripts.hard_bot.hover_bot import (
    create_paddleocr_reader,
    process_image,  # reuse dots/boxes generation
)


def _ensure_agent(port: int) -> subprocess.Popen:
    cfg = PROJECT_ROOT / "scripts" / "control_agent" / "train.json"
    agent = PROJECT_ROOT / "scripts" / "control_agent" / "control_agent.py"
    if not agent.exists():
        raise FileNotFoundError(f"control_agent.py not found at {agent}")
    cmd = [sys.executable, str(agent), "--config", str(cfg), "--port", str(port)]
    # Start detached so we can terminate on ESC
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        creationflags = 0
    proc = subprocess.Popen(cmd, creationflags=creationflags)
    return proc


def _as_rect(box: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _group_lines(seqs: List[Dict[str, Any]]) -> List[List[int]]:
    entries = []  # (idx, min_y, max_y, center_y, height)
    for i, s in enumerate(seqs):
        box = s.get("box") or []
        if not box:
            continue
        ys = [p[1] for p in box]
        min_y, max_y = float(min(ys)), float(max(ys))
        center_y = 0.5 * (min_y + max_y)
        entries.append((i, min_y, max_y, center_y, max(1.0, max_y - min_y)))

    groups: List[List[int]] = []
    ranges: List[Tuple[float, float]] = []
    for i, min_y, max_y, center_y, h in sorted(entries, key=lambda t: t[3]):
        placed = False
        for gi, (gmin, gmax) in enumerate(ranges):
            gc = 0.5 * (gmin + gmax)
            gh = max(1.0, gmax - gmin)
            if abs(center_y - gc) <= 0.45 * max(h, gh):
                ranges[gi] = (min(gmin, min_y), max(gmax, max_y))
                groups[gi].append(i)
                placed = True
                break
        if not placed:
            ranges.append((min_y, max_y))
            groups.append([i])
    return groups


def _build_path_and_meta(seqs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, int]], List[Tuple[int,int,int,int]], List[int]]:
    gaps: List[Tuple[int,int,int,int]] = []
    for s in seqs:
        if float(s.get("confidence", 0.0)) < 0.0:
            box = s.get("box") or []
            if box:
                gaps.append(_as_rect([[float(x), float(y)] for x, y in box]))

    real_idx = [i for i, s in enumerate(seqs) if float(s.get("confidence", 0.0)) >= 0.0]
    groups_local = _group_lines([seqs[i] for i in real_idx])

    points: List[Dict[str, int]] = []
    line_jump_indices: List[int] = []
    seg_index = -1
    for g in groups_local:
        order = sorted(g, key=lambda i: min(p[0] for p in seqs[real_idx[i]].get("box", [[0,0]])))
        first_in_line = True
        for i_local in order:
            s = seqs[real_idx[i_local]]
            dots = s.get("dots") or []
            if not dots:
                continue
            if first_in_line:
                if points:
                    line_jump_indices.append(max(0, seg_index))
                first_in_line = False
            for d in dots:
                x, y = int(round(d[0])), int(round(d[1]))
                points.append({"x": x, "y": y})
                seg_index += 1
    return points, gaps, line_jump_indices


def _send_path(
    port: int,
    points: List[Dict[str, int]],
    gaps: List[Tuple[int,int,int,int]],
    line_jumps: List[int],
    *,
    speed: str = "normal",
    speed_factor: float = 1.0,
    gap_boost: float = 3.0,
    line_jump_boost: float = 1.5,
    min_dt: float = 0.004,
    min_total_ms: float = 0.0,
):
    payload = {
        "cmd": "path",
        "points": points,
        "speed": speed,
        "speed_factor": float(speed_factor),
        "gap_rects": [list(r) for r in gaps],
        "gap_boost": float(gap_boost),
        "line_jump_indices": line_jumps,
        "line_jump_boost": float(line_jump_boost),
        "min_dt": float(min_dt),
        "min_total_ms": float(min_total_ms),
    }
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = json.dumps(payload).encode("utf-8")
    sock.sendto(data, ("127.0.0.1", port))

def _load_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Brak pliku zrzutu ekranu: {path}")
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Nie udało się wczytać obrazu: {path}")
    return frame

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Ścieżka do gotowego zrzutu ekranu (domyślnie: data/screen/current_run/screenshot.png).",
    )
    ap.add_argument("--speed-factor", type=float, default=1.0)
    ap.add_argument("--gap-boost", type=float, default=3.0)
    ap.add_argument("--line-jump-boost", type=float, default=1.5)
    ap.add_argument("--min-dt", type=float, default=0.004)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    # 1) start agent
    proc = _ensure_agent(args.port)

    def _cleanup():
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
    atexit.register(_cleanup)

    # 2) wait 2s
    time.sleep(2.0)

    # ESC handler
    stop = False
    listener = None
    if keyboard is not None:
        def on_press(key):
            nonlocal stop
            try:
                if key == keyboard.Key.esc:
                    stop = True
                    return False
            except Exception:
                return False
            return True
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    # 3) screenshot (reuse the one captured by the main pipeline)
    image_path = Path(args.image) if args.image else DEFAULT_SCREENSHOT
    frame = _load_image(image_path)

    # 4) OCR + dots
    try:
        reader = create_paddleocr_reader(lang="en")
    except OCRNotAvailableError as e:
        raise ImportError("paddleocr is required. Install with `pip install paddleocr`.") from e

    sequences, annotated = process_image(image_path, reader=reader)

    # 5) build path and send to agent
    points, gaps, line_jumps = _build_path_and_meta([{
        "index": s.index,
        "text": s.text,
        "confidence": s.confidence,
        "box": s.box,
        "dots": s.dots,
    } for s in sequences])

    _send_path(
        args.port,
        points,
        gaps,
        line_jumps,
        speed="normal",
        speed_factor=float(getattr(args, "speed_factor", 1.0)),
        gap_boost=float(getattr(args, "gap_boost", 3.0)),
        line_jump_boost=float(getattr(args, "line_jump_boost", 1.5)),
        min_dt=float(getattr(args, "min_dt", 0.004)),
    )

    # --- Debug dumps: raw, annotated (dots/boxes), and trajectory with lines ---
    try:
        ts = time.strftime("%Y%m%d_%H%M%S")
        dbg_dir = PROJECT_ROOT / "debug"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg_dir / f"screen_{ts}.png"), frame)
        cv2.imwrite(str(dbg_dir / f"annotated_{ts}.png"), annotated)
        # draw trajectory lines over a copy of frame
        traj = annotated.copy()
        prev = None
        for p in points:
            x, y = int(p["x"]), int(p["y"]) 
            cv2.circle(traj, (x, y), radius=3, color=(0, 255, 255), thickness=-1)
            if prev is not None:
                cv2.line(traj, prev, (x, y), color=(0, 200, 0), thickness=1)
            prev = (x, y)
        cv2.imwrite(str(dbg_dir / f"trajectory_{ts}.png"), traj)
        # save payload meta for reproducibility
        meta = {
            "points": points,
            "gaps": [list(r) for r in gaps],
            "line_jumps": line_jumps,
        }
        (dbg_dir / f"trajectory_{ts}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    if getattr(args, "show", False):
        try:
            cv2.imshow("hover annotated", annotated)
            cv2.waitKey(500)
        except Exception:
            pass

    # 8) wait for ESC or small delay for movement to complete
    wait_s = max(20.0, len(points) * 0.02)
    t0 = time.time()
    while time.time() - t0 < wait_s and not stop:
        time.sleep(0.05)

    if listener is not None:
        try:
            listener.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
