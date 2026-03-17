#!/usr/bin/env python3
"""
Hover speed recorder (standalone helper).

- Samples real mouse positions on a fixed interval (default 50 ms).
- Builds a speed heatmap (px/s) on top of the latest screenshot.
- Draws extra overlays with sampled points:
    * points + speed heatmap -> data/screen/hover/hover_points_on_speed_current + history
    * points + hover path    -> data/screen/hover/hover_points_on_path_current + history

Autosave is ON by default (every 0.1 s). Use --noautosave to save only at the end.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

try:
    from pynput import mouse  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        f"[hover_speed_recorder] ERROR: cannot import pynput.mouse ({exc}). "
        "Install 'pynput' inside ai_cuda env."
    )

try:
    from PIL import Image, ImageDraw  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        f"[hover_speed_recorder] ERROR: cannot import Pillow ({exc}). "
        "Install 'Pillow' inside ai_cuda env."
    )

try:
    import numpy as np  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        f"[hover_speed_recorder] ERROR: cannot import numpy ({exc}). "
        "Install 'numpy' inside ai_cuda env."
    )


ROOT = Path(__file__).resolve().parents[2]
DATA_SCREEN_DIR = ROOT / "data" / "screen"
HOVER_DIR = DATA_SCREEN_DIR / "hover"
RAW_DIR = DATA_SCREEN_DIR / "raw"
RAW_CURRENT_DIR = RAW_DIR / "raw_screens_current"
HOVER_INPUT_CURRENT_DIR = HOVER_DIR / "hover_input_current"
HOVER_SPEED_DIR = HOVER_DIR / "hover_speed"
HOVER_SPEED_CURRENT_DIR = HOVER_DIR / "hover_speed_current"
HOVER_PATH_DIR = HOVER_DIR / "hover_path"
HOVER_PATH_CURRENT_DIR = HOVER_DIR / "hover_path_current"
HOVER_OUTPUT_CURRENT_DIR = HOVER_DIR / "hover_output_current"
HOVER_POINTS_SPEED_DIR = HOVER_DIR / "hover_points_on_speed"
HOVER_POINTS_SPEED_CURRENT_DIR = HOVER_DIR / "hover_points_on_speed_current"
HOVER_POINTS_PATH_DIR = HOVER_DIR / "hover_points_on_path"
HOVER_POINTS_PATH_CURRENT_DIR = HOVER_DIR / "hover_points_on_path_current"


@dataclass
class Sample:
    t: float
    x: int
    y: int


def _resolve_background(image_arg: str | None) -> Path:
    if image_arg:
        p = Path(image_arg)
        if not p.is_file():
            raise FileNotFoundError(f"Podany obraz nie istnieje: {p}")
        return p
    # Priority: raw/raw_screens_current/screenshot.png -> hover_input_current/hover_input.png
    cand1 = RAW_CURRENT_DIR / "screenshot.png"
    if cand1.is_file():
        return cand1
    cand2 = HOVER_INPUT_CURRENT_DIR / "hover_input.png"
    if cand2.is_file():
        return cand2
    raise FileNotFoundError(
        f"Nie znaleziono domyslnego screenshota.\n"
        f"Oczekiwano {cand1} lub {cand2}. Uruchom main.py i zrob jeden screenshot."
    )


def record_positions(interval: float, duration: float | None = None) -> List[Sample]:
    ctrl = mouse.Controller()
    samples: List[Sample] = []
    print(f"[hover_speed_recorder] Start probkowania kursora co {interval:.3f} s (Ctrl+C aby zakonczyc)...")
    t0 = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            t = now - t0
            if duration is not None and t >= duration:
                break
            x, y = ctrl.position
            samples.append(Sample(t=t, x=int(x), y=int(y)))
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    print(f"[hover_speed_recorder] Zebrano {len(samples)} probek.")
    return samples


def build_speed_segments(samples: List[Sample]) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
    segs: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []
    if len(samples) < 2:
        return segs
    for s0, s1 in zip(samples, samples[1:]):
        dt = float(s1.t - s0.t)
        if dt <= 1e-4:
            continue
        v = math.hypot(float(s1.x - s0.x), float(s1.y - s0.y)) / dt
        segs.append(((s0.x, s0.y), (s1.x, s1.y), v))
    return segs


def render_speed_heatmap(
    background: Path,
    segments: List[Tuple[Tuple[int, int], Tuple[int, int], float]],
    out_hist: Path | None,
    out_current: Path | None,
) -> Tuple[Path | None, Path | None]:
    if not segments:
        return None, None
    try:
        img = Image.open(background).convert("RGB")
    except Exception as exc:
        print(f"[hover_speed_recorder] ERROR: nie mozna otworzyc tla {background}: {exc}")
        return None, None

    arr = np.array(img, dtype=np.uint8)
    h, w, _ = arr.shape

    speeds = [v for _, _, v in segments]
    v_min = min(speeds)
    v_max = max(speeds)
    if v_max <= v_min + 1e-6:
        v_max = v_min + 1.0

    def _clip_xy(x: int, y: int) -> Tuple[int, int]:
        return int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))

    out = arr.copy()
    for (x0, y0), (x1, y1), v in segments:
        x0, y0 = _clip_xy(x0, y0)
        x1, y1 = _clip_xy(x1, y1)
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        if steps <= 0:
            steps = 1
        xs = np.linspace(x0, x1, steps + 1, dtype=np.int32)
        ys = np.linspace(y0, y1, steps + 1, dtype=np.int32)

        # Normalize speed to [0,1].
        t = (v - v_min) / (v_max - v_min)
        t = max(0.0, min(1.0, t))
        if t < 0.5:
            # blue -> green
            k = t / 0.5
            r, g, b = 0, int(255 * k), int(255 * (1.0 - k))
        else:
            # green -> red
            k = (t - 0.5) / 0.5
            r, g, b = int(255 * k), int(255 * (1.0 - k)), 0
        colour = np.array([r, g, b], dtype=np.uint8)
        out[ys, xs] = colour

    if out_hist:
        out_hist = out_hist.with_suffix(".png")
    if out_current:
        out_current = out_current.with_suffix(".png")

    hist_saved: Path | None = None
    current_saved: Path | None = None

    try:
        if out_hist:
            out_hist.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(out).save(out_hist)
            hist_saved = out_hist
        if out_current:
            out_current.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(out).save(out_current)
            current_saved = out_current
    except Exception as exc:
        print(f"[hover_speed_recorder] ERROR: nie mozna zapisac map predkosci: {exc}")
        return hist_saved, current_saved

    if hist_saved:
        print(f"[hover_speed_recorder] Zapisano heatmape predkosci -> {hist_saved}")
    if current_saved:
        print(f"[hover_speed_recorder] Ostatni run -> {current_saved}")
    return hist_saved, current_saved


def _draw_points_overlay(
    background: Path,
    points: List[Tuple[int, int]],
    out_hist: Path | None,
    out_current: Path | None,
    colour: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 3,
    draw_lines: bool = False,
    paths: List[List[Tuple[int, int]]] | None = None,
    line_colour: Tuple[int, int, int] | None = None,
    line_width: int = 2,
) -> Tuple[Path | None, Path | None]:
    if not points:
        return None, None
    if not background.exists():
        return None, None
    try:
        img = Image.open(background).convert("RGB")
    except Exception as exc:
        print(f"[hover_speed_recorder] WARN: nie mozna otworzyc tla do overlay {background}: {exc}")
        return None, None

    draw = ImageDraw.Draw(img)

    if draw_lines and paths:
        lc = line_colour if line_colour is not None else colour
        lw = max(1, int(line_width))
        for p in paths:
            if len(p) >= 2:
                draw.line(p, fill=lc, width=lw)
    elif draw_lines and len(points) >= 2:
        lc = line_colour if line_colour is not None else colour
        draw.line(points, fill=lc, width=max(1, int(line_width)))
    r = max(1, int(radius))
    for x, y in points:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=None, outline=colour)

    if out_hist:
        out_hist = out_hist.with_suffix(".png")
    if out_current:
        out_current = out_current.with_suffix(".png")

    hist_saved: Path | None = None
    current_saved: Path | None = None
    try:
        if out_hist:
            out_hist.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_hist)
            hist_saved = out_hist
        if out_current:
            out_current.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_current)
            current_saved = out_current
    except Exception as exc:
        print(f"[hover_speed_recorder] WARN: nie mozna zapisac overlay z kropkami: {exc}")
    return hist_saved, current_saved


def _overlay_on_speed(
    heatmap_hist_path: Path | None,
    heatmap_current_path: Path | None,
    points: List[Tuple[int, int]],
    paths: List[List[Tuple[int, int]]] | None,
    stem: str,
    hist: bool,
) -> None:
    hist_out = HOVER_POINTS_SPEED_DIR / f"{stem}_hover_points_on_speed" if hist else None
    current_out = HOVER_POINTS_SPEED_CURRENT_DIR / "hover_points_on_speed"

    # Zapisz current z current tĹ‚a, a historiÄ™ z historycznego tĹ‚a (ĹĽeby kolory heatmapy byĹ‚y 1:1).
    if heatmap_current_path is not None:
        _draw_points_overlay(
            heatmap_current_path,
            points,
            None,
            current_out,
            colour=(0, 220, 255),      # punkty cyan
            radius=3,
            draw_lines=True,
            paths=paths,
            line_colour=(255, 200, 0), # linia zĂłĹ‚to-pomaraĹ„czowa
            line_width=3,
        )
    elif heatmap_hist_path is not None:
        # Fallback: brak current tĹ‚a -> uĹĽyj hist do current.
        _draw_points_overlay(
            heatmap_hist_path,
            points,
            None,
            current_out,
            colour=(0, 220, 255),
            radius=3,
            draw_lines=True,
            paths=paths,
            line_colour=(255, 200, 0),
            line_width=3,
        )

    if hist and heatmap_hist_path is not None:
        _draw_points_overlay(
            heatmap_hist_path,
            points,
            hist_out,
            None,
            colour=(0, 220, 255),
            radius=3,
            draw_lines=True,
            paths=paths,
            line_colour=(255, 200, 0),
            line_width=3,
        )


def _overlay_on_path(
    points: List[Tuple[int, int]],
    paths: List[List[Tuple[int, int]]] | None,
    stem: str,
    hist: bool,
) -> None:
    bg_hist = HOVER_PATH_DIR / f"{stem}_hover_path.png"
    bg_current = HOVER_PATH_CURRENT_DIR / "hover_path.png"
    if not bg_hist.exists() and not bg_current.exists():
        return

    hist_out = HOVER_POINTS_PATH_DIR / f"{stem}_hover_points_on_path" if hist else None
    current_out = HOVER_POINTS_PATH_CURRENT_DIR / "hover_points_on_path"

    # Current zawsze na aktualnym hover_path.png, ĹĽeby byĹ‚o 1:1
    if bg_current.exists():
        _draw_points_overlay(
            bg_current,
            points,
            None,
            current_out,
            colour=(255, 0, 255),
            radius=3,
            draw_lines=True,
            paths=paths,
            line_colour=(200, 0, 255),
            line_width=3,
        )
    elif bg_hist.exists():
        _draw_points_overlay(
            bg_hist,
            points,
            None,
            current_out,
            colour=(255, 0, 255),
            radius=3,
            draw_lines=True,
            paths=paths,
            line_colour=(200, 0, 255),
            line_width=3,
        )

    if hist and bg_hist.exists():
        _draw_points_overlay(
            bg_hist,
            points,
            hist_out,
            None,
            colour=(255, 0, 255),
            radius=3,
            draw_lines=True,
            paths=paths,
            line_colour=(200, 0, 255),
            line_width=3,
        )


def _load_hover_output_paths() -> List[List[Tuple[int, int]]]:
    """Wczytaj Ĺ›cieĹĽki (lista punktĂłw na element) z hover_output_current/hover_output.json (pole dots)."""
    json_path = HOVER_OUTPUT_CURRENT_DIR / "hover_output.json"
    if not json_path.exists():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    paths: List[List[Tuple[int, int]]] = []
    for item in data:
        dots = item.get("dots") if isinstance(item, dict) else None
        if not dots:
            continue
        seg: List[Tuple[int, int]] = []
        for d in dots:
            if not (isinstance(d, (list, tuple)) and len(d) == 2):
                continue
            try:
                x = int(round(float(d[0])))
                y = int(round(float(d[1])))
                seg.append((x, y))
            except Exception:
                continue
        if seg:
            paths.append(seg)
    return paths


def _flatten_paths(paths: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for p in paths:
        out.extend(p)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rejestrator predkosci hovera (heatmapa px/s).")
    ap.add_argument(
        "--interval",
        type=float,
        default=0.002,
        help="Interwal probkowania kursora (s, domyslnie 0.002).",
    )
    ap.add_argument(
        "--image",
        type=str,
        default=None,
        help="Sciezka do tla (screenshota). Domyslnie raw/raw_screens_current/screenshot.png.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(HOVER_SPEED_CURRENT_DIR),
        help="Katalog wyjsciowy dla heatmapy (domyslnie hover_speed_current).",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Czas nagrywania w sekundach (0 = az do Ctrl+C).",
    )
    ap.add_argument(
        "--noautosave",
        action="store_true",
        help="Wylacz autosave w trakcie (zapisz tylko na koncu).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    bg = _resolve_background(args.image)
    dur = float(args.duration)
    duration = dur if dur > 0.0 else None
    interval = max(0.001, float(args.interval))
    autosave = 0.0 if args.noautosave else 0.1

    ctrl = mouse.Controller()
    samples: List[Sample] = []
    t0 = time.perf_counter()
    last_save = t0
    reference_paths = _load_hover_output_paths()

    stem = bg.stem
    out_dir = Path(args.out_dir)
    hist_path = HOVER_SPEED_DIR / f"{stem}_hover_speed"
    current_path = out_dir / "hover_speed"

    print(
        f"[hover_speed_recorder] Start probkowania co {interval:.3f}s "
        f"(autosave={autosave:.3f}s, Ctrl+C aby zakonczyc)..."
    )
    try:
        while True:
            now = time.perf_counter()
            t = now - t0
            if duration is not None and t >= duration:
                break
            x, y = ctrl.position
            samples.append(Sample(t=t, x=int(x), y=int(y)))

            if autosave > 0 and (now - last_save) >= autosave and len(samples) >= 2:
                # JeĹ›li hover_output_current pojawi siÄ™ pĂłĹşniej, doĹ‚aduj raz.
                if not reference_paths:
                    reference_paths = _load_hover_output_paths()
                segs_live = build_speed_segments(samples)
                _, current_saved = render_speed_heatmap(bg, segs_live, None, current_path)
                if reference_paths:
                    points_list = _flatten_paths(reference_paths)
                    paths = reference_paths
                else:
                    points_list = [(s.x, s.y) for s in samples]
                    paths = [points_list]
                _overlay_on_speed(None, current_saved, points_list, paths, stem, hist=False)
                last_save = now

            time.sleep(interval)
    except KeyboardInterrupt:
        pass

    print(f"[hover_speed_recorder] Zebrano {len(samples)} probek.")
    segments = build_speed_segments(samples)
    hist_saved, current_saved = render_speed_heatmap(bg, segments, hist_path, current_path)

    # Overlay: points + speed heatmap (preferuj "prawdziwe kropki" z hover_output_current).
    if not reference_paths:
        reference_paths = _load_hover_output_paths()
    if reference_paths:
        points_list = _flatten_paths(reference_paths)
        paths = reference_paths
    else:
        points_list = [(s.x, s.y) for s in samples]
        paths = [points_list]
    _overlay_on_speed(hist_saved, current_saved, points_list, paths, stem, hist=True)

    # Overlay: points + hover path render
    _overlay_on_path(points_list, paths, stem, hist=True)


if __name__ == "__main__":
    main()
