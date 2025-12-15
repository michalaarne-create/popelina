#!/usr/bin/env python3
"""
Prosty profiler ludzkiego hovera sterowany klawiszem ']'.

Jak używać:
    (ai_cuda) python scripts/control_agent/record_hover_profile.py

Instrukcja:
    - Ruszaj myszką po ekranie.
    - Naciśnij ']' aby WŁĄCZYĆ nagrywanie trajektorii.
    - Naciśnij ']' ponownie aby JE WYŁĄCZYĆ (zamyka segment).
    - Zatrzymaj skrypt Ctrl+C w konsoli, gdy nazbierasz kilka prób.

Skrypt:
    - Zapisuje trajektorie i statystyki do data/hover_profile/profile.json
    - Na końcu wypisuje sugerowane:
        * docelową prędkość w px/s
        * minimalny dt między punktami
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, median
from typing import List, Optional

try:
    from pynput import mouse, keyboard  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        f"[record_hover_profile] ERROR: nie można zaimportować pynput ({exc}). "
        "Zainstaluj paczkę 'pynput' w envie ai_cuda."
    )


ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "data" / "hover_profile"
PROFILE_PATH = PROFILE_DIR / "profile.json"


@dataclass
class Sample:
    t: float
    x: int
    y: int


@dataclass
class Segment:
    start_time: float
    end_time: float
    samples: List[Sample]

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "samples": [asdict(s) for s in self.samples],
        }


def _percentile(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def _compute_stats(segments: List[Segment]) -> dict:
    dts: List[float] = []
    speeds: List[float] = []

    for seg in segments:
        pts = seg.samples
        if len(pts) < 2:
            continue
        for i in range(1, len(pts)):
            dt = pts[i].t - pts[i - 1].t
            if dt <= 0:
                continue
            dx = pts[i].x - pts[i - 1].x
            dy = pts[i].y - pts[i - 1].y
            dist = math.hypot(dx, dy)
            if dist <= 0:
                dts.append(dt)
                continue
            v = dist / dt
            dts.append(dt)
            speeds.append(v)

    dts_sorted = sorted(dts)
    speeds_sorted = sorted(speeds)

    def _stats_block(values_sorted: List[float]) -> dict:
        if not values_sorted:
            return {"count": 0}
        return {
            "count": len(values_sorted),
            "mean": float(mean(values_sorted)),
            "median": float(median(values_sorted)),
            "p10": _percentile(values_sorted, 0.10),
            "p50": _percentile(values_sorted, 0.50),
            "p90": _percentile(values_sorted, 0.90),
        }

    stats = {
        "segments": len(segments),
        "samples_total": len(dts),
        "dt_stats_s": _stats_block(dts_sorted),
        "speed_px_s_stats": _stats_block(speeds_sorted),
    }

    rec_speed = stats["speed_px_s_stats"].get("median") or stats["speed_px_s_stats"].get("mean") or 800.0
    rec_dt = stats["dt_stats_s"].get("median") or stats["dt_stats_s"].get("mean") or 0.008
    rec_speed = float(max(200.0, min(1800.0, rec_speed)))
    rec_dt = float(max(0.0015, min(0.03, rec_dt * 0.5)))

    stats["recommended"] = {
        "DEFAULT_SPEED_PX_PER_S": rec_speed,
        "DEFAULT_MIN_DT": rec_dt,
        "DEFAULT_SPEED_FACTOR": 1.0,
        "DEFAULT_SPEED_MODE": "normal",
        "DEFAULT_GAP_BOOST": None,
        "DEFAULT_LINE_JUMP_BOOST": None,
    }
    return stats


def main() -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("[record_hover_profile] Start.")
    print("Ruszaj myszką po ekranie.")
    print("Nagrywanie włączasz/wyłączasz klawiszem ']'.")
    print("Zakończ nagrywanie Ctrl+C.\n")

    segments: List[Segment] = []
    current: Optional[Segment] = None
    last_sample: Optional[Sample] = None
    last_pos: Optional[Sample] = None
    recording = False

    PIXEL_STEP = 8.0
    MAX_DT = 0.05

    def _start_segment(t: float, x_i: int, y_i: int) -> None:
        nonlocal current, last_sample
        s = Sample(t=t, x=x_i, y=y_i)
        current = Segment(start_time=t, end_time=t, samples=[s])
        last_sample = s
        print(f"[record_hover_profile] START segment (x={x_i}, y={y_i})")

    def _stop_segment() -> None:
        nonlocal current, last_sample
        if current is not None and len(current.samples) >= 2:
            current.end_time = current.samples[-1].t
            segments.append(current)
            print(
                f"[record_hover_profile] END segment: "
                f"samples={len(current.samples)} duration={current.end_time - current.start_time:.3f}s"
            )
        current = None
        last_sample = None

    def on_move(x, y):
        nonlocal current, last_sample, last_pos, recording
        t = time.perf_counter()
        x_i, y_i = int(x), int(y)
        last_pos = Sample(t=t, x=x_i, y=y_i)

        if not recording:
            return

        if last_sample is None:
            _start_segment(t, x_i, y_i)
            return

        dt = t - last_sample.t
        dx = x_i - last_sample.x
        dy = y_i - last_sample.y
        dist = math.hypot(dx, dy)

        if dist >= PIXEL_STEP or dt >= MAX_DT:
            if current is None:
                current = Segment(start_time=t, end_time=t, samples=[])
            s = Sample(t=t, x=x_i, y=y_i)
            current.samples.append(s)
            current.end_time = t
            last_sample = s

    def on_press(key):
        nonlocal recording
        try:
            ch = key.char
        except AttributeError:
            return
        if ch == "]":
            recording = not recording
            if recording:
                print("[record_hover_profile] >>> REC ON ('])'")
            else:
                _stop_segment()
                print("[record_hover_profile] >>> REC OFF ('])'")

    mouse_listener = mouse.Listener(on_move=on_move)
    key_listener = keyboard.Listener(on_press=on_press)
    mouse_listener.start()
    key_listener.start()

    try:
        while mouse_listener.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[record_hover_profile] Stop requested (Ctrl+C).")
    finally:
        mouse_listener.stop()
        key_listener.stop()

    if current is not None and len(current.samples) >= 2:
        current.end_time = current.samples[-1].t
        segments.append(current)

    if not segments:
        print("[record_hover_profile] Brak zarejestrowanych segmentów — nic do profilu.")
        return

    stats = _compute_stats(segments)
    profile = {
        "created_at": time.time(),
        "segments": [seg.to_dict() for seg in segments],
        "stats": stats,
    }
    PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[record_hover_profile] Zapisano profil hovera ->", PROFILE_PATH)
    print("Podsumowanie:")
    s_dt = stats["dt_stats_s"]
    s_sp = stats["speed_px_s_stats"]
    print(
        f"  segments={stats['segments']}  samples={stats['samples_total']}\n"
        f"  dt median = {s_dt.get('median', 0):.4f} s (p10={s_dt.get('p10', 0):.4f}, p90={s_dt.get('p90', 0):.4f})\n"
        f"  speed median = {s_sp.get('median', 0):.1f} px/s (p10={s_sp.get('p10', 0):.1f}, p90={s_sp.get('p90', 0):.1f})"
    )
    rec = stats["recommended"]
    print("\nSugerowane ustawienia zapisane w profile.json:")
    print(f"  DEFAULT_SPEED_PX_PER_S ≈ {rec['DEFAULT_SPEED_PX_PER_S']:.1f}")
    print(f"  DEFAULT_MIN_DT ≈ {rec['DEFAULT_MIN_DT']:.4f} s")


if __name__ == "__main__":
    main()

