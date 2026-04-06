#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "mouse_trajectories"


@dataclass
class RawSample:
    t: float
    x: int
    y: int


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _distance(a: RawSample, b: RawSample) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def _heading(a: RawSample, b: RawSample) -> float:
    return math.atan2(b.y - a.y, b.x - a.x)


def _angle_diff(a: float, b: float) -> float:
    d = b - a
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def _moving_average_points(samples: Sequence[RawSample], radius: int = 2) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    n = len(samples)
    for i in range(n):
        sx = 0.0
        sy = 0.0
        cnt = 0
        for j in range(max(0, i - radius), min(n, i + radius + 1)):
            sx += float(samples[j].x)
            sy += float(samples[j].y)
            cnt += 1
        out.append((sx / max(1, cnt), sy / max(1, cnt)))
    return out


def _compress_samples(samples: Sequence[RawSample], *, min_step_px: float, keepalive_s: float) -> List[RawSample]:
    if not samples:
        return []
    out = [samples[0]]
    last = samples[0]
    for sample in samples[1:]:
        dt = float(sample.t - last.t)
        dist = _distance(last, sample)
        if dist >= min_step_px or dt >= keepalive_s:
            out.append(sample)
            last = sample
    if out[-1].t != samples[-1].t or out[-1].x != samples[-1].x or out[-1].y != samples[-1].y:
        out.append(samples[-1])
    return out


def _segment_stats(samples: Sequence[RawSample]) -> Dict[str, float]:
    if len(samples) < 2:
        return {
            "duration_s": 0.0,
            "path_length_px": 0.0,
            "net_displacement_px": 0.0,
            "straightness": 0.0,
            "mean_speed_px_s": 0.0,
            "peak_speed_px_s": 0.0,
            "abs_turn_rad": 0.0,
            "turn_per_sec": 0.0,
            "direction_flips": 0.0,
        }
    path_length = 0.0
    speeds: List[float] = []
    headings: List[float] = []
    for i in range(1, len(samples)):
        dt = max(1e-6, samples[i].t - samples[i - 1].t)
        dist = _distance(samples[i - 1], samples[i])
        path_length += dist
        speeds.append(dist / dt)
        headings.append(_heading(samples[i - 1], samples[i]))
    abs_turn = 0.0
    flips = 0
    prev_sign = 0
    for i in range(1, len(headings)):
        turn = _angle_diff(headings[i - 1], headings[i])
        abs_turn += abs(turn)
        sign = 1 if turn > 1e-4 else (-1 if turn < -1e-4 else 0)
        if sign != 0 and prev_sign != 0 and sign != prev_sign:
            flips += 1
        if sign != 0:
            prev_sign = sign
    duration_s = max(1e-6, samples[-1].t - samples[0].t)
    net = _distance(samples[0], samples[-1])
    straightness = net / max(1e-6, path_length)
    return {
        "duration_s": float(duration_s),
        "path_length_px": float(path_length),
        "net_displacement_px": float(net),
        "straightness": float(straightness),
        "mean_speed_px_s": float(sum(speeds) / max(1, len(speeds))),
        "peak_speed_px_s": float(max(speeds) if speeds else 0.0),
        "abs_turn_rad": float(abs_turn),
        "turn_per_sec": float(abs_turn / duration_s),
        "direction_flips": float(flips),
    }


def _classify_segment(stats: Dict[str, float]) -> str:
    net = float(stats["net_displacement_px"])
    straight = float(stats["straightness"])
    abs_turn = float(stats["abs_turn_rad"])
    flips = float(stats["direction_flips"])
    if abs_turn >= 1.6 * math.pi and straight <= 0.55:
        return "orbit"
    if net <= 60.0 and (straight <= 0.72 or flips >= 2.0):
        return "micro_adjust"
    if net >= 180.0 and straight >= 0.84:
        return "travel"
    if flips >= 2.0 or abs_turn >= 0.9 * math.pi:
        return "swing"
    return "mixed"


def _choose_split_index(
    samples: Sequence[RawSample],
    metrics: Sequence[Dict[str, float]],
    start_idx: int,
    end_idx: int,
    *,
    min_points: int,
) -> int:
    lo = start_idx + min_points - 1
    hi = max(lo, end_idx)
    best_idx = hi
    best_score = -1e18
    local_speeds = [float(metrics[i]["speed"]) for i in range(max(start_idx + 1, lo - 4), min(len(metrics), hi + 5))]
    speed_ref = max(1.0, median(local_speeds) if local_speeds else 1.0)
    for idx in range(lo, hi + 1):
        row = metrics[idx]
        speed_score = 1.0 - _clamp(float(row["speed"]) / speed_ref, 0.0, 1.8)
        turn_score = _clamp(abs(float(row["turn"])) / math.radians(135.0), 0.0, 1.0)
        progress = (idx - start_idx) / max(1, hi - start_idx)
        progress_score = 1.0 - abs(progress - 0.72)
        drift_score = _clamp(float(row["abs_turn_accum"]) / (1.35 * math.pi), 0.0, 1.0)
        score = (speed_score * 0.45) + (turn_score * 0.30) + (progress_score * 0.10) + (drift_score * 0.15)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _prepare_metrics(samples: Sequence[RawSample]) -> List[Dict[str, float]]:
    n = len(samples)
    metrics: List[Dict[str, float]] = []
    headings: List[float] = [0.0] * n
    speeds: List[float] = [0.0] * n
    for i in range(1, n):
        dt = max(1e-6, samples[i].t - samples[i - 1].t)
        dist = _distance(samples[i - 1], samples[i])
        speeds[i] = dist / dt
        headings[i] = _heading(samples[i - 1], samples[i])
    abs_turn_accum = 0.0
    sign_flips_accum = 0.0
    prev_sign = 0
    for i in range(n):
        turn = 0.0
        if i >= 2:
            turn = _angle_diff(headings[i - 1], headings[i])
            abs_turn_accum += abs(turn)
            sign = 1 if turn > 1e-4 else (-1 if turn < -1e-4 else 0)
            if sign != 0 and prev_sign != 0 and sign != prev_sign:
                sign_flips_accum += 1.0
            if sign != 0:
                prev_sign = sign
        metrics.append(
            {
                "speed": float(speeds[i]),
                "heading": float(headings[i]),
                "turn": float(turn),
                "abs_turn_accum": float(abs_turn_accum),
                "sign_flips_accum": float(sign_flips_accum),
            }
        )
    return metrics


def _segment_stream(
    samples: Sequence[RawSample],
    *,
    min_points: int,
    min_duration_s: float,
    max_duration_s: float,
    max_path_px: float,
    max_net_disp_px: float,
    split_turn_deg: float,
) -> List[List[RawSample]]:
    if len(samples) < min_points:
        return []
    metrics = _prepare_metrics(samples)
    out: List[List[RawSample]] = []
    start = 0
    while start + min_points <= len(samples):
        seg_path = 0.0
        hard_cut_idx: Optional[int] = None
        for i in range(start + 1, len(samples)):
            seg_path += _distance(samples[i - 1], samples[i])
            duration_s = samples[i].t - samples[start].t
            net_disp = _distance(samples[start], samples[i])
            abs_turn = metrics[i]["abs_turn_accum"] - metrics[start]["abs_turn_accum"]
            flips = metrics[i]["sign_flips_accum"] - metrics[start]["sign_flips_accum"]
            strong_turn = abs(metrics[i]["turn"]) >= math.radians(split_turn_deg)
            dynamic_split = (
                i >= start + min_points
                and duration_s >= min_duration_s
                and strong_turn
                and net_disp >= 28.0
                and (abs_turn >= math.radians(160.0) or flips >= 2.0)
            )
            too_long = duration_s >= max_duration_s or seg_path >= max_path_px or net_disp >= max_net_disp_px
            orbit_like = duration_s >= min_duration_s and abs_turn >= 1.85 * math.pi and net_disp <= max(120.0, seg_path * 0.40)
            if dynamic_split or too_long or orbit_like:
                hard_cut_idx = i
                break
        if hard_cut_idx is None:
            hard_cut_idx = len(samples) - 1
        if hard_cut_idx - start + 1 < min_points:
            break
        cut_idx = _choose_split_index(samples, metrics, start, hard_cut_idx, min_points=min_points)
        if cut_idx - start + 1 < min_points:
            cut_idx = hard_cut_idx
        segment = list(samples[start : cut_idx + 1])
        stats = _segment_stats(segment)
        if (
            len(segment) >= min_points
            and stats["duration_s"] >= min_duration_s
            and stats["path_length_px"] >= 18.0
            and stats["net_displacement_px"] >= 8.0
        ):
            out.append(segment)
        start = max(cut_idx, start) + 1
    return out


def _smooth_and_rebuild(samples: Sequence[RawSample]) -> List[RawSample]:
    if len(samples) < 3:
        return list(samples)
    smoothed = _moving_average_points(samples, radius=2)
    rebuilt: List[RawSample] = []
    for src, (x, y) in zip(samples, smoothed):
        rebuilt.append(RawSample(t=float(src.t), x=int(round(x)), y=int(round(y))))
    return rebuilt


def _segment_to_record(samples: Sequence[RawSample], idx: int) -> Dict[str, Any]:
    stats = _segment_stats(samples)
    record = {
        "id": f"seg_{idx:06d}",
        "kind": _classify_segment(stats),
        "start_time": float(samples[0].t),
        "end_time": float(samples[-1].t),
        "duration_s": float(stats["duration_s"]),
        "path_length_px": float(stats["path_length_px"]),
        "net_displacement_px": float(stats["net_displacement_px"]),
        "straightness": float(stats["straightness"]),
        "mean_speed_px_s": float(stats["mean_speed_px_s"]),
        "peak_speed_px_s": float(stats["peak_speed_px_s"]),
        "abs_turn_rad": float(stats["abs_turn_rad"]),
        "direction_flips": int(round(stats["direction_flips"])),
        "points": [{"x": int(s.x), "y": int(s.y)} for s in samples],
        "trajectory": [[int(s.x), int(s.y), round(float(s.t - samples[0].t), 6)] for s in samples],
    }
    return record


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record continuous mouse motion and split it into short trajectory segments.")
    parser.add_argument("--hz", type=float, default=100.0, help="Sampling rate. Typical values: 60 or 100.")
    parser.add_argument("--start-paused", action="store_true", help="Start paused; use toggle key to begin recording.")
    parser.add_argument("--toggle-key", type=str, default="]", help="Global key used to toggle pause/resume.")
    parser.add_argument("--stop-key", type=str, default="}", help="Global key used to stop and save.")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR, help="Output directory for raw and segmented recordings.")
    parser.add_argument("--session-name", type=str, default=None, help="Optional fixed session name.")
    parser.add_argument("--min-step-px", type=float, default=1.0, help="Compression threshold for consecutive samples.")
    parser.add_argument("--keepalive-ms", type=float, default=35.0, help="Max time between kept compressed samples.")
    parser.add_argument("--min-points", type=int, default=8, help="Minimum points per trajectory segment.")
    parser.add_argument("--min-duration-ms", type=float, default=80.0, help="Minimum duration of a saved segment.")
    parser.add_argument("--max-duration-ms", type=float, default=900.0, help="Soft upper bound for segment duration.")
    parser.add_argument("--max-path-px", type=float, default=900.0, help="Soft upper bound for path length in one segment.")
    parser.add_argument("--max-net-disp-px", type=float, default=700.0, help="Soft upper bound for start->end displacement.")
    parser.add_argument("--split-turn-deg", type=float, default=105.0, help="Turning angle that can trigger a boundary.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        from pynput import keyboard, mouse  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            f"[record_mouse_trajectories] ERROR: cannot import pynput ({exc}). "
            "Install 'pynput' in the Python environment used for recording."
        )
    sample_interval = 1.0 / max(1.0, float(args.hz))
    keepalive_s = max(sample_interval, float(args.keepalive_ms) / 1000.0)
    min_duration_s = max(0.03, float(args.min_duration_ms) / 1000.0)
    max_duration_s = max(min_duration_s + 0.05, float(args.max_duration_ms) / 1000.0)
    session_name = str(args.session_name or time.strftime("mouse_session_%Y%m%d_%H%M%S"))
    session_dir = Path(args.out_dir).expanduser().resolve() / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("[record_mouse_trajectories] Continuous recorder started.")
    print(f"[record_mouse_trajectories] Sampling at {float(args.hz):.1f} Hz")
    print(f"[record_mouse_trajectories] Toggle key: {args.toggle_key!r} | Stop key: {args.stop_key!r}")
    print(f"[record_mouse_trajectories] Output dir: {session_dir}")
    print("[record_mouse_trajectories] Segmentation uses motion dynamics, not clicks or idle.")
    print("=" * 70)

    mouse_ctrl = mouse.Controller()
    state = {
        "recording": not bool(args.start_paused),
        "stop": False,
    }
    raw: List[RawSample] = []

    def _key_char(key: Any) -> Optional[str]:
        try:
            return getattr(key, "char", None)
        except Exception:
            return None

    def on_press(key: Any) -> Optional[bool]:
        ch = _key_char(key)
        if ch == str(args.toggle_key):
            state["recording"] = not state["recording"]
            status = "ON" if state["recording"] else "PAUSED"
            print(f"[record_mouse_trajectories] Recording {status}")
        elif ch == str(args.stop_key):
            state["stop"] = True
            print("[record_mouse_trajectories] Stop requested.")
            return False
        return None

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    started_at = time.perf_counter()
    last_print = started_at
    try:
        while not state["stop"]:
            now = time.perf_counter()
            x, y = mouse_ctrl.position
            if state["recording"]:
                raw.append(RawSample(t=float(now), x=int(x), y=int(y)))
            if (now - last_print) >= 5.0:
                elapsed = now - started_at
                print(
                    f"[record_mouse_trajectories] elapsed={elapsed:.1f}s "
                    f"raw_samples={len(raw)} mode={'REC' if state['recording'] else 'PAUSE'}"
                )
                last_print = now
            time.sleep(sample_interval)
    except KeyboardInterrupt:
        print("\n[record_mouse_trajectories] Ctrl+C received. Finishing session.")
    finally:
        state["stop"] = True
        listener.stop()

    if not raw:
        print("[record_mouse_trajectories] No samples captured.")
        return 1

    compressed = _compress_samples(raw, min_step_px=float(args.min_step_px), keepalive_s=keepalive_s)
    smoothed = _smooth_and_rebuild(compressed)
    segments = _segment_stream(
        smoothed,
        min_points=int(args.min_points),
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        max_path_px=float(args.max_path_px),
        max_net_disp_px=float(args.max_net_disp_px),
        split_turn_deg=float(args.split_turn_deg),
    )
    records = [_segment_to_record(seg, idx) for idx, seg in enumerate(segments, start=1)]

    raw_path = session_dir / "raw_samples.json"
    compressed_path = session_dir / "compressed_samples.json"
    segments_jsonl = session_dir / "segments.jsonl"
    summary_path = session_dir / "summary.json"

    _write_json(raw_path, [{"t": round(s.t, 6), "x": s.x, "y": s.y} for s in raw])
    _write_json(compressed_path, [{"t": round(s.t, 6), "x": s.x, "y": s.y} for s in smoothed])
    _write_jsonl(segments_jsonl, records)

    durations = sorted(float(r["duration_s"]) for r in records)
    net_disps = sorted(float(r["net_displacement_px"]) for r in records)
    path_lengths = sorted(float(r["path_length_px"]) for r in records)
    summary = {
        "session_name": session_name,
        "created_at": time.time(),
        "sampling_hz": float(args.hz),
        "toggle_key": str(args.toggle_key),
        "stop_key": str(args.stop_key),
        "raw_samples": len(raw),
        "compressed_samples": len(smoothed),
        "segments": len(records),
        "median_segment_duration_s": float(median(durations)) if durations else 0.0,
        "median_segment_net_disp_px": float(median(net_disps)) if net_disps else 0.0,
        "median_segment_path_length_px": float(median(path_lengths)) if path_lengths else 0.0,
        "kind_counts": {
            kind: sum(1 for r in records if r["kind"] == kind)
            for kind in sorted({str(r["kind"]) for r in records})
        },
        "files": {
            "raw_samples": str(raw_path),
            "compressed_samples": str(compressed_path),
            "segments_jsonl": str(segments_jsonl),
        },
        "segmentation": {
            "min_points": int(args.min_points),
            "min_duration_ms": float(args.min_duration_ms),
            "max_duration_ms": float(args.max_duration_ms),
            "max_path_px": float(args.max_path_px),
            "max_net_disp_px": float(args.max_net_disp_px),
            "split_turn_deg": float(args.split_turn_deg),
            "uses_idle": False,
            "uses_clicks": False,
        },
    }
    _write_json(summary_path, summary)

    print("=" * 70)
    print(f"[record_mouse_trajectories] Saved session: {session_dir}")
    print(
        f"[record_mouse_trajectories] raw={len(raw)} compressed={len(smoothed)} "
        f"segments={len(records)} median_duration={summary['median_segment_duration_s']:.3f}s"
    )
    print(f"[record_mouse_trajectories] segments_jsonl -> {segments_jsonl}")
    print(f"[record_mouse_trajectories] summary -> {summary_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
