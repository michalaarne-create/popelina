from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.runtime_control_agent import send_control_agent


Point = Tuple[float, float]


PROFILES: Dict[str, Dict[str, float]] = {
    "natural": {
        "curve_scale": 0.16,
        "curve_cap": 115.0,
        "overshoot_chance": 0.34,
        "overshoot_min": 4.0,
        "overshoot_max": 12.0,
        "noise_px": 1.6,
        "min_points": 24.0,
        "points_per_px": 0.11,
        "min_duration_ms": 180.0,
        "max_duration_ms": 1450.0,
        "min_dt": 0.008,
        "swing_chance": 0.80,
        "loop_chance": 0.48,
        "ellipse_chance": 0.62,
        "lobe_scale": 0.19,
        "lobe_cap": 160.0,
        "orbit_long_scale": 0.11,
        "turn_slowdown": 0.72,
        "phase_power": 1.55,
    },
    "balanced": {
        "curve_scale": 0.12,
        "curve_cap": 85.0,
        "overshoot_chance": 0.22,
        "overshoot_min": 3.0,
        "overshoot_max": 8.0,
        "noise_px": 1.0,
        "min_points": 18.0,
        "points_per_px": 0.085,
        "min_duration_ms": 140.0,
        "max_duration_ms": 1100.0,
        "min_dt": 0.006,
        "swing_chance": 0.62,
        "loop_chance": 0.24,
        "ellipse_chance": 0.40,
        "lobe_scale": 0.13,
        "lobe_cap": 105.0,
        "orbit_long_scale": 0.08,
        "turn_slowdown": 0.45,
        "phase_power": 1.35,
    },
}

CONTROL_AGENT_PATH = PROJECT_ROOT / "scripts" / "click" / "control_agent" / "control_agent.py"
CONTROL_AGENT_CONFIG = PROJECT_ROOT / "scripts" / "click" / "control_agent" / "train.json"


def _log(msg: str) -> None:
    print(msg)


def _resolve_agent_python() -> str:
    venv_python = PROJECT_ROOT.parent / ".venv312" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    if importlib.util.find_spec("pynput") is not None:
        return sys.executable
    return sys.executable


def _is_udp_port_open(port: int, *, host: str = "127.0.0.1", timeout_s: float = 0.15) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, int(port)))
        return False
    except OSError:
        return True
    finally:
        sock.close()


def _wait_for_udp_listener(port: int, *, timeout_s: float = 6.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _is_udp_port_open(port):
            return True
        time.sleep(0.12)
    return False


def _start_control_agent(port: int, *, verbose: bool = False) -> subprocess.Popen[Any]:
    creationflags = 0
    popen_kwargs: Dict[str, Any] = {
        "cwd": str(PROJECT_ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        popen_kwargs["creationflags"] = creationflags
    cmd = [
        _resolve_agent_python(),
        str(CONTROL_AGENT_PATH),
        "--port",
        str(int(port)),
        "--config",
        str(CONTROL_AGENT_CONFIG),
    ]
    if verbose:
        cmd.append("--verbose")
    return subprocess.Popen(cmd, **popen_kwargs)


def _ensure_control_agent_running(port: int, *, verbose: bool = False) -> None:
    if _is_udp_port_open(port):
        return
    _log(f"[INFO] control_agent is not listening on UDP {int(port)}. Starting it in background.")
    try:
        proc = _start_control_agent(port, verbose=verbose)
    except Exception as exc:
        raise RuntimeError(f"Could not start control_agent: {exc}") from exc
    if proc.poll() is not None:
        raise RuntimeError(f"control_agent exited immediately with code {proc.returncode}.")
    if not _wait_for_udp_listener(port):
        raise RuntimeError(f"control_agent did not open UDP port {int(port)} in time.")
    _log(f"[INFO] control_agent is ready on UDP {int(port)}.")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _normalize(dx: float, dy: float) -> Point:
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return 1.0, 0.0
    return dx / norm, dy / norm


def _ease_in_out(t: float) -> float:
    return 3.0 * t * t - 2.0 * t * t * t


def _bezier_point(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    u = 1.0 - t
    tt = t * t
    uu = u * u
    uuu = uu * u
    ttt = tt * t
    x = (uuu * p0[0]) + (3.0 * uu * t * p1[0]) + (3.0 * u * tt * p2[0]) + (ttt * p3[0])
    y = (uuu * p0[1]) + (3.0 * uu * t * p1[1]) + (3.0 * u * tt * p2[1]) + (ttt * p3[1])
    return x, y


def _dedupe_points(points: Sequence[Point]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    last: Optional[Tuple[int, int]] = None
    for px, py in points:
        item = (int(round(px)), int(round(py)))
        if item != last:
            out.append(item)
            last = item
    return out


def _smoothstep_window(t: float, start: float, end: float) -> float:
    if t <= start or t >= end:
        return 0.0
    mid = (start + end) * 0.5
    if t <= mid:
        x = (t - start) / max(1e-6, mid - start)
        return x * x * (3.0 - 2.0 * x)
    x = (end - t) / max(1e-6, end - mid)
    return x * x * (3.0 - 2.0 * x)


def _gauss_window(t: float, center: float, width: float) -> float:
    width = max(0.03, width)
    z = (t - center) / width
    return math.exp(-0.5 * z * z)


def _build_control_points(start: Point, end: Point, *, rnd: random.Random, profile: Dict[str, float]) -> Tuple[Point, Point]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = max(1.0, math.hypot(dx, dy))
    ux = dx / dist
    uy = dy / dist
    px = -uy
    py = ux
    lateral = min(profile["curve_cap"], dist * profile["curve_scale"])
    if dist < 80.0:
        lateral *= 0.45
    elif dist < 180.0:
        lateral *= 0.75
    sign = -1.0 if rnd.random() < 0.5 else 1.0
    lateral *= sign * rnd.uniform(0.55, 1.0)
    c1_forward = dist * rnd.uniform(0.18, 0.34)
    c2_forward = dist * rnd.uniform(0.64, 0.86)
    c1 = (
        start[0] + ux * c1_forward + px * lateral * rnd.uniform(0.75, 1.0),
        start[1] + uy * c1_forward + py * lateral * rnd.uniform(0.75, 1.0),
    )
    c2 = (
        start[0] + ux * c2_forward - px * lateral * rnd.uniform(0.45, 0.9),
        start[1] + uy * c2_forward - py * lateral * rnd.uniform(0.45, 0.9),
    )
    return c1, c2


def _sample_curve(start: Point, end: Point, *, rnd: random.Random, profile: Dict[str, float], point_count: int) -> List[Point]:
    c1, c2 = _build_control_points(start, end, rnd=rnd, profile=profile)
    points: List[Point] = []
    dist = max(1.0, _distance(start, end))
    noise_cap = min(profile["noise_px"], dist * 0.008)
    for idx in range(point_count):
        raw_t = idx / max(1, point_count - 1)
        t = _ease_in_out(raw_t)
        x, y = _bezier_point(start, c1, c2, end, t)
        if 0 < idx < point_count - 1 and noise_cap > 0.0:
            bell = math.sin(math.pi * raw_t)
            x += rnd.gauss(0.0, noise_cap * 0.35) * bell
            y += rnd.gauss(0.0, noise_cap * 0.35) * bell
        points.append((x, y))
    return points


def _build_swing_points(start: Point, end: Point, *, rnd: random.Random, profile: Dict[str, float], point_count: int) -> List[Point]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = max(1.0, math.hypot(dx, dy))
    ux, uy = _normalize(dx, dy)
    px, py = -uy, ux
    lateral = min(profile["lobe_cap"], dist * profile["lobe_scale"]) * rnd.uniform(0.55, 1.0)
    if dist < 120.0:
        lateral *= 0.45
    sign = -1.0 if rnd.random() < 0.5 else 1.0
    freq = 1 if dist < 260.0 else (2 if rnd.random() < 0.7 else 3)
    phase = rnd.uniform(-0.45, 0.45)
    long_wobble = min(dist * profile["orbit_long_scale"], lateral * 0.75) * rnd.uniform(0.35, 1.0)
    points: List[Point] = []
    for idx in range(point_count):
        raw_t = idx / max(1, point_count - 1)
        t = _ease_in_out(raw_t)
        base_x = start[0] + dx * t
        base_y = start[1] + dy * t
        envelope = math.sin(math.pi * raw_t) ** 1.2
        osc = math.sin(freq * math.pi * raw_t + phase)
        osc2 = math.sin((freq + 0.5) * math.pi * raw_t + phase * 0.5)
        lat = sign * lateral * envelope * osc
        lon = long_wobble * envelope * osc2 * 0.35
        x = base_x + px * lat + ux * lon
        y = base_y + py * lat + uy * lon
        points.append((x, y))
    return points


def _build_orbit_points(start: Point, end: Point, *, rnd: random.Random, profile: Dict[str, float], point_count: int) -> List[Point]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = max(1.0, math.hypot(dx, dy))
    ux, uy = _normalize(dx, dy)
    px, py = -uy, ux
    center_t = rnd.uniform(0.32, 0.68)
    center_x = start[0] + dx * center_t
    center_y = start[1] + dy * center_t
    radius_lat = min(profile["lobe_cap"], dist * profile["lobe_scale"]) * rnd.uniform(0.6, 1.05)
    radius_long = min(dist * 0.18, max(12.0, dist * profile["orbit_long_scale"])) * rnd.uniform(0.8, 1.2)
    if dist < 180.0:
        radius_lat *= 0.55
        radius_long *= 0.5
    direction = -1.0 if rnd.random() < 0.5 else 1.0
    loops = 1.0 if rnd.random() < 0.72 else rnd.uniform(0.72, 1.35)
    theta_start = rnd.uniform(-0.9 * math.pi, -0.35 * math.pi)
    theta_end = theta_start + direction * (loops * 2.0 * math.pi * rnd.uniform(0.7, 1.0))
    width = rnd.uniform(0.12, 0.22)
    points: List[Point] = []
    for idx in range(point_count):
        raw_t = idx / max(1, point_count - 1)
        t = _ease_in_out(raw_t)
        base_x = start[0] + dx * t
        base_y = start[1] + dy * t
        orbit_w = _gauss_window(raw_t, center_t, width)
        theta = theta_start + (theta_end - theta_start) * raw_t
        orbit_long = math.cos(theta) * radius_long * orbit_w
        orbit_lat = math.sin(theta) * radius_lat * orbit_w
        x = base_x + ux * orbit_long + px * orbit_lat
        y = base_y + uy * orbit_long + py * orbit_lat
        points.append((x, y))
    return points


def _blend_paths(primary: Sequence[Point], secondary: Sequence[Point], *, amount: float) -> List[Point]:
    n = min(len(primary), len(secondary))
    if n <= 0:
        return list(primary)
    out: List[Point] = []
    for idx in range(n):
        ax, ay = primary[idx]
        bx, by = secondary[idx]
        out.append((ax * (1.0 - amount) + bx * amount, ay * (1.0 - amount) + by * amount))
    return out


def _overshoot_target(start: Point, end: Point, *, rnd: random.Random, profile: Dict[str, float]) -> Optional[Point]:
    dist = _distance(start, end)
    if dist < 220.0:
        return None
    if rnd.random() >= profile["overshoot_chance"]:
        return None
    overshoot = rnd.uniform(profile["overshoot_min"], profile["overshoot_max"])
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    unit = max(1.0, math.hypot(dx, dy))
    ux = dx / unit
    uy = dy / unit
    return end[0] + ux * overshoot, end[1] + uy * overshoot


def _generate_path_points(start: Point, end: Point, *, rnd: random.Random, profile: Dict[str, float]) -> List[Tuple[int, int]]:
    dist = _distance(start, end)
    if dist < 2.0:
        return _dedupe_points([start, end])
    point_count = int(
        max(
            profile["min_points"],
            min(140.0, profile["min_points"] + dist * profile["points_per_px"]),
        )
    )
    primary_style = "curve"
    style_roll = rnd.random()
    if dist >= 80.0 and style_roll < profile.get("loop_chance", 0.0):
        primary_style = "orbit"
    elif dist >= 35.0 and style_roll < profile.get("ellipse_chance", 0.0):
        primary_style = "swing"
    if primary_style == "orbit":
        points = _build_orbit_points(start, end, rnd=rnd, profile=profile, point_count=point_count)
    elif primary_style == "swing":
        points = _build_swing_points(start, end, rnd=rnd, profile=profile, point_count=point_count)
    else:
        points = _sample_curve(start, end, rnd=rnd, profile=profile, point_count=point_count)
    if dist >= 70.0 and rnd.random() < profile.get("swing_chance", 0.0):
        swing = _build_swing_points(start, end, rnd=rnd, profile=profile, point_count=point_count)
        points = _blend_paths(points, swing, amount=rnd.uniform(0.25, 0.55))
    overshoot = _overshoot_target(start, end, rnd=rnd, profile=profile)
    if overshoot is None:
        points[0] = start
        points[-1] = end
        return _dedupe_points(points)
    first_count = max(12, int(point_count * rnd.uniform(0.72, 0.84)))
    second_count = max(8, point_count - first_count + 6)
    first = _generate_path_points(start, overshoot, rnd=rnd, profile={**profile, "overshoot_chance": 0.0},)
    correction_profile = dict(profile)
    correction_profile["curve_scale"] = max(0.03, profile["curve_scale"] * 0.35)
    correction_profile["curve_cap"] = max(10.0, profile["curve_cap"] * 0.18)
    correction_profile["noise_px"] = max(0.0, profile["noise_px"] * 0.2)
    correction_profile["loop_chance"] = 0.0
    correction_profile["ellipse_chance"] = min(0.18, correction_profile.get("ellipse_chance", 0.0))
    second = _sample_curve(overshoot, end, rnd=rnd, profile=correction_profile, point_count=second_count)
    merged = [(float(x), float(y)) for x, y in first[:-1]] + second
    merged[0] = start
    merged[-1] = end
    return _dedupe_points(merged)


def _default_duration_ms(points: Sequence[Tuple[int, int]], profile: Dict[str, float], *, rnd: random.Random) -> int:
    if len(points) < 2:
        return int(profile["min_duration_ms"])
    total = 0.0
    turn_energy = 0.0
    for p0, p1 in zip(points, points[1:]):
        total += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    for p0, p1, p2 in zip(points, points[1:], points[2:]):
        v1x, v1y = p1[0] - p0[0], p1[1] - p0[1]
        v2x, v2y = p2[0] - p1[0], p2[1] - p1[1]
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 <= 1e-6 or n2 <= 1e-6:
            continue
        cos_a = _clamp((v1x * v2x + v1y * v2y) / (n1 * n2), -1.0, 1.0)
        angle = math.acos(cos_a)
        turn_energy += angle / math.pi
    base = 120.0 + total * rnd.uniform(1.1, 1.7)
    base *= 1.0 + min(0.55, turn_energy / max(12.0, len(points)))
    return int(round(_clamp(base, profile["min_duration_ms"], profile["max_duration_ms"])))


def build_payload(
    start: Tuple[int, int],
    end: Tuple[int, int],
    *,
    profile_name: str,
    seed: Optional[int],
    duration_ms: Optional[int],
    speed_px_per_s: Optional[float],
    speed_factor: Optional[float],
    min_dt: Optional[float],
    trace_stem: Optional[str],
    press: Optional[str],
) -> Dict[str, Any]:
    profile = PROFILES[profile_name]
    actual_seed = int(seed if seed is not None else time.time_ns() & 0xFFFFFFFF)
    rnd = random.Random(actual_seed)
    points = _generate_path_points((float(start[0]), float(start[1])), (float(end[0]), float(end[1])), rnd=rnd, profile=profile)
    payload: Dict[str, Any] = {
        "cmd": "path",
        "points": [{"x": px, "y": py} for px, py in points],
        "min_dt": float(min_dt if min_dt is not None else profile["min_dt"]),
        "turn_slowdown": float(profile.get("turn_slowdown", 0.0)),
        "phase_power": float(profile.get("phase_power", 1.4)),
    }
    if duration_ms is not None:
        payload["duration_ms"] = int(duration_ms)
    elif speed_px_per_s is None:
        payload["duration_ms"] = _default_duration_ms(points, profile, rnd=rnd)
    if speed_px_per_s is not None:
        payload["speed_px_per_s"] = float(speed_px_per_s)
    if speed_factor is not None:
        payload["speed_factor"] = float(speed_factor)
    if trace_stem:
        payload["trace_stem"] = str(trace_stem)
    if press and press != "none":
        payload["press"] = press
    payload["_meta"] = {
        "profile": profile_name,
        "seed": actual_seed,
        "distance_px": round(_distance(start, end), 2),
        "points_count": len(points),
    }
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a human-like mouse path and send it to control_agent.")
    parser.add_argument("x1", type=int)
    parser.add_argument("y1", type=int)
    parser.add_argument("x2", type=int)
    parser.add_argument("y2", type=int)
    parser.add_argument("--port", type=int, default=8765, help="UDP port used by control_agent.")
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="natural")
    parser.add_argument("--duration-ms", type=int, default=None, help="Force total path duration in ms.")
    parser.add_argument("--speed-px-per-s", type=float, default=None, help="Let control_agent rescale timing to a target pixels/sec speed.")
    parser.add_argument("--speed-factor", type=float, default=None, help="Additional timing multiplier consumed by control_agent.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for repeatable trajectories.")
    parser.add_argument("--min-dt", type=float, default=None, help="Minimum dt between control_agent steps.")
    parser.add_argument("--trace-stem", type=str, default=None, help="Optional debug trace stem understood by control_agent.")
    parser.add_argument("--press", choices=("none", "mouse"), default="none", help="Optionally hold left mouse during the path.")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to save the generated payload.")
    parser.add_argument("--debug", action="store_true", help="Print payload details and metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Generate payload only, do not send it.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    start = (int(args.x1), int(args.y1))
    end = (int(args.x2), int(args.y2))
    if start == end:
        _log("[ERROR] Start and end points must be different.")
        return 2
    payload = build_payload(
        start,
        end,
        profile_name=str(args.profile),
        seed=args.seed,
        duration_ms=args.duration_ms,
        speed_px_per_s=args.speed_px_per_s,
        speed_factor=args.speed_factor,
        min_dt=args.min_dt,
        trace_stem=args.trace_stem,
        press=args.press,
    )
    meta = payload.pop("_meta", {})
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.debug or args.dry_run:
        _log(
            f"[INFO] profile={meta.get('profile')} seed={meta.get('seed')} "
            f"distance_px={meta.get('distance_px')} points={meta.get('points_count')}"
        )
        _log(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    try:
        _ensure_control_agent_running(int(args.port), verbose=bool(args.debug))
    except Exception as exc:
        _log(f"[ERROR] {exc}")
        return 1
    sent = send_control_agent(payload, int(args.port), log=_log)
    if not sent:
        _log(f"[ERROR] Failed to send path to control_agent on UDP port {int(args.port)}.")
        return 1
    _log(
        f"[OK] Sent human path to control_agent port {int(args.port)} "
        f"(profile={meta.get('profile')}, seed={meta.get('seed')}, points={meta.get('points_count')})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
