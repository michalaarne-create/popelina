"""
Pick a random recorded mouse trajectory (JSONL), reinterpret it as relative deltas
anchored at the current cursor position, and send it to control_agent over UDP.

Usage (PowerShell):
  python utils/send_random_behavior_path.py `
    --data data/raw_recordings.jsonl `
    --port 8765 `
    --scale 1.0 `
    --screen_width 1920 --screen_height 1080 `
    --auto-start-agent

Assumptions:
- Each JSONL line: {"points": [{"x": float, "y": float, ...}, ...]} optionally with "screen".
- If coords look normalized (max <= 1.5), they are scaled by screen_width/height.
- control_agent listens for UDP {"cmd": "path"} payloads on given port.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pynput.mouse import Controller

try:
    import psutil  # optional, for process detection
except Exception:
    psutil = None


def load_sequences(path: Path, min_points: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pts = rec.get("points") or []
            if len(pts) < min_points:
                continue
            samples.append(rec)
    return samples


def maybe_denormalize(coords: np.ndarray, screen_w: float, screen_h: float) -> np.ndarray:
    # If all values are small, assume normalized [0,1]
    if coords.max() <= 1.5:
        coords[:, 0] *= screen_w
        coords[:, 1] *= screen_h
    return coords


def build_path_from_sample(
    sample: Dict[str, Any],
    current_pos: Tuple[float, float],
    scale: float,
    screen_w: int,
    screen_h: int,
) -> List[Tuple[int, int]]:
    pts = np.array([[float(p["x"]), float(p["y"])] for p in sample.get("points", [])], dtype=float)
    sw = float(sample.get("screen", {}).get("width", screen_w))
    sh = float(sample.get("screen", {}).get("height", screen_h))
    pts = maybe_denormalize(pts, sw, sh)

    deltas = np.zeros_like(pts)
    deltas[1:] = np.diff(pts, axis=0) * float(scale)

    cx, cy = current_pos
    path = [(int(cx), int(cy))]
    x, y = float(cx), float(cy)
    for dx, dy in deltas[1:]:
        x = max(0, min(screen_w, x + dx))
        y = max(0, min(screen_h, y + dy))
        path.append((int(round(x)), int(round(y))))
    return path


def send_path(path: List[Tuple[int, int]], port: int) -> None:
    payload = {
        "cmd": "path",
        "points": [{"x": int(x), "y": int(y)} for x, y in path],
        "speed": "normal",
        "min_total_ms": 0.0,
        "speed_factor": 1.0,
        "min_dt": 0.004,
        "gap_rects": [],
        "gap_boost": 1.0,
        "line_jump_indices": [],
        "line_jump_boost": 1.0,
    }
    data = json.dumps(payload).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(data, ("127.0.0.1", port))


def _control_agent_running() -> bool:
    target = "control_agent.py"
    if psutil:
        try:
            for p in psutil.process_iter(["cmdline"]):
                cmd = " ".join(p.info.get("cmdline") or [])
                if target in cmd:
                    return True
        except Exception:
            pass
    if platform.system() == "Windows":
        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.CommandLine -like '*control_agent.py*' } | "
                "Select-Object -First 1 -ExpandProperty CommandLine",
            ]
            out = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if out.returncode == 0 and out.stdout.strip():
                return True
        except Exception:
            pass
    return False


def ensure_control_agent(port: int) -> None:
    """Start control_agent.py if not running. Console window hidden."""
    if _control_agent_running():
        return
    root = Path(__file__).resolve().parents[1]
    agent = root / "scripts" / "control_agent" / "control_agent.py"
    config = root / "scripts" / "control_agent" / "train.json"
    if not agent.exists():
        print(f"[send_random_behavior_path] control_agent.py not found at {agent}")
        return
    creationflags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
    cmd = [sys.executable, str(agent), "--config", str(config), "--port", str(port)]
    log_path = root / "control_agent_autostart.log"
    try:
        with log_path.open("a", encoding="utf-8") as logf:
            subprocess.Popen(
                cmd,
                cwd=str(root),
                stdout=logf,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                start_new_session=True,
            )
        time.sleep(0.8)
        if not _control_agent_running():
            print("[send_random_behavior_path] auto-start FAILED, see log:", log_path)
        else:
            print("[send_random_behavior_path] auto-started control_agent (hidden). Log:", log_path)
    except Exception as e:
        print(f"[send_random_behavior_path] failed to start control_agent: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick random trajectory from JSONL and send to control_agent as path.")
    ap.add_argument("--data", type=str, required=True, help="JSONL with trajectories (e.g., data/raw_recordings.jsonl).")
    ap.add_argument("--port", type=int, default=8765, help="UDP port of control_agent.")
    ap.add_argument("--min_points", type=int, default=8, help="Minimum points in a sequence.")
    ap.add_argument("--scale", type=float, default=1.0, help="Scale factor for step lengths.")
    ap.add_argument("--screen_width", type=int, default=1920, help="Screen width in pixels.")
    ap.add_argument("--screen_height", type=int, default=1080, help="Screen height in pixels.")
    ap.add_argument("--auto-start-agent", action="store_true", help="If control_agent is down, start it (hidden).")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {data_path}")

    samples = load_sequences(data_path, min_points=args.min_points)
    if not samples:
        raise RuntimeError(f"No sequences with >= {args.min_points} points in {data_path}")

    sample = random.choice(samples)
    mouse = Controller()
    current_pos = mouse.position
    path = build_path_from_sample(
        sample,
        current_pos=current_pos,
        scale=args.scale,
        screen_w=args.screen_width,
        screen_h=args.screen_height,
    )

    if len(path) < 2:
        raise RuntimeError("Path has too few points after processing.")

    if args.auto_start_agent:
        ensure_control_agent(port=args.port)

    send_path(path, args.port)
    print(
        f"[send_random_behavior_path] Sent path with {len(path)} points to port {args.port} "
        f"(anchored at cursor {current_pos}, scale={args.scale})"
    )


if __name__ == "__main__":
    main()

