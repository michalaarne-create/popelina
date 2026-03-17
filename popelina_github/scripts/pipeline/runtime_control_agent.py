from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict


def send_udp_payload(payload: Dict[str, Any], port: int, *, log) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = json.dumps(payload).encode("utf-8")
        sock.sendto(data, ("127.0.0.1", port))
        try:
            pts = payload.get("points", [])
            log(f"[DEBUG] UDP sent to {port}: size={len(data)} bytes, points={len(pts)}")
        except Exception:
            log(f"[DEBUG] UDP sent to {port}: size={len(data)} bytes")
        return True
    except Exception as exc:
        log(f"[WARN] Failed to send payload to control agent: {exc}")
        return False


def send_control_agent(payload: Dict[str, Any], port: int, *, log) -> bool:
    cmd = payload.get("cmd")
    if cmd == "path":
        try:
            size = len(json.dumps(payload))
            log(f"[DEBUG] control_agent path payload size={size} bytes, points={len(payload.get('points', []))}")
        except Exception:
            pass
    else:
        log(f"[DEBUG] control_agent send generic cmd={cmd} -> {payload}")
    return send_udp_payload(payload, port, log=log)


def send_key(combo: str, port: int, *, log) -> bool:
    combo = str(combo or "").strip()
    if not combo:
        return False
    return send_control_agent({"cmd": "keys", "combo": combo}, port, log=log)


def send_key_repeat(key: str, count: int, port: int, *, log, delay_s: float = 0.05) -> bool:
    key = str(key or "").strip()
    total = max(0, int(count or 0))
    if not key or total <= 0:
        return False
    ok = True
    for _ in range(total):
        ok = send_control_agent({"cmd": "keys", "combo": key}, port, log=log) and ok
        if delay_s > 0:
            time.sleep(delay_s)
    return ok


def send_type(text: str, port: int, *, log, delay: float = 0.0) -> bool:
    text = str(text or "")
    if not text:
        return False
    return send_control_agent({"cmd": "type", "text": text, "delay": float(delay)}, port, log=log)


def send_wait(ms: int, *, log) -> bool:
    delay = max(0.0, float(ms or 0) / 1000.0)
    if delay > 0:
        log(f"[DEBUG] control_agent wait {delay:.3f}s")
        time.sleep(delay)
    return True
