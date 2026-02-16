from __future__ import annotations

import json
import socket
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
