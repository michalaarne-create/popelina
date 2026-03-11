from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Optional

from PIL import Image


def send_click_from_bbox(
    bbox,
    image_path: Optional[Path],
    context_label: str,
    *,
    send_control_agent,
    control_agent_port: int,
    log,
    update_overlay_status,
    screen_click_offset_x: int = 0,
    screen_click_offset_y: int = 0,
) -> bool:
    if not bbox or len(bbox) != 4:
        log(f"[WARN] {context_label}: invalid bbox.")
        update_overlay_status(f"{context_label}: invalid bbox.")
        return False
    if image_path and image_path.exists():
        try:
            with Image.open(image_path) as im:
                screen_w, screen_h = im.size
        except Exception:
            screen_w, screen_h = (1920, 1080)
    else:
        screen_w, screen_h = (1920, 1080)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(screen_w, x2), min(screen_h, y2)
    if x2 - x1 <= 4 or y2 - y1 <= 4:
        log(f"[WARN] {context_label}: bounding box too small.")
        update_overlay_status(f"{context_label}: bbox too small.")
        return False
    margin = 10
    inner_x1, inner_x2 = (x1 + margin, x2 - margin) if (x2 - x1) > 2 * margin + 1 else (x1, x2)
    inner_y1, inner_y2 = (y1 + margin, y2 - margin) if (y2 - y1) > 2 * margin + 1 else (y1, y2)
    deterministic = str(os.environ.get("FULLBOT_QUIZ_MODE", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
    if inner_x2 <= inner_x1 or inner_y2 <= inner_y1 or deterministic:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    else:
        cx, cy = random.randint(inner_x1, inner_x2), random.randint(inner_y1, inner_y2)
    tx = int(cx + int(screen_click_offset_x))
    ty = int(cy + int(screen_click_offset_y))
    if send_control_agent({"cmd": "move", "x": tx, "y": ty, "press": "mouse"}, control_agent_port):
        if "[FALLBACK]" in str(context_label):
            log(f"[FALLBACK] {context_label}: click at ({tx}, {ty})")
        else:
            log(f"[INFO] {context_label}: click at ({tx}, {ty})")
        update_overlay_status(f"{context_label} click at ({tx}, {ty})")
        return True
    update_overlay_status(f"{context_label}: failed to send click.")
    return False


def scroll_on_box(
    bbox,
    image_path: Optional[Path],
    context_label: str,
    *,
    send_control_agent,
    control_agent_port: int,
    log,
    total_notches: int = 8,
    direction: str = "down",
    screen_click_offset_x: int = 0,
    screen_click_offset_y: int = 0,
) -> bool:
    if not bbox or len(bbox) != 4:
        log(f"[WARN] {context_label}: invalid bbox for scroll_on_box.")
        return False
    if image_path and image_path.exists():
        try:
            with Image.open(image_path) as im:
                screen_w, screen_h = im.size
        except Exception:
            screen_w, screen_h = (1920, 1080)
    else:
        screen_w, screen_h = (1920, 1080)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(screen_w, x2), min(screen_h, y2)
    if x2 - x1 <= 4 or y2 - y1 <= 4:
        log(f"[WARN] {context_label}: bbox too small for scroll_on_box.")
        return False
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    tx = int(cx + int(screen_click_offset_x))
    ty = int(cy + int(screen_click_offset_y))
    if not send_control_agent({"cmd": "move", "x": tx, "y": ty}, control_agent_port):
        log(f"[WARN] {context_label}: failed to move before scroll_on_box.")
        return False
    time.sleep(random.uniform(0.06, 0.14))
    fast_duration = max(0.05, 0.35 / 4.0)
    remaining = max(1, int(total_notches))
    while remaining > 0:
        send_control_agent(
            {"cmd": "scroll", "direction": direction, "amount": 1, "duration": fast_duration * random.uniform(0.75, 1.25)},
            control_agent_port,
        )
        remaining -= 1
        time.sleep(fast_duration * random.uniform(0.6, 1.4))
    log(f"[INFO] {context_label}: scroll_on_box done (notches={total_notches})")
    return True
