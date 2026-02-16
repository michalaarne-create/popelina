from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image


def find_screenshot_for_summary(summary_path: Path, *, screenshot_dir: Path, image_exts: Tuple[str, ...]) -> Optional[Path]:
    stem = summary_path.stem
    if stem.endswith("_summary"):
        stem = stem[: -len("_summary")]
    for ext in image_exts:
        candidate = screenshot_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def send_random_click(
    summary_path: Path,
    image_path: Path,
    *,
    send_control_agent,
    control_agent_port: int,
    cancel_hover_fallback_timer,
    log,
    update_overlay_status,
) -> None:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"[WARN] Could not read summary JSON {summary_path}: {exc}")
        update_overlay_status("Summary JSON missing or invalid.")
        return
    top = data.get("top_labels") or {}
    candidates = [entry for entry in top.values() if isinstance(entry, dict) and entry.get("bbox")]
    if not candidates:
        log("[WARN] Summary has no candidates for random click.")
        update_overlay_status("No candidates for random click.")
        return
    try:
        with Image.open(image_path) as im:
            screen_w, screen_h = im.size
    except Exception:
        screen_w, screen_h = (1920, 1080)
    bbox = random.choice(candidates).get("bbox") or []
    if not bbox or len(bbox) != 4:
        log("[WARN] Candidate without bbox for random click.")
        update_overlay_status("Candidate without bbox for random click.")
        return
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(screen_w, x2), min(screen_h, y2)
    if x2 - x1 <= 10 or y2 - y1 <= 10:
        log("[WARN] Bounding box too small for random click.")
        update_overlay_status("Bounding box too small for random click.")
        return
    rx_min, rx_max = max(5, x1 + 5), min(screen_w - 5, x2 - 5)
    ry_min, ry_max = max(5, y1 + 5), min(screen_h - 5, y2 - 5)
    if rx_max <= rx_min or ry_max <= ry_min:
        log("[WARN] No space to place random click inside bbox.")
        update_overlay_status("No space for random click.")
        return
    rand_x, rand_y = random.randint(rx_min, rx_max), random.randint(ry_min, ry_max)
    if send_control_agent({"cmd": "move", "x": rand_x, "y": rand_y}, control_agent_port):
        cancel_hover_fallback_timer()
        log(f"[INFO] Random click sent at ({rand_x}, {rand_y}) from {summary_path.name}")
        update_overlay_status(f"Random click at ({rand_x}, {rand_y})")
    else:
        update_overlay_status("Failed to send random click.")


def send_best_click(
    summary_path: Path,
    image_path: Optional[Path],
    *,
    send_control_agent,
    control_agent_port: int,
    log,
    update_overlay_status,
) -> None:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"[WARN] Could not read summary JSON {summary_path}: {exc}")
        update_overlay_status("Summary JSON missing.")
        return
    top = data.get("top_labels") or {}
    candidates = [entry for entry in top.values() if isinstance(entry, dict) and entry.get("bbox")]
    if not candidates:
        log("[WARN] Summary has no candidates for best click.")
        update_overlay_status("No candidates for best click.")
        return
    bbox = max(candidates, key=lambda e: float(e.get("score", e.get("confidence", 0.0)))).get("bbox") or []
    if not bbox or len(bbox) != 4:
        log("[WARN] Candidate without bbox for best click.")
        update_overlay_status("Candidate without bbox.")
        return
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
        log("[WARN] Bounding box too small for best click.")
        update_overlay_status("Bounding box too small.")
        return
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    if send_control_agent({"cmd": "move", "x": cx, "y": cy}, control_agent_port):
        update_overlay_status(f"Best click at ({cx}, {cy})")
        log(f"[INFO] Best click sent for {summary_path.name}")
    else:
        update_overlay_status("Failed to send best click.")
