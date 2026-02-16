from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def dispatch_hover_to_control_agent(
    points_json: Path,
    *,
    hover_input_current_dir: Path,
    raw_current_dir: Path,
    hold_left_button: bool,
    control_agent_port: int,
    build_hover_path,
    save_hover_path_visual,
    save_hover_overlay_from_json,
    send_control_agent,
    start_hover_fallback_timer,
    log,
) -> None:
    try:
        data = json.loads(points_json.read_text(encoding="utf-8"))
        seqs = data.get("sequences") if isinstance(data, dict) else data
        seqs = seqs or []
        if not isinstance(seqs, list):
            raise ValueError("Hover JSON must contain a list or dict with 'sequences'")
    except Exception as exc:
        log(f"[WARN] Could not read hover JSON {points_json}: {exc}")
        return
    if not seqs:
        fallback_img = hover_input_current_dir / "hover_input.png"
        if not fallback_img.exists():
            fallback_img = raw_current_dir / "screenshot.png"
        try:
            with Image.open(fallback_img) as im:
                w, h = im.size
        except Exception:
            w, h = 1920, 1080
        cx, cy = int(w * 0.5), int(h * 0.5)
        seqs = [{"confidence": 1.0, "box": [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]], "dots": [(cx - 20, cy), (cx - 10, cy + 5), (cx, cy), (cx + 10, cy - 5), (cx + 20, cy)]}]
    payload = build_hover_path(seqs, offset_x=0, offset_y=0)
    if not payload:
        log("[WARN] hover_bot produced insufficient points.")
        return
    pts = payload.get("points") or []
    if pts:
        save_hover_path_visual(pts, points_json)
    save_hover_overlay_from_json(points_json)
    trace_stem = points_json.stem[:-6] if points_json.stem.endswith("_hover") else points_json.stem
    if trace_stem:
        payload["trace_stem"] = trace_stem
    if hold_left_button:
        payload["press"] = "mouse"
    sent = send_control_agent(payload, control_agent_port)
    if sent:
        log(f"[INFO] Sent hover path ({len(payload['points'])} pts) to control agent port {control_agent_port}")
        start_hover_fallback_timer()
    else:
        log("[WARN] hover dispatch failed (control_agent send returned False)")
