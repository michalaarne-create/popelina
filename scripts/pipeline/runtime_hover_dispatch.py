from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def _with_point_offset(payload: dict, *, offset_x: int, offset_y: int) -> dict:
    if not isinstance(payload, dict):
        return payload
    if not offset_x and not offset_y:
        return payload
    points = payload.get("points")
    if not isinstance(points, list):
        return payload
    shifted = []
    for p in points:
        if isinstance(p, dict):
            px = int(p.get("x", 0))
            py = int(p.get("y", 0))
            out = dict(p)
            out["x"] = int(px + offset_x)
            out["y"] = int(py + offset_y)
            shifted.append(out)
        else:
            shifted.append(p)
    out_payload = dict(payload)
    out_payload["points"] = shifted
    return out_payload


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
    screen_click_offset_x: int = 0,
    screen_click_offset_y: int = 0,
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
    payload_send = _with_point_offset(
        payload,
        offset_x=int(screen_click_offset_x),
        offset_y=int(screen_click_offset_y),
    )
    sent = send_control_agent(payload_send, control_agent_port)
    if sent:
        log(
            f"[INFO] Sent hover path ({len(payload['points'])} pts) to control agent port {control_agent_port} "
            f"(offset=({int(screen_click_offset_x)},{int(screen_click_offset_y)}))"
        )
        start_hover_fallback_timer()
    else:
        log("[WARN] hover dispatch failed (control_agent send returned False)")
