from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Callable, Optional, Tuple

from PIL import Image


_BOX4_KEYS = {
    "bbox",
    "text_box",
    "dropdown_box",
    "coarse_bbox_xyxy",
    "refined_bbox_xyxy",
}


def _scale_xy(v: float, scale: float) -> int:
    return int(round(float(v) * float(scale)))


def _scale_box4(box: list, sx: float, sy: float) -> list:
    if len(box) != 4:
        return box
    x1, y1, x2, y2 = box
    return [_scale_xy(x1, sx), _scale_xy(y1, sy), _scale_xy(x2, sx), _scale_xy(y2, sy)]


def _scale_points(points: list, sx: float, sy: float) -> list:
    out = []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([_scale_xy(p[0], sx), _scale_xy(p[1], sy)])
        else:
            out.append(p)
    return out


def _scale_payload_obj(obj, sx: float, sy: float):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _BOX4_KEYS and isinstance(v, list) and len(v) == 4:
                out[k] = _scale_box4(v, sx, sy)
                continue
            if k == "laser_endpoints" and isinstance(v, list):
                out[k] = _scale_points(v, sx, sy)
                continue
            if k == "bbox" and isinstance(v, dict):
                out[k] = {
                    **v,
                    "x": _scale_xy(v.get("x", 0), sx),
                    "y": _scale_xy(v.get("y", 0), sy),
                    "width": _scale_xy(v.get("width", 0), sx),
                    "height": _scale_xy(v.get("height", 0), sy),
                }
                continue
            out[k] = _scale_payload_obj(v, sx, sy)
        return out
    if isinstance(obj, list):
        return [_scale_payload_obj(x, sx, sy) for x in obj]
    return obj


def _rescale_region_json_to_screenshot(json_path: Path, region_image: Path, screenshot_path: Path, log) -> None:
    try:
        with Image.open(region_image) as im_rg:
            rw, rh = im_rg.size
        with Image.open(screenshot_path) as im_sc:
            sw, sh = im_sc.size
    except Exception as exc:
        log(f"[WARN] region output rescale skipped (image read failed): {exc}")
        return

    if rw <= 0 or rh <= 0 or sw <= 0 or sh <= 0:
        return
    if rw == sw and rh == sh:
        return

    sx = float(sw) / float(rw)
    sy = float(sh) / float(rh)
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        log(f"[WARN] region output rescale skipped (json read failed): {exc}")
        return
    if not isinstance(payload, dict):
        return

    scaled = _scale_payload_obj(payload, sx, sy)
    scaled["image"] = str(screenshot_path)
    scaled["_coord_space"] = {
        "source_image": str(region_image),
        "target_image": str(screenshot_path),
        "scale_x": round(sx, 6),
        "scale_y": round(sy, 6),
    }
    try:
        json_path.write_text(json.dumps(scaled, ensure_ascii=False, indent=2), encoding="utf-8")
        log(
            "[INFO] region_grow coords rescaled "
            f"{rw}x{rh} -> {sw}x{sh} (sx={sx:.4f}, sy={sy:.4f})"
        )
    except Exception as exc:
        log(f"[WARN] region output rescale write failed: {exc}")


def run_region_and_rating(
    *,
    screenshot_path: Path,
    fast_skip: bool,
    downscale_for_region: Callable[[Path], Path],
    run_region_grow: Callable[[Path], Optional[Path]],
    build_hover_from_region_results: Callable[[Path], Optional[Path]],
    dispatch_hover_to_control_agent: Callable[[Path], None],
    run_arrow_post: Callable[[Path], None],
    run_rating: Callable[[Path], bool],
    log: Callable[[str], None],
    update_overlay_status: Callable[[str], None],
    is_abort_requested: Callable[[], bool],
) -> Tuple[Optional[Path], bool]:
    turbo_mode = str(os.environ.get("FULLBOT_TURBO_MODE", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
    try:
        budget_ms = float(os.environ.get("FULLBOT_REGION_RATING_BUDGET_MS", "1000") or 1000.0)
    except Exception:
        budget_ms = 1000.0
    budget_ms = max(100.0, min(30000.0, float(budget_ms)))
    t_total = time.perf_counter()
    if fast_skip:
        log("[INFO] Fast skip: bypassing region_grow/rating for timing test.")
        return None, False

    t_downscale = time.perf_counter()
    region_image = downscale_for_region(screenshot_path)
    log(f"[TIMER] stage_region_rating.downscale {time.perf_counter() - t_downscale:.3f}s")
    if is_abort_requested():
        update_overlay_status("Iteration aborted.")
        return None, False

    t_rg = time.perf_counter()
    json_path = run_region_grow(region_image)
    log(f"[TIMER] stage_region_rating.region_grow {time.perf_counter() - t_rg:.3f}s")
    if not json_path:
        update_overlay_status("region_grow failed.")
        return None, False
    _rescale_region_json_to_screenshot(json_path, region_image, screenshot_path, log)
    if is_abort_requested():
        update_overlay_status("Iteration aborted.")
        return None, False

    try:
        t_hover = time.perf_counter()
        hover_json = build_hover_from_region_results(json_path)
        if hover_json:
            dispatch_hover_to_control_agent(hover_json)
        log(f"[TIMER] stage_region_rating.hover_dispatch {time.perf_counter() - t_hover:.3f}s")
    except Exception as exc:
        log(f"[WARN] build_hover_from_region_results failed: {exc}")
    if is_abort_requested():
        update_overlay_status("Iteration aborted.")
        return None, False

    update_overlay_status("region_grow done. Running rating...")
    if turbo_mode:
        log("[INFO] Turbo mode: skipping arrow_post.")
    else:
        t_arrow = time.perf_counter()
        run_arrow_post(json_path)
        log(f"[TIMER] stage_region_rating.arrow_post {time.perf_counter() - t_arrow:.3f}s")
    t_rating = time.perf_counter()
    rating_ok = run_rating(json_path)
    log(f"[TIMER] stage_region_rating.rating {time.perf_counter() - t_rating:.3f}s")
    if not rating_ok:
        update_overlay_status("rating failed.")
    dt_total = time.perf_counter() - t_total
    log(f"[TIMER] stage_region_rating.total {dt_total:.3f}s")
    if (dt_total * 1000.0) > budget_ms:
        log(f"[WARN] stage_region_rating budget exceeded: {dt_total*1000.0:.1f}ms > {budget_ms:.1f}ms")
    return json_path, bool(rating_ok)
