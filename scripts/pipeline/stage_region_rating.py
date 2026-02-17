from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Callable, Optional, Tuple


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

    t_rg = time.perf_counter()
    json_path = run_region_grow(region_image)
    log(f"[TIMER] stage_region_rating.region_grow {time.perf_counter() - t_rg:.3f}s")
    if not json_path:
        update_overlay_status("region_grow failed.")
        return None, False

    try:
        t_hover = time.perf_counter()
        hover_json = build_hover_from_region_results(json_path)
        if hover_json:
            dispatch_hover_to_control_agent(hover_json)
        log(f"[TIMER] stage_region_rating.hover_dispatch {time.perf_counter() - t_hover:.3f}s")
    except Exception as exc:
        log(f"[WARN] build_hover_from_region_results failed: {exc}")

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
