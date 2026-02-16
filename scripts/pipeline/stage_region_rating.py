from __future__ import annotations

from pathlib import Path
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
    if fast_skip:
        log("[INFO] Fast skip: bypassing region_grow/rating for timing test.")
        return None, False

    region_image = downscale_for_region(screenshot_path)
    json_path = run_region_grow(region_image)
    if not json_path:
        update_overlay_status("region_grow failed.")
        return None, False

    try:
        hover_json = build_hover_from_region_results(json_path)
        if hover_json:
            dispatch_hover_to_control_agent(hover_json)
    except Exception as exc:
        log(f"[WARN] build_hover_from_region_results failed: {exc}")

    update_overlay_status("region_grow done. Running rating...")
    run_arrow_post(json_path)
    rating_ok = run_rating(json_path)
    if not rating_ok:
        update_overlay_status("rating failed.")
    return json_path, bool(rating_ok)

