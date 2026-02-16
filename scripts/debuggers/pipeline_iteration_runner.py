from __future__ import annotations

from pathlib import Path
from typing import Optional

from scripts.pipeline.iteration_orchestrator import run_iteration


def run_pipeline_iteration(
    loop_idx: int,
    screenshot_prefix: str = "screen",
    input_image: Optional[Path] = None,
    fast_skip: bool = False,
) -> None:
    """
    Executes one full pipeline iteration via main runtime implementation.

    This module is intentionally separated for easier debugging/routing.
    """
    import __main__ as m

    dbg = getattr(m, "debug", None)
    debug_mode = bool(getattr(m, "DEBUG_MODE", False))
    if callable(dbg) and debug_mode:
        dbg(
            f"pipeline_iteration_runner start idx={loop_idx} "
            f"prefix={screenshot_prefix!r} input_image={str(input_image) if input_image else None} "
            f"fast_skip={bool(fast_skip)}"
        )

    try:
        globals_fn = getattr(m, "globals", None)
        if not callable(globals_fn):
            globals_fn = lambda: getattr(m, "__dict__", {})
        run_iteration(
            loop_idx=loop_idx,
            screenshot_prefix=screenshot_prefix,
            input_image=input_image,
            fast_skip=fast_skip,
            deps={
                "globals_fn": globals_fn,
                "SCREENSHOT_DIR": getattr(m, "SCREENSHOT_DIR"),
                "RAW_CURRENT_DIR": getattr(m, "RAW_CURRENT_DIR"),
                "CURRENT_RUN_DIR": getattr(m, "CURRENT_RUN_DIR"),
                "capture_fullscreen": getattr(m, "capture_fullscreen"),
                "write_current_artifact": getattr(m, "_write_current_artifact"),
                "DEBUG_MODE": bool(getattr(m, "DEBUG_MODE", False)),
                "debug": getattr(m, "debug"),
                "log": getattr(m, "log"),
                "update_overlay_status": getattr(m, "update_overlay_status"),
                "prepare_hover_image": getattr(m, "prepare_hover_image"),
                "downscale_for_region": getattr(m, "_downscale_for_region"),
                "run_region_grow": getattr(m, "run_region_grow"),
                "build_hover_from_region_results": getattr(m, "build_hover_from_region_results"),
                "dispatch_hover_to_control_agent": getattr(m, "dispatch_hover_to_control_agent"),
                "run_arrow_post": getattr(m, "run_arrow_post"),
                "run_rating": getattr(m, "run_rating"),
                "RATE_RESULTS_DIR": getattr(m, "RATE_RESULTS_DIR"),
                "RATE_RESULTS_DEBUG_DIR": getattr(m, "RATE_RESULTS_DEBUG_DIR"),
                "RATE_RESULTS_CURRENT_DIR": getattr(m, "RATE_RESULTS_CURRENT_DIR"),
                "RATE_RESULTS_DEBUG_CURRENT_DIR": getattr(m, "RATE_RESULTS_DEBUG_CURRENT_DIR"),
                "RATE_SUMMARY_DIR": getattr(m, "RATE_SUMMARY_DIR"),
                "RATE_SUMMARY_CURRENT_DIR": getattr(m, "RATE_SUMMARY_CURRENT_DIR"),
                "BRAIN_AGENT": getattr(m, "BRAIN_AGENT"),
                "send_click_from_bbox": getattr(m, "_send_click_from_bbox"),
                "scroll_on_box": getattr(m, "scroll_on_box"),
                "find_screenshot_for_summary": getattr(m, "_find_screenshot_for_summary"),
                "send_best_click": getattr(m, "send_best_click"),
                "send_random_click": getattr(m, "send_random_click"),
                "cancel_hover_fallback_timer": getattr(m, "cancel_hover_fallback_timer"),
            },
        )
        if callable(dbg) and debug_mode:
            dbg(f"pipeline_iteration_runner done idx={loop_idx}")
    except Exception:
        if callable(dbg) and debug_mode:
            dbg(f"pipeline_iteration_runner failed idx={loop_idx}")
        raise
