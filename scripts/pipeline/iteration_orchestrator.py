from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from scripts.debuggers.files_reader import collect_and_dispatch_to_brain
from .stage_brain_action import run_brain_action
from .stage_capture import capture_iteration_image
from .stage_hover import run_hover_stage
from .stage_region_rating import run_region_and_rating


def run_iteration(
    *,
    loop_idx: int,
    screenshot_prefix: str,
    input_image: Optional[Path],
    fast_skip: bool,
    deps: dict,
) -> None:
    deps["globals_fn"]()["_hover_fallback_allowed"] = True
    t_iter_start = time.perf_counter()

    screenshot_path = capture_iteration_image(
        screenshot_prefix=screenshot_prefix,
        input_image=input_image,
        screenshot_dir=deps["SCREENSHOT_DIR"],
        raw_current_dir=deps["RAW_CURRENT_DIR"],
        current_run_dir=deps["CURRENT_RUN_DIR"],
        capture_fullscreen=deps["capture_fullscreen"],
        write_current_artifact=deps["write_current_artifact"],
        debug_mode=deps["DEBUG_MODE"],
        debug=deps["debug"],
        log=deps["log"],
        update_overlay_status=deps["update_overlay_status"],
    )
    if screenshot_path is None:
        return

    run_hover_stage(
        screenshot_path=screenshot_path,
        prepare_hover_image=deps["prepare_hover_image"],
        log=deps["log"],
    )

    try:
        json_path, rating_ok = run_region_and_rating(
            screenshot_path=screenshot_path,
            fast_skip=fast_skip,
            downscale_for_region=deps["downscale_for_region"],
            run_region_grow=deps["run_region_grow"],
            build_hover_from_region_results=deps["build_hover_from_region_results"],
            dispatch_hover_to_control_agent=deps["dispatch_hover_to_control_agent"],
            run_arrow_post=deps["run_arrow_post"],
            run_rating=deps["run_rating"],
            log=deps["log"],
            update_overlay_status=deps["update_overlay_status"],
        )
        if not json_path or not rating_ok:
            return

        dispatch = collect_and_dispatch_to_brain(
            screenshot_path=screenshot_path,
            json_path=json_path,
            rate_results_dir=deps["RATE_RESULTS_DIR"],
            rate_results_debug_dir=deps["RATE_RESULTS_DEBUG_DIR"],
            rate_results_current_dir=deps["RATE_RESULTS_CURRENT_DIR"],
            rate_results_debug_current_dir=deps["RATE_RESULTS_DEBUG_CURRENT_DIR"],
            rate_summary_dir=deps["RATE_SUMMARY_DIR"],
            rate_summary_current_dir=deps["RATE_SUMMARY_CURRENT_DIR"],
            write_current_artifact=deps["write_current_artifact"],
            brain_agent=deps["BRAIN_AGENT"],
            log=deps["log"],
        )
        if dispatch is None:
            deps["update_overlay_status"]("rating completed (no summary).")
            return
        run_brain_action(
            decision=dispatch.decision,
            summary_path=dispatch.summary_path,
            screenshot_path=screenshot_path,
            send_click_from_bbox=deps["send_click_from_bbox"],
            scroll_on_box=deps["scroll_on_box"],
            find_screenshot_for_summary=deps["find_screenshot_for_summary"],
            send_best_click=deps["send_best_click"],
            send_random_click=deps["send_random_click"],
            log=deps["log"],
            update_overlay_status=deps["update_overlay_status"],
        )
    finally:
        deps["cancel_hover_fallback_timer"]()
        deps["log"](f"[TIMER] iteration {loop_idx} total {time.perf_counter() - t_iter_start:.3f}s")

