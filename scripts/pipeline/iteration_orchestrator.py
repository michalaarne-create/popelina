from __future__ import annotations

import time
import os
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
    deps["globals_fn"]()["_iteration_in_progress"] = True
    final_json_path: Optional[Path] = None
    final_screenshot_path: Optional[Path] = None
    def _is_abort_requested() -> bool:
        try:
            return bool(deps["globals_fn"]().get("_abort_iteration_requested"))
        except Exception:
            return False

    def _clear_abort_requested() -> None:
        try:
            deps["globals_fn"]()["_abort_iteration_requested"] = False
        except Exception:
            pass

    def _abort_iteration() -> None:
        deps["log"]("[WARN] Iteration aborted by hotkey '\"'.")
        deps["update_overlay_status"]("Iteration aborted.")
        _clear_abort_requested()

    deps["globals_fn"]()["_hover_fallback_allowed"] = True
    ts_iter = time.strftime("%Y%m%d_%H%M%S")
    os.environ["FULLBOT_OCR_ITERATION_ID"] = f"iter_{int(loop_idx)}_{ts_iter}"
    t_iter_start = time.perf_counter()
    if _is_abort_requested():
        _abort_iteration()
        return

    t_capture = time.perf_counter()
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
    deps["log"](f"[TIMER] iter.capture {time.perf_counter() - t_capture:.3f}s")
    if screenshot_path is None:
        return
    if _is_abort_requested():
        _abort_iteration()
        return

    t_hover = time.perf_counter()
    run_hover_stage(
        screenshot_path=screenshot_path,
        prepare_hover_image=deps["prepare_hover_image"],
        log=deps["log"],
    )
    deps["log"](f"[TIMER] iter.hover {time.perf_counter() - t_hover:.3f}s")
    if _is_abort_requested():
        _abort_iteration()
        return

    try:
        t_region_rating = time.perf_counter()
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
            is_abort_requested=_is_abort_requested,
        )
        deps["log"](f"[TIMER] iter.region_rating {time.perf_counter() - t_region_rating:.3f}s")
        if _is_abort_requested():
            _abort_iteration()
            return
        final_json_path = json_path
        final_screenshot_path = screenshot_path
        if not json_path or not rating_ok:
            return

        t_dispatch = time.perf_counter()
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
        deps["log"](f"[TIMER] iter.files_reader_dispatch {time.perf_counter() - t_dispatch:.3f}s")
        if dispatch is None:
            deps["update_overlay_status"]("rating completed (no summary).")
            return
        if _is_abort_requested():
            _abort_iteration()
            return
        t_action = time.perf_counter()
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
        deps["log"](f"[TIMER] iter.brain_action {time.perf_counter() - t_action:.3f}s")
    finally:
        with_ocr_iter = os.environ.get("FULLBOT_OCR_ITERATION_ID")
        if with_ocr_iter:
            os.environ.pop("FULLBOT_OCR_ITERATION_ID", None)
        try:
            annotate_fn = deps.get("run_region_annotation")
            if callable(annotate_fn) and final_json_path is not None and final_screenshot_path is not None:
                annotate_fn(final_json_path, final_screenshot_path)
        except Exception as exc:
            deps["log"](f"[WARN] Deferred region annotation step failed: {exc}")
        try:
            deps["globals_fn"]()["_iteration_in_progress"] = False
        except Exception:
            pass
        deps["cancel_hover_fallback_timer"]()
        deps["log"](f"[TIMER] iteration {loop_idx} total {time.perf_counter() - t_iter_start:.3f}s")
