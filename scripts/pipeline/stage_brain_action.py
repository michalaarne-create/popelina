from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional


def run_brain_action(
    *,
    decision: Any,
    summary_path: Path,
    screenshot_path: Path,
    send_click_from_bbox: Callable[..., bool],
    scroll_on_box: Callable[..., bool],
    find_screenshot_for_summary: Callable[[Path], Optional[Path]],
    send_best_click: Callable[[Path, Optional[Path]], None],
    send_random_click: Callable[[Path, Path], None],
    log: Callable[[str], None],
    update_overlay_status: Callable[[str], None],
) -> None:
    if decision.recommended_action == "click_cookies_accept" and decision.target_bbox:
        if send_click_from_bbox(decision.target_bbox, screenshot_path, "Brain cookies accept"):
            scroll_on_box(decision.target_bbox, screenshot_path, "Brain cookies scroll", total_notches=10, direction="down")
    elif decision.target_bbox:
        label = "Brain next" if decision.recommended_action == "click_next" else "Brain answer"
        send_click_from_bbox(decision.target_bbox, screenshot_path, label)
    elif decision.recommended_action == "click_next":
        screenshot = find_screenshot_for_summary(summary_path) or screenshot_path
        send_best_click(summary_path, screenshot)
    elif decision.recommended_action in ("click_answer", "fallback_random"):
        send_random_click(summary_path, screenshot_path)
    else:
        log("[INFO] Brain recommended idle; no click sent.")
    update_overlay_status("rating completed.")

