from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Callable, Optional


def capture_iteration_image(
    *,
    screenshot_prefix: str,
    input_image: Optional[Path],
    screenshot_dir: Path,
    raw_current_dir: Path,
    current_run_dir: Path,
    capture_fullscreen: Callable[[Path], Path],
    write_current_artifact: Callable[..., Optional[Path]],
    debug_mode: bool,
    debug: Callable[[str], None],
    log: Callable[[str], None],
    update_overlay_status: Callable[[str], None],
) -> Optional[Path]:
    t0 = time.perf_counter()
    if input_image:
        screenshot_path = Path(input_image)
        if not screenshot_path.exists():
            log(f"[ERROR] Provided input image does not exist: {screenshot_path}")
            return None
        write_current_artifact(screenshot_path, raw_current_dir, "screenshot.png")
        write_current_artifact(screenshot_path, current_run_dir, "screenshot.png")
        log(f"[INFO] Using provided screenshot -> {screenshot_path}")
        log(f"[TIMER] stage_capture {time.perf_counter() - t0:.3f}s (input_image)")
        return screenshot_path

    name = f"{screenshot_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    screenshot_path = screenshot_dir / name
    capture_fullscreen(screenshot_path)
    write_current_artifact(screenshot_path, raw_current_dir, "screenshot.png")
    write_current_artifact(screenshot_path, current_run_dir, "screenshot.png")
    log(f"[INFO] Saved screenshot -> {screenshot_path}")
    if debug_mode:
        debug(f"RAW_CURRENT_DIR now: {raw_current_dir / 'screenshot.png'} exists={(raw_current_dir / 'screenshot.png').exists()}")
    update_overlay_status(f"Screenshot captured ({screenshot_path.name})")
    log(f"[TIMER] stage_capture {time.perf_counter() - t0:.3f}s")
    return screenshot_path
