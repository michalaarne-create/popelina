from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import os
import time
from typing import Callable, Optional


def capture_iteration_image(
    *,
    loop_idx: int,
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
    runtime_profile = str(os.environ.get("FULLBOT_RUNTIME_PROFILE", "ultra_fast") or "ultra_fast").strip().lower()
    ultra_fast = runtime_profile == "ultra_fast"
    try:
        archive_every_n = int(os.environ.get("FULLBOT_CAPTURE_ARCHIVE_EVERY_N", "0") or 0)
    except Exception:
        archive_every_n = 0

    if input_image:
        screenshot_path = Path(input_image)
        if not screenshot_path.exists():
            log(f"[ERROR] Provided input image does not exist: {screenshot_path}")
            return None
        write_current_artifact(screenshot_path, current_run_dir, "screenshot.png")
        if not ultra_fast:
            write_current_artifact(screenshot_path, raw_current_dir, "screenshot.png")
        log(f"[INFO] Using provided screenshot -> {screenshot_path}")
        dt = time.perf_counter() - t0
        log(f"[TIMER] stage_capture {dt:.3f}s (input_image)")
        try:
            budget_ms = float(os.environ.get("FULLBOT_STAGE_CAPTURE_BUDGET_MS", "180") or 180.0)
        except Exception:
            budget_ms = 180.0
        if dt * 1000.0 > budget_ms:
            log(f"[WARN] stage_capture budget exceeded: {dt*1000.0:.1f}ms > {budget_ms:.1f}ms")
        return screenshot_path

    if ultra_fast:
        current_run_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = current_run_dir / "screenshot.png"
        capture_fullscreen(screenshot_path)
        if archive_every_n > 0 and loop_idx > 0 and (loop_idx % archive_every_n) == 0:
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            archived = screenshot_dir / f"{screenshot_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            shutil.copy2(screenshot_path, archived)
            log(f"[INFO] Saved screenshot archive -> {archived}")
    else:
        name = f"{screenshot_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        screenshot_path = screenshot_dir / name
        capture_fullscreen(screenshot_path)
        write_current_artifact(screenshot_path, raw_current_dir, "screenshot.png")
        write_current_artifact(screenshot_path, current_run_dir, "screenshot.png")
        log(f"[INFO] Saved screenshot -> {screenshot_path}")

    if debug_mode:
        debug(f"RAW_CURRENT_DIR now: {raw_current_dir / 'screenshot.png'} exists={(raw_current_dir / 'screenshot.png').exists()}")
    update_overlay_status(f"Screenshot captured ({screenshot_path.name})")
    dt = time.perf_counter() - t0
    log(f"[TIMER] stage_capture {dt:.3f}s")
    try:
        budget_ms = float(os.environ.get("FULLBOT_STAGE_CAPTURE_BUDGET_MS", "180") or 180.0)
    except Exception:
        budget_ms = 180.0
    if dt * 1000.0 > budget_ms:
        log(f"[WARN] stage_capture budget exceeded: {dt*1000.0:.1f}ms > {budget_ms:.1f}ms")
    return screenshot_path
