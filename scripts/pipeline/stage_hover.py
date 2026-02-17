from __future__ import annotations

from pathlib import Path
import time
from typing import Callable


def run_hover_stage(
    *,
    screenshot_path: Path,
    prepare_hover_image: Callable[[Path], object],
    log: Callable[[str], None],
) -> None:
    t0 = time.perf_counter()
    try:
        _ = prepare_hover_image(screenshot_path)
    except Exception as exc:
        log(f"[WARN] hover_prepare failed: {exc}")
    finally:
        log(f"[TIMER] stage_hover {time.perf_counter() - t0:.3f}s")
