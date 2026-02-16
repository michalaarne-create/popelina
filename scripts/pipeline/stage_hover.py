from __future__ import annotations

from pathlib import Path
from typing import Callable


def run_hover_stage(
    *,
    screenshot_path: Path,
    prepare_hover_image: Callable[[Path], object],
    log: Callable[[str], None],
) -> None:
    try:
        _ = prepare_hover_image(screenshot_path)
    except Exception as exc:
        log(f"[WARN] hover_prepare failed: {exc}")

