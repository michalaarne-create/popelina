from __future__ import annotations

from pathlib import Path
import os
import time
from typing import Callable, Optional


def run_hover_stage(
    *,
    screenshot_path: Path,
    prepare_hover_image: Callable[[Path], object],
    log: Callable[[str], None],
) -> Optional[Path]:
    hover_enabled = str(os.environ.get("FULLBOT_HOVER_ENABLED", "0") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not hover_enabled:
        log("[INFO] Hover stage disabled (FULLBOT_HOVER_ENABLED=0).")
        return None
    t0 = time.perf_counter()
    hover_image: Optional[Path] = None
    try:
        out = prepare_hover_image(screenshot_path)
        if isinstance(out, Path):
            hover_image = out
    except Exception as exc:
        log(f"[WARN] hover_prepare failed: {exc}")
    finally:
        dt = time.perf_counter() - t0
        log(f"[TIMER] stage_hover {dt:.3f}s")
        try:
            budget_ms = float(os.environ.get("FULLBOT_STAGE_HOVER_BUDGET_MS", "120") or 120.0)
        except Exception:
            budget_ms = 120.0
        if dt * 1000.0 > budget_ms:
            log(f"[WARN] stage_hover budget exceeded: {dt*1000.0:.1f}ms > {budget_ms:.1f}ms")
    return hover_image
