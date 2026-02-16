from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class PipelineConfig:
    interval_s: float = 3.0
    auto: bool = False
    loop_count: Optional[int] = None
    fast_skip: bool = False
    disable_recorder: bool = False
    safe_test: bool = False


@dataclass
class IterationInput:
    loop_idx: int
    screenshot_prefix: str = "screen"
    input_image: Optional[Path] = None
    fast_skip: bool = False


@dataclass
class IterationResult:
    loop_idx: int
    screenshot_path: Optional[Path]
    region_json_path: Optional[Path]
    summary_path: Optional[Path]
    decision_action: Optional[str]
    rating_ok: bool
    elapsed_s: float
    metadata: dict[str, Any]

