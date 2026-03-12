from __future__ import annotations

from pathlib import Path
from typing import Optional

from scripts.debuggers.pipeline_iteration_runner import run_pipeline_iteration

def pipeline_iteration(
    loop_idx: int,
    screenshot_prefix: str = "screen",
    input_image: Optional[Path] = None,
    fast_skip: bool = False,
) -> None:
    """
    External pipeline entrypoint.

    main.py should call this function; the actual implementation remains in
    main.py as `_pipeline_iteration_impl` to keep compatibility with the
    existing global runtime wiring.
    """
    run_pipeline_iteration(
        loop_idx=loop_idx,
        screenshot_prefix=screenshot_prefix,
        input_image=input_image,
        fast_skip=fast_skip,
    )
