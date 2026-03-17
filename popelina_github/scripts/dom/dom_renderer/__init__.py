"""dom_renderer package exports.

Provides a single entry (
    LiveRecorder, main
) that stitches together the split modules.
"""

from .ai_recorder_live import LiveRecorder  # re-export


async def main():
    # Delegate to the entry module's main
    from .ai_recorder_live import main as _main

    return await _main()


__all__ = [
    "LiveRecorder",
    "main",
]
