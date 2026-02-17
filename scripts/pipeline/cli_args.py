from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch ai_recorder_live and run OCR -> region_grow -> rating every few seconds.",
    )
    parser.add_argument("--interval", type=float, default=3.0, help="Delay between pipeline iterations.")
    parser.add_argument(
        "--loop-count",
        type=int,
        default=None,
        help="Number of iterations to run (default: infinite). Useful for testing.",
    )
    parser.add_argument(
        "--disable-recorder",
        action="store_true",
        help="Do not spawn ai_recorder_live.py (keeps only the pipeline).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run pipeline continuously without waiting for hotkey.",
    )
    parser.add_argument(
        "--recorder-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Additional args passed to ai_recorder_live.py. Use as: --recorder-args -- --url https://example.com",
    )
    parser.add_argument(
        "--left",
        action="store_true",
        help="Hold left mouse button throughout the hover path.",
    )
    debug_group = parser.add_argument_group("DEBUG")
    debug_group.add_argument(
        "--debug",
        action="store_true",
        help="Verbose debug logging for hover/control flow.",
    )
    debug_group.add_argument(
        "--debug-control-agent",
        action="store_true",
        help="Enable verbose logging in control_agent.",
    )
    advanced_debug_group = parser.add_argument_group("ADVANCED DEBUG")
    advanced_debug_group.add_argument(
        "--advanced-debug",
        action="store_true",
        help=(
            "Enable very verbose diagnostics and timers across region_grow, rating, "
            "rating_fast, OCR and control_agent."
        ),
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        default=True,
        help="Deprecated: on-screen overlay is removed; logs are shown in console.",
    )
    parser.add_argument(
        "--autostart-control-agent",
        action="store_true",
        help="Auto-launch control_agent if not running (disabled by default).",
    )
    parser.add_argument(
        "--safe-test",
        action="store_true",
        help="Disable overlay and control_agent autostart to keep mouse untouched during tests.",
    )
    parser.add_argument(
        "--first-agent",
        action="store_true",
        help=(
            "Test-only: run a single iteration focused on hover -> control_agent. "
            "Skips region_grow/rating and measures only path dispatch."
        ),
    )
    parser.add_argument(
        "--notime",
        action="store_true",
        help="Auto-run a single iteration immediately, measure end-to-end time (full pipeline).",
    )
    parser.add_argument(
        "--fast-skip",
        action="store_true",
        help="Skip region_grow + rating (hover-only timing).",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help="Use an existing screenshot instead of capturing the screen.",
    )
    return parser.parse_args()
