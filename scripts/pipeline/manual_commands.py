from __future__ import annotations

from typing import Optional
import argparse
import queue
import subprocess


def handle_manual_command(
    command: str,
    *,
    args: argparse.Namespace,
    recorder_proc: Optional[subprocess.Popen],
    run_region_grow_latest,
    run_rating_latest,
    start_ai_recorder,
    trigger_best_click_from_summary,
    abort_current_iteration,
    log,
    update_overlay_status,
) -> Optional[subprocess.Popen]:
    if command == "region":
        run_region_grow_latest()
    elif command == "rating":
        run_rating_latest()
    elif command == "recorder":
        if recorder_proc and recorder_proc.poll() is None:
            log("[INFO] ai_recorder_live already running.")
            update_overlay_status("Recorder already running.")
        else:
            recorder_proc = start_ai_recorder(args.recorder_args)
            if recorder_proc:
                update_overlay_status("Recorder launched.")
    elif command == "control":
        trigger_best_click_from_summary()
    elif command == "abort":
        abort_current_iteration()
    return recorder_proc


def drain_manual_commands(
    cmd_queue: "queue.Queue[str]",
    *,
    args: argparse.Namespace,
    recorder_proc: Optional[subprocess.Popen],
    **deps,
) -> Optional[subprocess.Popen]:
    while True:
        try:
            command = cmd_queue.get_nowait()
        except queue.Empty:
            break
        recorder_proc = handle_manual_command(command, args=args, recorder_proc=recorder_proc, **deps)
    return recorder_proc
