from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import os
import subprocess
import sys


def start_ai_recorder(
    *,
    ai_recorder_script: Path,
    root: Path,
    subprocess_kw: dict,
    log,
    extra_args: Optional[Iterable[str]] = None,
) -> Optional[subprocess.Popen]:
    if not ai_recorder_script.exists():
        log(f"[WARN] ai_recorder_live not found at {ai_recorder_script}")
        return None
    args = [sys.executable, str(ai_recorder_script)]
    if extra_args:
        args.extend(extra_args)
    log(f"[INFO] Launching ai_recorder_live ({' '.join(args[2:]) or 'default args'})")
    return subprocess.Popen(args, cwd=str(root), **subprocess_kw)


def ensure_control_agent(
    *,
    control_agent_running,
    control_agent_script: Path,
    control_agent_port: int,
    root: Path,
    subprocess_kw: dict,
    log,
) -> Optional[subprocess.Popen]:
    if control_agent_running():
        log("[INFO] control_agent already running.")
        return None
    if not control_agent_script.exists():
        log(f"[WARN] control_agent.py not found at {control_agent_script}")
        return None
    try:
        args = [sys.executable, str(control_agent_script), "--port", str(control_agent_port)]
        if str(os.environ.get("CONTROL_AGENT_VERBOSE", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}:
            args.append("--verbose")
        proc = subprocess.Popen(
            args,
            cwd=str(root),
            **subprocess_kw,
        )
        log(f"[INFO] control_agent launched (pid={proc.pid}, port={control_agent_port}, verbose={'--verbose' in args})")
        return proc
    except Exception as exc:
        log(f"[WARN] Failed to launch control_agent: {exc}")
        return None


def stop_process(proc: Optional[subprocess.Popen], *, timeout: float = 5.0, log=None) -> None:
    if proc is None or proc.poll() is not None:
        return
    if callable(log):
        log("[INFO] Stopping ai_recorder_live...")
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if callable(log):
            log("[WARN] ai_recorder_live did not stop in time; killing.")
        proc.kill()
