from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import os
import subprocess
import sys
import threading


def start_ai_recorder(
    *,
    ai_recorder_script: Path,
    root: Path,
    subprocess_kw: dict,
    log,
    extra_args: Optional[Iterable[str]] = None,
    python_executable: Optional[str] = None,
) -> Optional[subprocess.Popen]:
    if not ai_recorder_script.exists():
        log(f"[WARN] ai_recorder_live not found at {ai_recorder_script}")
        return None
    args = [str(python_executable or sys.executable), str(ai_recorder_script)]
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
        configured_python = str(os.environ.get("FULLBOT_CONTROL_AGENT_PYTHON", "") or "").strip()
        python_path = configured_python or sys.executable
        args = [python_path, str(control_agent_script), "--port", str(control_agent_port)]
        if str(os.environ.get("CONTROL_AGENT_VERBOSE", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}:
            args.append("--verbose")
        spawn_timeout_s = float(os.environ.get("FULLBOT_CONTROL_AGENT_SPAWN_TIMEOUT_S", "3.0") or "3.0")
        proc_holder: dict = {"proc": None, "exc": None}

        def _spawn() -> None:
            try:
                proc_holder["proc"] = subprocess.Popen(
                    args,
                    cwd=str(root),
                    **subprocess_kw,
                )
            except Exception as exc:  # pragma: no cover - defensive capture
                proc_holder["exc"] = exc

        worker = threading.Thread(target=_spawn, daemon=True)
        worker.start()
        worker.join(timeout=max(0.2, spawn_timeout_s))
        if worker.is_alive():
            log(
                "[WARN] control_agent launch timed out "
                f"after {spawn_timeout_s:.1f}s; continuing without control_agent."
            )
            return None
        if proc_holder.get("exc") is not None:
            raise proc_holder["exc"]  # type: ignore[misc]
        proc = proc_holder.get("proc")
        if proc is None:
            log("[WARN] control_agent launch returned no process; continuing without control_agent.")
            return None
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
