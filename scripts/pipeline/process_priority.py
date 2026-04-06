from __future__ import annotations

import contextlib
import os
import subprocess
from typing import Any, Dict, Optional


def windows_normal_priority_flag() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "NORMAL_PRIORITY_CLASS", 0x00000020))


def merge_creationflags(*flags: int) -> int:
    out = 0
    for flag in flags:
        try:
            out |= int(flag or 0)
        except Exception:
            continue
    return out


def with_windows_normal_priority(
    base: Optional[Dict[str, Any]] = None,
    *,
    extra_flags: int = 0,
) -> Dict[str, Any]:
    merged = dict(base or {})
    if os.name != "nt":
        return merged
    merged["creationflags"] = merge_creationflags(
        int(merged.get("creationflags", 0) or 0),
        windows_normal_priority_flag(),
        int(extra_flags or 0),
    )
    return merged


def lower_current_process_to_normal(*, log=None) -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetCurrentProcess()
        ok = bool(kernel32.SetPriorityClass(handle, windows_normal_priority_flag()))
        if ok and callable(log):
            log("[INFO] Process priority forced to NORMAL_PRIORITY_CLASS.")
        return ok
    except Exception as exc:
        if callable(log):
            log(f"[WARN] Could not force NORMAL_PRIORITY_CLASS: {exc}")
        return False


def lower_process_to_normal(proc: Any, *, log=None, label: str = "process") -> bool:
    if os.name != "nt" or proc is None:
        return False
    pid = getattr(proc, "pid", None)
    if not pid:
        return False
    try:
        import psutil  # type: ignore

        psutil.Process(int(pid)).nice(windows_normal_priority_flag())
        return True
    except Exception:
        pass
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_SET_INFORMATION = 0x0200
        PROCESS_QUERY_INFORMATION = 0x0400
        handle = kernel32.OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            ok = bool(kernel32.SetPriorityClass(handle, windows_normal_priority_flag()))
            if ok and callable(log):
                log(f"[INFO] {label} priority forced to NORMAL_PRIORITY_CLASS (pid={pid}).")
            return ok
        finally:
            with contextlib.suppress(Exception):
                kernel32.CloseHandle(handle)
    except Exception as exc:
        if callable(log):
            log(f"[WARN] Could not lower {label} priority: {exc}")
        return False
