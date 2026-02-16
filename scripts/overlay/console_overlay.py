from __future__ import annotations

import contextlib
import os
import sys
from typing import Any, Dict


_VT_ENABLED = False


def init_console_overlay(alpha: int = 220) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "os": os.name,
        "vt_enabled": False,
        "green_theme_applied": False,
        "transparency_applied": False,
        "console_window_found": False,
    }
    _set_line_buffering()
    if os.name != "nt":
        return status
    status["vt_enabled"] = bool(_enable_vt_mode_windows())
    status["green_theme_applied"] = bool(_set_green_console_theme_windows())
    transp = _set_console_transparency_windows(alpha=alpha)
    status["transparency_applied"] = bool(transp.get("ok"))
    status["console_window_found"] = bool(transp.get("window_found"))
    if bool(transp.get("reason")):
        status["transparency_reason"] = str(transp.get("reason"))
    return status


def colorize_log_message(message: str) -> str:
    text = str(message or "")
    if not text:
        return text
    if "[TIMER]" in text:
        return _wrap_ansi(text, "94")
    if "[WARN]" in text:
        return _wrap_ansi(text, "91")
    if "[ERROR]" in text:
        return _wrap_ansi(text, "31")
    return text


def _wrap_ansi(text: str, code: str) -> str:
    if os.name == "nt" and not _VT_ENABLED:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _set_line_buffering() -> None:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    with contextlib.suppress(Exception):
        sys.stderr.reconfigure(line_buffering=True)


def _enable_vt_mode_windows() -> bool:
    global _VT_ENABLED
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle in (0, -1):
            return False
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return False
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if kernel32.SetConsoleMode(handle, new_mode) != 0:
            _VT_ENABLED = True
            return True
        return False
    except Exception:
        _VT_ENABLED = False
        return False


def _set_green_console_theme_windows() -> bool:
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle in (0, -1):
            return False
        FOREGROUND_GREEN = 0x0002
        FOREGROUND_INTENSITY = 0x0008
        attr = FOREGROUND_GREEN | FOREGROUND_INTENSITY
        return bool(kernel32.SetConsoleTextAttribute(handle, attr))
    except Exception:
        return False


def _set_console_transparency_windows(alpha: int = 220) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False, "window_found": False, "reason": ""}
    try:
        import ctypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = kernel32.GetConsoleWindow()
        if not hwnd:
            out["reason"] = "no_console_window"
            return out
        out["window_found"] = True

        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x00080000
        LWA_ALPHA = 0x00000002

        alpha_i = int(max(80, min(255, alpha)))
        ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style | WS_EX_LAYERED)
        ok = user32.SetLayeredWindowAttributes(hwnd, 0, alpha_i, LWA_ALPHA)
        out["ok"] = bool(ok)
        if not out["ok"]:
            out["reason"] = "SetLayeredWindowAttributes_failed"
        return out
    except Exception:
        out["reason"] = "exception"
        return out
