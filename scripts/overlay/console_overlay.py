from __future__ import annotations

import contextlib
import os
import queue
import re
import sys
import threading
from typing import Any, Dict


_VT_ENABLED = False
_PSEUDO_OVERLAY_ENABLED = False
_STREAM_MIRROR_INSTALLED = False
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def init_console_overlay(alpha: int = 220) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "os": os.name,
        "vt_enabled": False,
        "green_theme_applied": False,
        "transparency_applied": False,
        "console_window_found": False,
        "pseudo_overlay_applied": False,
        "status_overlay": None,
        "gui_overlay_applied": False,
    }
    _set_line_buffering()
    if os.name != "nt":
        return status
    status["vt_enabled"] = bool(_enable_vt_mode_windows())
    if (not status["vt_enabled"]) and _detect_ansi_capable_terminal_windows():
        # ConPTY/VSCode/Windows Terminal can support ANSI even if SetConsoleMode fails.
        _force_vt_enabled()
        status["vt_enabled"] = True
        status["vt_reason"] = "env-detected-ansi-terminal"
    status["green_theme_applied"] = bool(_set_green_console_theme_windows())
    transp = _set_console_transparency_windows(alpha=alpha)
    status["transparency_applied"] = bool(transp.get("ok"))
    status["console_window_found"] = bool(transp.get("window_found"))
    if bool(transp.get("reason")):
        status["transparency_reason"] = str(transp.get("reason"))
    if (not status["transparency_applied"]) and status["vt_enabled"]:
        status["pseudo_overlay_applied"] = bool(_enable_pseudo_overlay())
    enable_gui_overlay = str(os.environ.get("FULLBOT_ENABLE_TK_OVERLAY", "1") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if enable_gui_overlay:
        overlay = _start_status_overlay_fallback()
        if overlay is not None:
            status["status_overlay"] = overlay
            status["gui_overlay_applied"] = True
            if str(os.environ.get("FULLBOT_OVERLAY_MIRROR_STDIO", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}:
                _install_stream_mirror(overlay)
        else:
            status["gui_overlay_reason"] = "tk_overlay_unavailable"
    return status


def colorize_log_message(message: str) -> str:
    text = str(message or "")
    if not text:
        return text
    if "TIMER" in text:
        return _wrap_ansi(text, "94")
    if "[WARN]" in text:
        return _wrap_ansi(text, "91")
    if "[ERROR]" in text:
        return _wrap_ansi(text, "31")
    if _PSEUDO_OVERLAY_ENABLED and "[main " in text:
        # VSCode/ConPTY fallback: emulate overlay with subtle green background.
        return _wrap_ansi(text, "30;102")
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
            out["reason"] = "no_console_window (likely VSCode/ConPTY terminal)"
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
    except Exception as exc:
        out["reason"] = f"exception: {exc}"
        return out


def _enable_pseudo_overlay() -> bool:
    global _PSEUDO_OVERLAY_ENABLED
    _PSEUDO_OVERLAY_ENABLED = True
    return True


def _force_vt_enabled() -> None:
    global _VT_ENABLED
    _VT_ENABLED = True


def _detect_ansi_capable_terminal_windows() -> bool:
    env = os.environ
    term_program = str(env.get("TERM_PROGRAM", "")).strip().lower()
    term = str(env.get("TERM", "")).strip().lower()
    return any(
        [
            bool(env.get("WT_SESSION")),
            term_program == "vscode",
            bool(env.get("ANSICON")),
            str(env.get("ConEmuANSI", "")).strip().upper() == "ON",
            "xterm" in term,
            "vt100" in term,
            "ansi" in term,
        ]
    )


class _TkStatusOverlay:
    """Topmost right-side status strip (semi-transparent green)."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[tuple[str, str | None]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="tk-status-overlay", daemon=True)
        self._started = threading.Event()
        self._ready = False
        self._thread.start()
        self._started.wait(timeout=2.0)

    @property
    def ready(self) -> bool:
        return bool(self._ready)

    def update(self, text: str) -> None:
        with contextlib.suppress(Exception):
            self._queue.put_nowait(("status", str(text or "")))

    def append_log(self, text: str) -> None:
        with contextlib.suppress(Exception):
            self._queue.put_nowait(("log", str(text or "")))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._queue.put_nowait(("close", None))

    def hide(self) -> None:
        with contextlib.suppress(Exception):
            self._queue.put_nowait(("hide", None))

    def show(self) -> None:
        with contextlib.suppress(Exception):
            self._queue.put_nowait(("show", None))

    def _run(self) -> None:
        try:
            import tkinter as tk
        except Exception:
            self._ready = False
            self._started.set()
            return

        root = None
        try:
            root = tk.Tk()
            root.title("FULL BOT STATUS")
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            with contextlib.suppress(Exception):
                root.attributes("-alpha", 0.58)
            root.configure(bg="#01240d")

            sw = int(root.winfo_screenwidth() or 1920)
            sh = int(root.winfo_screenheight() or 1080)
            strip_w = max(240, min(360, int(sw * 0.18)))
            strip_h = max(420, int(sh * 0.92))
            margin = 8
            x = max(0, sw - strip_w - margin)
            y = max(0, int((sh - strip_h) / 2))
            root.geometry(f"{strip_w}x{strip_h}+{x}+{y}")

            container = tk.Frame(root, bg="#01240d", highlightthickness=0, bd=0)
            container.pack(fill="both", expand=True)
            text = tk.Text(
                container,
                bg="#01240d",
                fg="#39ff88",
                insertbackground="#39ff88",
                font=("Consolas", 10),
                wrap="word",
                padx=8,
                pady=8,
                relief="flat",
                bd=0,
                highlightthickness=0,
            )
            sb = tk.Scrollbar(container, orient="vertical", command=text.yview)
            text.configure(yscrollcommand=sb.set)
            text.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            text.insert("end", "Initializing...\n")
            text.see("end")
            # Przydatne skróty jak w konsoli.
            text.bind("<Control-a>", lambda e: (text.tag_add("sel", "1.0", "end-1c"), "break"))
            text.bind("<Control-A>", lambda e: (text.tag_add("sel", "1.0", "end-1c"), "break"))
            text.bind("<Control-c>", lambda e: None)
            text.bind("<Control-C>", lambda e: None)
            default_fg = "#39ff88"
            default_bg = "#01240d"
            tag_cache: Dict[str, str] = {}

            def _map_fg(code: int) -> str | None:
                return {
                    30: "#000000",
                    31: "#cc2222",
                    91: "#ff6666",
                    94: "#66aaff",
                }.get(code)

            def _map_bg(code: int) -> str | None:
                return {
                    40: "#000000",
                    102: "#66ff66",
                }.get(code)

            def _tag_for(fg: str | None, bg: str | None) -> str:
                key = f"{fg or ''}|{bg or ''}"
                if key in tag_cache:
                    return tag_cache[key]
                name = f"ansi_{len(tag_cache)}"
                cfg: Dict[str, str] = {}
                if fg:
                    cfg["foreground"] = fg
                if bg:
                    cfg["background"] = bg
                if cfg:
                    text.tag_configure(name, **cfg)
                tag_cache[key] = name
                return name

            def _append_ansi_line(line: str) -> None:
                pos = 0
                cur_fg = default_fg
                cur_bg = None
                for m in _ANSI_RE.finditer(line):
                    if m.start() > pos:
                        seg = line[pos:m.start()]
                        tag = _tag_for(cur_fg, cur_bg)
                        text.insert("end", seg, (tag,))
                    sgr = m.group(0)[2:-1]
                    codes = [c for c in sgr.split(";") if c]
                    if not codes:
                        codes = ["0"]
                    for cs in codes:
                        try:
                            c = int(cs)
                        except Exception:
                            continue
                        if c == 0:
                            cur_fg = default_fg
                            cur_bg = None
                        else:
                            fg = _map_fg(c)
                            bg = _map_bg(c)
                            if fg is not None:
                                cur_fg = fg
                            if bg is not None:
                                cur_bg = bg
                    pos = m.end()
                if pos < len(line):
                    tag = _tag_for(cur_fg, cur_bg)
                    text.insert("end", line[pos:], (tag,))
                text.insert("end", "\n")

            self._ready = True
            self._started.set()

            def _poll() -> None:
                try:
                    while True:
                        kind, item = self._queue.get_nowait()
                        if kind == "close":
                            root.destroy()
                            return
                        if kind == "hide":
                            with contextlib.suppress(Exception):
                                root.withdraw()
                            continue
                        if kind == "show":
                            with contextlib.suppress(Exception):
                                root.deiconify()
                                root.attributes("-topmost", True)
                                root.lift()
                            continue
                        if kind == "status":
                            root.title(f"FULL BOT STATUS - {str(item or '')[:80]}")
                            continue
                        if kind == "log":
                            line = str(item or "")
                            if line:
                                _append_ansi_line(line.rstrip("\n"))
                                text.see("end")
                except queue.Empty:
                    pass
                root.after(80, _poll)

            root.after(80, _poll)
            root.mainloop()
        except Exception:
            self._ready = False
            self._started.set()
            if root is not None:
                with contextlib.suppress(Exception):
                    root.destroy()


def _start_status_overlay_fallback() -> Any:
    # Explicit opt-out for environments where any extra window is unwanted.
    if str(os.environ.get("FULLBOT_DISABLE_TK_OVERLAY", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return None
    try:
        overlay = _TkStatusOverlay()
        if overlay.ready:
            return overlay
        return None
    except Exception:
        return None


class _OverlayTee:
    def __init__(self, stream: Any, overlay: Any) -> None:
        self._stream = stream
        self._overlay = overlay

    def write(self, s: str) -> int:
        txt = str(s or "")
        n = 0
        with contextlib.suppress(Exception):
            n = self._stream.write(txt)
        if txt.strip():
            with contextlib.suppress(Exception):
                self._overlay.append_log(txt)
        return n if isinstance(n, int) else len(txt)

    def flush(self) -> None:
        with contextlib.suppress(Exception):
            self._stream.flush()

    def isatty(self) -> bool:
        with contextlib.suppress(Exception):
            return bool(self._stream.isatty())
        return False

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")


def _install_stream_mirror(overlay: Any) -> None:
    global _STREAM_MIRROR_INSTALLED
    if _STREAM_MIRROR_INSTALLED:
        return
    if overlay is None or not hasattr(overlay, "append_log"):
        return
    try:
        sys.stdout = _OverlayTee(sys.stdout, overlay)
        sys.stderr = _OverlayTee(sys.stderr, overlay)
        _STREAM_MIRROR_INSTALLED = True
    except Exception:
        _STREAM_MIRROR_INSTALLED = False
