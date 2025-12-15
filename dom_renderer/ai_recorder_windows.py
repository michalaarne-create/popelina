import sys
from typing import Dict, Optional, Tuple

try:
    if __package__:
        from .ai_recorder_common import TabInfo, WindowInfo
    else:
        from ai_recorder_common import TabInfo, WindowInfo
except Exception:
    from ai_recorder_common import TabInfo, WindowInfo

if sys.platform.startswith("win"):
    try:
        import uiautomation as auto
    except ImportError:
        auto = None
    try:
        import psutil
    except ImportError:
        psutil = None
    import ctypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    def get_foreground_hwnd() -> int:
        return user32.GetForegroundWindow()

    def get_hwnd_pid(hwnd: int) -> Optional[int]:
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value) if pid.value else None

    def get_window_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        rect = RECT()
        ok = user32.GetWindowRect(hwnd, ctypes.byref(rect))
        if not ok:
            return None
        return rect.left, rect.top, rect.right, rect.bottom

    def get_process_name(pid: int) -> str:
        try:
            p = psutil.Process(pid)
            return p.name().lower()
        except Exception:
            return ""
else:
    auto = None
    psutil = None

    def get_foreground_hwnd() -> int:
        return 0

    def get_hwnd_pid(hwnd: int) -> Optional[int]:
        return None

    def get_window_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        return None

    def get_process_name(pid: int) -> str:
        return ""


class UIATracker:
    def __init__(self):
        if not sys.platform.startswith("win"):
            return
        self.browsers = {"chrome.exe", "msedge.exe", "brave.exe", "opera.exe", "firefox.exe"}
        if auto:
            try:
                auto.SetGlobalSearchTimeout(0.3)
            except Exception:
                pass
        self.cache_windows: Dict[int, WindowInfo] = {}
        self.last_refresh = 0.0
        self.refresh_interval = 1.5

    def get_active_window_info(self) -> Optional[WindowInfo]:
        if not sys.platform.startswith("win"):
            return None
        try:
            hwnd = get_foreground_hwnd()
            if not hwnd:
                return None
            pid = get_hwnd_pid(hwnd)
            if not pid:
                return None
            pname = get_process_name(pid)
            if pname not in self.browsers:
                return None
            rect = get_window_rect(hwnd)
            if not rect:
                return None
            title = ""
            if auto:
                try:
                    win_ctrl = auto.ControlFromHandle(hwnd)
                    title = (win_ctrl.Name or "").strip()
                except Exception:
                    title = ""
            return WindowInfo(
                hwnd=hwnd,
                pid=pid,
                process_name=pname,
                title=title,
                rect=rect,
                tabs=[],
                selected_tab_title=None,
                address_bar_url=None,
                is_active=True,
            )
        except Exception:
            return None
