import os
import sys
import time
import hashlib
import re as _re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

START_URL_DEFAULT = os.environ.get("START_URL", "")
SNAPSHOT_FPS_DEFAULT = 3.0
OCR_INTERVAL = 1.0
RECORD_SCREENSHOTS = True
OCR_LANG = os.environ.get("OCR_LANG", "pol+eng")
SLOW_LOG_THRESHOLD_S = float(os.environ.get("SLOW_LOG_THRESHOLD_S", "1.5"))

DEFAULT_TIMEOUT_MS = int(os.environ.get("PW_DEFAULT_TIMEOUT_MS", "1000"))
CONTENT_TIMEOUT_MS = 1500
SCREENSHOT_TIMEOUT_MS = 2500
CDP_TIMEOUT_MS = 300
GOTO_TIMEOUT_MS = int(os.environ.get("PW_GOTO_TIMEOUT_MS", "60000"))
WATCHDOG_TIMEOUT = 30.0

ROI_WIDTH = int(os.environ.get("ROI_WIDTH", "800"))
ROI_HEIGHT = int(os.environ.get("ROI_HEIGHT", "180"))

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

LOG_FILE: Optional[str] = None
DEBUG_TO_FILE = True
VERBOSE_DEBUG = True
# Ogranicz kosztowne logowanie na konsolę: INFO tylko gdy wyraźnie włączone
LOG_INFO_TO_CONSOLE = int(os.environ.get("LOG_INFO_TO_CONSOLE", "0"))


def _strip_ansi(s: str) -> str:
    return _re.sub(r"\x1b\[[0-9;]*m", "", s)


def set_log_file(path: Path) -> None:
    global LOG_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE = str(path)
    except Exception:
        LOG_FILE = None


def set_verbose_debug(enabled: bool) -> None:
    global VERBOSE_DEBUG
    VERBOSE_DEBUG = enabled


def log(msg: str, level: str = "INFO") -> None:
    if level == "DEBUG" and not VERBOSE_DEBUG:
        return

    colors = {
        "INFO": "",
        "SUCCESS": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "DEBUG": "\033[94m",
    }
    color = colors.get(level, "")
    reset = "\033[0m" if color else ""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"

    try:
        if level in ("ERROR", "WARNING") or VERBOSE_DEBUG or LOG_INFO_TO_CONSOLE:
            print(f"{color}{line}{reset}", flush=True)
    except Exception:
        try:
            if level in ("ERROR", "WARNING") or VERBOSE_DEBUG or LOG_INFO_TO_CONSOLE:
                print(line, flush=True)
        except Exception:
            pass

    if DEBUG_TO_FILE and LOG_FILE:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(_strip_ansi(line) + "\n")
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def norm_text(s: Optional[str]) -> str:
    return " ".join((s or "").split())


def build_bbox(x: float, y: float, w: float, h: float) -> Dict[str, int]:
    x = int(round(x))
    y = int(round(y))
    w = int(round(max(0, w)))
    h = int(round(max(0, h)))
    return {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "center_x": x + w // 2,
        "center_y": y + h // 2,
    }


def bbox_overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["width"], by1 + b["height"]
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    a_area = max(1, a["width"] * a["height"])
    return inter / a_area


def fuzzy_ratio(a: str, b: str) -> float:
    if not sys.platform.startswith("win"):
        return 0.0
    import difflib

    a = norm_text(a).lower()
    b = norm_text(b).lower()
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def extract_url_candidates(text: str) -> List[str]:
    if not text:
        return []
    text = text.replace("\n", " ")
    url_re = _re.compile(r"((?:https?://)?[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?)", _re.IGNORECASE)
    matches = url_re.findall(text)
    out: List[str] = []
    for m in matches:
        m = m.strip().strip(".,;:()[]{}<>\"'|")
        if len(m) >= 4:
            out.append(m)
    seen = set()
    res = []
    for u in out:
        key = u.lower()
        if key not in seen:
            seen.add(key)
            res.append(u)
    return res


def domain_from_url(u: str) -> str:
    if not u:
        return ""
    u2 = _re.sub(r"^[a-z]+://", "", u, flags=_re.IGNORECASE)
    u2 = u2.split("/")[0]
    u2 = u2.split(":")[0]
    return u2.lower()


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

@dataclass
class ClickableElem:
    id: str
    tag: str
    role: str
    type: Optional[str]
    text: str
    bbox: Dict[str, int]
    attributes: Dict[str, Any] = field(default_factory=dict)
    label_text: Optional[str] = None
    href: Optional[str] = None
    category: str = "control"


@dataclass
class OcrLine:
    text: str
    conf: float
    bbox: Dict[str, int]


@dataclass
class QuestionGroup:
    id: int
    question_text: str
    ocr_bbox: Dict[str, int]
    dom_candidates: List[Dict[str, Any]]
    answers: List[Dict[str, Any]]


@dataclass
class PageTrack3r:
    page: Any
    page_id: str
    url: str
    title: str
    last_activity: float
    last_user_interaction: float
    is_visible: bool = False
    has_focus: bool = False
    target_id: Optional[str] = None
    context_id: str = ""
    interaction_count: int = 0
    last_check: float = 0.0
    clicks_count: int = 0
    keys_count: int = 0
    mutations_count: int = 0
    last_click_ts: float = 0.0
    last_key_ts: float = 0.0
    last_mutation_ts: float = 0.0
    last_any_ts: float = 0.0
    newly_opened: bool = False
    last_switch_ts: float = 0.0
    storage_key: str = ""


@dataclass
class TabInfo:
    title: str
    selected: bool = False


@dataclass
class WindowInfo:
    hwnd: int
    pid: int
    process_name: str
    title: str
    rect: Tuple[int, int, int, int]
    tabs: List[TabInfo] = field(default_factory=list)
    selected_tab_title: Optional[str] = None
    address_bar_url: Optional[str] = None
    is_active: bool = False


class PerformanceMonitor:
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.counts: Dict[str, int] = {}

    def start(self, operation: str) -> float:
        return time.perf_counter()

    def end(self, operation: str, start_time: float) -> None:
        duration = time.perf_counter() - start_time
        if operation not in self.timings:
            self.timings[operation] = []
            self.counts[operation] = 0
        self.timings[operation].append(duration)
        self.counts[operation] += 1
        if len(self.timings[operation]) > 100:
            self.timings[operation] = self.timings[operation][-100:]
        if duration > SLOW_LOG_THRESHOLD_S:
            log(f"âš ď¸Ź Slow operation: {operation} took {duration:.2f}s", "WARNING")

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for op, times in self.timings.items():
            if times:
                stats[op] = {
                    "count": self.counts[op],
                    "avg": float(sum(times)/(len(times) or 1)),
                    "max": float(max(times)),
                    "last": float(times[-1]),
                }
        return stats


__all__ = [
    "START_URL_DEFAULT",
    "SNAPSHOT_FPS_DEFAULT",
    "OCR_INTERVAL",
    "RECORD_SCREENSHOTS",
    "OCR_LANG",
    "DEFAULT_TIMEOUT_MS",
    "CONTENT_TIMEOUT_MS",
    "SCREENSHOT_TIMEOUT_MS",
    "CDP_TIMEOUT_MS",
    "GOTO_TIMEOUT_MS",
    "WATCHDOG_TIMEOUT",
    "ROI_WIDTH",
    "ROI_HEIGHT",
    "set_log_file",
    "set_verbose_debug",
    "log",
    "ensure_dir",
    "md5",
    "norm_text",
    "build_bbox",
    "bbox_overlap_ratio",
    "fuzzy_ratio",
    "extract_url_candidates",
    "domain_from_url",
    "ClickableElem",
    "OcrLine",
    "QuestionGroup",
    "PageTrack3r",
    "TabInfo",
    "WindowInfo",
    "PerformanceMonitor",
]




