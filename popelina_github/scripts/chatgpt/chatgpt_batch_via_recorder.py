from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

try:
    from playwright.sync_api import Browser, Page, sync_playwright
except Exception:  # pragma: no cover - runtime dependency
    Browser = Any  # type: ignore[assignment]
    Page = Any  # type: ignore[assignment]
    sync_playwright = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.process_control import ensure_control_agent  # noqa: E402
from scripts.pipeline.runtime_control_agent import send_control_agent, send_type  # noqa: E402


AI_RECORDER_SCRIPT = PROJECT_ROOT / "scripts" / "dom" / "dom_renderer" / "ai_recorder_live.py"
CONTROL_AGENT_SCRIPT = PROJECT_ROOT / "scripts" / "click" / "control_agent" / "control_agent.py"
DOM_LIVE_DIR = PROJECT_ROOT / "scripts" / "dom" / "dom_live"
RECORDER_PROFILE_DIR = PROJECT_ROOT / "_recorder_profile"
DEFAULT_CDP_ENDPOINT = "http://127.0.0.1:9222"
DEFAULT_CHATGPT_URL = "https://chatgpt.com/"
DEFAULT_CONTROL_AGENT_PORT = int(os.environ.get("CONTROL_AGENT_PORT", "8765") or "8765")
RUNS_ROOT = PROJECT_ROOT / "data" / "chatgpt_runs"
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


@dataclass
class PromptResult:
    index: int
    prompt: str
    response_text: str
    status: str
    capture_method: str
    page_url: str
    started_at: str
    finished_at: str
    duration_s: float
    error: Optional[str]


class ClipboardError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)


def _jsonl_append(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _probe_http(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = int(getattr(response, "status", 0) or 0)
            return 200 <= status < 500
    except Exception:
        return False


def _probe_cdp(endpoint: str, timeout: float = 1.0) -> bool:
    endpoint = str(endpoint or "").rstrip("/")
    if not endpoint:
        return False
    return _probe_http(endpoint + "/json/version", timeout=timeout)


def _read_devtools_endpoint(user_data_dir: Path) -> Optional[str]:
    active_file = user_data_dir / "DevToolsActivePort"
    if not active_file.exists():
        return None
    try:
        lines = active_file.read_text(encoding="utf-8").strip().splitlines()
    except Exception:
        return None
    if not lines:
        return None
    port = lines[0].strip()
    if not port.isdigit():
        return None
    endpoint = f"http://127.0.0.1:{port}"
    return endpoint if _probe_cdp(endpoint) else None


def _wait_for_cdp_endpoint(
    preferred_endpoint: Optional[str],
    user_data_dir: Path,
    *,
    timeout_s: float,
    started_proc: Optional[subprocess.Popen[Any]],
) -> str:
    if preferred_endpoint and _probe_cdp(preferred_endpoint):
        return preferred_endpoint
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        endpoint = _read_devtools_endpoint(user_data_dir)
        if endpoint:
            return endpoint
        if started_proc is not None and started_proc.poll() is not None:
            raise RuntimeError(f"ai_recorder exited early with code {started_proc.returncode}")
        time.sleep(0.25)
    raise RuntimeError(f"CDP endpoint not ready within {timeout_s:.1f}s")


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _host_matches(url: str, expected_host: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    return expected_host.lower() in netloc


def _list_all_pages(browser: Browser) -> List[Page]:
    pages: List[Page] = []
    for context in list(browser.contexts):
        with contextlib.suppress(Exception):
            for page in list(context.pages):
                with contextlib.suppress(Exception):
                    if not page.is_closed():
                        pages.append(page)
    return pages


def _pick_chatgpt_page(browser: Browser, chatgpt_url: str) -> Optional[Page]:
    host = urlparse(chatgpt_url).netloc or "chatgpt.com"
    candidates: List[Page] = []
    for page in _list_all_pages(browser):
        with contextlib.suppress(Exception):
            if _host_matches(page.url, host):
                candidates.append(page)
    if candidates:
        candidates.sort(key=lambda item: 0 if item.url.rstrip("/") == chatgpt_url.rstrip("/") else 1)
        return candidates[0]
    return None


def _get_or_create_chatgpt_page(browser: Browser, chatgpt_url: str) -> Page:
    page = _pick_chatgpt_page(browser, chatgpt_url)
    if page is not None:
        return page
    contexts = list(browser.contexts)
    context = contexts[0] if contexts else browser.new_context()
    page = context.new_page()
    page.goto(chatgpt_url, wait_until="domcontentloaded", timeout=30000)
    return page


def _ensure_live_chatgpt_page(browser: Browser, chatgpt_url: str, current_page: Optional[Page] = None) -> Page:
    if current_page is not None:
        with contextlib.suppress(Exception):
            if not current_page.is_closed() and _host_matches(current_page.url, urlparse(chatgpt_url).netloc or "chatgpt.com"):
                return current_page
    return _get_or_create_chatgpt_page(browser, chatgpt_url)


def _find_visible_composer_selector(page: Page) -> Optional[str]:
    selectors = [
        "#prompt-textarea",
        "#thread-bottom textarea",
        "[data-testid='composer'] textarea",
        "form textarea[data-testid*='composer']",
        "textarea[data-testid*='prompt']",
        "div[contenteditable='true'][data-testid*='composer']",
        "div[contenteditable='plaintext-only']",
        "[role='textbox'][data-testid*='composer']",
        "div.ProseMirror[id='prompt-textarea']",
        "div.ProseMirror",
    ]
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.count() and locator.is_visible():
                return selector
        except Exception:
            continue
    try:
        return page.evaluate(
            """
            () => {
              const selectors = [
                "#prompt-textarea",
                "#thread-bottom textarea",
                "[data-testid='composer'] textarea",
                "form textarea[data-testid*='composer']",
                "textarea[data-testid*='prompt']",
                "div[contenteditable='true'][data-testid*='composer']",
                "div[contenteditable='plaintext-only']",
                "[role='textbox'][data-testid*='composer']",
                "div.ProseMirror[id='prompt-textarea']",
                "div.ProseMirror",
              ];
              const visible = (el) => {
                if (!el) return false;
                const style = window.getComputedStyle(el);
                if (!style) return false;
                if (style.display === "none" || style.visibility === "hidden") return false;
                const r = el.getBoundingClientRect();
                return r.width > 4 && r.height > 4;
              };
              for (const selector of selectors) {
                const el = document.querySelector(selector);
                if (visible(el)) return selector;
              }
              const fallback = Array.from(document.querySelectorAll("textarea,[contenteditable='true'],[contenteditable='plaintext-only'],[role='textbox']")).find(visible);
              if (!fallback) return null;
              if (fallback.id) return "#" + fallback.id;
              return fallback.tagName.toLowerCase();
            }
            """
        )
    except Exception:
        return None


def _find_send_button_bbox(page: Page) -> Optional[Tuple[int, int, int, int]]:
    try:
        box = page.evaluate(
            """
            () => {
              const visible = (el) => {
                if (!el) return false;
                const style = window.getComputedStyle(el);
                if (!style) return false;
                if (style.display === "none" || style.visibility === "hidden") return false;
                const r = el.getBoundingClientRect();
                return r.width > 4 && r.height > 4;
              };
              const buttons = Array.from(document.querySelectorAll("form button, button"));
              const match = buttons.find((btn) => {
                if (!visible(btn) || btn.disabled) return false;
                const text = ((btn.innerText || btn.textContent || "") + " " + (btn.getAttribute("aria-label") || "") + " " + (btn.getAttribute("data-testid") || "")).toLowerCase();
                return text.includes("send") || text.includes("wyslij") || text.includes("submit");
              });
              if (!match) return null;
              const r = match.getBoundingClientRect();
              return { x1: Math.round(r.left), y1: Math.round(r.top), x2: Math.round(r.right), y2: Math.round(r.bottom) };
            }
            """
        )
        if not box:
            return None
        return (int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"]))
    except Exception:
        return None


def _read_chat_state(page: Page) -> Dict[str, Any]:
    return page.evaluate(
        """
        () => {
          const visible = (el) => {
            if (!el) return false;
            const style = window.getComputedStyle(el);
            if (!style) return false;
            if (style.display === "none" || style.visibility === "hidden") return false;
            const r = el.getBoundingClientRect();
            return r.width > 4 && r.height > 4;
          };
          const textOf = (el) => ((el.innerText || el.textContent || "").replace(/\\s+/g, " ").trim());
          const buttons = Array.from(document.querySelectorAll("button")).filter(visible);
          const stopVisible = buttons.some((btn) => {
            const combined = ((btn.innerText || btn.textContent || "") + " " + (btn.getAttribute("aria-label") || "") + " " + (btn.getAttribute("data-testid") || "")).toLowerCase();
            return combined.includes("stop generating") || combined.includes("stop streaming") || combined === "stop" || combined.includes("stop response");
          });
          const seen = new Set();
          const assistantItems = [];
          const push = (el, source) => {
            if (!visible(el)) return;
            const text = textOf(el);
            if (!text) return;
            const r = el.getBoundingClientRect();
            const key = Math.round(r.top) + "|" + Math.round(r.height) + "|" + text.slice(0, 120);
            if (seen.has(key)) return;
            seen.add(key);
            const hasCopy = !!(
              el.querySelector("button[aria-label*='Copy']") ||
              el.querySelector("button[aria-label*='copy']") ||
              el.querySelector("button[data-testid*='copy']")
            );
            assistantItems.push({
              text,
              top: Math.round(r.top),
              bottom: Math.round(r.bottom),
              has_copy: hasCopy,
              source,
            });
          };
          document.querySelectorAll("[data-message-author-role='assistant']").forEach((el) => push(el, "data-role"));
          document.querySelectorAll("[data-testid*='conversation-turn']").forEach((el) => {
            const role = (el.getAttribute("data-message-author-role") || el.getAttribute("data-testid") || "").toLowerCase();
            if (role.includes("assistant")) push(el, "turn");
          });
          document.querySelectorAll("article").forEach((el) => {
            if (el.querySelector("button[aria-label*='Copy'], button[aria-label*='copy'], button[data-testid*='copy']")) {
              push(el, "article");
            }
          });
          assistantItems.sort((a, b) => (a.top - b.top) || (a.bottom - b.bottom));
          const last = assistantItems.length ? assistantItems[assistantItems.length - 1] : null;
          return {
            assistant_count: assistantItems.length,
            last_text: last ? last.text : "",
            last_has_copy: last ? !!last.has_copy : false,
            stop_visible: stopVisible,
          };
        }
        """
    )


def _find_copy_button_bbox(page: Page) -> Optional[Tuple[int, int, int, int]]:
    try:
        box = page.evaluate(
            """
            () => {
              const visible = (el) => {
                if (!el) return false;
                const style = window.getComputedStyle(el);
                if (!style) return false;
                if (style.display === "none" || style.visibility === "hidden") return false;
                const r = el.getBoundingClientRect();
                return r.width > 4 && r.height > 4;
              };
              const assistants = Array.from(document.querySelectorAll("[data-message-author-role='assistant']")).filter(visible);
              let target = null;
              if (assistants.length) {
                const last = assistants[assistants.length - 1];
                target = Array.from(last.querySelectorAll("button")).find((btn) => {
                  const combined = ((btn.innerText || btn.textContent || "") + " " + (btn.getAttribute("aria-label") || "") + " " + (btn.getAttribute("data-testid") || "")).toLowerCase();
                  return combined.includes("copy") && visible(btn);
                }) || null;
              }
              if (!target) {
                const all = Array.from(document.querySelectorAll("button")).filter((btn) => {
                  if (!visible(btn)) return false;
                  const combined = ((btn.innerText || btn.textContent || "") + " " + (btn.getAttribute("aria-label") || "") + " " + (btn.getAttribute("data-testid") || "")).toLowerCase();
                  return combined.includes("copy");
                });
                target = all.length ? all[all.length - 1] : null;
              }
              if (!target) return null;
              const r = target.getBoundingClientRect();
              return { x1: Math.round(r.left), y1: Math.round(r.top), x2: Math.round(r.right), y2: Math.round(r.bottom) };
            }
            """
        )
        if not box:
            return None
        return (int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"]))
    except Exception:
        return None


def _wait_for_response_completion(
    page: Page,
    previous_count: int,
    previous_text: str,
    *,
    timeout_s: float,
    stable_seconds: float,
) -> Dict[str, Any]:
    deadline = time.time() + float(timeout_s)
    stable_since: Optional[float] = None
    last_text = previous_text
    last_state: Dict[str, Any] = _read_chat_state(page)
    while time.time() < deadline:
        page.wait_for_timeout(350)
        state = _read_chat_state(page)
        last_state = state
        current_text = str(state.get("last_text") or "")
        assistant_count = int(state.get("assistant_count") or 0)
        if assistant_count <= previous_count and current_text == previous_text:
            stable_since = None
            continue
        if current_text != last_text:
            stable_since = time.time()
            last_text = current_text
            continue
        if state.get("stop_visible"):
            stable_since = None
            continue
        if stable_since is None:
            stable_since = time.time()
            continue
        if time.time() - stable_since >= float(stable_seconds):
            return state
    raise TimeoutError(f"Response did not stabilize within {timeout_s:.1f}s")


def _locate_bbox(page: Page, selector: str) -> Optional[Tuple[int, int, int, int]]:
    try:
        box = page.locator(selector).first.bounding_box()
    except Exception:
        box = None
    if not box:
        return None
    x1 = int(round(box["x"]))
    y1 = int(round(box["y"]))
    x2 = int(round(box["x"] + box["width"]))
    y2 = int(round(box["y"] + box["height"]))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


class WindowsClipboard:
    def __init__(self) -> None:
        if os.name != "nt":
            raise ClipboardError("Clipboard helper implemented only for Windows")
        self.kernel32 = ctypes.windll.kernel32
        self.user32 = ctypes.windll.user32
        self.kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
        self.kernel32.GlobalAlloc.restype = ctypes.c_void_p
        self.kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        self.kernel32.GlobalLock.restype = ctypes.c_void_p
        self.kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        self.kernel32.GlobalUnlock.restype = ctypes.c_int
        self.user32.OpenClipboard.argtypes = [ctypes.c_void_p]
        self.user32.OpenClipboard.restype = ctypes.c_int
        self.user32.CloseClipboard.argtypes = []
        self.user32.CloseClipboard.restype = ctypes.c_int
        self.user32.EmptyClipboard.argtypes = []
        self.user32.EmptyClipboard.restype = ctypes.c_int
        self.user32.GetClipboardData.argtypes = [ctypes.c_uint]
        self.user32.GetClipboardData.restype = ctypes.c_void_p
        self.user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        self.user32.SetClipboardData.restype = ctypes.c_void_p

    def _open(self) -> None:
        for _ in range(20):
            if self.user32.OpenClipboard(None):
                return
            time.sleep(0.05)
        raise ClipboardError("OpenClipboard failed")

    def read_text(self) -> str:
        self._open()
        try:
            handle = self.user32.GetClipboardData(CF_UNICODETEXT)
            if not handle:
                return ""
            locked = self.kernel32.GlobalLock(handle)
            if not locked:
                raise ClipboardError("GlobalLock failed")
            try:
                return ctypes.wstring_at(locked) or ""
            finally:
                self.kernel32.GlobalUnlock(handle)
        finally:
            self.user32.CloseClipboard()

    def write_text(self, text: str) -> None:
        payload = ctypes.create_unicode_buffer(str(text or ""))
        size = ctypes.sizeof(payload)
        handle = self.kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle:
            raise ClipboardError("GlobalAlloc failed")
        locked = self.kernel32.GlobalLock(handle)
        if not locked:
            raise ClipboardError("GlobalLock failed")
        try:
            ctypes.memmove(locked, ctypes.addressof(payload), size)
        finally:
            self.kernel32.GlobalUnlock(handle)
        self._open()
        try:
            if not self.user32.EmptyClipboard():
                raise ClipboardError("EmptyClipboard failed")
            if not self.user32.SetClipboardData(CF_UNICODETEXT, handle):
                raise ClipboardError("SetClipboardData failed")
        finally:
            self.user32.CloseClipboard()


class FallbackController:
    def __init__(self, port: int, root: Path) -> None:
        self.port = int(port)
        self.root = root
        self.proc: Optional[subprocess.Popen[Any]] = None
        self.ready = False

    def _control_agent_running(self) -> bool:
        if self.proc is not None and self.proc.poll() is None:
            return True
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
            sock.settimeout(0.1)
            try:
                sock.sendto(b"{}", ("127.0.0.1", self.port))
            except Exception:
                return False
        return False

    def ensure_ready(self) -> None:
        if self.ready:
            return
        subprocess_kw: Dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        proc = ensure_control_agent(
            control_agent_running=self._control_agent_running,
            control_agent_script=CONTROL_AGENT_SCRIPT,
            control_agent_port=self.port,
            root=self.root,
            subprocess_kw=subprocess_kw,
            log=log,
        )
        if proc is not None:
            self.proc = proc
            time.sleep(1.0)
        self.ready = True

    def click_bbox(self, bbox: Tuple[int, int, int, int], label: str) -> None:
        self.ensure_ready()
        x1, y1, x2, y2 = bbox
        tx = int((x1 + x2) / 2)
        ty = int((y1 + y2) / 2)
        ok = send_control_agent({"cmd": "move", "x": tx, "y": ty, "press": "mouse"}, self.port, log=log)
        if not ok:
            raise RuntimeError(f"Fallback click failed for {label}")

    def type_text(self, text: str) -> None:
        self.ensure_ready()
        if not send_type(text, self.port, log=log, delay=0.0):
            raise RuntimeError("Fallback typing failed")

    def keys(self, combo: str) -> None:
        self.ensure_ready()
        if not send_control_agent({"cmd": "keys", "combo": combo}, self.port, log=log):
            raise RuntimeError(f"Fallback keys failed for {combo}")

    def copy_selection(self) -> None:
        self.keys("ctrl+c")

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _ensure_output_dir(requested: Optional[str]) -> Path:
    if requested:
        out_dir = Path(requested).expanduser().resolve()
    else:
        out_dir = (RUNS_ROOT / _timestamp_now()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _copy_recorder_artifacts(out_dir: Path) -> None:
    artifacts = ["current_page.json", "current_clickables.json", "current_snapshot.json", "current_controls.json", "recorder_debug.log"]
    debug_dir = out_dir / "recorder_artifacts"
    for name in artifacts:
        src = DOM_LIVE_DIR / name
        if not src.exists():
            continue
        debug_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(Exception):
            shutil.copy2(src, debug_dir / name)


def _start_ai_recorder(
    *,
    chatgpt_url: str,
    user_data_dir: Path,
    cdp_endpoint: Optional[str],
    verbose: bool,
) -> subprocess.Popen[Any]:
    if not AI_RECORDER_SCRIPT.exists():
        raise FileNotFoundError(f"Missing ai_recorder_live.py at {AI_RECORDER_SCRIPT}")
    attach_existing = bool(cdp_endpoint and _probe_cdp(cdp_endpoint))
    args = [
        sys.executable,
        str(AI_RECORDER_SCRIPT),
        "--user-data-dir",
        str(user_data_dir),
        "--disable-ocr",
        "--dom-only",
    ]
    if attach_existing:
        args.extend(["--cdp-endpoint", cdp_endpoint, "--connect-existing"])
    else:
        args.extend(["--url", chatgpt_url])
    if verbose:
        args.append("--verbose")
    log(f"[INFO] Launching ai_recorder_live ({' '.join(args[2:])})")
    return subprocess.Popen(
        args,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _bring_chatgpt_page_front(page: Page, chatgpt_url: str) -> None:
    if page.is_closed():
        raise RuntimeError("ChatGPT page handle is already closed")
    if not _host_matches(page.url, urlparse(chatgpt_url).netloc or "chatgpt.com"):
        raise RuntimeError(f"Selected page is not ChatGPT: {page.url}")
    with contextlib.suppress(Exception):
        page.bring_to_front()
    page.wait_for_timeout(1000)


def _inject_prompt_userlike(
    page: Page,
    selector: str,
    prompt: str,
    fallback: FallbackController,
    clipboard: WindowsClipboard,
) -> str:
    bbox = _locate_bbox(page, selector)
    if not bbox:
        raise RuntimeError("Composer bbox not available")
    fallback.click_bbox(bbox, "composer")
    page.wait_for_timeout(250)
    with contextlib.suppress(Exception):
        fallback.keys("ctrl+a")
        page.wait_for_timeout(100)
    with contextlib.suppress(Exception):
        fallback.keys("backspace")
        page.wait_for_timeout(100)
    clipboard.write_text(prompt)
    page.wait_for_timeout(100)
    fallback.keys("ctrl+v")
    page.wait_for_timeout(250)
    return "control_agent_paste"


def _send_prompt_userlike(page: Page, fallback: FallbackController) -> str:
    fallback.keys("enter")
    page.wait_for_timeout(700)
    after_enter = _read_chat_state(page)
    if after_enter.get("stop_visible") or int(after_enter.get("assistant_count") or 0) > 0:
        return "control_agent_enter"
    send_bbox = _find_send_button_bbox(page)
    if send_bbox:
        fallback.click_bbox(send_bbox, "send_button")
        page.wait_for_timeout(500)
        return "control_agent_click_send"
    return "control_agent_enter"


def _extract_response_text(
    page: Page,
    state: Dict[str, Any],
    clipboard: WindowsClipboard,
    fallback: FallbackController,
) -> Tuple[str, str]:
    text = str(state.get("last_text") or "").strip()
    if text:
        return text, "dom"

    previous_clipboard = clipboard.read_text()
    copy_bbox = _find_copy_button_bbox(page)
    if copy_bbox:
        fallback.click_bbox(copy_bbox, "copy_button")
        page.wait_for_timeout(600)
        copied = clipboard.read_text().strip()
        if copied and copied != previous_clipboard:
            return copied, "copy_button_physical"

    raise RuntimeError("Failed to extract assistant response")


def _save_prompt_response(out_dir: Path, index: int, response_text: str) -> None:
    filename = f"response_{index:03d}.txt"
    (out_dir / filename).write_text(response_text, encoding="utf-8")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ChatGPT automation via ai_recorder + CDP.")
    parser.add_argument("--prompt", action="append", required=True, help="Prompt to send to ChatGPT. Repeat for batch mode.")
    parser.add_argument("--chatgpt-url", default=DEFAULT_CHATGPT_URL, help=f"ChatGPT URL (default: {DEFAULT_CHATGPT_URL})")
    parser.add_argument("--cdp-endpoint", default=DEFAULT_CDP_ENDPOINT, help=f"Preferred CDP endpoint (default: {DEFAULT_CDP_ENDPOINT})")
    parser.add_argument("--user-data-dir", default=str(RECORDER_PROFILE_DIR), help=f"Chrome profile dir used by ai_recorder (default: {RECORDER_PROFILE_DIR})")
    parser.add_argument("--out-dir", default=None, help="Directory for run artifacts/results.")
    parser.add_argument("--response-timeout", type=float, default=120.0, help="Max seconds to wait for one ChatGPT response.")
    parser.add_argument("--stable-seconds", type=float, default=2.0, help="How long response text must stay unchanged.")
    parser.add_argument("--control-agent-port", type=int, default=DEFAULT_CONTROL_AGENT_PORT, help=f"Control agent UDP port (default: {DEFAULT_CONTROL_AGENT_PORT})")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose recorder startup.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if sync_playwright is None:
        raise RuntimeError("playwright is not installed in this interpreter")
    if os.name != "nt":
        raise RuntimeError("This script currently supports Windows only")

    user_data_dir = Path(args.user_data_dir).expanduser().resolve()
    user_data_dir.mkdir(parents=True, exist_ok=True)
    out_dir = _ensure_output_dir(args.out_dir)
    results_path = out_dir / "results.jsonl"
    log_path = out_dir / "run.log"
    clipboard = WindowsClipboard()
    fallback = FallbackController(port=args.control_agent_port, root=PROJECT_ROOT)
    recorder_proc: Optional[subprocess.Popen[Any]] = None
    browser: Optional[Browser] = None
    pw = None
    summary: Dict[str, Any] = {
        "started_at": _utc_iso_now(),
        "chatgpt_url": args.chatgpt_url,
        "cdp_endpoint_requested": args.cdp_endpoint,
        "user_data_dir": str(user_data_dir),
        "prompts_total": len(args.prompt),
        "results_ok": 0,
        "results_failed": 0,
        "run_dir": str(out_dir),
    }

    with log_path.open("a", encoding="utf-8") as log_handle:
        def run_log(message: str) -> None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            line = f"[{timestamp}] {message}"
            print(line, flush=True)
            log_handle.write(line + "\n")
            log_handle.flush()

        global log
        log = run_log  # type: ignore[assignment]

        try:
            recorder_proc = _start_ai_recorder(
                chatgpt_url=args.chatgpt_url,
                user_data_dir=user_data_dir,
                cdp_endpoint=args.cdp_endpoint,
                verbose=bool(args.verbose),
            )
            endpoint = _wait_for_cdp_endpoint(
                args.cdp_endpoint if _probe_cdp(args.cdp_endpoint) else None,
                user_data_dir,
                timeout_s=60.0,
                started_proc=recorder_proc,
            )
            log(f"[INFO] Using CDP endpoint {endpoint}")

            pw = sync_playwright().start()
            browser = pw.chromium.connect_over_cdp(endpoint, timeout=10000)
            page = _ensure_live_chatgpt_page(browser, args.chatgpt_url)
            try:
                _bring_chatgpt_page_front(page, args.chatgpt_url)
            except Exception:
                page = _pick_chatgpt_page(browser, args.chatgpt_url) or _get_or_create_chatgpt_page(browser, args.chatgpt_url)
                _bring_chatgpt_page_front(page, args.chatgpt_url)

            composer_selector = _find_visible_composer_selector(page)
            if not composer_selector:
                raise RuntimeError("ChatGPT composer not found. Ensure session is logged in and chat UI is visible.")
            log(f"[INFO] Composer selector: {composer_selector}")

            for index, prompt in enumerate(args.prompt, start=1):
                started_at = _utc_iso_now()
                started_perf = time.perf_counter()
                page = _ensure_live_chatgpt_page(browser, args.chatgpt_url, page)
                try:
                    _bring_chatgpt_page_front(page, args.chatgpt_url)
                except Exception:
                    page = _pick_chatgpt_page(browser, args.chatgpt_url)
                    if page is None:
                        raise RuntimeError("Open ChatGPT in the existing browser session before running the next prompt.")
                    _bring_chatgpt_page_front(page, args.chatgpt_url)
                composer_selector = _find_visible_composer_selector(page) or composer_selector
                if not composer_selector:
                    raise RuntimeError("ChatGPT composer not found during prompt iteration")
                state_before = _read_chat_state(page)
                previous_count = int(state_before.get("assistant_count") or 0)
                previous_text = str(state_before.get("last_text") or "")
                capture_method = ""
                send_method = ""
                error_text: Optional[str] = None
                response_text = ""
                status = "ok"

                try:
                    prompt_method = _inject_prompt_userlike(page, composer_selector, prompt, fallback, clipboard)
                    send_method = _send_prompt_userlike(page, fallback)
                    final_state = _wait_for_response_completion(
                        page,
                        previous_count,
                        previous_text,
                        timeout_s=float(args.response_timeout),
                        stable_seconds=float(args.stable_seconds),
                    )
                    response_text, capture_method = _extract_response_text(page, final_state, clipboard, fallback)
                    clipboard.write_text(response_text)
                    _save_prompt_response(out_dir, index, response_text)
                    log(f"[INFO] Prompt {index}: prompt_method={prompt_method}, send_method={send_method}, capture={capture_method}, chars={len(response_text)}")
                    summary["results_ok"] = int(summary.get("results_ok", 0)) + 1
                except Exception as prompt_exc:
                    status = "failed"
                    error_text = str(prompt_exc)
                    log(f"[WARN] Prompt {index} failed: {error_text}")
                    _copy_recorder_artifacts(out_dir / f"failed_{index:03d}")
                    summary["results_failed"] = int(summary.get("results_failed", 0)) + 1

                finished_at = _utc_iso_now()
                result = PromptResult(
                    index=index,
                    prompt=prompt,
                    response_text=response_text,
                    status=status,
                    capture_method=capture_method or send_method or "",
                    page_url=str(getattr(page, "url", "")),
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_s=round(time.perf_counter() - started_perf, 3),
                    error=error_text,
                )
                _jsonl_append(results_path, asdict(result))

            summary["finished_at"] = _utc_iso_now()
            _json_dump(out_dir / "run_summary.json", summary)
            return 0 if int(summary.get("results_failed", 0)) == 0 else 1
        finally:
            if browser is not None:
                with contextlib.suppress(Exception):
                    browser.close()
            if pw is not None:
                with contextlib.suppress(Exception):
                    pw.stop()
            fallback.stop()
            if recorder_proc is not None and recorder_proc.poll() is None:
                recorder_proc.terminate()
                with contextlib.suppress(Exception):
                    recorder_proc.wait(timeout=5.0)


if __name__ == "__main__":
    raise SystemExit(main())
