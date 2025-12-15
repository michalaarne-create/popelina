import asyncio

import contextlib

import gc

import json

import os

import random

import signal

import subprocess

import sys

import time

from datetime import datetime

from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from typing import Any, Dict, List, Optional, Set



try:

    import mss

except ImportError:

    mss = None



if __package__:

    from .ai_recorder_common import (

        CONTENT_TIMEOUT_MS,

        SCREENSHOT_TIMEOUT_MS,

        DEFAULT_TIMEOUT_MS,

        ROI_HEIGHT,

        ROI_WIDTH,

        GOTO_TIMEOUT_MS,

        ensure_dir,

        build_bbox,

        log,

        ClickableElem,

        OcrLine,

        PageTrack3r,

        PerformanceMonitor,

        md5,

    )

    from .ai_recorder_windows import UIATracker

else:

    from ai_recorder_common import (

        CONTENT_TIMEOUT_MS,

        SCREENSHOT_TIMEOUT_MS,

        DEFAULT_TIMEOUT_MS,

        ROI_HEIGHT,

        ROI_WIDTH,

        GOTO_TIMEOUT_MS,

        ensure_dir,

        build_bbox,

        log,

        ClickableElem,

        OcrLine,

        PageTrack3r,

        PerformanceMonitor,

        md5,

    )

    from ai_recorder_windows import UIATracker



try:

    from playwright.async_api import Browser, BrowserContext, Page

except ImportError:

    raise SystemExit("playwright jest wymagany: pip install playwright \u0026\u0026 playwright install")





DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.6778.86 Safari/537.36"
)


class LiveRecorderBase:

    def __init__(

        self,

        output_dir: str,

        start_url: str,

        user_data_dir: str,

        fps: float,

        screenshots: bool,

        verbose: bool,

        viewport_mode: str = "os-max",

        chrome_exe: Optional[str] = None,

        connect_existing: bool = False,

        cdp_endpoint: Optional[str] = None,

        enable_ocr: Optional[bool] = None,

        profile_directory: Optional[str] = None,
        user_agent: Optional[str] = None,

        proxy_server: Optional[str] = None,

        json_out_path: Optional[str] = None,

        ndjson_out_path: Optional[str] = None,

        json_compact: bool = False,

        dom_only: bool = False,

        extra_urls: Optional[List[str]] = None,

    ):

        self.output_dir = Path(output_dir).resolve()

        self.start_url = start_url
        self.extra_urls = [u for u in (extra_urls or []) if isinstance(u, str) and u.strip()]

        self.user_data_dir = Path(user_data_dir).resolve()
        self.profile_directory = profile_directory
        ua_env = os.environ.get("RECORDER_USER_AGENT")
        self._user_agent_custom = bool(user_agent or ua_env)
        self.user_agent = user_agent or ua_env or None
        self.stealth_mode = (os.environ.get("RECORDER_STEALTH_MODE", "off") or "off").lower()
        self.hardware_concurrency = os.cpu_count() or 8
        self.snapshot_interval = 1.0 / max(0.5, float(fps))
        self.goto_timeout_ms = GOTO_TIMEOUT_MS

        self.record_screenshots = screenshots

        self.verbose = verbose

        self.viewport_mode = viewport_mode

        self.connect_existing = connect_existing

        self.cdp_endpoint = cdp_endpoint

        self.chrome_exe = chrome_exe

        # Allow environment override without changing CLI: set RECORDER_PROXY=scheme://host:port

        try:

            import os as _os

        except Exception:

            _os = None



        def _env_bool(name: str) -> Optional[bool]:

            if not _os:

                return None

            raw = _os.environ.get(name)

            if raw is None:

                return None

            raw = raw.strip().lower()

            if raw in ("0", "false", "off", "no", ""):

                return False

            return True



        self.proxy_server = proxy_server or (_os.environ.get("RECORDER_PROXY") if _os else None)

        env_ocr = _env_bool("RECORDER_ENABLE_OCR")

        if enable_ocr is None:

            self.enable_ocr = env_ocr if env_ocr is not None else True

        else:

            self.enable_ocr = bool(enable_ocr)

        self.debug_detection = bool(_env_bool("RECORDER_DEBUG_DETECTION") or False)

        self.dom_only = bool(dom_only)



        ensure_dir(self.output_dir)

        ensure_dir(self.user_data_dir)



        self.pw = None

        self.browser: Optional[Browser] = None

        self.context: Optional[BrowserContext] = None

        self.page: Optional[Page] = None



        self.chrome_process: Optional[subprocess.Popen] = None



        self.uia = UIATracker() if sys.platform.startswith("win") else None

        # W trybie DOM-only nie uruchamiamy MSS (zero zrzutów ekranu) – opcjonalne jeśli mss dostępne

        self.sct = (mss.mss() if (mss and not self.dom_only) else None)

        if not mss and not self.dom_only and self.record_screenshots:

            log("MSS not installed – screenshots disabled (pip install mss to enable)", "WARNING")

        self.roi_width = ROI_WIDTH

        self.roi_height = ROI_HEIGHT



        self.tracked_pages: Dict[str, PageTrack3r] = {}

        self.active_page_id: Optional[str] = None



        self.auto_switch_cooldown = 0.75

        self.user_is_active = False



        # pliki wyjĹ›ciowe

        self.file_snapshot = self.output_dir / "current_snapshot.json"

        self.file_clickables = self.output_dir / "current_clickables.json"

        data_screen = Path(__file__).resolve().parents[1] / "data" / "screen"
        ensure_dir(data_screen)
        self.page_current_dir = data_screen / "page_current"
        self.page_dir = data_screen / "page"
        ensure_dir(self.page_current_dir)
        ensure_dir(self.page_dir)
        self.file_page_current = self.page_current_dir / "page_current.json"
        self.file_page_static = self.page_dir / "page.json"
        self.file_page = self.file_page_current

        self.file_stats = self.output_dir / "stats.json"

        self.file_screenshot = self.output_dir / "current_screenshot.png"

        self.file_tabs = self.output_dir / "current_tabs.json"



        def _resolve_out(p: Optional[str]) -> Optional[Path]:

            if not p:

                return None

            pp = Path(p)

            return pp if pp.is_absolute() else (self.output_dir / pp)



        self.file_out_json = _resolve_out(json_out_path) or (self.output_dir / "out.json")

        self.file_out_ndjson = _resolve_out(ndjson_out_path)

        self.json_compact = bool(json_compact)



        self.recording = True

        self.last_snapshot_hash = ""

        self.last_url = ""

        self.last_title = ""

        self.page_state = "idle"

        self.stable_snapshots_count = 0

        self.last_dom_change_time = time.time()

        self.force_next_snapshot = True



        self.snapshots_written = 0

        self.last_write_time = 0.0



        self.clickables: List[ClickableElem] = []

        self.clickables_cache_time = 0.0

        self.clickables_cache_duration = 0.5



        self.ocr_lines_raw: List[OcrLine] = []

        self.ocr_lines_filtered: List[OcrLine] = []



        self.ocr_executor = ThreadPoolExecutor(max_workers=1)

        self.ocr_in_progress = False

        self._last_ocr_poll = 0.0

        self._ocr_dbg_t = 0.0

        self.ocr_strip_interval = 3.0

        self.ocr_debug_dump_interval = 5.0



        self.perf = PerformanceMonitor()



        self.last_successful_operation = time.time()

        self.watchdog_task: Optional[asyncio.Task] = None

        self.focus_poll_task: Optional[asyncio.Task] = None

        self.ocr_poll_task: Optional[asyncio.Task] = None



        self.stats = {

            "session_start": datetime.now().isoformat(),

            "last_update": datetime.now().isoformat(),

            "total_snapshots": 0,

            "files_written": 0,

            "ocr_runs": 0,

            "navigation_errors": 0,

            "page_switches": 0,

            "manual_switches": 0,

            "timeouts": 0,

            "watchdog_resets": 0

        }



        # STEALTH: Bardziej generyczny klucz storage (wyglÄ…da jak zwykĹ‚a aplikacja)

        import hashlib

        domain_hash = hashlib.md5(start_url.encode()).hexdigest()[:6]

        timestamp_part = str(int(time.time()) % 100000)

        self.activity_key = f"app_state_{domain_hash}_{timestamp_part}"

        

        self._ctx_ids_configured: Set[int] = set()



        signal.signal(signal.SIGINT, self._signal_handler)

        signal.signal(signal.SIGTERM, self._signal_handler)





    def _signal_handler(self, signum, frame):

        print("\nđź›‘ Stopping...")

        self.recording = False



    async def stop(self):

        # miÄ™kkie zatrzymanie â€“ nie zamykamy Chrome/Context/Browser; tylko taski i Playwright driver

        self.recording = False

        if self.watchdog_task:

            self.watchdog_task.cancel()

            with contextlib.suppress(Exception):

                await asyncio.sleep(0)

        if self.focus_poll_task:

            self.focus_poll_task.cancel()

            with contextlib.suppress(Exception):

                await asyncio.sleep(0)

        if self.ocr_poll_task:

            self.ocr_poll_task.cancel()

            with contextlib.suppress(Exception):

                await asyncio.sleep(0)

        try:

            if self.pw:

                await self.pw.stop()

        except Exception:

            pass



    async def eval_js(self, script: str, timeout_ms: int = 3000, arg=None, default=None, page: Optional[Page] = None):

        p = page or self.page

        if not p or p.is_closed():

            return default

        try:

            if arg is None:

                result = await asyncio.wait_for(p.evaluate(script), timeout=timeout_ms / 1000)

            else:

                result = await asyncio.wait_for(p.evaluate(script, arg), timeout=timeout_ms / 1000)

            self.last_successful_operation = time.time()

            return result

        except asyncio.TimeoutError:

            self.stats["timeouts"] += 1

            return default

        except Exception:

            return default



    async def safe_eval(self, script: str, arg=None, default=None, page: Optional[Page] = None):

        p = page or self.page

        if not p or p.is_closed():

            return default

        try:

            if arg is None:

                result = await asyncio.wait_for(

                    p.evaluate(script),

                    timeout=DEFAULT_TIMEOUT_MS/1000

                )

            else:

                result = await asyncio.wait_for(

                    p.evaluate(script, arg),

                    timeout=DEFAULT_TIMEOUT_MS/1000

                )

            self.last_successful_operation = time.time()

            return result

        except asyncio.TimeoutError:

            self.stats["timeouts"] += 1

            return default

        except Exception:

            return default



    async def _collect_extra_json(self) -> Dict[str, Any]:

        """

        Zbiera dane do zwiÄ™zĹ‚ego JSON: window.__BROWSER_SCAN__.config oraz <symbol> z SVG.

        Zero inwazyjnoĹ›ci â€” tylko odczyt.

        """

        if not self.page or self.page.is_closed():

            return {"browserScan": None, "svgSymbols": []}

        js = r"""

        () => {

            const cfg = (window.__BROWSER_SCAN__ && window.__BROWSER_SCAN__.config) || null;

            const symbols = [...document.querySelectorAll('symbol')].map(s => ({

                id: s.id || s.getAttribute('id') || null,

                viewBox: s.getAttribute('viewBox') || null,

                paths: [...s.querySelectorAll('path')].map(p => p.getAttribute('d')).filter(Boolean)

            }));

            return { cfg, symbols };

        }

        """

        try:

            res = await self.safe_eval(js, default=None)

            if not res or not isinstance(res, dict):

                return {"browserScan": None, "svgSymbols": []}

            return {

                "browserScan": res.get("cfg"),

                "svgSymbols": res.get("symbols") or []

            }

        except Exception:

            return {"browserScan": None, "svgSymbols": []}



    async def _write_out_json_records(self, record: Dict[str, Any]) -> None:

        """

        Zapisuje:

        - out.json (ostatni peĹ‚ny stan â€” nadpisywany)

        - out.ndjson (jeĹ›li skonfigurowano â€” dopisywana 1 linia/rekord)

        """

        def _writer():

            try:

                # out.json

                if self.file_out_json:

                    with open(self.file_out_json, "w", encoding="utf-8") as f:

                        json.dump(record, f, ensure_ascii=False,

                                  indent=None if self.json_compact else 2)

                # out.ndjson

                if self.file_out_ndjson:

                    with open(self.file_out_ndjson, "a", encoding="utf-8") as f:

                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                self.stats["files_written"] += 1

            except Exception as e:

                log(f"âťŚ out.json/out.ndjson write error: {e}", "ERROR")



        loop = asyncio.get_running_loop()

        await loop.run_in_executor(None, _writer)



    async def check_detection_extended(self):

        """Rozszerzona wersja z dodatkowymi sprawdzeniami (tylko na życzenie)."""

        if not getattr(self, "debug_detection", False):

            return {"score": None, "issues": ["detection_check_disabled"], "details": {}}

        if not self.page or self.page.is_closed():

            return {"score": None, "issues": ["no_page"], "details": {}}

        script = """

        () => {

            const checks = {};

            checks.webdriver = navigator.webdriver === true;

            checks.chromeRuntime = !!(window.chrome && window.chrome.runtime);

            try {

                checks.pluginsCount = navigator.plugins ? navigator.plugins.length : 0;

            } catch (e) {

                checks.pluginsCount = -1;

            }

            try {

                checks.hasCdcVariables = Object.keys(window || {}).some((k) => k.startsWith('cdc_'));

            } catch (e) {

                checks.hasCdcVariables = false;

            }

            checks.hasDomAutomation = typeof window.domAutomation !== "undefined";

            return checks;

        }}

        """

        result = await self.safe_eval(script, default={})

        score = 100

        issues: List[str] = []

        if result.get("webdriver"):

            score -= 60

            issues.append("❌ navigator.webdriver = true")

        if result.get("hasCdcVariables"):

            score -= 30

            issues.append("❌ cdc_* artifacts present")

        if result.get("hasDomAutomation"):

            score -= 15

            issues.append("⚠️ window.domAutomation detected")

        if not result.get("chromeRuntime"):

            score -= 10

            issues.append("⚠️ chrome.runtime missing or incomplete")

        if result.get("pluginsCount", 0) <= 0:

            score -= 5

            issues.append("⚠️ navigator.plugins empty")

        log(f"🎯 STEALTH SCORE: {score}/100", "SUCCESS" if score >= 80 else "WARNING")

        for issue in issues:

            log(f"  {issue}", "WARNING")

        return {"score": score, "issues": issues, "details": result}


    async def safe_content(self, default="", page: Optional[Page] = None):

        """STEALTH VERSION - z naturalnym timingiem"""

        p = page or self.page

        if not p or p.is_closed():

            return default

        

        # Losowy timeout (nie staĹ‚y!)

        timeout = random.uniform(

            CONTENT_TIMEOUT_MS * 0.8,

            CONTENT_TIMEOUT_MS * 1.2

        ) / 1000

        

        try:

            # Czasem (10%) dodaj maĹ‚e opĂłĹşnienie przed odczytem

            if random.random() < 0.1:

                await asyncio.sleep(random.uniform(0.05, 0.15))

            

            content = await asyncio.wait_for(p.content(), timeout=timeout)

            self.last_successful_operation = time.time()

            return content

            

        except asyncio.TimeoutError:

            self.stats["timeouts"] += 1

            self.stats["navigation_errors"] += 1

            return default

        except Exception:

            self.stats["navigation_errors"] += 1

            return default



    



    async def safe_screenshot_bytes(self) -> Optional[bytes]:

        """STEALTH VERSION - ludzkie screenshoty"""

        if not self.page or self.page.is_closed():

            return None

        

        # Losowy timeout

        timeout = random.uniform(

            SCREENSHOT_TIMEOUT_MS * 0.8,

            SCREENSHOT_TIMEOUT_MS * 1.2

        ) / 1000

        

        try:

            # === OPCJA: Czasem rĂłb full_page (jak czĹ‚owiek scrolluje) ===

            full_page = random.random() < 0.15  # 15% czasu

            

            b = await asyncio.wait_for(

                self.page.screenshot(full_page=full_page),

                timeout=timeout

            )

            

            if self.record_screenshots and b:

                # Czasem (90%) zapisz

                if random.random() < 0.9:

                    with contextlib.suppress(Exception):

                        with open(self.file_screenshot, "wb") as f:

                            f.write(b)

            

            self.last_successful_operation = time.time()

            return b

            

        except asyncio.TimeoutError:

            self.stats["timeouts"] += 1

            return None

        except Exception:

            return None









    async def extract_clickables(self, page: Optional[Page] = None) -> List[ClickableElem]:

        """STEALTH VERSION - bogatsze heurystyki i trudniejsze fingerprintowanie."""

        now = time.time()



        cache_duration = random.uniform(

            self.clickables_cache_duration * 0.7,

            self.clickables_cache_duration * 1.3,

        )

        if self.clickables and (now - self.clickables_cache_time < cache_duration):

            return self.clickables



        base_selectors = [
            "button",
            "a[href]",
            "input:not([type='hidden'])",
            "select",
            "textarea",
            "[contenteditable='true']",
            "[contenteditable='plaintext-only']",
            "[role='textbox']",
        ]
        extra_selectors = [

            "[role='button']:not([aria-disabled='true'])",

            "[role='checkbox']:not([aria-disabled='true'])",

            "[role='radio']:not([aria-disabled='true'])",

            "[onclick]",

            "label[for]",

            "[role='option']",

            "[data-value]",

            "[data-answer]",

            "div[style*='cursor: pointer']",

            "[class*='clickable']",

            "[class*='selectable']",

        ]

        num_extra = max(1, random.randint(len(extra_selectors) // 2, int(len(extra_selectors) * 0.8)))

        selected_extra = random.sample(extra_selectors, num_extra)

        all_selectors = base_selectors + selected_extra

        random.shuffle(all_selectors)

        selectors_js = json.dumps(all_selectors)



        js = f"""
        () => {{
            const selectors = {selectors_js};
            const seen = new WeakSet();
            const out = [];
            const honeypot = /(honeypot|hp-field|hidden-input|trap)/i;
            const skipRoles = /(presentation|menu|listbox)/i;
            const composerSelectors = [
                "#prompt-textarea",
                "#thread-bottom textarea",
                "[data-testid='composer'] textarea",
                "form textarea[data-testid*='composer']",
                "textarea[data-testid*='prompt']",
                "div[contenteditable='true'][data-testid*='composer']",
                "div[contenteditable='plaintext-only']",
                "[role='textbox'][data-testid*='composer']",
                "div.ProseMirror[id='prompt-textarea']",
            ];


            const sample = (nodes, selector) => {{

                if (!nodes) return;

                for (const el of nodes) {{

                    if (!el || seen.has(el)) continue;

                    seen.add(el);

                    const r = el.getBoundingClientRect();

                    if (!r || r.width <= 1 || r.height <= 1) continue;

                    const style = window.getComputedStyle(el);

                    if (!style) continue;

                    if (style.display === 'none' || style.visibility === 'hidden') continue;

                    if (style.pointerEvents === 'none') continue;

                    if (parseFloat(style.opacity || '1') < 0.05) continue;

                    const cls = (el.className || '').toString().toLowerCase();

                    if (honeypot.test(cls)) continue;

                    if (skipRoles.test(el.getAttribute('role') || '')) continue;

                    const text = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ');

                    out.push({{

                        selector,

                        tag: (el.tagName || '').toLowerCase(),

                        type: el.getAttribute('type') || el.getAttribute('role') || style.cursor || 'unknown',

                        text: text.substring(0, 160),

                        id: el.id || null,

                        href: el.href || null,

                        tabindex: el.tabIndex,

                        aria: {{

                            label: el.getAttribute('aria-label'),

                            pressed: el.getAttribute('aria-pressed'),

                            expanded: el.getAttribute('aria-expanded')

                        }},

                        corners: {{

                            top_left: {{x: Math.round(r.left), y: Math.round(r.top)}},

                            top_right: {{x: Math.round(r.right), y: Math.round(r.top)}},

                            bottom_left: {{x: Math.round(r.left), y: Math.round(r.bottom)}},

                            bottom_right: {{x: Math.round(r.right), y: Math.round(r.bottom)}}

                        }},

                        center: {{x: Math.round(r.left + r.width/2), y: Math.round(r.top + r.height/2)}},

                        size: {{width: Math.round(r.width), height: Math.round(r.height)}},

                        isInteractive: true

                    }});

                }}

            }};



            for (const sel of selectors) {{
                try {{
                    sample(document.querySelectorAll(sel), sel);
                }} catch (e) {{}}
            }}

            for (const sel of composerSelectors) {{
                try {{
                    const el = document.querySelector(sel);
                    if (el) sample([el], sel + "::composer");
                }} catch (e) {{}}
            }}

            const ensureComposer = () => {{
                if (out.some(item => item.selector === "__composer__")) return;
                const candidates = [
                    document.querySelector("#prompt-textarea"),
                    document.querySelector("#thread-bottom textarea:not([style*='display: none'])"),
                    document.querySelector("form textarea[name='prompt-textarea']"),
                    document.querySelector("textarea[data-testid*='composer']"),
                    document.querySelector("div.ProseMirror[id='prompt-textarea']"),
                    document.querySelector("div.ProseMirror"),
                ].filter(Boolean);
                const el = candidates.length ? candidates[0] : null;
                const host = el ? el.closest("form") || el.closest("[data-testid*='composer']") || el
                                 : document.querySelector("#thread-bottom form");
                const target = el || host;
                if (!target) return;
                let rect = target.getBoundingClientRect();
                if (!rect || rect.width <= 4 || rect.height <= 4) {{
                    rect = host && host !== target ? host.getBoundingClientRect() : rect;
                }}
                if (!rect || rect.width <= 4 || rect.height <= 4) return;
                const style = window.getComputedStyle(target);
                if (style && (style.display === "none" || style.visibility === "hidden")) return;
                const centerX = Math.round(rect.left + rect.width / 2);
                const centerY = Math.round(rect.top + rect.height / 2);
                const tagName = (target.tagName || "div").toLowerCase();
                const id = target.id ? ("#" + target.id) : "";
                const dataTestid = target.getAttribute("data-testid") || "";
                const nameAttr = target.getAttribute("name") || "";
                const hostId = host && host.id ? ("#" + host.id) : "";
                const hostDataTestid = (host && host.getAttribute("data-testid"))
                    ? "[data-testid='" + host.getAttribute("data-testid") + "']"
                    : "";
                const selectorParts = [];
                if (tagName) selectorParts.push(tagName);
                if (id) selectorParts.push(id);
                if (dataTestid) selectorParts.push("[data-testid='" + dataTestid + "']");
                if (nameAttr) selectorParts.push("[name='" + nameAttr + "']");
                if (hostId) selectorParts.push(hostId);
                if (hostDataTestid) selectorParts.push(hostDataTestid);
                selectorParts.push("__composer__");
                const selectorHint = selectorParts.join(" ").trim();
                out.push({{
                    selector: selectorHint || "__composer__",
                    tag: tagName,
                    type: "textbox",
                    text: (target.innerText || target.textContent || "").trim().slice(0, 160),
                    id: target.id || null,
                    href: null,
                    tabindex: target.tabIndex,
                    aria: {{
                        label: target.getAttribute("aria-label") || "composer",
                        pressed: target.getAttribute("aria-pressed"),
                        expanded: target.getAttribute("aria-expanded"),
                    }},
                    corners: {{
                        top_left: {{x: Math.round(rect.left), y: Math.round(rect.top)}},
                        top_right: {{x: Math.round(rect.right), y: Math.round(rect.top)}},
                        bottom_left: {{x: Math.round(rect.left), y: Math.round(rect.bottom)}},
                        bottom_right: {{x: Math.round(rect.right), y: Math.round(rect.bottom)}},
                    }},
                    center: {{x: centerX, y: centerY}},
                    size: {{width: Math.round(rect.width), height: Math.round(rect.height)}},
                    isInteractive: true,
                }});
            }};

            ensureComposer();


            out.sort((a, b) => {{

                const dy = a.corners.top_left.y - b.corners.top_left.y;

                if (Math.abs(dy) > 5) return dy;

                return a.corners.top_left.x - b.corners.top_left.x;

            }});

            return out;

        }}

        """
        start_time = self.perf.start("extract_clickables")
        timeout_ms = random.randint(3000, 5000)
        out: List[ClickableElem] = []

        try:
            raw = await self.eval_js(js, timeout_ms=timeout_ms, default=[], page=page)
            count = len(raw or [])
            if self.verbose:
                log(f"extract_clickables -> {count} nodes", "DEBUG")
            for idx, item in enumerate(raw or []):
                try:
                    corners = item.get("corners") or {}
                    top_left = corners.get("top_left") or {}
                    size = item.get("size") or {}
                    width = max(1, int(size.get("width") or 0))
                    height = max(1, int(size.get("height") or 0))
                    extended_bbox = build_bbox(
                        int(top_left.get("x") or 0),
                        int(top_left.get("y") or 0),
                        width,
                        height,
                    )
                    attrs = {
                        "selector": item.get("selector"),
                        "tabindex": item.get("tabindex"),
                        "aria": item.get("aria"),
                        "is_interactive": item.get("isInteractive", False),
                    }
                    elem = ClickableElem(
                        id=md5(f"{item.get('type')}_{idx}_{top_left}"),
                        tag=item.get("tag", ""),
                        role=item.get("type", ""),
                        type=item.get("type"),
                        text=item.get("text", ""),
                        bbox=extended_bbox,
                        href=item.get("href"),
                        category="control",
                        attributes=attrs,
                    )
                    out.append(elem)
                except Exception as item_err:
                    if self.verbose:
                        log(f"extract_clickables item error: {item_err}", "DEBUG")
                    continue
        except Exception as e:
            log(f"extract_clickables js error: {e}", "DEBUG")
            out = []

        self.clickables = out
        self.clickables_cache_time = time.time()
        self.perf.end("extract_clickables", start_time)
        return out



    def _write_tabs_file(self) -> None:

        """Zapisz aktualny stan kart do current_tabs.json."""

        try:

            active_id = getattr(self, "active_page_id", None)

            tabs_payload: List[Dict[str, Any]] = []

            for tr in list(getattr(self, "tracked_pages", {}).values()):

                title = (tr.title or tr.url or "Tab").strip() or "Tab"

                url = tr.url or ""

                tabs_payload.append({

                    "title": title,

                    "url": url,

                    "active": bool(tr.page_id == active_id),

                })

            with open(self.file_tabs, "w", encoding="utf-8") as f:

                json.dump(

                    tabs_payload,

                    f,

                    ensure_ascii=False,

                    indent=None if self.json_compact else 2,

                )

        except Exception:

            pass

    async def start(self):
        """STEALTH START - symuluje ludzkie otwieranie strony"""
        log("Connecting via CDP...", "INFO")
        await asyncio.sleep(random.uniform(0.5, 1.5))
        await self._connect_cdp()
        # Optional viewport tweak before any navigation
        if random.random() < 0.7:
            await asyncio.sleep(random.uniform(0.1, 0.3))
            try:
                await self._apply_viewport_mode("pre-goto", self.page)
            except Exception:
                pass
        url = (self.start_url or "").strip()
        try:
            page_url = (self.page.url or "").strip()
        except Exception:
            page_url = ""
        if url and getattr(self, "connect_existing", False):
            if page_url and page_url.lower() not in ("about:blank", "chrome://newtab/", "chrome://new-tab-page/", "edge://newtab/"):
                log(f"Skipping initial navigation (already at {page_url[:80]})", "INFO")
                url = ""
        if url:
            log(f"Navigating to: {url}", "INFO")
            typing_delay = random.uniform(0.8, 2.0)
            await asyncio.sleep(typing_delay)
            wait_strategies = [
                ("domcontentloaded", 0.3),
                ("load", 0.4),
                ("networkidle", 0.2),
                (None, 0.1),
            ]
            strategy = random.choices(
                [s[0] for s in wait_strategies],
                weights=[s[1] for s in wait_strategies],
            )[0]
            try:
                if strategy:
                    await self.page.goto(url, wait_until=strategy, timeout=self.goto_timeout_ms)
                else:
                    await self.page.goto(url, wait_until="commit", timeout=self.goto_timeout_ms)
                    await asyncio.sleep(random.uniform(0.5, 1.5))
            except Exception as e:
                log(f"go to failed ({e}); retrying with delay", "WARNING")
                await asyncio.sleep(random.uniform(1.0, 3.0))
                try:
                    self.page = await self.context.new_page()
                    await self._handle_new_page(self.page)
                    await asyncio.sleep(random.uniform(0.3, 0.8))
                    await self.page.goto(url, wait_until="domcontentloaded", timeout=self.goto_timeout_ms)
                except Exception as e2:
                    log(f"goto retry failed: {e2}", "ERROR")
                    raise
            settle_time = random.uniform(0.5, 2.0)
            await asyncio.sleep(settle_time)
            if random.random() < 0.3:
                try:
                    await self._apply_viewport_mode("post-goto", self.page)
                except Exception:
                    pass
        else:
            log("Initial navigation skipped (no start_url provided).", "INFO")
        await self._register_page(self.page)
        self.active_page_id = str(id(self.page))
        # Opcjonalnie otwórz dodatkowe karty na starcie
        for extra_url in self.extra_urls:
            url2 = (extra_url or "").strip()
            if not url2:
                continue
            try:
                p_extra = await self.context.new_page()
                await self._handle_new_page(p_extra)
                await asyncio.sleep(random.uniform(0.2, 0.6))
                await p_extra.goto(url2, wait_until="domcontentloaded", timeout=self.goto_timeout_ms)
                await asyncio.sleep(random.uniform(0.2, 0.6))
                log(f"Extra tab opened: {url2}", "INFO")
            except Exception as extra_err:
                log(f"Extra tab failed ({url2}): {extra_err}", "WARNING")
        await asyncio.sleep(random.uniform(0.2, 0.5))
        if self.watchdog_task:
            self.watchdog_task.cancel()
        if self.focus_poll_task:
            self.focus_poll_task.cancel()
        if self.ocr_poll_task:
            self.ocr_poll_task.cancel()
        self.watchdog_task = asyncio.create_task(self._watchdog_loop())
        self.focus_poll_task = asyncio.create_task(self._focus_poll_loop())
        self.ocr_poll_task = asyncio.create_task(self._ocr_strip_poll_loop())
        log(f"Ready: {self.page.url[:80]} | mode={self.viewport_mode}", "SUCCESS")

    async def run(self):

        """STEALTH RUN - naturalne wzorce aktywno?ci"""

        await self.start()

        log(f"?? Recorder running | Output: {self.output_dir}", "SUCCESS")

        # === ZMIENNE PARAMETRY (nie sta?e!) ===
        base_snapshot_interval = self.snapshot_interval
        last_snapshot_time = 0.0
        last_gc_time = time.time()

        # Losowy GC interval (45-90s zamiast zawsze 60)
        gc_interval = random.uniform(45, 90)

        # Activity pattern (symuluje uwag? u?ytkownika)
        activity_level = 1.0  # 1.0 = aktywny, 0.1 = nieaktywny
        last_activity_change = time.time()
        next_activity_change = last_activity_change + random.uniform(20, 60)

        while self.recording:

            try:

                now = time.time()

                # === SYMULACJA AKTYWNO?CI U?YTKOWNIKA ===
                # Co 20-60s zmie? poziom aktywno?ci
                if now >= next_activity_change:

                    # 70% aktywny, 30% nieaktywny/AFK
                    activity_level = random.choices([1.0, 0.1], weights=[0.7, 0.3])[0]

                    last_activity_change = now
                    next_activity_change = now + random.uniform(20, 60)

                    if activity_level < 0.5:

                        log("?? User inactive/AFK simulation", "DEBUG")

                # === NATURALNY SNAPSHOT INTERVAL ===
                # Dodaj jitter i uwzgl?dnij aktywno??
                jittered_interval = base_snapshot_interval * random.uniform(0.8, 1.2)

                # Je?li nieaktywny, rzadsze snapshoty
                if activity_level < 0.5:

                    jittered_interval *= random.uniform(2.0, 4.0)

                time_since_snapshot = now - last_snapshot_time
                force_snap = bool(getattr(self, "force_next_snapshot", False))
                if not force_snap and time_since_snapshot < jittered_interval:
                    # je?li nie pora na snapshot, po?pij i oddaj CPU
                    remaining = jittered_interval - time_since_snapshot
                    idle_floor = 0.02 if activity_level > 0.5 else 0.08
                    idle_ceiling = 0.6 if activity_level > 0.5 else 1.0
                    await asyncio.sleep(min(max(idle_floor, remaining * 0.5), idle_ceiling))
                    continue

                # Czasem (5%) pomi? snapshot (lag, rozproszenie)
                if force_snap or random.random() > 0.05:

                    self.force_next_snapshot = False

                    snapshot_task = asyncio.create_task(self._do_snapshot())

                    # Timeout te? z jitterem
                    timeout = jittered_interval * random.uniform(1.8, 2.5)
                    # Adjust timeout to realistic upper bounds based on DOM ops
                    timeout = max(

                        timeout,

                        3.0,

                        (CONTENT_TIMEOUT_MS / 1000.0) * 2.2,

                        (SCREENSHOT_TIMEOUT_MS / 1000.0) * 1.5,

                        jittered_interval * 3.0,

                    )

                    try:

                        await asyncio.wait_for(snapshot_task, timeout=timeout)

                    except asyncio.TimeoutError:

                        log("?? Snapshot timeout", "WARNING")

                        self.stats["timeouts"] += 1

                else:

                    log("?? Skipped snapshot (distracted)", "DEBUG")

                last_snapshot_time = time.time()

                # === NATURALNY GC PATTERN ===
                # Nie co dok?adnie X sekund!
                if time.time() - last_gc_time > gc_interval:

                    # 80% szans na GC (czasem zapomina)
                    if random.random() < 0.8:

                        gc.collect()

                        last_gc_time = time.time()

                        # Nast?pny GC za losowy czas

                        gc_interval = random.uniform(45, 90)

                    # Performance stats tylko czasem (30%)
                    if random.random() < 0.3:

                        perf_stats = self.perf.get_stats()

                        if perf_stats:

                            slowest = max(perf_stats.items(), key=lambda x: x[1].get("max", 0))

                            log(f"?? Slowest: {slowest[0]} ({slowest[1]['max']:.2f}s)", "DEBUG")

                # === NATURALNY SLEEP PATTERN ===
                if activity_level > 0.5:

                    # Aktywny: 20-80ms
                    sleep_time = random.uniform(0.02, 0.08)

                else:

                    # Nieaktywny: 120-600ms
                    sleep_time = random.uniform(0.12, 0.6)

                await asyncio.sleep(sleep_time)

            except Exception as e:

                log(f"? Loop error: {e}", "ERROR")

                # Losowy backoff

                await asyncio.sleep(random.uniform(0.3, 1.0))
