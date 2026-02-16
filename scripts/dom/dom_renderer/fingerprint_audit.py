#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fingerprint_audit.py

Headed, DOM-only fingerprint audit. Does not modify the page.
Runs a comprehensive set of client-side checks to report signals that
sites often use to classify automation/webdriver. Produces a concise
human-readable summary and (optionally) a JSON report.

Usage examples (PowerShell/CMD):

  # Connect to existing headed browser started with --remote-debugging-port
  python -m dom_renderer.fingerprint_audit \
      --cdp-endpoint http://127.0.0.1:9222 --attach-only --verbose

  # Launch a headed browser just for auditing a URL (least recommended for parity)
  python -m dom_renderer.fingerprint_audit --url https://example.com --verbose
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _print(msg: str) -> None:
    print(msg, flush=True)


JS_AUDIT = r"""
(() => {
  const out = { ok: true };
  const safe = (fn, fallback=null) => { try { return fn(); } catch(e) { return fallback === undefined ? String(e) : fallback; } };

  out.navigator = {};
  out.navigator.webdriver = safe(() => navigator.webdriver, null);
  out.navigator.webdriverDesc = safe(() => !!Object.getOwnPropertyDescriptor(navigator, 'webdriver'), null);
  out.navigator.userAgent = safe(() => navigator.userAgent, null);
  out.navigator.headlessUA = safe(() => /HeadlessChrome/i.test(navigator.userAgent), null);
  out.navigator.languages = safe(() => navigator.languages, null);
  out.navigator.language = safe(() => navigator.language, null);
  out.navigator.platform = safe(() => navigator.platform, null);
  out.navigator.pluginsLength = safe(() => navigator.plugins && navigator.plugins.length, null);
  out.navigator.mimeTypesLength = safe(() => navigator.mimeTypes && navigator.mimeTypes.length, null);
  out.navigator.hardwareConcurrency = safe(() => navigator.hardwareConcurrency, null);
  out.navigator.deviceMemory = safe(() => navigator.deviceMemory, null);
  out.navigator.maxTouchPoints = safe(() => navigator.maxTouchPoints, null);

  out.uaData = safe(() => (
    navigator.userAgentData ? {
      mobile: navigator.userAgentData.mobile,
      brands: navigator.userAgentData.brands
    } : null
  ), null);

  out.timezone = safe(() => Intl.DateTimeFormat().resolvedOptions().timeZone, null);

  out.env = {};
  out.env.hasChromeObj = safe(() => typeof window.chrome !== 'undefined', null);
  out.env.hasChromeRuntime = safe(() => !!(window.chrome && window.chrome.runtime), null);
  out.env.devtools = null; // not reliably detectable without heuristics

  out.perms = {};
  out.perms.queryNative = safe(() => {
    const q = navigator.permissions && navigator.permissions.query;
    if (!q) return null;
    const s = Function.prototype.toString.call(q);
    return { includesNative: s.includes('[native code]'), length: s.length };
  }, null);

  out.keys = {};
  out.keys.cdc = safe(() => Object.keys(window).filter(k => k.startsWith('cdc_')), []);
  out.keys.playwright = safe(() => !!window.__playwright, null);
  out.keys.puppeteer = safe(() => !!window.__puppeteer_evaluation_script__, null);

  out.viewport = {
    inner: { w: safe(() => innerWidth, null), h: safe(() => innerHeight, null) },
    outer: { w: safe(() => outerWidth, null), h: safe(() => outerHeight, null) },
    screen: {
      w: safe(() => screen.width, null), h: safe(() => screen.height, null),
      aw: safe(() => screen.availWidth, null), ah: safe(() => screen.availHeight, null)
    },
    dpr: safe(() => devicePixelRatio, null)
  };

  // WebGL vendor/renderer (optional)
  out.webgl = { vendor: null, renderer: null };
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (gl) {
      const dbg = gl.getExtension('WEBGL_debug_renderer_info');
      out.webgl.vendor = dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : null;
      out.webgl.renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : null;
    }
  } catch(e) {}

  // Quick stack check (may show 'evaluate' in certain engines)
  out.stack = safe(() => { const err = new Error(); return (err.stack || '').slice(0, 200); }, null);

  // Notification permission as a simple baseline
  out.notification = safe(() => Notification && Notification.permission, null);

  return out;
})()
"""


def score(report: Dict[str, Any]) -> Dict[str, Any]:
    s = 100
    reasons = []
    n = report.get("navigator", {})
    keys = report.get("keys", {})
    env = report.get("env", {})

    if n.get("webdriver") is True:
        s -= 50; reasons.append("navigator.webdriver = true")
    if n.get("webdriverDesc") is True:
        s -= 15; reasons.append("navigator.webdriver descriptor present")
    if n.get("headlessUA"):
        s -= 30; reasons.append("HeadlessChrome in UA")
    if (n.get("pluginsLength") == 0) or (n.get("mimeTypesLength") == 0):
        s -= 15; reasons.append("plugins/mimeTypes empty")
    if keys.get("cdc"):
        s -= 20; reasons.append(f"cdc keys: {len(keys['cdc'])}")
    if env.get("hasChromeObj") and (env.get("hasChromeRuntime") is False):
        s -= 5; reasons.append("chrome present but runtime missing")
    qn = report.get("perms", {}).get("queryNative")
    if qn and qn.get("includesNative") is False:
        s -= 10; reasons.append("permissions.query not native")

    return { "score": max(0, s), "issues": reasons }


async def audit_async(args) -> int:
    try:
        from playwright.async_api import async_playwright
    except Exception:
        _print("Playwright is required: pip install playwright && playwright install")
        return 2

    endpoint = args.cdp_endpoint
    launch_mode = endpoint is None
    result: Dict[str, Any] = { "launch": {}, "js": {}, "summary": {} }

    async with async_playwright() as p:
        browser = None
        context = None
        page = None
        try:
            if launch_mode:
                # Launch headed browser for auditing a specific URL (optional)
                browser = await p.chromium.launch(headless=False)
                context = await browser.new_context()
                page = await context.new_page()
                if args.url:
                    await page.goto(args.url, wait_until="domcontentloaded")
                result["launch"] = { "mode": "launch", "headless": False }
            else:
                # Connect to an existing headed browser
                browser = await p.chromium.connect_over_cdp(endpoint)
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                pages = list(context.pages)
                if pages:
                    page = pages[0]
                elif args.url and not args.attach_only:
                    page = await context.new_page()
                    await page.goto(args.url, wait_until="domcontentloaded")
                else:
                    raise RuntimeError("No page to audit: open a tab or provide --url without --attach-only")
                result["launch"] = { "mode": "connect", "endpoint": endpoint }

            # Perform DOM-only audit
            js = await page.evaluate(JS_AUDIT)
            result["js"] = js
            result["summary"] = score(js)

            # Output
            if args.out_json:
                Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            # Human summary
            _print(f"Score: {result['summary']['score']}/100")
            if result["summary"]["issues"]:
                for issue in result["summary"]["issues"]:
                    _print(f" - {issue}")
            else:
                _print("No obvious automation signals detected.")

            if args.verbose:
                _print(json.dumps(js, ensure_ascii=False, indent=2))

            return 0
        finally:
            if launch_mode and browser:
                with contextlib.suppress(Exception):
                    await browser.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Headed DOM-only fingerprint audit (no page modifications)")
    ap.add_argument("--cdp-endpoint", type=str, default=None, help="Connect to existing browser CDP endpoint (e.g. http://127.0.0.1:9222)")
    ap.add_argument("--url", type=str, default=None, help="URL to navigate if needed (launch/connect modes)")
    ap.add_argument("--out-json", type=str, default=None, help="Write full JSON report to this path")
    ap.add_argument("--attach-only", action="store_true", help="Do not create a new page when connecting (error if none)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    return asyncio.run(audit_async(args))


if __name__ == "__main__":
    sys.exit(main())

