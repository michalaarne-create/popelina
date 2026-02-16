#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
stealth_audit.py
Audit Playwright + Opera GX stealthiness while ONLY reading DOM.
Windows 10 oriented. Requires: pip install playwright; playwright install
Usage (PowerShell/CMD):
  python stealth_audit.py --exe "C:\Users\YOU\AppData\Local\Programs\Opera GX\launcher.exe" --profile "C:\Users\YOU\AppData\Roaming\Opera Software\Opera GX Stable" --url "https://example.com" --keep-extensions
Notes:
- Close Opera GX before running (cannot reuse same profile concurrently).
- This script does not click or modify the page; it navigates and reads.
"""
import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

def risk(label, ok, msg_ok, msg_bad):
    return {
        "label": label,
        "status": "PASS" if ok else "WARN",
        "details": msg_ok if ok else msg_bad
    }

def main():
    ap = argparse.ArgumentParser(description="Audit stealthiness for Playwright + Opera GX (DOM-read only).")
    ap.add_argument("--exe", required=True, help="Path to Opera GX launcher.exe")
    ap.add_argument("--profile", required=True, help="Path to Opera GX profile directory (e.g. ...\\Opera GX Stable)")
    ap.add_argument("--url", required=True, help="Target URL to load (must be legal/allowed).")
    ap.add_argument("--keep-extensions", action="store_true", help="Keep profile extensions enabled (more realistic but may inject content)")
    ap.add_argument("--headless", action="store_true", help="Run headless (NOT recommended for stealth)")
    args = ap.parse_args()

    exe = Path(args.exe)
    prof = Path(args.profile)
    if not exe.exists():
        print(f"[ERROR] Opera executable not found: {exe}", file=sys.stderr)
        sys.exit(2)
    if not prof.exists():
        print(f"[ERROR] Profile path not found: {prof}", file=sys.stderr)
        sys.exit(2)

    ignore_args = ["--enable-automation"]
    if args.keep_extensions:
        ignore_args.append("--disable-extensions")

    # Extra Chromium args to minimize automation signals
    chromium_args = [
        "--disable-blink-features=AutomationControlled",
        "--disable-infobars",
        "--no-sandbox"
    ]

    report = {"launch": {}, "js_fingerprint": {}, "checks": []}

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(prof),
            executable_path=str(exe),
            headless=args.headless,
            args=chromium_args,
            ignore_default_args=ignore_args,
            viewport={"width": 1366, "height": 768},
            locale="pl-PL",
            timezone_id="Europe/Warsaw",
            java_script_enabled=True,
        )
        report["launch"] = {
            "executable_path": str(exe),
            "profile_dir": str(prof),
            "headless": args.headless,
            "ignored_default_args": ignore_args,
            "extra_args": chromium_args,
        }

        page = context.pages[0] if context.pages else context.new_page()
        page.set_extra_http_headers({"Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8"})
        page.goto(args.url, wait_until="networkidle")

        # Evaluate various fingerprint vectors without modifying the page.
        js = r"""
(() => {
  const out = {};
  try { out.webdriver = navigator.webdriver; } catch(e) { out.webdriver = 'error'; }

  out.ua = navigator.userAgent;
  out.uaHeadless = /HeadlessChrome/i.test(navigator.userAgent);
  out.languages = (navigator.languages || null);
  out.language = (navigator.language || null);
  out.pluginsLength = (navigator.plugins ? navigator.plugins.length : null);
  out.platform = navigator.platform || null;
  out.hardwareConcurrency = navigator.hardwareConcurrency || null;
  out.deviceMemory = (typeof navigator.deviceMemory !== 'undefined') ? navigator.deviceMemory : null;
  out.maxTouchPoints = (typeof navigator.maxTouchPoints !== 'undefined') ? navigator.maxTouchPoints : null;
  out.timezone = (() => { try { return Intl.DateTimeFormat().resolvedOptions().timeZone; } catch(e) { return null; }})();
  out.hasChromeObj = (typeof window.chrome !== 'undefined');

  try {
    out.uaData = navigator.userAgentData ? {
      mobile: navigator.userAgentData.mobile,
      brands: navigator.userAgentData.brands
    } : null;
  } catch(e) { out.uaData = 'error'; }

  // WebGL vendor/renderer (may be null if disabled)
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (gl) {
      const dbg = gl.getExtension('WEBGL_debug_renderer_info');
      out.webglVendor = dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR);
      out.webglRenderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    } else {
      out.webglVendor = null; out.webglRenderer = null;
    }
  } catch(e) { out.webglError = String(e); }

  // Property descriptor for navigator.webdriver (advanced detectors sometimes check descriptor presence)
  try {
    const proto = Object.getPrototypeOf(navigator);
    const desc = Object.getOwnPropertyDescriptor(proto, "webdriver");
    out.webdriverDescriptorExists = !!desc;
  } catch(e) { out.webdriverDescriptorExists = 'error'; }

  // Screen metrics
  out.screen = {
    width: screen.width, height: screen.height,
    availWidth: screen.availWidth, availHeight: screen.availHeight,
    colorDepth: screen.colorDepth,
    pixelRatio: window.devicePixelRatio,
    innerWidth: window.innerWidth, innerHeight: window.innerHeight
  };

  // Notifications permission (some headless bugs show inconsistent states)
  try { out.notificationPermission = (typeof Notification !== "undefined") ? Notification.permission : null; } catch(e) { out.notificationPermission = 'error'; }

  return out;
})();
"""
        fp = page.evaluate(js)
        report["js_fingerprint"] = fp

        # Heuristic checks
        checks = []
        checks.append(risk("Headless UA", not fp.get("uaHeadless", False),
                           "UA does not contain HeadlessChrome",
                           "UA indicates HeadlessChrome -> switch to headed mode"))

        checks.append(risk("navigator.webdriver falsy", not bool(fp.get("webdriver", False)),
                           "navigator.webdriver is falsy/undefined",
                           "navigator.webdriver is true -> automation detectable"))

        # If descriptor exists, some detectors may still flag; many real browsers do not have it explicitly.
        desc_exists = fp.get("webdriverDescriptorExists", False)
        checks.append(risk("webdriver descriptor", (desc_exists is False) or (desc_exists == "error"),
                           "No explicit webdriver descriptor on navigator proto",
                           "webdriver descriptor present -> possible automation signal"))

        # Languages present and plausible
        langs = fp.get("languages") or []
        checks.append(risk("languages present", bool(langs) and isinstance(langs, list),
                           f"navigator.languages looks normal: {langs}",
                           "navigator.languages empty/missing -> suspicious"))

        # Plugins length (0 can be legit on many modern setups; warn only if null/missing)
        pl = fp.get("pluginsLength")
        checks.append(risk("plugins enumerable", pl is not None,
                           f"navigator.plugins length={pl}",
                           "navigator.plugins missing -> suspicious in Chromium"))

        # Timezone check (you can change 'Europe/Warsaw' if different)
        tz_ok = (fp.get("timezone") in (None, "Europe/Warsaw"))  # None acceptable if blocked
        checks.append(risk("timezone", tz_ok,
                           f"Timezone OK: {fp.get('timezone')}",
                           f"Unexpected timezone: {fp.get('timezone')}"))

        # WebGL renderer: SwiftShader indicates software rendering -> can be suspicious
        renderer = (fp.get("webglRenderer") or "").lower()
        swiftshader = "swiftshader" in renderer
        checks.append(risk("webgl renderer", not swiftshader,
                           f"WebGL renderer: {fp.get('webglVendor')} / {fp.get('webglRenderer')}",
                           f"Software WebGL renderer (SwiftShader) -> may be flagged"))

        # has window.chrome object in Chromium-family
        has_chrome_obj = bool(fp.get("hasChromeObj"))
        checks.append(risk("window.chrome exists", has_chrome_obj,
                           "window.chrome present (Chromium-like)",
                           "window.chrome missing -> can be suspicious in Chromium-family"))

        # Notification permission sanity
        np = fp.get("notificationPermission")
        checks.append(risk("Notification permission", np in ("default", "granted", "denied", None),
                           f"Notification.permission={np} looks normal",
                           f"Unusual Notification.permission value: {np}"))

        report["checks"] = checks

        # Simple grade: % of PASS
        total = len(checks)
        passed = sum(1 for c in checks if c["status"] == "PASS")
        report["grade"] = {
            "passed": passed,
            "total": total,
            "percent": round(100.0 * passed / max(1, total), 1)
        }

        print(json.dumps(report, ensure_ascii=False, indent=2))

        context.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:  # brak argumentów => testowy run
        sys.argv += [
            "--exe", "C:/Users/User/AppData/Local/Programs/Opera GX/opera.exe",
            "--profile", "C:/Users/User/AppData/Roaming/Opera Software/Opera GX Stable",
            "--url", "https://example.com"
        ]
    main()
