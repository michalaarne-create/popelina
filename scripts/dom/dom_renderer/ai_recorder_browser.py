import asyncio  
import contextlib  
import os  
import socket  
import subprocess  
import sys  
import time  
from pathlib import Path  
from typing import Optional  
import re as _re  
import re as _re  
try:  
    import urllib.request as _urlreq  
except Exception:  
    _urlreq = None  
try:  
    import numpy as _np  
except Exception:  
    _np = None  
try:  
    import mss as _mss  
    import mss.tools as _msstools  
except Exception:  
    _mss = None  
_PaddleOCR = None  
try:  
    from PIL import Image as _PILImage  # type: ignore  
except Exception:  
    _PILImage = None  
  
try:  
    if __package__:  
        from .ai_recorder_common import ensure_dir, log, PageTrack3r, norm_text, domain_from_url, fuzzy_ratio  
    else:  
        from ai_recorder_common import ensure_dir, log, PageTrack3r, norm_text, domain_from_url, fuzzy_ratio  
except Exception:  
    from ai_recorder_common import ensure_dir, log, PageTrack3r, norm_text, domain_from_url, fuzzy_ratio  # type: ignore  

ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_SCREEN_DIR = ROOT_PATH / "data" / "screen"
OCR_DEBUG_DIR = DATA_SCREEN_DIR / "debug"

def _ocr_debug_dir() -> Path:
    try:
        ensure_dir(OCR_DEBUG_DIR)
    except Exception:
        pass
    return OCR_DEBUG_DIR
  
  
_HEADER_PROFILE = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.5",
    "Cache-Control": "max-age=0",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "0",
}

_STEALTH_INIT_JS_TEMPLATE = r"""
(() => {
  if (window.__aiRecorderProfileApplied) {
    return;
  }
  window.__aiRecorderProfileApplied = true;

  const profile = {
    userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.86 Safari/537.36",
    fullVersion: "131.0.6778.86",
    platform: "Win32",
    languages: ["en-US", "en"],
    timeZone: "Europe/Warsaw",
    timezoneOffset: -60,
    vendor: "Google Inc.",
    product: "Gecko",
    productSub: "20030107",
    buildId: "20181001000000",
    hardwareConcurrency: 4,
    deviceMemory: 8,
    screen: {
      width: 1920,
      height: 1080,
      availWidth: 1920,
      availHeight: 1080,
      colorDepth: 24,
      pixelDepth: 24
    },
    webgl: {
      vendor: "Google Inc.",
      renderer: "ANGLE (Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0)",
      extensions: [
        "ANGLE_instanced_arrays",
        "EXT_blend_minmax",
        "EXT_color_buffer_half_float",
        "EXT_disjoint_timer_query",
        "EXT_float_blend",
        "EXT_texture_filter_anisotropic",
        "EXT_sRGB",
        "KHR_parallel_shader_compile",
        "OES_element_index_uint",
        "OES_fbo_render_mipmap",
        "OES_standard_derivatives",
        "OES_texture_float",
        "OES_texture_float_linear",
        "OES_texture_half_float",
        "OES_texture_half_float_linear",
        "OES_vertex_array_object",
        "WEBGL_color_buffer_float",
        "WEBGL_compressed_texture_s3tc",
        "WEBGL_debug_renderer_info",
        "WEBGL_debug_shaders",
        "WEBGL_depth_texture",
        "WEBGL_draw_buffers",
        "WEBGL_lose_context"
      ]
    },
    canvasSeed: [3, 5, 7, 11, 13, 17, 19, 23],
    canvasFallback: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAMCAYAAABbayygAAAAhUlEQVR4nM2RwQ2AMAxE33QAV3AAV3AAV3AAF8BAV1AcMFIq0DzGJM1kLqT7jHAAzbluFhvgQ9lELyhlHLLJoCq114F8CbnMD4HzyBbs6k8ZZrACG4AQwJNG0AxQAnCM1j7rwhhtjfiPgk41IrTnfjp+1hhZCvGSqtEtFPiMGDzfrxGX2YeliYg+6TSGT+bVIRVKiWOgAAAABJRU5ErkJggg==",
    audio: {
      sampleRate: 48000
    },
    mediaDevices: [
      {
        deviceId: "default",
        kind: "audiooutput",
        label: "Speakers (High Definition Audio)",
        groupId: "audio-output-1"
      }
    ],
    plugins: [
      {
        name: "Chrome PDF Viewer",
        filename: "internal-pdf-viewer",
        description: "Portable Document Format",
        mime: {
          type: "application/pdf",
          suffixes: "pdf",
          description: "Portable Document Format"
        }
      }
    ]
  };

  const patchResult = (fn) => {
    try {
      fn();
    } catch (err) {
      console.debug("stealth patch error", err);
    }
  };

  const overrideGetter = (obj, prop, value) => {
    if (!obj) return;
    try {
      const existing = Object.getOwnPropertyDescriptor(obj, prop);
      if (existing && !existing.configurable) {
        return;
      }
      Object.defineProperty(obj, prop, {
        configurable: true,
        get: () => (typeof value === "function" ? value() : value)
      });
    } catch (err) {
      console.debug("overrideGetter err", prop, err);
    }
  };

  const overrideValue = (obj, prop, value) => {
    if (!obj) return;
    try {
      const existing = Object.getOwnPropertyDescriptor(obj, prop);
      if (existing && !existing.configurable) {
        return;
      }
      Object.defineProperty(obj, prop, {
        configurable: true,
        writable: false,
        value
      });
    } catch (err) {
      console.debug("overrideValue err", prop, err);
    }
  };

  // Canvas/WebGL spoof disabled – allow native fingerprint
  /*
  patchResult(() => {
    const navProto = Object.getPrototypeOf(navigator) || navigator;
    overrideGetter(navProto, "platform", () => profile.platform);
    overrideGetter(navProto, "userAgent", () => profile.userAgent);
    overrideGetter(navProto, "appVersion", () => profile.userAgent.replace("Mozilla/", ""));
    overrideGetter(navProto, "appName", () => "Netscape");
    overrideGetter(navProto, "languages", () => profile.languages.slice());
    overrideGetter(navProto, "language", () => profile.languages[0]);
    overrideGetter(navProto, "hardwareConcurrency", () => profile.hardwareConcurrency);
    overrideGetter(navProto, "deviceMemory", () => profile.deviceMemory);
    overrideGetter(navProto, "vendor", () => profile.vendor);
    overrideGetter(navProto, "vendorSub", () => "");
    overrideGetter(navProto, "product", () => profile.product);
    overrideGetter(navProto, "productSub", () => profile.productSub);
    overrideGetter(navProto, "buildID", () => profile.buildId);
    overrideGetter(navProto, "maxTouchPoints", () => 0);
    overrideGetter(navProto, "cookieEnabled", () => true);
    overrideGetter(navProto, "doNotTrack", () => "unspecified");
    overrideValue(navProto, "webdriver", undefined);
    overrideValue(navProto, "javaEnabled", function javaEnabled() {
      return false;
    });
    // navigator.connection stays native (no override) to avoid obvious throttling fingerprints

    if ("userAgentData" in navigator) {
      const brands = [
        { brand: "Not/A)Brand", version: "8" },
        { brand: "Chromium", version: "131" },
        { brand: "Google Chrome", version: "131" }
      ];
      const uaData = {
        brands: brands.map((b) => ({ ...b })),
        mobile: false,
        platform: "Windows",
        getHighEntropyValues: (hints) =>
          Promise.resolve({
            architecture: "x86",
            bitness: "64",
            model: "",
            platform: "Windows",
            platformVersion: "15.0.0",
            uaFullVersion: profile.fullVersion,
            fullVersionList: brands.map((b) => ({ ...b }))
          }),
        toJSON() {
          return { brands: this.brands, mobile: this.mobile, platform: this.platform };
        }
      };
      overrideGetter(navProto, "userAgentData", () => uaData);
    }

    const pluginEntries = profile.plugins.map((plugin) => {
      const mime = {
        type: plugin.mime.type,
        suffixes: plugin.mime.suffixes,
        description: plugin.mime.description,
        enabledPlugin: null
      };
      const pluginObj = {
        description: plugin.description,
        filename: plugin.filename,
        length: 1,
        name: plugin.name,
        0: mime,
        item: (idx) => (idx === 0 ? mime : null),
        namedItem: (name) => (name === plugin.mime.type ? mime : null)
      };
      mime.enabledPlugin = pluginObj;
      return pluginObj;
    });
    const makeArrayLike = (items, proto) => {
      const arr = items.slice();
      arr.length = items.length;
      arr.item = (idx) => arr[idx] || null;
      arr.namedItem = (name) => arr.find((entry) => entry && (entry.name === name || entry.type === name)) || null;
      arr.refresh = () => undefined;
      if (proto) {
        try {
          Object.setPrototypeOf(arr, proto.prototype);
        } catch (err) {
          console.debug("proto set err", err);
        }
      }
      return arr;
    };
    const pluginArray = makeArrayLike(pluginEntries, window.PluginArray);
    overrideGetter(navProto, "plugins", () => pluginArray);
    const mimeEntries = profile.plugins.map((plugin, idx) => {
      const mime = {
        type: plugin.mime.type,
        suffixes: plugin.mime.suffixes,
        description: plugin.mime.description,
        enabledPlugin: pluginEntries[idx]
      };
      return mime;
    });
    const mimeArray = makeArrayLike(mimeEntries, window.MimeTypeArray);
    overrideGetter(navProto, "mimeTypes", () => mimeArray);

    if (navigator.mediaDevices && typeof navigator.mediaDevices === "object") {
      const fakeDevices = profile.mediaDevices.map((d) => ({ ...d }));
      const devicesClone = () => fakeDevices.map((device) => ({ ...device }));
      overrideValue(navigator.mediaDevices, "enumerateDevices", () => Promise.resolve(devicesClone()));
    }

    if (navigator.getBattery) {
      try {
        delete navigator.getBattery;
      } catch (err) {
        overrideValue(navigator, "getBattery", undefined);
      }
    }

    if (navigator.permissions && navigator.permissions.query) {
      const originalQuery = navigator.permissions.query.bind(navigator.permissions);
      overrideValue(navigator.permissions, "query", (parameters) => {
        const targetName = parameters && parameters.name;
        const forcePrompt = new Set([
          "notifications",
          "camera",
          "microphone",
          "geolocation",
          "accelerometer",
          "gyroscope",
          "magnetometer"
        ]);
        if (!targetName || !forcePrompt.has(targetName)) {
          try {
            return originalQuery(parameters);
          } catch (err) {
            return Promise.resolve({ state: "prompt" });
          }
        }
        return Promise.resolve({
          state: "prompt",
          onchange: null,
          addEventListener() {},
          removeEventListener() {},
          dispatchEvent() {
            return true;
          }
        });
      });
    }
  });
  */

  patchResult(() => {
    const screenProto = Object.getPrototypeOf(screen) || screen;
    overrideGetter(screenProto, "width", () => profile.screen.width);
    overrideGetter(screenProto, "height", () => profile.screen.height);
    overrideGetter(screenProto, "availWidth", () => profile.screen.availWidth);
    overrideGetter(screenProto, "availHeight", () => profile.screen.availHeight);
    overrideGetter(screenProto, "colorDepth", () => profile.screen.colorDepth);
    overrideGetter(screenProto, "pixelDepth", () => profile.screen.pixelDepth);
    overrideGetter(screenProto, "availTop", () => 0);
    overrideGetter(screenProto, "availLeft", () => 0);
    if (screen.orientation) {
      overrideGetter(screen.orientation, "angle", () => 0);
      overrideGetter(screen.orientation, "type", () => "landscape-primary");
    }
  });

  patchResult(() => {
    if (Intl && Intl.DateTimeFormat) {
      const originalResolved = Intl.DateTimeFormat.prototype.resolvedOptions;
      overrideValue(Intl.DateTimeFormat.prototype, "resolvedOptions", function resolvedOptions(...args) {
        const result = originalResolved.apply(this, args);
        result.timeZone = profile.timeZone;
        result.locale = profile.languages[0];
        return result;
      });
    }
    overrideValue(Date.prototype, "getTimezoneOffset", function getTimezoneOffset() {
      return profile.timezoneOffset;
    });
  });

  patchResult(() => {
    const patchGL = (ctx) => {
      if (!ctx || ctx.__aiRecorderWebGl) {
        return;
      }
      ctx.__aiRecorderWebGl = true;
      const originalGetParameter = ctx.getParameter.bind(ctx);
      overrideValue(ctx, "getParameter", function getParameter(parameter) {
        if (parameter === 37445) {
          return profile.webgl.vendor;
        }
        if (parameter === 37446) {
          return profile.webgl.renderer;
        }
        if (parameter === 7936) {
          return profile.webgl.vendor;
        }
        if (parameter === 7937) {
          return profile.webgl.renderer;
        }
        return originalGetParameter(parameter);
      });
      if (ctx.getSupportedExtensions) {
        const originalExtensions = ctx.getSupportedExtensions.bind(ctx);
        overrideValue(ctx, "getSupportedExtensions", () => {
          const real = originalExtensions() || [];
          const merged = new Set(real.concat(profile.webgl.extensions));
          return Array.from(merged);
        });
      }
    };

    if (HTMLCanvasElement && HTMLCanvasElement.prototype && HTMLCanvasElement.prototype.getContext) {
      const originalGetContext = HTMLCanvasElement.prototype.getContext;
      overrideValue(HTMLCanvasElement.prototype, "getContext", function getContext(type, attrs) {
        const context = originalGetContext.call(this, type, attrs);
        if (context && /(webgl|experimental-webgl|webgl2)/i.test(type)) {
          patchGL(context);
        }
        return context;
      });
    }
  });

  patchResult(() => {
    const audioCtx = window.AudioContext || window.webkitAudioContext;
    if (audioCtx && audioCtx.prototype) {
      // Leave native sampleRate to avoid suspicious overrides
      // overrideGetter(audioCtx.prototype, "sampleRate", () => profile.audio.sampleRate);
    }
    if ("Notification" in window) {
      overrideGetter(Notification, "permission", () => "default");
    }
    const mediaProto = HTMLMediaElement && HTMLMediaElement.prototype;
    if (mediaProto && mediaProto.canPlayType) {
      const originalCanPlay = mediaProto.canPlayType;
      overrideValue(mediaProto, "canPlayType", function canPlayType(type) {
        if (type && /mp4|avc1|h264/i.test(type)) {
          return "probably";
        }
        return originalCanPlay.call(this, type);
      });
    }
    ["DeviceMotionEvent", "DeviceOrientationEvent", "AbsoluteOrientationSensor", "Accelerometer", "Gyroscope", "Magnetometer"].forEach(
      (api) => {
        if (api in window) {
          try {
            window[api] = undefined;
          } catch (err) {
            console.debug("sensor override err", api, err);
          }
        }
      }
    );
  });
})();
"""

class LiveRecorderBrowserMixin:  
    _STEALTH_INIT_JS = _STEALTH_INIT_JS_TEMPLATE
  
    async def _install_activity_listeners_init_script(self, ctx) -> None:
        # Keep activity listeners disabled in DOM-only mode;
        # this method intentionally does nothing.
        return

    def _build_stealth_script(self) -> str:
        stealth_mode = getattr(self, "stealth_mode", "off")
        if stealth_mode in ("off", "none", "native"):
            self._STEALTH_INIT_JS = ""
            return ""
        return self._STEALTH_INIT_JS or ""

    async def _install_stealth_init_script(self, ctx) -> None:
        try:
            script = self._build_stealth_script()
            await ctx.add_init_script(script)
        except Exception:
            pass
  
    async def _inject_activity_observer(self, page, page_id: str) -> bool:  
        # DOM-only: no injection  
        return True  
  
    async def _apply_viewport_mode(self, label: str = "", page: Optional[object] = None):
        # DOM-only: no emulation
        return

    def _close_existing_browsers(self) -> None:
        if getattr(self, "connect_existing", False):
            return
        if not sys.platform.startswith("win"):
            return
        kill_list = [
            "chrome.exe",
            "chrome_proxy.exe",
            "chrome_crashpad_handler.exe",
        ]
        for proc in kill_list:
            try:
                subprocess.run(
                    ["taskkill", "/IM", proc, "/F", "/T"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception:
                pass

    async def _launch_chrome_cdp(self) -> str:
        exe = getattr(self, 'chrome_exe', None)
        if exe and os.path.isfile(exe):
            browser_exe = exe
        else:
            if sys.platform.startswith('win'):  
                candidates = [  
                    r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",  
                    r"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",  
                    r"C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",  
                    r"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",  
                ]  
                local_app = os.environ.get("LOCALAPPDATA")  
                if local_app:  
                    candidates.extend([  
                        os.path.join(local_app, "Programs", "Opera GX", "launcher.exe"),  
                        os.path.join(local_app, "Programs", "Opera GX", "opera.exe"),  
                        os.path.join(local_app, "Programs", "Opera", "launcher.exe"),  
                        os.path.join(local_app, "Programs", "Opera", "opera.exe"),  
                    ])  
            elif sys.platform == 'darwin':  
                candidates = [  
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  
                    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",  
                ]  
            else:  
                candidates = [  
                    "/usr/bin/google-chrome",  
                    "/usr/bin/google-chrome-stable",  
                    "/usr/bin/chromium",  
                    "/usr/bin/chromium-browser",  
                ]  
            browser_exe = next((p for p in candidates if os.path.isfile(p)), None)  
            if not browser_exe:  
                raise RuntimeError("Chrome/Edge executable not found")  
  
        ensure_dir(self.user_data_dir)
        active_file = self.user_data_dir / "DevToolsActivePort"
        # Try reusing existing endpoint first regardless of flag
        existing_endpoint = await self._reuse_existing_chrome(active_file)
        if existing_endpoint:
            log(f"Reusing existing browser via DevToolsActivePort: {existing_endpoint}", "INFO")
            return existing_endpoint

        # Only close browsers when we explicitly don't want to attach
        if not getattr(self, "connect_existing", False):
            self._close_existing_browsers()
        exe_name = os.path.basename(browser_exe).lower() if browser_exe else ""
        is_opera = "opera" in exe_name
        self._is_opera = is_opera
        existing_endpoint = await self._reuse_existing_chrome(active_file)
        if existing_endpoint:
            return existing_endpoint

        # Remove stale DevToolsActivePort so we don't reuse an old, dead port.
        try:
            if active_file.exists():
                active_file.unlink()
        except Exception:
            pass
  
        args = [  
            browser_exe,  
            f"--user-data-dir={self.user_data_dir}",  
            "--remote-debugging-port=0",  
            "--no-first-run",  
            "--no-default-browser-check",  
            "--start-maximized",  
            "--lang=en-US",
        ]  
        stealth_flags = [  
            "--disable-blink-features=AutomationControlled",  
            "--disable-infobars",  
            "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",  
            "--disable-quic",  
            "--dns-over-https-mode=off",  
            "--disable-features=DnsOverHttps,UseDnsHttpsSvcb,UseDnsHttpsSvcbAlpn,AsyncDns,BuiltInDnsClient",  
            "--disable-dns-prefetch",  
        ]  
        if is_opera:  
            stealth_flags = [f for f in stealth_flags if f != "--disable-blink-features=AutomationControlled"]  
        args.extend(stealth_flags)  
        profile_dir = getattr(self, "profile_directory", None)
        if profile_dir and not is_opera:
            args.append(f"--profile-directory={profile_dir}")
        elif is_opera:
            args.append("--profile-directory=Default")
        # Optional proxy (via RECORDER_PROXY env or CLI). Forces DNS via proxy using host resolver rules.  
        proxy = getattr(self, 'proxy_server', None)  
        if proxy:  
            args.extend([  
                f"--proxy-server={proxy}",  
                "--proxy-bypass-list=<-loopback>",  
                # Ensure Chrome does not resolve hostnames locally when proxy in use  
                "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE localhost",  
            ])  
        if getattr(self, "user_agent", None):
            args.append(f"--user-agent={self.user_agent}")
        env = os.environ.copy()  

        max_wait = 90 if not is_opera else 120
        launch_attempts = 1
        last_error = None
        endpoint_port = None

        for attempt in range(launch_attempts):
            self.chrome_process = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
            )

            for remaining in range(5, 0, -1):
                log(f"[Chrome] Czekam {remaining}s na inicjalizację profilu...", "INFO")
                await asyncio.sleep(1.0)

            start_t = time.time()
            launch_epoch = start_t
            success = False
            while time.time() - start_t < max_wait:
                if active_file.exists():
                    try:
                        # Skip stale file from previous runs and wait for a fresh CDP listener.
                        if active_file.stat().st_mtime < launch_epoch:
                            await asyncio.sleep(0.05)
                            continue
                        txt = active_file.read_text(encoding="utf-8").strip().splitlines()
                        if txt and txt[0].strip().isdigit():
                            port_candidate = int(txt[0].strip())
                            endpoint_candidate = f"http://127.0.0.1:{port_candidate}"
                            if await self._probe_cdp_endpoint(endpoint_candidate, attempts=3):
                                endpoint_port = port_candidate
                                success = True
                                break
                    except Exception:
                        pass
                if self.chrome_process.poll() is not None:
                    last_error = RuntimeError(f"Chrome exited early ({self.chrome_process.returncode})")
                    break
                await asyncio.sleep(0.05)

            if success:
                break

            await asyncio.sleep(1.0)
        else:
            log("Chrome nie uruchomił się automatycznie. Sprawdzam, czy istnieje już ręcznie otwarte okno...", "WARNING")
            manual_endpoint = await self._wait_for_manual_chrome(
                active_file, timeout=max_wait if max_wait > 60 else 90
            )
            if manual_endpoint:
                log(f"Reusing manual Chrome session: {manual_endpoint}", "INFO")
                return manual_endpoint
            raise last_error or RuntimeError(
                "DevToolsActivePort not found. Upewnij się, że wszystkie okna Chrome/Opera są zamknięte i spróbuj ponownie."
            )
  
        if endpoint_port is None:
            txt = active_file.read_text(encoding="utf-8").strip().splitlines()
            if txt and txt[0].strip().isdigit():
                endpoint_port = int(txt[0].strip())
        if endpoint_port is None:
            raise RuntimeError("DevToolsActivePort is empty or invalid after Chrome launch")

        endpoint = f"http://127.0.0.1:{endpoint_port}"  
        log(f"CDP endpoint: {endpoint}", "SUCCESS")  
        return endpoint  
  
    async def _connect_cdp(self):  
        from playwright.async_api import async_playwright  
  
        if getattr(self, 'pw', None) is None:  
            self.pw = await async_playwright().start()  
  
        endpoint = getattr(self, 'cdp_endpoint', None)  
        if not endpoint:  
            endpoint = await self._launch_chrome_cdp()  
  
        # Optional probe to avoid silent hangs  
        try:  
            if _urlreq and endpoint and endpoint.startswith("http"):  
                ver_url = endpoint.rstrip("/") + "/json/version"  
                start = time.time()  
                ok = False  
                while time.time() - start < 3.0:  
                    try:  
                        with _urlreq.urlopen(ver_url, timeout=1) as r:  
                            if getattr(r, "status", 200) == 200:  
                                ok = True  
                                break  
                    except Exception:  
                        await asyncio.sleep(0.2)  
                if not ok:  
                    raise RuntimeError(f"CDP probe failed at {ver_url}")  
        except Exception as e:  
            log(f"CDP probe warning: {e}", "WARNING")  
  
        # Connect with bounded timeout (więcej prób dla Opery)
        exe_name = os.path.basename(getattr(self, "chrome_exe", "") or "").lower()
        is_opera_connect = "opera" in exe_name or getattr(self, "_is_opera", False)

        connect_attempts = 5 if is_opera_connect else 3
        last_err = None
        for attempt in range(connect_attempts):
            try:
                self.browser = await self.pw.chromium.connect_over_cdp(endpoint, timeout=5000)
                break
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.5 * (attempt + 1))
        else:
            raise last_err
  
        # choose context and register across ALL contexts  
        contexts = list(self.browser.contexts)  
        if contexts:  
            # pick a primary for legacy fields, but we will register all  
            self.context = max(contexts, key=lambda c: len(c.pages))  
        else:  
            self.context = await self.browser.new_context()  
            contexts = [self.context]  
  
        # add stealth init to every context; then optional activity listeners  
        for ctx in contexts:  
            if getattr(self, "stealth_mode", "off") not in ("off", "none", "native"):
                with contextlib.suppress(Exception):
                    await ctx.set_extra_http_headers(dict(_HEADER_PROFILE))
                with contextlib.suppress(Exception):  
                    await self._install_stealth_init_script(ctx)  
                if not getattr(self, 'dom_only', False):  
                    with contextlib.suppress(Exception):  
                        await self._install_activity_listeners_init_script(ctx)  
            with contextlib.suppress(Exception):  
                ctx.on("page", lambda p, _ctx=ctx: asyncio.create_task(self._handle_new_page(p)))  
  
        # Register all existing pages; pick a primary  
        pages = []  
        for ctx in contexts:  
            with contextlib.suppress(Exception):  
                pages.extend(list(ctx.pages))  
        if pages:  
            self.page = pages[0]  
            for p in pages:  
                if getattr(self, "stealth_mode", "off") not in ("off", "none", "native"):
                    with contextlib.suppress(Exception):  
                        await p.add_init_script(self._STEALTH_INIT_JS)  
                    with contextlib.suppress(Exception):  
                        await p.evaluate(self._STEALTH_INIT_JS)  
                with contextlib.suppress(Exception):  
                    await self._handle_new_page(p)  
            # Choose best active once after initial registration  
            with contextlib.suppress(Exception):  
                await self._reselect_active_page()  
            log("CDP connected + attached to existing page(s)", "SUCCESS")  
        else:  
            # brief wait for user to open one  
            wait_start = time.time()  
            while time.time() - wait_start < 5:  
                pages = list(self.context.pages)  
                if pages:  
                    self.page = pages[0]  
                    for p in pages:  
                        if getattr(self, "stealth_mode", "off") not in ("off", "none", "native"):
                            with contextlib.suppress(Exception):  
                                await p.add_init_script(self._STEALTH_INIT_JS)  
                            with contextlib.suppress(Exception):  
                                await p.evaluate(self._STEALTH_INIT_JS)  
                        with contextlib.suppress(Exception):  
                            await self._handle_new_page(p)  
                    with contextlib.suppress(Exception):  
                        await self._reselect_active_page()  
                    log("CDP connected + attached to existing page(s)", "SUCCESS")  
                    break  
                await asyncio.sleep(0.2)  
            if not pages:  
                self.page = await self.context.new_page()  
                if getattr(self, "stealth_mode", "off") not in ("off", "none", "native"):
                    with contextlib.suppress(Exception):  
                        await self.page.add_init_script(self._STEALTH_INIT_JS)  
                await self._handle_new_page(self.page)  
                log("CDP connected + fresh page created", "SUCCESS")  

    async def _reuse_existing_chrome(self, active_file):
        """
        Detect and reuse an already running browser that shares the same
        user-data-dir instead of spawning a duplicate instance.
        """
        endpoint = self._read_devtools_endpoint(active_file)
        if not endpoint:
            return None
        if await self._probe_cdp_endpoint(endpoint):
            log(f"Detected existing browser at {endpoint} - reusing DevTools endpoint", "INFO")
            return endpoint
        return None

    async def _wait_for_manual_chrome(self, active_file, timeout=90):
        start = time.time()
        while time.time() - start < timeout:
            endpoint = await self._reuse_existing_chrome(active_file)
            if endpoint:
                return endpoint
            await asyncio.sleep(0.5)
        return None

    def _read_devtools_endpoint(self, active_file):
        try:
            txt = active_file.read_text(encoding="utf-8").strip().splitlines()
        except FileNotFoundError:
            return None
        except Exception:
            return None
        if not txt:
            return None
        port_line = txt[0].strip()
        if not port_line.isdigit():
            return None
        return f"http://127.0.0.1:{port_line}"

    async def _probe_cdp_endpoint(self, endpoint: str, attempts: int = 3) -> bool:
        if not endpoint:
            return False
        if _urlreq:
            ver_url = endpoint.rstrip("/") + "/json/version"
            for _ in range(attempts):
                try:
                    ok = await asyncio.to_thread(self._fetch_url_ok, ver_url)
                    if ok:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(0.2)
        port = self._port_from_endpoint(endpoint)
        if port is None:
            return False
        for _ in range(attempts):
            ok = await asyncio.to_thread(self._socket_probe, port)
            if ok:
                return True
            await asyncio.sleep(0.2)
        return False

    @staticmethod
    def _fetch_url_ok(url: str, timeout: float = 0.5) -> bool:
        if not _urlreq:
            return False
        with _urlreq.urlopen(url, timeout=timeout) as resp:
            return getattr(resp, "status", 200) == 200

    @staticmethod
    def _socket_probe(port: int, timeout: float = 0.3) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=timeout):
                return True
        except OSError:
            return False

    @staticmethod  
    def _port_from_endpoint(endpoint: str) -> Optional[int]:  
        try:  
            return int(endpoint.rsplit(":", 1)[1])  
        except (ValueError, IndexError):  
            return None  
  
    async def _handle_new_page(self, p):  
        # minimal registration only  
        try:  
            await self._apply_viewport_mode("new-page", p)  
        except Exception:  
            pass  
        if getattr(self, "stealth_mode", "off") not in ("off", "none", "native"):
            try:  
                await p.set_extra_http_headers(dict(_HEADER_PROFILE))  
            except Exception:  
                pass  
            # ensure stealth on new pages  
            try:  
                await p.add_init_script(self._STEALTH_INIT_JS)  
            except Exception:  
                pass  
            try:  
                await p.evaluate(self._STEALTH_INIT_JS)  
            except Exception:  
                pass  
        if not getattr(self, 'dom_only', False):  
            with contextlib.suppress(Exception):  
                await self._inject_activity_observer(p, str(id(p)))  
        page_id = str(id(p))  
        ctx_id = ""  
        try:  
            ctx_id = str(id(p.context)) if hasattr(p, 'context') else (str(id(self.context)) if getattr(self, 'context', None) else "")  
        except Exception:  
            ctx_id = str(id(self.context)) if getattr(self, 'context', None) else ""  
  
        self.tracked_pages[page_id] = PageTrack3r(  
            page=p,  
            page_id=page_id,  
            url=p.url,  
            title=await p.title() if hasattr(p, 'title') else '',  
            last_activity=time.time(),  
            last_user_interaction=0.0,  
            context_id=ctx_id,  
            storage_key=getattr(self, 'activity_key', ''),  
        )  
        # keep tracker fresh on navigation/content events (no DOM injection)  
        try:  
            p.on("framenavigated", lambda frame, _p=p: asyncio.create_task(self._update_tracked_page(_p)))  
        except Exception:  
            pass  
        try:  
            p.on("domcontentloaded", lambda _p=p: asyncio.create_task(self._update_tracked_page(_p)))  
        except Exception:  
            pass  
        try:  
            p.on("load", lambda _p=p: asyncio.create_task(self._update_tracked_page(_p)))  
        except Exception:  
            pass  
  
    async def _register_page(self, page):  
        """Register page in internal tracker (no injection)."""  
        try:  
            pid = str(id(page))  
            if pid in getattr(self, 'tracked_pages', {}):  
                return  
            try:  
                title = await page.title()  
            except Exception:  
                title = ''  
            self.tracked_pages[pid] = PageTrack3r(  
                page=page,  
                page_id=pid,  
                url=page.url,  
                title=title,  
                last_activity=time.time(),  
                last_user_interaction=0.0,  
                context_id=str(id(self.context)) if getattr(self, 'context', None) else "",  
                storage_key=getattr(self, 'activity_key', ''),  
            )  
            # Cleanup on close  
            try:  
                def _on_close(pid=pid):  
                    try:  
                        self.tracked_pages.pop(pid, None)  
                    except Exception:  
                        pass  
                page.on("close", lambda: _on_close())  
            except Exception:  
                pass  
        except Exception:  
            pass  
  
    async def _update_tracked_page(self, page):  
        """Refresh URL/title for a tracked page and reselect active if needed."""  
        try:  
            pid = str(id(page))  
            tr = self.tracked_pages.get(pid)  
            if not tr:  
                return  
            tr.url = getattr(page, 'url', tr.url)  
            try:  
                tr.title = await page.title()  
            except Exception:  
                pass  
            tr.last_activity = time.time()  
            await self._reselect_active_page()  
        except Exception:  
            pass  
  
    async def _page_state_quick(self, tracker):  
        try:  
            p = tracker.page  
            if not p or p.is_closed():  
                return None  
            data = await p.evaluate("""() => ({  
                url: location.href,  
                title: document.title,  
                hasFocus: document.hasFocus(),  
                isVisible: document.visibilityState === 'visible',  
                ts: Date.now()  
            })""")  
            now = time.time()  
            tracker.has_focus = bool(data.get('hasFocus', False))  
            tracker.is_visible = bool(data.get('isVisible', False))  
            tracker.url = data.get('url') or tracker.url  
            tracker.title = data.get('title') or tracker.title  
            tracker.last_activity = now  
            return data  
        except Exception:  
            # On failure, mark as not focused/visible to avoid stale preference  
            try:  
                tracker.has_focus = False  
                tracker.is_visible = False  
            except Exception:  
                pass  
            return None  
  
    async def _get_active_page(self):  
        # Prefer explicitly selected active page id, if alive  
        try:  
            apid = getattr(self, 'active_page_id', None)  
            if apid and apid in getattr(self, 'tracked_pages', {}):  
                tr = self.tracked_pages.get(apid)  
                if tr and tr.page and not tr.page.is_closed():  
                    return tr.page  
        except Exception:  
            pass  
  
        # Fallback to primary page  
        if getattr(self, 'page', None) and not self.page.is_closed():  
            return self.page  
  
        # Any alive tracked page  
        for tr in getattr(self, 'tracked_pages', {}).values():  
            if tr.page and not tr.page.is_closed():  
                return tr.page  
        return None  
  
    async def _get_most_visible_page(self):  
        """Actively scan all tracked pages and return the most likely visible tab.  
        Zero injection; uses document.visibilityState + hasFocus.  
        """  
        best = None  
        best_score = -1.0  
        for tr in list(getattr(self, 'tracked_pages', {}).values()):  
            try:  
                if not tr.page or tr.page.is_closed():  
                    continue  
                with contextlib.suppress(Exception):  
                    await self._page_state_quick(tr)  
                score = 0.0  
                if getattr(tr, 'has_focus', False):  
                    score += 2.0  
                if getattr(tr, 'is_visible', False):  
                    score += 1.0  
                # Slightly prefer recent activity  
                score += max(0.0, (time.time() - float(tr.last_activity or 0.0)) * -0.01)  
                if score > best_score:  
                    best_score = score  
                    best = tr.page  
            except Exception:  
                continue  
        return best  
  
    # ===== OCR-based tiebreaker (optional, Windows only) =====  
    def _ensure_paddle(self):
        if not hasattr(self, "_paddle"):
            self._paddle = None
        global _PaddleOCR
        if _PaddleOCR is None:
            try:
                import logging as _logging  
                _logging.getLogger("ppocr").setLevel(_logging.ERROR)  
            except Exception:  
                pass  
            try:  
                from paddleocr import PaddleOCR as _PO  
                _PaddleOCR = _PO
            except Exception:
                _PaddleOCR = None
        if self._paddle is None and _PaddleOCR is not None:
            try:
                gpu_id = int(os.environ.get("PADDLEOCR_GPU_ID", "0"))
                self._paddle = _PaddleOCR(
                    use_angle_cls=False,
                    lang='en',
                    show_log=False,
                    use_gpu=True,
                    gpu_mem=2000,
                    gpu_id=gpu_id,
                    rec_batch_num=16,
                )
            except Exception:
                self._paddle = None

    def _ensure_paddle_pl(self):
        if not hasattr(self, "_paddle_pl"):
            self._paddle_pl = None
        self._ensure_paddle()
        if self._paddle_pl is None and _PaddleOCR is not None:
            try:
                gpu_id = int(os.environ.get("PADDLEOCR_GPU_ID", "0"))
                self._paddle_pl = _PaddleOCR(
                    use_angle_cls=False,
                    lang='pl',
                    show_log=False,
                    use_gpu=True,
                    gpu_mem=2000,
                    gpu_id=gpu_id,
                    rec_batch_num=16,
                )
            except Exception:  
                self._paddle_pl = None  
  
    def _ocr_image_tokens(self, img_bgr) -> list:
        """Prosty OCR: bez dodatkowych modyfikacji obrazu (jak wcześniej).
        Tylko próba PL, gdy EN da bardzo mało tekstu.
        """
        # Wymuszamy OCR niezależnie od flag, żeby Paddle pozostał w RAM (CPU)
        tokens = []
        self._ensure_paddle()
        if self._paddle is not None:
            try:
                res = self._paddle.ocr(img_bgr, det=True, rec=True, cls=False)
                for block in res:  
                    for (_pts, (txt, _prob)) in block:  
                        t = (txt or "").strip()  
                        if t:  
                            tokens.append(t)  
            except Exception:  
                pass  
        if len(tokens) < 3:  
            self._ensure_paddle_pl()  
            if self._paddle_pl is not None:  
                try:  
                    res = self._paddle_pl.ocr(img_bgr, det=True, rec=True, cls=False)  
                    for block in res:  
                        for (_pts, (txt, _prob)) in block:  
                            t = (txt or "").strip()  
                            if t:  
                                tokens.append(t)  
                except Exception:  
                    pass  
        return tokens  
  
    def _norm_url_for_match(self, url: str) -> str:  
        u = (url or "").strip().lower()  
        if not u:  
            return ""  
        if u.startswith("http://"):  
            u = u[7:]  
        elif u.startswith("https://"):  
            u = u[8:]  
        if u.startswith("www."):  
            u = u[4:]  
        return u.strip().strip('/ ')  
  
    async def _ocr_active_url_hint_text(self) -> Optional[str]:  
        # usunięto bramkę enable_ocr – zapis STRIPów ma się odbywać zawsze  
        if not sys.platform.startswith('win'):  
            log("OCR: disabled (non-Windows)", "DEBUG")  
            return None  
        if _mss is None or _np is None:  
            log("OCR: missing libs (mss/numpy)", "DEBUG")  
            return None  
        try:  
            # Screen-based OCR: detect active tab bbox and OCR only that + centered address slice  
            # Reuse a single MSS instance to reduce overhead  
            prev_sig = getattr(self, "_last_ocr_sig", None)  
            sct = getattr(self, '_mss_inst', None)  
            if sct is None:  
                self._mss_inst = _mss.mss()  
                sct = self._mss_inst  
            mon = sct.monitors[1]  
            mleft, mtop = int(mon['left']), int(mon['top'])  
            mwidth, mheight = int(mon['width']), int(mon['height'])  
            if mwidth < 100 or mheight < 60:  
                log(f"OCR: skip (monitor too small {mwidth}x{mheight})", "DEBUG")  
                return None  
            tabs_h = max(40, int(mheight * 0.12))  
            addr_h = max(10, int(mheight * 0.06))  
            shot_tabs = sct.grab({"left": mleft, "top": mtop, "width": mwidth, "height": tabs_h})  
            shot_addr = sct.grab({"left": mleft, "top": mtop + tabs_h, "width": mwidth, "height": addr_h})  
            # Light cache: if strips look identical, return last hint without re-running OCR  
            skip_ocr = False  
            try:  
                v1 = int(_np.frombuffer(shot_tabs.rgb, dtype=_np.uint8)[::64].sum())  
                v2 = int(_np.frombuffer(shot_addr.rgb, dtype=_np.uint8)[::64].sum())  
                sig = (v1, v2, shot_tabs.size, shot_addr.size)  
                self._last_ocr_sig = sig  
                if prev_sig == sig:  
                    cached = getattr(self, "_last_ocr_hint", None)  
                    if cached is not None:  
                        return cached  
            except Exception:  
                pass  
            finally:  
                # Nowa metoda: kolumnowy profil jasności, szukamy rozjaśnionego pasa aktywnej karty.  
                tabs_img = _np.array(shot_tabs)[..., :3]  
                gray = (0.114 * tabs_img[..., 0] + 0.587 * tabs_img[..., 1] + 0.299 * tabs_img[..., 2]).astype("float32")  
                x0 = 40  
                x1 = max(80, mwidth - 80)  
                sub = gray[:, x0:x1]  
                top_band = sub[: max(6, int(sub.shape[0] * 0.55)), :]  
                rest_band = sub[max(6, int(sub.shape[0] * 0.35)) :, :]  
                if rest_band.size == 0:  
                    rest_band = sub  
                col_top = top_band.mean(axis=0)  
                col_rest = rest_band.mean(axis=0)  
                col_score = _np.clip(col_top - (col_rest * 0.85), 0, None)  
                if col_score.size < 5:  
                    col_score = _np.pad(col_score, (0, max(0, 5 - col_score.size)), mode="edge")  
                win = max(40, int(col_score.shape[0] * 0.06))  
                if win % 2 == 0:  
                    win += 1  
                kernel = _np.ones(win, dtype=_np.float32) / float(win)  
                smooth = _np.convolve(col_score, kernel, mode="same")  
                best_start = int(max(0, smooth.argmax() - win // 2))  
                span = x1 - x0  
                best_len = max(int(span * 0.18), 160)  
                best_len = min(best_len, span)  
                if best_start + best_len > span:  
                    best_start = max(0, span - best_len)  
                if best_len <= 0:  
                    best_start = int(span * 0.30)  
                    best_len = max(1, int(span * 0.40))  
                tab_x_start = x0 + best_start  
                tab_x_end = min(x1, tab_x_start + best_len)  
            # Dedykowane wąskie zrzuty aktywnej karty + środka paska adresu
            tab_w = max(1, tab_x_end - tab_x_start)
            shot_tabs_active = sct.grab({
                "left": mleft + tab_x_start,
                "top": mtop,
                "width": tab_w,
                "height": tabs_h,
            })
            addr_center_w = min(mwidth, max(120, int(mwidth * 0.35)))
            addr_center_left = mleft + max(0, (mwidth - addr_center_w) // 2)
            shot_addr_center = sct.grab({
                "left": addr_center_left,
                "top": mtop + tabs_h,
                "width": addr_center_w,
                "height": addr_h,
            })
            log(f"OCR: monitor tabs {mwidth}x{tabs_h} addr {mwidth}x{addr_h} @({mleft},{mtop}); active X=({tab_x_start},{tab_x_end})", "DEBUG")  

            img_tabs_active = _np.array(shot_tabs_active)[..., :3]  
            img_addr_center = _np.array(shot_addr_center)[..., :3]  
            tokens_tabs = self._ocr_image_tokens(img_tabs_active)  
            tokens_addr = self._ocr_image_tokens(img_addr_center)  
            tokens = tokens_tabs + tokens_addr  
            full = " ".join(tokens).strip()  
            if not full:  
                fallback = getattr(self, '_last_ocr_hint', '') or ''  
                if not fallback:  
                    try:  
                        active_id = getattr(self, 'active_page_id', None)  
                        tr = getattr(self, 'tracked_pages', {}).get(active_id) if active_id else None  
                        if tr:  
                            fallback = (tr.title or tr.url or '')[:80]  
                    except Exception:  
                        fallback = ''  
                full = fallback  
            self._last_ocr_hint = full  
            # Save debug crops and tokens  
            try:  
                if getattr(self, 'output_dir', None):  
                    debug_dir = _ocr_debug_dir()  
                    p_strip = (debug_dir / 'ocr_strip1.png')  
                    p_strip_act = (debug_dir / 'ocr_strip1_active.png')  
                    p_addr = (debug_dir / 'ocr_strip2_center.png')  
                    if _msstools:  
                        with open(p_strip, 'wb') as f:  
                            f.write(_msstools.to_png(shot_tabs.rgb, shot_tabs.size))  
                        with open(p_strip_act, 'wb') as f:  
                            f.write(_msstools.to_png(shot_tabs_active.rgb, shot_tabs_active.size))  
                        with open(p_addr, 'wb') as f:  
                            f.write(_msstools.to_png(shot_addr_center.rgb, shot_addr_center.size))  
                    else:  
                        # Fallback bez mss.tools: PIL z bufora RGB (bez zmiany logiki)  
                        try:  
                            from PIL import Image as _PILImage  # type: ignore  
                            img1 = _PILImage.frombytes('RGB', shot_tabs.size, shot_tabs.rgb)  
                            img2 = _PILImage.frombytes('RGB', shot_tabs_active.size, shot_tabs_active.rgb)  
                            img3 = _PILImage.frombytes('RGB', shot_addr_center.size, shot_addr_center.rgb)  
                            img1.save(str(p_strip))  
                            img2.save(str(p_strip_act))  
                            img3.save(str(p_addr))  
                        except Exception:  
                            pass  
                    self._last_ocr_debug = {  
                        'tab_bbox': {'x_start': int(tab_x_start), 'x_end': int(tab_x_end), 'width': int(mwidth)},  
                        'crops': [  
                            {'file': str(p_strip), 'width': int(shot_tabs.size[0]), 'height': int(shot_tabs.size[1])},  
                            {'file': str(p_strip_act), 'width': int(shot_tabs_active.size[0]), 'height': int(shot_tabs_active.size[1]), 'tokens': tokens_tabs},  
                            {'file': str(p_addr), 'width': int(shot_addr_center.size[0]), 'height': int(shot_addr_center.size[1]), 'tokens': tokens_addr},  
                        ]  
                    }  
            except Exception:  
                pass  
            if not full:  
                # Fallback: jeśli OCR nic nie znalazł, spróbuj użyć tytułu okna przeglądarki  
                try:  
                    wt = (getattr(self,'uia',None) and getattr(self.uia.get_active_window_info(),'title',''), '') or ''  
                    if wt:  
                        log("OCR: fallback to window title", "DEBUG")  
                        full = wt  
                except Exception:  
                    pass  
            ln = (full or "")  
            ln_short = (ln[:80] + "…") if len(ln) > 80 else ln  
            log(f"OCR: extracted '{ln_short}' (tokens={len(tokens)})", "INFO")  
            try:  
                self._last_ocr_hint = full or None  
            except Exception:  
                pass  
            return full or None  
        except Exception:  
            return None  
  
    async def _ocr_pick_visible_page(self) -> Optional[str]:  
        try:  
            cands = []  
            for pid, tr in list(self.tracked_pages.items()):  
                if not tr.page or tr.page.is_closed():  
                    continue  
                if getattr(tr, 'is_visible', False) or getattr(tr, 'has_focus', False):  
                    cands.append(pid)  
            if len(cands) <= 1:  
                return cands[0] if cands else None  
            hint = await self._ocr_active_url_hint_text()  
            if not hint:  
                log("OCR: no hint text", "DEBUG")  
                return None  
            hint_norm = self._norm_url_for_match(hint)  
            if not hint_norm:  
                log("OCR: empty normalized hint", "DEBUG")  
                return None  
            # core token z OCR (najdłuższy alfanumeryczny bez www/http/com/pl)  
            toks = [t for t in _re.findall(r"[a-z0-9]+", hint_norm) if t]  
            stop = {"www","http","https","com","pl","net","org","co","io","app"}  
            core = None  
            if toks:  
                toks2 = [t for t in toks if t not in stop and len(t) >= 2]  
                core = max(toks2 or toks, key=len)  
            try:  
                log(f"OCR: core token='{core or ''}' from hint='{hint_norm[:80]}'", "INFO")  
            except Exception:  
                pass  
            # core token z OCR (najdłuższy alfanum. bez www/http/com/pl itp.)  
            toks = [t for t in _re.findall(r"[a-z0-9]+", hint_norm) if t]  
            stop = {"www","http","https","com","pl","net","org","co","io","app"}  
            core = None  
            if toks:  
                toks2 = [t for t in toks if t not in stop and len(t) >= 3]  
                core = max(toks2 or toks, key=len)  
            try:  
                log(f"OCR: core='{(core or '')}' from hint='{hint_norm[:80]}'", "INFO")  
            except Exception:  
                pass  
            best_pid = None  
            best_score = -1.0  
            debug_cands = []  
            for pid in cands:  
                tr = self.tracked_pages.get(pid)  
                if not tr:  
                    continue  
                url_norm = self._norm_url_for_match(getattr(tr, 'url', ''))  
                dom_norm = domain_from_url(tr.url)  
                s_url = float(fuzzy_ratio(hint_norm, url_norm) or 0.0)  
                s_dom = float(fuzzy_ratio(hint_norm, dom_norm) or 0.0)  
                s_title = float(fuzzy_ratio(hint_norm, norm_text(tr.title or '')) or 0.0)  
                # Twarde dopasowania „zawiera”: domena/brand w tekście → wynik bliski 1.0  
                brand = (dom_norm.split('.')[-2] if dom_norm else "")  
                contains_dom = 1.0 if (dom_norm and dom_norm in hint_norm) else 0.0  
                contains_brand = 1.0 if (brand and brand in hint_norm) else 0.0  
                # Preferencja odwrotna: czy DOM/URL zawierają rdzeń z OCR (bo OCR bywa krótszy)  
                url_has_core = 1.0 if (core and url_norm and core in url_norm) else 0.0  
                dom_has_core = 1.0 if (core and dom_norm and core in dom_norm) else 0.0  
                title_has_core = 1.0 if (core and tr.title and core in (norm_text(tr.title or '').lower())) else 0.0  
                # Wynik końcowy (bez pozycyjnych bonusów)  
                score = max(  
                    url_has_core,  
                    dom_has_core,  
                    title_has_core * 0.9,  
                    contains_dom,  
                    contains_brand * 0.95,  
                    s_dom,  
                    s_url * 0.8,  
                    s_title * 0.5,  
                )  
                log(  
                    f"OCR: cand pid={pid[-6:]} core='{core or ''}' s_url={s_url:.2f} s_dom={s_dom:.2f} s_title={s_title:.2f} "  
                    f"url_has_core={int(url_has_core)} dom_has_core={int(dom_has_core)} title_has_core={int(title_has_core)} "  
                    f"contains(dom)={int(contains_dom)} contains(brand)={int(contains_brand)} total={score:.2f} | "  
                    f"url='{url_norm[:50]}' dom='{dom_norm[:30]}' title='{(tr.title or '')[:40]}'",  
                    "INFO",  
                )  
                debug_cands.append({  
                    'pid': pid,  
                    'url': getattr(tr, 'url', ''),  
                    'title': tr.title,  
                    'scores': {  
                        's_url': s_url,  
                        's_dom': s_dom,  
                        's_title': s_title,  
                        'url_has_core': url_has_core,  
                        'dom_has_core': dom_has_core,  
                        'title_has_core': title_has_core,  
                        'contains_dom': contains_dom,  
                        'contains_brand': contains_brand,  
                        'total': score,  
                    }  
                })  
                if score > best_score:  
                    best_score = score  
                    best_pid = pid  
            if best_pid and best_score >= 0.50:  
                tr = self.tracked_pages.get(best_pid)  
                log(f"OCR: picked pid={best_pid[-6:]} score={best_score:.2f} -> {(tr.title or '')[:60]} | {(tr.url or '')[:120]}", "SUCCESS")  
                # Write JSON debug summary (what OCR saw and how it scored)  
                try:  
                    if getattr(self, 'output_dir', None):  
                        dbg = getattr(self, '_last_ocr_debug', {}) or {}  
                        dbg['hint'] = hint  
                        dbg['hint_norm'] = hint_norm  
                        dbg['core'] = core  
                        dbg['candidates'] = debug_cands  
                        import json  
                        debug_dir = _ocr_debug_dir()  
                        with open(debug_dir / 'ocr_debug.json', 'w', encoding='utf-8') as f:  
                            json.dump(dbg, f, ensure_ascii=False, indent=2)  
                except Exception:  
                    pass  
                return best_pid  
            log(f"OCR: no confident pick (best={best_score:.2f})", "DEBUG")  
            return None  
        except Exception:  
            return None  
  
    # ===== Color-based tabs hint (no OCR, Windows only) =====  
    async def _color_tabs_active_hint(self) -> Optional[dict]:  
        """Detect presence of a bright active tab region in the top strip.  
        Returns dict with metrics or None if unsupported/unreliable.  
        Does not map to a specific page; used only as an extra hint/log.  
        """  
        if not sys.platform.startswith('win'):  
            return None  
        if _mss is None or _np is None:  
            return None  
        try:  
            win = getattr(self, 'uia', None) and self.uia.get_active_window_info()  
            if not win or not win.rect:  
                return None  
            left, top, right, bottom = win.rect  
            width = max(0, int(right - left))  
            height = max(0, int(bottom - top))  
            if width < 200 or height < 60:  
                return None  
            # Crop only the tabs strip (~12% of window height)  
            h = max(40, min(140, int(height * 0.12)))  
            box = {"left": int(left), "top": int(top), "width": int(width), "height": int(h)}  
            with _mss.mss() as sct:  
                shot = sct.grab(box)  
            img = _np.array(shot)[..., :3]  # BGR  
            # Convert to gray (BT.601): Y = 0.299R + 0.587G + 0.114B  
            gray = (0.114*img[...,0] + 0.587*img[...,1] + 0.299*img[...,2]).astype('float32')  
            # Adaptive threshold: mean + small delta  
            thr = float(gray.mean() + 0.15*gray.std())  
            mask = (gray >= thr).astype('uint8')  # 1 for bright  
            # Ignore leftmost 40 px (window controls) and rightmost 80 px (profile/controls)  
            x0, x1 = 40, max(80, width-80)  
            sub = mask[:, x0:x1]  
            # Column-wise bright ratio  
            col_ratio = sub.mean(axis=0)  
            # Find longest run of columns with ratio >= 0.30 (30%)  
            run_min = 0.30  
            best_len = 0  
            best_start = 0  
            cur_len = 0  
            cur_start = 0  
            for i, v in enumerate(col_ratio):  
                if v >= run_min:  
                    if cur_len == 0:  
                        cur_start = i  
                    cur_len += 1  
                else:  
                    if cur_len > best_len:  
                        best_len = cur_len  
                        best_start = cur_start  
                    cur_len = 0  
            if cur_len > best_len:  
                best_len = cur_len  
                best_start = cur_start  
            # Metrics  
            cov = float(sub.mean())  # overall bright coverage  
            run_cov = float(col_ratio[best_start:best_start+best_len].mean()) if best_len > 0 else 0.0  
            run_width_ratio = best_len / max(1.0, (x1 - x0))  
            info = {  
                "ok": bool(best_len > 0),  
                "coverage": cov,  
                "run_width_ratio": run_width_ratio,  
                "run_bright_ratio": run_cov,  
                "strip_height": h,  
                "width": width,  
            }  
            log(  
                f"COLOR: ok={int(info['ok'])} cov={info['coverage']:.2f} runW={info['run_width_ratio']:.2f} runBright={info['run_bright_ratio']:.2f}",  
                "INFO",  
            )  
            return info  
        except Exception:  
            return None  
  
    async def _reselect_active_page(self):  
        """Choose the best candidate for the currently visible/focused tab without injecting anything."""  
        try:  
            if not getattr(self, 'tracked_pages', None):  
                return  
  
            # 1) Prefer OS foreground window title if available (Windows UIA)  
            win_info = None  
            try:  
                win_info = getattr(self, 'uia', None) and self.uia.get_active_window_info()  
            except Exception:  
                win_info = None  
  
            def _title_similarity(a: str, b: str) -> float:  
                try:  
                    from .ai_recorder_common import fuzzy_ratio, norm_text  
                except Exception:  
                    from ai_recorder_common import fuzzy_ratio, norm_text  # type: ignore  
                return float(fuzzy_ratio(norm_text(a or ''), norm_text(b or '')) or 0.0)  
  
            best_id: Optional[str] = None  
            # 1a) If we know the foreground window title, pick the page with highest title similarity  
            if win_info and getattr(win_info, 'title', None):  
                try:  
                    sim_best = -1.0  
                    win_title_norm = (win_info.title or "").strip().lower()  
                    for pid, tr in list(self.tracked_pages.items()):  
                        if not tr.page or tr.page.is_closed():  
                            continue  
                        with contextlib.suppress(Exception):  
                            await self._page_state_quick(tr)  
                        sim = _title_similarity(tr.title, win_info.title)  
                        # Strong match if window title contains domain/brand of the tab  
                        try:  
                            dom = domain_from_url(tr.url)  
                            brand = (dom.split('.')[-2] if dom else "")  
                        except Exception:  
                            dom = ""; brand = ""  
                        strong = 0.0  
                        if dom and dom.lower() in win_title_norm:  
                            strong = 1.0  
                        elif brand and brand.lower() in win_title_norm:  
                            strong = 0.95  
                        sim_final = max(sim, strong)  
                        # Log each UIA candidate with similarity  
                        try:  
                            log(  
                                f"UIA: cand pid={pid[-6:]} sim={sim:.2f} strong={strong:.2f} final={sim_final:.2f} | "  
                                f"winTitle='{(win_info.title or '')[:40]}' tabTitle='{(tr.title or '')[:40]}' dom='{(dom or '')[:30]}'",  
                                "INFO",  
                            )  
                        except Exception:  
                            pass  
                        if sim_final > sim_best:  
                            sim_best = sim_final  
                            best_id = pid  
                    # If similarity is too low, we'll fall back to DOM signals below  
                except Exception:  
                    best_id = None  
  
            # 2) If UIA could not resolve, use DOM focus/visibility+recency; if tie of multiple visible/focused, try OCR  
            if not best_id:  
                best_score = -1.0  
                vis_focus_pids = []  
                now = time.time()  
                for pid, tr in list(self.tracked_pages.items()):  
                    if not tr.page or tr.page.is_closed():  
                        continue  
                    with contextlib.suppress(Exception):  
                        await self._page_state_quick(tr)  
                    score = 0.0  
                    f = bool(getattr(tr, 'has_focus', False))  
                    v = bool(getattr(tr, 'is_visible', False))  
                    if f:  
                        score += 2.0  
                    if v:  
                        score += 1.0  
                        vis_focus_pids.append(pid)  
                    # slight preference for recency (as implemented: non-negative)  
                    rec_add = max(0.0, (now - float(tr.last_activity or 0.0)) * -0.01)  
                    score += rec_add  
                    try:  
                        log(  
                            f"DOM: cand pid={pid[-6:]} focus={int(f)}(+{2.0 if f else 0.0}) "  
                            f"visible={int(v)}(+{1.0 if v else 0.0}) rec={rec_add:+.2f} total={score:.2f} | "  
                            f"title='{(tr.title or '')[:40]}' url='{(tr.url or '')[:80]}'",  
                            "INFO",  
                        )  
                    except Exception:  
                        pass  
                    if score > best_score:  
                        best_score = score  
                        best_id = pid  
                try:  
                    log(f"DOM: best pid={(best_id or '')[-6:]} score={best_score:.2f}", "INFO")  
                except Exception:  
                    pass  
                # If we have more than one visible/focus candidate, optionally try OCR tie-breaker  
                if len(vis_focus_pids) > 1 and getattr(self, 'enable_ocr', False):  
                    with contextlib.suppress(Exception):  
                        ocr_pid = await self._ocr_pick_visible_page()  
                        if ocr_pid:  
                            best_id = ocr_pid  
  
            # 3) If still unresolved, optionally try OCR tiebreaker on the OS top strip  
            if not best_id and getattr(self, 'enable_ocr', False):  
                with contextlib.suppress(Exception):  
                    ocr_pid = await self._ocr_pick_visible_page()  
                    if ocr_pid:  
                        best_id = ocr_pid  
  
            # 4) Last resort: explicit most-visible scan  
            if not best_id:  
                with contextlib.suppress(Exception):  
                    mv = await self._get_most_visible_page()  
                    if mv:  
                        best_id = str(id(mv))  
  
            if best_id and best_id != getattr(self, 'active_page_id', None):  
                self.active_page_id = best_id  
                tr = self.tracked_pages.get(best_id)  
                if tr:  
                    title = (tr.title or "").strip()  
                    url = (tr.url or "").strip()  
                    log(f"Active tab -> {title[:80]} | {url[:140]}", "INFO")  
        except Exception:  
            pass  
  
    async def _focus_poll_loop(self):  
        while getattr(self, 'recording', False):  
            try:  
                # Stop sentinel file (OUTPUT_DIR/STOP)  
                try:  
                    if getattr(self, 'stop_file', None) and self.stop_file.exists():  
                        log("Stop requested via STOP file", "WARNING")  
                        self.recording = False  
                        break  
                except Exception:  
                    pass  
                # Initialize with the main page if not set  
                if getattr(self, 'active_page_id', None) is None and getattr(self, 'page', None):  
                    self.active_page_id = str(id(self.page))  
                # Reselect best active page periodically (no injection)  
                await self._reselect_active_page()  
                # Extra diagnostic log: current focus/visibility across tabs (no injection)  
                with contextlib.suppress(Exception):  
                    await self._log_focus_matrix()  
                with contextlib.suppress(Exception):  
                    self._write_tabs_file()  
                await asyncio.sleep(1.0)  
            except asyncio.CancelledError:  
                break  
            except Exception:  
                await asyncio.sleep(1.0)  
  
    async def _ocr_strip_poll_loop(self):  
        """Wymusza generowanie OCR_STRIP oraz OCR_STRIP_ACTIVE dokładnie co 2 sekundy, bezwarunkowo."""  
        interval = 2.0  # bezwzględnie co 2 sekundy  
        next_run = time.time()  
        # Pre-warm PaddleOCR aby utrzymać modele w RAM  
        with contextlib.suppress(Exception):  
            self._ensure_paddle()  
            self._ensure_paddle_pl()  
        while getattr(self, "recording", False):  
            try:  
                # Zawsze wykonuj przechwyt i dump (niezależnie od enable_ocr)  
                hint = await self._ocr_active_url_hint_text()  
                await self._ocr_debug_dump(hint)  
                self._last_ocr_poll = time.time()  
            except asyncio.CancelledError:  
                break  
            except Exception:  
                pass  
            next_run += interval  
            delay = max(0.0, next_run - time.time())  
            try:  
                await asyncio.sleep(delay if delay > 0 else 0)  
            except asyncio.CancelledError:  
                break  
  
    async def _log_focus_matrix(self):  
        """Log summary of all tracked tabs: which one is visible/focused/active.  
        Only reads minimal DOM state; no injections, no modifications.  
        """  
        try:  
            if not getattr(self, 'tracked_pages', None):  
                return  
            rows = []  
            active_id = getattr(self, 'active_page_id', None)  
            # Refresh quick state before logging  
            for pid, tr in list(self.tracked_pages.items()):  
                if not tr or not getattr(tr, 'page', None) or tr.page.is_closed():  
                    continue  
                with contextlib.suppress(Exception):  
                    await self._page_state_quick(tr)  
                is_active = (pid == active_id)  
                mark = '*' if is_active else ' '  
                vis = 'V' if getattr(tr, 'is_visible', False) else '-'  
                foc = 'F' if getattr(tr, 'has_focus', False) else '-'  
                title = (tr.title or '').strip().replace('\n',' ')[:60]  
                url = (tr.url or '').strip().replace('\n',' ')[:120]  
                rows.append(f"[{mark}{vis}{foc}] {title} | {url}")  
            if rows:  
                # One compact line to keep log tidy  
                summary = ' || '.join(rows)  
                log(f"Tabs: {summary}", "INFO")  
        except Exception:  
            pass  
  
    async def _ocr_debug_dump(self, hint: Optional[str] = None):  
        """Always produce OCR debug artifacts (strips + JSON) into <output>/debug.  
        Does not influence selection; for diagnostics only.  
        """  
        try:  
            # Capture hint and save strips (handled inside)  
            if hint is None:  
                hint = await self._ocr_active_url_hint_text()  
            if hint is None:  
                hint = ""  
            hint_norm = self._norm_url_for_match(hint)  
            # Build candidate scoring snapshot for all tracked pages  
            debug_cands = []  
            core = None  
            if hint_norm:  
                toks = [t for t in _re.findall(r"[a-z0-9]+", hint_norm) if t]  
                stop = {"www","http","https","com","pl","net","org","co","io","app"}  
                if toks:  
                    toks2 = [t for t in toks if t not in stop and len(t) >= 2]  
                    core = max(toks2 or toks, key=len)  
            for pid, tr in list(getattr(self, 'tracked_pages', {}).items()):  
                if not tr or not getattr(tr, 'page', None) or tr.page.is_closed():  
                    continue  
                url_norm = self._norm_url_for_match(getattr(tr, 'url', ''))  
                dom_norm = domain_from_url(tr.url)  
                try:  
                    from .ai_recorder_common import fuzzy_ratio, norm_text  
                except Exception:  
                    from ai_recorder_common import fuzzy_ratio, norm_text  # type: ignore  
                s_url = float(fuzzy_ratio(hint_norm, url_norm) or 0.0)  
                s_dom = float(fuzzy_ratio(hint_norm, dom_norm) or 0.0)  
                s_title = float(fuzzy_ratio(hint_norm, norm_text(tr.title or '')) or 0.0)  
                brand = (dom_norm.split('.')[-2] if dom_norm else "")  
                url_has_core = 1.0 if (core and url_norm and core in url_norm) else 0.0  
                dom_has_core = 1.0 if (core and dom_norm and core in dom_norm) else 0.0  
                title_has_core = 1.0 if (core and tr.title and core in (norm_text(tr.title or '').lower())) else 0.0  
                contains_dom = 1.0 if (dom_norm and dom_norm in hint_norm) else 0.0  
                contains_brand = 1.0 if (brand and brand in hint_norm) else 0.0  
                total = max(url_has_core, dom_has_core, title_has_core*0.9, contains_dom, contains_brand*0.95, s_dom, s_url*0.8, s_title*0.5)  
                debug_cands.append({  
                    'pid': pid,  
                    'url': getattr(tr, 'url', ''),  
                    'title': tr.title,  
                    'scores': {  
                        's_url': s_url,  
                        's_dom': s_dom,  
                        's_title': s_title,  
                        'url_has_core': url_has_core,  
                        'dom_has_core': dom_has_core,  
                        'title_has_core': title_has_core,  
                        'contains_dom': contains_dom,  
                        'contains_brand': contains_brand,  
                        'total': total,  
                    }  
                })  
            # Persist JSON next to strips  
            if getattr(self, 'output_dir', None):  
                debug_dir = _ocr_debug_dir()  
                import json  
                dbg = getattr(self, '_last_ocr_debug', {}) or {}  
                dbg['hint'] = hint  
                dbg['hint_norm'] = hint_norm  
                dbg['core'] = core  
                dbg['candidates'] = debug_cands  
                with open(debug_dir / 'ocr_debug.json', 'w', encoding='utf-8') as f:  
                    json.dump(dbg, f, ensure_ascii=False, indent=2)  
        except Exception:  
            pass  
