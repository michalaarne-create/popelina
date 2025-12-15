# ai_recorder_live.py - WERSJA STEALTH CDP (clean) [czÄ™Ĺ›Ä‡ 1/2]
import os
import sys
import asyncio
from pathlib import Path

if __package__:
    from .ai_recorder_common import (
        START_URL_DEFAULT,
        SNAPSHOT_FPS_DEFAULT,
        ensure_dir,
        log,
        set_log_file,
        set_verbose_debug,
    )
    from .ai_recorder_base import LiveRecorderBase
    from .ai_recorder_browser import LiveRecorderBrowserMixin
    from .ai_recorder_capture import LiveRecorderCaptureMixin
else:
    from ai_recorder_common import (
        START_URL_DEFAULT,
        SNAPSHOT_FPS_DEFAULT,
        ensure_dir,
        log,
        set_log_file,
        set_verbose_debug,
    )
    from ai_recorder_base import LiveRecorderBase
    from ai_recorder_browser import LiveRecorderBrowserMixin
    from ai_recorder_capture import LiveRecorderCaptureMixin


class LiveRecorder(LiveRecorderCaptureMixin, LiveRecorderBrowserMixin, LiveRecorderBase):
    pass

# ========== MAIN ==========

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Recorder Live - STEALTH CDP connect (clean)")
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        type=str,
        default=str(Path.cwd() / "dom_live"),
        help="Directory used for recorder output (alias: --output-dir)",
    )
    parser.add_argument("--user-data-dir", type=str, default=str(Path.cwd() / "_recorder_profile"))
    parser.add_argument(
        "--profile-dir",
        "--profile-directory",
        dest="profile_dir",
        type=str,
        default=None,
        help="Chrome profile directory name (Default, Profile 1, etc.)"
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=None,
        help="Override browser User-Agent header/string"
    )
    parser.add_argument(
        "--extra-url",
        dest="extra_urls",
        action="append",
        default=[],
        help="Additional tab(s) to open on start (can be repeated)",
    )
    parser.add_argument("--url", type=str, default=START_URL_DEFAULT)
    parser.add_argument("--fps", type=float, default=SNAPSHOT_FPS_DEFAULT)
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path (defaults to <output>/recorder_debug.log)",
    )
    parser.add_argument(
        "--connect-existing",
        action="store_true",
        help="Attach to already running browser in user-data-dir (DevToolsActivePort) instead of launching new one",
    )
    screenshot_group = parser.add_mutually_exclusive_group()
    screenshot_group.add_argument(
        "--screenshots",
        dest="screenshots",
        action="store_true",
        help="Włącz screenshoty/canvas (domyślnie aktywne)",
    )
    screenshot_group.add_argument(
        "--no-screenshots",
        dest="screenshots",
        action="store_false",
        help="Wyłącz screenshoty/canvas nawet w trybie normalnym",
    )
    dom_group = parser.add_mutually_exclusive_group()
    dom_group.add_argument(
        "--dom-only",
        dest="dom_only",
        action="store_true",
        help="Zbieraj tylko DOM (bez screenshot/OCR) – stealth minimalny",
    )
    dom_group.add_argument(
        "--normal-mode",
        dest="dom_only",
        action="store_false",
        help="Wymuś pełny tryb canvas/screenshot niezależnie od konfiguracji",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cooldown", type=float, default=0.75)
    parser.add_argument("--viewport-mode", type=str, default=os.environ.get("VIEWPORT_MODE","os-max"),
                        choices=["os-max","cdp-1920","page-viewport"],
                        help="Tryb ustawiania rozmiaru okna/viewportu")
    parser.add_argument("--chrome-exe", type=str, default=None, help="ĹšcieĹĽka do Chrome/Edge (opcjonalnie)")
    parser.add_argument("--cdp-endpoint", type=str, default=None, help="CDP endpoint (np. http://127.0.0.1:9222). JeĹ›li brak, uruchomi Chrome i odczyta port.")
    parser.add_argument("--profile-cpu", action="store_true", help="Włącz profil CPU (cProfile) i zapisz raport do debug/")
    parser.add_argument("--enable-ocr", dest="enable_ocr", action="store_true", help="Włącz OCR zakładek/URL (uruchamia PaddleOCR/GPU).")
    parser.add_argument("--disable-ocr", dest="enable_ocr", action="store_false", help="Wyłącz OCR (tylko czysty DOM).")
    parser.set_defaults(enable_ocr=None, screenshots=True, dom_only=False)
    # Accept known args and allow a stealth --proxy without exposing in help
    args, _unknown = parser.parse_known_args()
    proxy_val = None
    # Parse optional --proxy from unknowns to avoid argparse errors and keep stealthy
    for i, tok in enumerate(_unknown):
        if isinstance(tok, str) and tok.startswith("--proxy="):
            proxy_val = tok.split("=", 1)[1]
            break
        if tok == "--proxy" and i + 1 < len(_unknown):
            proxy_val = _unknown[i + 1]
            break
    # Fallback to env var if not provided on CLI
    if not proxy_val:
        try:
            import os as _os
            proxy_val = _os.environ.get("RECORDER_PROXY")
        except Exception:
            proxy_val = None
    try:
        setattr(args, "proxy", proxy_val)
    except Exception:
        pass

    if args.enable_ocr is None:
        args.enable_ocr = True

    out_abs = Path(args.output).resolve()
    ud_abs = Path(args.user_data_dir).resolve()
    ensure_dir(out_abs)
    ensure_dir(ud_abs)

    log_path = Path(args.log_file).expanduser().resolve() if args.log_file else out_abs / "recorder_debug.log"
    set_log_file(log_path)
    set_verbose_debug(bool(args.verbose))
    log(f"ENV: Python={sys.version.split()[0]} | Platform={sys.platform}", "DEBUG")

    rec = LiveRecorder(
        output_dir=str(out_abs),
        start_url=args.url,
        user_data_dir=str(ud_abs),
        profile_directory=args.profile_dir,
        user_agent=args.user_agent,
        fps=args.fps,
        screenshots=args.screenshots,
        verbose=args.verbose,
        extra_urls=args.extra_urls,
        viewport_mode=args.viewport_mode,
        chrome_exe=args.chrome_exe,
        connect_existing=bool(args.cdp_endpoint) or bool(args.connect_existing),
        cdp_endpoint=args.cdp_endpoint,
        proxy_server=getattr(args, "proxy", None),
        enable_ocr=args.enable_ocr,
        dom_only=bool(args.dom_only),
    )
    rec.auto_switch_cooldown = args.cooldown

    try:
        rec._ensure_paddle()
        rec._ensure_paddle_pl()
        log("PaddleOCR preloaded and pinned in RAM (GPU inference)", "INFO")
    except Exception as preload_err:
        log(f"PaddleOCR preload failed: {preload_err}", "WARNING")

    log(f"đźš€ AI Recorder Live (STEALTH CDP)", "SUCCESS")
    log(f"đź›ˇď¸Ź CDP connect (browser stays alive on exit) + listener-only injection", "SUCCESS")
    log(f"đź“ Output: {out_abs}", "INFO")
    log(f"đź“¸ FPS: {args.fps}", "INFO")

    # Opcjonalne profilowanie CPU (bez zmiany logiki działania)
    prof = None
    if args.profile_cpu:
        try:
            import cProfile
            prof = cProfile.Profile()
            prof.enable()
            log("Profil CPU: start", "INFO")
        except Exception:
            prof = None

    try:
        await rec.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if prof is not None:
                prof.disable()
                debug_dir = (out_abs / "debug")
                ensure_dir(debug_dir)
                import pstats, json
                ps = pstats.Stats(prof)
                ps.sort_stats("tottime")
                total_tt = getattr(ps, "total_tt", None)
                if not total_tt:
                    total_tt = sum(v[2] for v in ps.stats.values()) or 1.0
                rows = []
                for (filename, lineno, funcname), stat in ps.stats.items():
                    cc, nc, tt, ct, callers = stat
                    pct_tt = (tt/total_tt)*100.0 if total_tt else 0.0
                    rows.append({"file": filename, "line": lineno, "func": funcname, "calls": nc, "prim_calls": cc, "tottime": tt, "cumtime": ct, "pct_tottime": pct_tt})
                rows.sort(key=lambda r: r["pct_tottime"], reverse=True)
                top = rows[:200]
                by_file = {}
                for r in rows:
                    by_file[r["file"]] = by_file.get(r["file"], 0.0) + r["tottime"]
                agg = [{"file": f, "tottime": tt, "pct_tottime": (tt/total_tt)*100.0 if total_tt else 0.0} for f, tt in by_file.items()]
                agg.sort(key=lambda a: a["pct_tottime"], reverse=True)
                with open(debug_dir / "cpu_profile.txt", "w", encoding="utf-8") as f:
                    f.write(f"Total tottime: {total_tt:.4f}s\n\n== Per function (top 200, by % tottime) ==\n")
                    for r in top:
                        f.write(f"{r['pct_tottime']:6.2f}%  {r['tottime']:.4f}s  {r['func']}  ({r['file']}:{r['line']}) calls={r['calls']}\n")
                    f.write("\n== Per file/module (by % tottime) ==\n")
                    for a in agg[:100]:
                        f.write(f"{a['pct_tottime']:6.2f}%  {a['tottime']:.4f}s  {a['file']}\n")
                with open(debug_dir / "cpu_profile.json", "w", encoding="utf-8") as jf:
                    json.dump({"total_tottime": total_tt, "top_functions": top, "by_file": agg}, jf, ensure_ascii=False, indent=2)
                log(f"Profil CPU zapisany: {debug_dir / 'cpu_profile.txt'}", "SUCCESS")
        finally:
            await rec.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nđź›‘ Stopped by user")
    except Exception as e:
        print(f"âťŚ Fatal error: {e}")
        import traceback
        traceback.print_exc()



