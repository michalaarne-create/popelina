from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

try:
    from mss import mss, tools
except Exception:  # pragma: no cover - optional fallback
    mss = None  # type: ignore[assignment]
    tools = None  # type: ignore[assignment]

try:
    from PIL import ImageGrab
except Exception:  # pragma: no cover - optional fallback
    ImageGrab = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MAIN_SCRIPT = ROOT / "main.py"
QUIZ_SERVER_SCRIPT = ROOT / "quiz" / "test_quiz_server.py"
LOG_FILE = ROOT / "quiz" / "logs" / "quiz_submissions.log"
QA_CACHE = ROOT / "quiz" / "data" / "qa_cache.json"
SESSION_ROOT = ROOT / "quiz" / "logs" / "screen_only_suite"

try:
    from quiz.run_test_suite import validate_log
except Exception:
    from run_test_suite import validate_log


def _probe(url: str, timeout: float = 0.5) -> bool:
    try:
        with urlopen(url, timeout=timeout) as response:
            status = int(getattr(response, "status", 0) or 0)
            return 200 <= status < 500
    except Exception:
        return False


def _wait_for_server(base_url: str, timeout_s: float = 15.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _probe(base_url):
            return
        time.sleep(0.25)
    raise RuntimeError(f"Quiz server did not start at {base_url} within {timeout_s:.1f}s")


def _start_server() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(QUIZ_SERVER_SCRIPT)],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _terminate(proc: Optional[subprocess.Popen], timeout: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


def _clear_log(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")


def _archive_log(log_path: Path) -> None:
    if not log_path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = log_path.with_name(f"{log_path.stem}_{stamp}{log_path.suffix}")
    log_path.replace(archive)


def _screen_capture(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mss is not None and tools is not None:
        with mss() as sct:
            monitor = sct.monitors[0]
            raw = sct.grab(monitor)
            tools.to_png(raw.rgb, raw.size, output=str(target))
            return target
    if ImageGrab is not None:
        img = ImageGrab.grab(all_screens=True)
        img.save(target)
        return target
    raise RuntimeError("No screen capture backend available (mss/ImageGrab missing).")


def _launch_visible_browser(url: str, *, width: int, height: int):
    pw = sync_playwright().start()
    browser = pw.chromium.launch(
        headless=False,
        args=["--start-maximized"],
    )
    context = browser.new_context(
        viewport={"width": int(width), "height": int(height)},
        device_scale_factor=1,
    )
    page = context.new_page()
    page.goto(url, wait_until="networkidle")
    time.sleep(0.5)
    return pw, browser, context, page


def _run_main(
    screenshot_path: Path,
    *,
    base_url: str,
    main_python: Path,
    interval: float,
    headless: int,
    viewport_width: int,
    viewport_height: int,
    main_timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["FULLBOT_ENVIRONMENT"] = "test"
    env["FULLBOT_START_CDP_BROWSER"] = "0"
    env["FULLBOT_ENABLE_RECORDER_STACK"] = "0"
    env["FULLBOT_DOM_FALLBACK_ON_DEMAND"] = "0"
    env["FULLBOT_DOM_FALLBACK_ACTIVE"] = "0"
    env["FULLBOT_RUNTIME_PROFILE"] = "ultra_fast"
    env["FULLBOT_ENABLE_TIMERS"] = "1"
    env["FULLBOT_HOVER_EARLY_ASYNC"] = "0"
    env["FULLBOT_CLICK_MONITOR_ENABLED"] = "0"
    env["FULLBOT_MINIMIZE_CONSOLE_ON_START"] = "0"

    cmd = [
        str(main_python),
        str(MAIN_SCRIPT),
        "--auto",
        "--notime",
        "--safe-test",
        "--interval",
        str(float(interval)),
        "--headless",
        str(int(headless)),
        "--width",
        str(int(viewport_width)),
        "--height",
        str(int(viewport_height)),
        "--quiz-mode",
        "--quiz-suite",
        "--quiz-answer-cache",
        str(QA_CACHE),
        "--quiz-lock-url-prefix",
        base_url,
        "--disable-recorder",
        "--input-image",
        str(screenshot_path),
    ]

    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=max(30.0, float(main_timeout_s)),
        check=False,
    )


def _scenario_url(base_url: str, type_id: int, question: int) -> str:
    return f"{base_url}t/{int(type_id)}/{int(question)}?reset=1"


def _run_one(
    *,
    session_dir: Path,
    base_url: str,
    main_python: Path,
    type_id: int,
    question: int,
    viewport_width: int,
    viewport_height: int,
    interval: float,
    headless: int,
    main_timeout_s: float,
) -> Dict[str, Any]:
    scenario_url = _scenario_url(base_url, type_id, question)
    shot_path = session_dir / f"type{type_id:02d}_q{question:02d}.png"
    pw = browser = context = page = None
    try:
        pw, browser, context, page = _launch_visible_browser(
            scenario_url,
            width=viewport_width,
            height=viewport_height,
        )
        _screen_capture(shot_path)
        result = _run_main(
            shot_path,
            base_url=base_url,
            main_python=main_python,
            interval=interval,
            headless=headless,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            main_timeout_s=main_timeout_s,
        )
        return {
            "type_id": int(type_id),
            "question": int(question),
            "scenario_url": scenario_url,
            "screenshot": str(shot_path),
            "returncode": int(result.returncode),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "ok": int(result.returncode) == 0,
        }
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        if pw is not None:
            try:
                pw.stop()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Full screen-only quiz suite for test_quiz_server.py.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--start-type", type=int, default=1)
    parser.add_argument("--end-type", type=int, default=13)
    parser.add_argument("--only-type", type=int, nargs="*", default=None)
    parser.add_argument("--question", type=int, default=1)
    parser.add_argument("--viewport-width", type=int, default=1440)
    parser.add_argument("--viewport-height", type=int, default=1400)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--main-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--main-timeout-s", type=float, default=90.0)
    parser.add_argument("--archive-log", action="store_true")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}/"
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = SESSION_ROOT / session_stamp
    session_dir.mkdir(parents=True, exist_ok=True)

    server_proc = _start_server()
    try:
        _wait_for_server(base_url)
        if args.archive_log:
            _archive_log(LOG_FILE)
        _clear_log(LOG_FILE)

        type_ids = list(range(max(1, int(args.start_type)), max(1, int(args.end_type)) + 1))
        if args.only_type:
            filtered: List[int] = []
            for item in args.only_type:
                try:
                    idx = int(item)
                except Exception:
                    continue
                if idx not in filtered and idx >= 1:
                    filtered.append(idx)
            if filtered:
                type_ids = filtered

        results: List[Dict[str, Any]] = []
        for type_id in type_ids:
            result = _run_one(
                session_dir=session_dir,
                base_url=base_url,
                main_python=Path(args.main_python),
                type_id=type_id,
                question=int(args.question),
                viewport_width=int(args.viewport_width),
                viewport_height=int(args.viewport_height),
                interval=float(args.interval),
                headless=int(args.headless),
                main_timeout_s=float(args.main_timeout_s),
            )
            results.append(result)
            if result["ok"]:
                continue
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(int(result["returncode"] or 1))

        validation = validate_log(LOG_FILE, type_ids=type_ids)
        payload = {
            "base_url": base_url,
            "type_ids": type_ids,
            "results": results,
            "validation": validation,
            "session_dir": str(session_dir),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if any(not item.get("ok") for item in results) or not validation.get("passed"):
            raise SystemExit(1)
    finally:
        _terminate(server_proc)


if __name__ == "__main__":
    main()
