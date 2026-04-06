from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = ROOT / "main.py"
QUIZ_SERVER_SCRIPT = ROOT / "quiz" / "test_quiz_server.py"
QA_CACHE = ROOT / "quiz" / "data" / "qa_cache.json"


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


def _start_server(host: str, port: int) -> subprocess.Popen:
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


def _capture_page_screenshot(url: str, screenshot_path: Path, *, width: int, height: int) -> None:
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                viewport={"width": int(width), "height": int(height)},
                device_scale_factor=1,
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle")
            page.screenshot(path=str(screenshot_path), full_page=False)
        finally:
            browser.close()


def _run_main(screenshot_path: Path, base_url: str, *, quiz_mode: bool, first_agent: bool) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["FULLBOT_START_CDP_BROWSER"] = "0"
    env["FULLBOT_ENABLE_RECORDER_STACK"] = "0"
    env["FULLBOT_DOM_FALLBACK_ON_DEMAND"] = "0"
    env["FULLBOT_RUNTIME_PROFILE"] = "ultra_fast"
    env["FULLBOT_ENABLE_TIMERS"] = "1"
    env["FULLBOT_HOVER_EARLY_ASYNC"] = "0"
    env["FULLBOT_CLICK_MONITOR_ENABLED"] = "0"
    env["FULLBOT_MINIMIZE_CONSOLE_ON_START"] = "0"

    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--auto",
        "--loop-count",
        "1",
        "--safe-test",
        "--disable-recorder",
        "--input-image",
        str(screenshot_path),
        "--quiz-lock-url-prefix",
        base_url,
    ]
    if quiz_mode:
        cmd.append("--quiz-mode")
    if first_agent:
        cmd.append("--first-agent")
    else:
        cmd.append("--notime")

    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen-only smoke test for quiz/test_quiz_server.py.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--type-id", type=int, default=1)
    parser.add_argument("--question", type=int, default=1)
    parser.add_argument("--viewport-width", type=int, default=1440)
    parser.add_argument("--viewport-height", type=int, default=1400)
    parser.add_argument("--screenshot", type=Path, default=ROOT / "data" / "screen" / "current_run" / "quiz_screen_only.png")
    parser.add_argument("--quiz-mode", action="store_true", default=True)
    parser.add_argument("--first-agent", action="store_true", default=True)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}/"
    scenario_url = f"{base_url}t/{int(args.type_id)}/{int(args.question)}?reset=1"

    server_proc = _start_server(args.host, args.port)
    try:
        _wait_for_server(base_url)
        _capture_page_screenshot(
            scenario_url,
            args.screenshot,
            width=int(args.viewport_width),
            height=int(args.viewport_height),
        )
        result = _run_main(
            args.screenshot,
            base_url,
            quiz_mode=bool(args.quiz_mode),
            first_agent=bool(args.first_agent),
        )
        payload = {
            "scenario_url": scenario_url,
            "screenshot": str(args.screenshot),
            "returncode": int(result.returncode),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
        if int(result.returncode) != 0:
            raise SystemExit(result.returncode)
    finally:
        _terminate(server_proc)


if __name__ == "__main__":
    main()
