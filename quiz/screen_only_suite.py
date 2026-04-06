from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from playwright.sync_api import sync_playwright
from scripts.brain.runtime.decision_core import build_decision_core

MAIN_SCRIPT = ROOT / "main.py"
QUIZ_SERVER_SCRIPT = ROOT / "quiz" / "test_quiz_server.py"
LOG_FILE = ROOT / "quiz" / "logs" / "quiz_submissions.log"
QA_CACHE = ROOT / "quiz" / "data" / "qa_cache.json"
SESSION_ROOT = ROOT / "quiz" / "logs" / "screen_only_suite"
BRAIN_STATE_FILE = ROOT / "data" / "brain_state.json"
CURRENT_RUN_DIR = ROOT / "data" / "screen" / "current_run"
CURRENT_RUN_PARSE_V2 = CURRENT_RUN_DIR / "screen_quiz_parse_v2.json"
CURRENT_RUN_PARSE = CURRENT_RUN_DIR / "screen_quiz_parse.json"
CHROME_CANDIDATES = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
)

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


def _capture_page_screenshot(url: str, screenshot_path: Path, *, width: int, height: int) -> None:
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        launch_kwargs: Dict[str, Any] = {"headless": True}
        for candidate in CHROME_CANDIDATES:
            if candidate.exists():
                launch_kwargs["executable_path"] = str(candidate)
                break
        browser = p.chromium.launch(**launch_kwargs)
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


def _run_main(
    screenshot_path: Path,
    *,
    scenario_url: str,
    base_url: str,
    main_python: Path,
    interval: float,
    main_timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["FULLBOT_ENVIRONMENT"] = "test"
    env["FULLBOT_ENABLE_RECORDER_STACK"] = "0"
    env["FULLBOT_START_CDP_BROWSER"] = "0"
    env["FULLBOT_RECORDER_HEADLESS"] = "1"
    env["FULLBOT_RECORDER_WINDOW_WIDTH"] = str(int(1440))
    env["FULLBOT_RECORDER_WINDOW_HEIGHT"] = str(int(1400))
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
        "--disable-recorder",
        "--loop-count",
        "1",
        "--interval",
        str(float(interval)),
        "--quiz-mode",
        "--quiz-suite",
        "--quiz-answer-cache",
        str(QA_CACHE),
        "--quiz-lock-url-prefix",
        base_url,
        "--input-image",
        str(screenshot_path),
        "--recorder-args",
        "--url",
        scenario_url,
        "--quiz-mode",
        "--quiz-lock-url-prefix",
        base_url,
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


def _read_brain_state() -> Dict[str, Any]:
    if not BRAIN_STATE_FILE.exists():
        return {}
    try:
        data = json.loads(BRAIN_STATE_FILE.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_current_screen_state() -> Dict[str, Any]:
    for candidate in (CURRENT_RUN_PARSE_V2, CURRENT_RUN_PARSE):
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if isinstance(data, dict):
            if "screen_state" in data and isinstance(data.get("screen_state"), dict):
                data = dict(data["screen_state"])
            else:
                data = dict(data)
            confidence = data.get("confidence")
            if isinstance(confidence, (int, float)):
                data["confidence"] = {"screen": float(confidence), "merged": float(confidence)}
            elif not isinstance(confidence, dict):
                data["confidence"] = {"screen": 0.0, "merged": 0.0}
            return data
    return {}


def _read_submission_count() -> int:
    if not LOG_FILE.exists():
        return 0
    try:
        with LOG_FILE.open("r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for line in fh if line.strip())
    except Exception:
        return 0


def _bbox_center(bbox: Any) -> Optional[tuple[float, float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _question_index_from_url(url: str) -> Optional[int]:
    try:
        parsed = urlparse(url)
        match = re.search(r"/t/\d+/(\d+)$", parsed.path or "")
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None


def _is_home_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.path in {"", "/", "/index", "/index.html"}
    except Exception:
        return False


def _fallback_actions_from_brain(brain: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    objects = brain.get("objects") if isinstance(brain.get("objects"), dict) else {}
    rec = str(brain.get("recommended_action") or "").strip().lower()
    answer_bbox = None
    next_bbox = None
    if isinstance(objects, dict):
        answers = objects.get("answers")
        if isinstance(answers, list) and answers:
            first = answers[0]
            if isinstance(first, dict):
                answer_bbox = first.get("bbox")
        next_obj = objects.get("next")
        if isinstance(next_obj, dict):
            next_bbox = next_obj.get("bbox")
    if rec == "click_answer" and answer_bbox:
        actions.append({"kind": "screen_click", "bbox": answer_bbox, "reason": "fallback_click_answer"})
        if next_bbox:
            actions.append({"kind": "wait", "amount": 120, "reason": "fallback_wait_before_next"})
            actions.append({"kind": "screen_click", "bbox": next_bbox, "reason": "fallback_click_next"})
    elif rec == "click_next" and next_bbox:
        actions.append({"kind": "screen_click", "bbox": next_bbox, "reason": "fallback_click_next"})
    elif rec == "scroll_page_down":
        actions.append({"kind": "screen_scroll", "bbox": None, "direction": "down", "amount": 4, "reason": "fallback_scroll"})
    elif rec == "click_cookies_accept":
        cookies = objects.get("cookies") if isinstance(objects, dict) else None
        if isinstance(cookies, dict) and cookies.get("bbox"):
            actions.append({"kind": "screen_click", "bbox": cookies.get("bbox"), "reason": "fallback_click_cookies"})
    return actions


def _execute_actions_on_page(page, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    executed: List[Dict[str, Any]] = []
    for action in actions:
        kind = str(action.get("kind") or "").strip().lower()
        if kind == "screen_click":
            center = _bbox_center(action.get("bbox"))
            if center is not None:
                x, y = center
                page.mouse.move(float(x), float(y))
                page.mouse.click(float(x), float(y), delay=25)
        elif kind == "screen_scroll":
            center = _bbox_center(action.get("bbox"))
            if center is not None:
                page.mouse.move(float(center[0]), float(center[1]))
            direction = str(action.get("direction") or "down").strip().lower()
            amount = max(1, int(action.get("amount") or 1))
            delta = 120 * amount
            if direction == "up":
                delta = -delta
            page.mouse.wheel(0, float(delta))
        elif kind == "key_press":
            combo = str(action.get("combo") or "").strip()
            if combo:
                page.keyboard.press(combo)
        elif kind == "key_repeat":
            combo = str(action.get("combo") or "").strip()
            repeat = max(1, int(action.get("repeat") or 1))
            for _ in range(repeat):
                if combo:
                    page.keyboard.press(combo)
        elif kind == "type_text":
            text = str(action.get("text") or "")
            if text:
                page.keyboard.type(text, delay=0)
        elif kind == "wait":
            ms = max(0, int(action.get("amount") or action.get("metadata", {}).get("ms") or 0))
            if ms:
                page.wait_for_timeout(ms)
        elif kind == "noop":
            pass
        else:
            # Hard stop for unsupported hidden-structure actions in screen-only mode.
            raise RuntimeError(f"Unsupported screen-only action kind: {kind or '<empty>'}")
        executed.append(dict(action))
    return executed


def _boxed_run_page(
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
    step_log: List[Dict[str, Any]] = []
    current_url = scenario_url
    current_question = int(question)
    start_submissions = _read_submission_count()
    runner_state: Dict[str, Any] = {}
    max_steps = 24
    result: Dict[str, Any] = {
        "type_id": int(type_id),
        "question": int(question),
        "scenario_url": scenario_url,
        "steps": step_log,
        "screenshot": None,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "ok": False,
        "submission_count_start": start_submissions,
        "submission_count_end": start_submissions,
    }

    with sync_playwright() as p:
        launch_kwargs: Dict[str, Any] = {"headless": bool(headless)}
        for candidate in CHROME_CANDIDATES:
            if candidate.exists():
                launch_kwargs["executable_path"] = str(candidate)
                break
        browser = p.chromium.launch(**launch_kwargs)
        try:
            context = browser.new_context(
                viewport={"width": int(viewport_width), "height": int(viewport_height)},
                device_scale_factor=1,
            )
            page = context.new_page()
            page.goto(current_url, wait_until="networkidle")
            for step_idx in range(max_steps):
                screenshot_path = session_dir / f"type{int(type_id):02d}_q{int(current_question):02d}_step{step_idx:02d}.png"
                page.screenshot(path=str(screenshot_path), full_page=False)
                result["screenshot"] = str(screenshot_path)

                main_result = _run_main(
                    screenshot_path,
                    scenario_url=page.url,
                    base_url=base_url,
                    main_python=main_python,
                    interval=interval,
                    main_timeout_s=main_timeout_s,
                )
                screen_state = _load_current_screen_state()
                decision = build_decision_core(
                    cache_path=QA_CACHE,
                    screen_state=screen_state or {},
                    prev_state=runner_state,
                    controls_data=None,
                    page_data=None,
                )
                actions = list(decision.actions or [])
                if not actions:
                    actions = _fallback_actions_from_brain(decision.brain_state or screen_state or {})

                before_submissions = _read_submission_count()
                before_url = page.url
                try:
                    executed = _execute_actions_on_page(page, actions)
                except Exception as exc:
                    step_log.append(
                        {
                            "step": int(step_idx),
                            "url_before": before_url,
                            "main_returncode": int(main_result.returncode),
                            "action_count": len(actions),
                            "error": str(exc),
                            "brain": {
                                "recommended_action": (decision.brain_state or {}).get("recommended_action"),
                                "question_text": (decision.brain_state or {}).get("question_text"),
                            },
                        }
                    )
                    result["returncode"] = 1
                    result["stdout"] = main_result.stdout
                    result["stderr"] = (main_result.stderr or "") + f"\n{exc}"
                    result["steps"] = step_log
                    return result

                with contextlib.suppress(Exception):
                    page.wait_for_load_state("networkidle", timeout=1500)
                page.wait_for_timeout(max(50, int(interval * 100)))
                after_submissions = _read_submission_count()
                after_url = page.url
                current_question_candidate = _question_index_from_url(after_url)
                if current_question_candidate is not None:
                    current_question = current_question_candidate

                step_log.append(
                    {
                        "step": int(step_idx),
                        "url_before": before_url,
                        "url_after": after_url,
                        "main_returncode": int(main_result.returncode),
                        "action_count": len(actions),
                        "executed_count": len(executed),
                        "progress": bool(after_submissions > before_submissions or after_url != before_url),
                        "submissions_before": int(before_submissions),
                        "submissions_after": int(after_submissions),
                        "brain": {
                            "recommended_action": (decision.brain_state or {}).get("recommended_action"),
                            "question_text": (decision.brain_state or {}).get("question_text"),
                            "detected_quiz_type": (decision.brain_state or {}).get("detected_quiz_type"),
                            "detected_operational_type": (decision.brain_state or {}).get("detected_operational_type"),
                            "answer_clicked": (decision.brain_state or {}).get("answer_clicked"),
                            "next_clicked": (decision.brain_state or {}).get("next_clicked"),
                            "has_answers": (decision.brain_state or {}).get("has_answers"),
                            "has_next": (decision.brain_state or {}).get("has_next"),
                        },
                    }
                )
                runner_state = dict(decision.brain_state or runner_state)
                result["stdout"] = main_result.stdout
                result["stderr"] = main_result.stderr
                result["returncode"] = int(main_result.returncode)
                result["submission_count_end"] = int(after_submissions)
                if _is_home_url(after_url):
                    result["ok"] = True
                    result["steps"] = step_log
                    return result
                if after_submissions >= start_submissions + 5:
                    # The quiz server logs one line per answered question.
                    result["ok"] = True
                    result["steps"] = step_log
                    return result
                if not bool(after_submissions > before_submissions or after_url != before_url):
                    # No observable progress from the screen-only loop.
                    break
            result["steps"] = step_log
            result["ok"] = bool(_is_home_url(page.url) or _read_submission_count() >= start_submissions + 5)
            return result
        finally:
            browser.close()


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
    return _boxed_run_page(
        session_dir=session_dir,
        base_url=base_url,
        main_python=main_python,
        type_id=type_id,
        question=question,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        interval=interval,
        headless=headless,
        main_timeout_s=main_timeout_s,
    )

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
