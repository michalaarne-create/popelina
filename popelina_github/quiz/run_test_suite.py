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

try:
    from .test_quiz_server import LOG_FILE, run as run_quiz_server
    from .validate_quiz_run import validate_log
except Exception:
    from test_quiz_server import LOG_FILE, run as run_quiz_server
    from validate_quiz_run import validate_log


ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = ROOT / "main.py"
CURRENT_PAGE_PATH = ROOT / "data" / "screen" / "page_current" / "page_current.json"
QA_CACHE = ROOT / "quiz" / "data" / "qa_cache.json"


def _probe(url: str, timeout: float = 0.5) -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = int(getattr(response, "status", 0) or 0)
            return 200 <= status < 500
    except Exception:
        return False


def _ensure_server(host: str, port: int) -> Optional[subprocess.Popen]:
    base_url = f"http://{host}:{port}/"
    if _probe(base_url):
        return None
    return subprocess.Popen(
        [sys.executable, str(ROOT / "quiz" / "test_quiz_server.py")],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_for_server(host: str, port: int, timeout_s: float = 15.0) -> None:
    base_url = f"http://{host}:{port}/"
    start = time.time()
    while (time.time() - start) < timeout_s:
        if _probe(base_url):
            return
        time.sleep(0.25)
    raise RuntimeError(f"Quiz server did not start at {base_url} within {timeout_s:.1f}s")


def _archive_log(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = log_path.with_name(f"{log_path.stem}_{stamp}{log_path.suffix}")
    log_path.replace(archive)


def _clear_log(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")


def _current_url() -> str:
    if not CURRENT_PAGE_PATH.exists():
        return ""
    try:
        payload = json.loads(CURRENT_PAGE_PATH.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return ""
    return str(payload.get("url") or "")


def _terminate(proc: Optional[subprocess.Popen], timeout: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


def _run_scenario(
    *,
    scenario_url: str,
    base_url: str,
    interval: float,
    timeout_s: float,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--auto",
        "--interval",
        str(interval),
        "--quiz-mode",
        "--quiz-suite",
        "--quiz-answer-cache",
        str(QA_CACHE),
        "--quiz-lock-url-prefix",
        base_url,
        "--recorder-args",
        "--url",
        scenario_url,
        "--quiz-mode",
        "--quiz-lock-url-prefix",
        base_url,
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    start = time.time()
    seen_quiz_page = False
    timed_out = False
    try:
        while (time.time() - start) < timeout_s:
            url = _current_url()
            if url.startswith(base_url) and url.rstrip("/") != base_url.rstrip("/"):
                seen_quiz_page = True
            if seen_quiz_page and url.rstrip("/") == base_url.rstrip("/"):
                break
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        else:
            timed_out = True
    finally:
        _terminate(proc)
    return {
        "scenario_url": scenario_url,
        "timed_out": timed_out,
        "elapsed_s": round(time.time() - start, 3),
        "seen_quiz_page": seen_quiz_page,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full test_quiz_server.py suite through main.py in quiz mode.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout-per-scenario", type=float, default=90.0)
    parser.add_argument("--archive-log", action="store_true")
    parser.add_argument("--start-type", type=int, default=1)
    parser.add_argument("--end-type", type=int, default=13)
    parser.add_argument("--only-type", type=int, nargs="*", default=None)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}/"
    server_proc = _ensure_server(args.host, args.port)
    _wait_for_server(args.host, args.port)
    if args.archive_log:
        _archive_log(LOG_FILE)
    _clear_log(LOG_FILE)

    scenario_ids = list(range(max(1, int(args.start_type)), min(13, int(args.end_type)) + 1))
    if args.only_type:
        filtered = []
        for item in args.only_type:
            try:
                idx = int(item)
            except Exception:
                continue
            if 1 <= idx <= 13 and idx not in filtered:
                filtered.append(idx)
        scenario_ids = filtered

    results: List[Dict[str, Any]] = []
    try:
        for idx in scenario_ids:
            scenario_url = f"{base_url}t/{idx}/1?reset=1"
            results.append(
                _run_scenario(
                    scenario_url=scenario_url,
                    base_url=base_url,
                    interval=args.interval,
                    timeout_s=args.timeout_per_scenario,
                )
            )
    finally:
        _terminate(server_proc)

    validation = validate_log(LOG_FILE, type_ids=scenario_ids)
    payload = {
        "base_url": base_url,
        "scenario_ids": scenario_ids,
        "results": results,
        "validation": validation,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if any(item.get("timed_out") for item in results) or not validation.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
