from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore[assignment]

from scripts.runtime_security.operational_watchdog_policy import build_failure_summary_contract


ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = ROOT / "main.py"
QUIZ_SERVER_SCRIPT = ROOT / "quiz" / "test_quiz_server.py"
QA_CACHE = ROOT / "quiz" / "data" / "qa_cache.json"
SESSIONS_DIR = ROOT / "quiz" / "logs" / "dev_sessions"

PAGE_CURRENT_PATH = ROOT / "data" / "screen" / "page_current" / "page_current.json"
DOM_CURRENT_PAGE_PATH = ROOT / "scripts" / "dom" / "dom_live" / "current_page.json"
RAW_SCREENSHOT_PATH = ROOT / "data" / "screen" / "current_run" / "screenshot.png"
REGION_GROW_CURRENT_PATH = ROOT / "data" / "screen" / "region_grow" / "region_grow_current" / "region_grow.json"
FAST_SUMMARY_PATH = ROOT / "data" / "screen" / "region_grow" / "region_grow_current" / "fast_summary.json"
CURRENT_QUESTION_PATH = ROOT / "scripts" / "dom" / "dom_live" / "current_question.json"
CURRENT_CONTROLS_PATH = ROOT / "scripts" / "dom" / "dom_live" / "current_controls.json"
CURRENT_RUN_PARSE_PATH = ROOT / "data" / "screen" / "current_run" / "screen_quiz_parse.json"
QUIZ_TRACE_PATH = ROOT / "data" / "screen" / "current_run" / "quiz_trace.jsonl"
BRAIN_STATE_PATH = ROOT / "data" / "brain_state.json"
RATE_SUMMARY_CURRENT_DIR = ROOT / "data" / "screen" / "rate" / "rate_summary_current"
CONTROL_AGENT_PORT = int(os.environ.get("CONTROL_AGENT_PORT", "8765") or "8765")
DEFAULT_HEADLESS = 1
DEFAULT_VIEWPORT_WIDTH = 1440
DEFAULT_VIEWPORT_HEIGHT = 1400
VIEWPORT_PROFILES: Dict[str, Dict[str, Any]] = {
    "desktop_standard": {"width": 1440, "height": 1400, "is_mobile_narrow": False},
    "tablet_portrait": {"width": 1024, "height": 1366, "is_mobile_narrow": False},
    "mobile_narrow": {"width": 430, "height": 932, "is_mobile_narrow": True},
}


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _print(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _clip(text: Any, limit: int = 120) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _latest_json(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    candidates = sorted((p for p in directory.glob("*.json") if p.is_file()), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _summarize_page(payload: Dict[str, Any]) -> str:
    return (
        "page "
        f"url={_clip(payload.get('url') or '', 90)} "
        f"title={_clip(payload.get('title') or '', 60)} "
        f"sig={_clip(payload.get('page_signature') or payload.get('tracking', {}).get('active_page_id') or '', 32)}"
    )


def _artifact_freshness(path: Path, session_start: float, fresh_window_s: float = 2.0) -> str:
    if not path.exists():
        return "missing"
    try:
        mtime = float(path.stat().st_mtime)
    except Exception:
        return "missing"
    if mtime < session_start:
        return "stale_previous_run"
    if (time.time() - mtime) <= max(0.1, float(fresh_window_s)):
        return "fresh_for_session"
    return "fresh_for_session"


def _port_alive(port: int) -> bool:
    try:
        import socket

        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.4):
            return True
    except Exception:
        return False


def _probe_http_alive(url: str, timeout_s: float = 0.6) -> bool:
    try:
        import urllib.request

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            status = int(getattr(response, "status", 0) or 0)
            return 200 <= status < 500
    except Exception:
        return False


def _python_has_module(python_path: Path, module_name: str) -> bool:
    if not python_path.exists():
        return False
    try:
        result = subprocess.run(
            [
                str(python_path),
                "-c",
                f"import importlib.util as u, sys; sys.exit(0 if u.find_spec({module_name!r}) else 1)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(ROOT),
            timeout=8.0,
            check=False,
        )
        return int(result.returncode) == 0
    except Exception:
        return False


def _candidate_python(raw: str) -> Optional[Path]:
    value = str(raw or "").strip()
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _resolve_recorder_python() -> Path:
    candidates = [
        _candidate_python(os.environ.get("FULLBOT_RECORDER_PYTHON", "")),
        ROOT.parent / ".venv312" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists() and _python_has_module(candidate, "playwright"):
            return candidate
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return Path(sys.executable)


def _resolve_region_grow_python() -> Path:
    candidates = [
        _candidate_python(os.environ.get("FULLBOT_REGION_GROW_PYTHON", "")),
        ROOT.parent / ".venv312" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists() and _python_has_module(candidate, "cupy"):
            return candidate
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return Path(sys.executable)


def _normalize_environment(raw: Any) -> str:
    value = str(raw or os.environ.get("FULLBOT_ENVIRONMENT") or "test").strip().lower()
    if value in {"prod", "production"}:
        return "production"
    if value in {"stage", "staging"}:
        return "staging"
    if value in {"dev", "development"}:
        return "development"
    return "test"


def _resolve_viewport_profile(args: argparse.Namespace) -> Dict[str, Any]:
    requested = str(getattr(args, "viewport_profile", "") or "").strip().lower()
    if requested and requested in VIEWPORT_PROFILES:
        profile = dict(VIEWPORT_PROFILES[requested])
        profile["name"] = requested
        profile["source"] = "preset"
        return profile

    raw_viewport_width = getattr(args, "viewport_width", DEFAULT_VIEWPORT_WIDTH)
    raw_viewport_height = getattr(args, "viewport_height", DEFAULT_VIEWPORT_HEIGHT)
    width = max(1, int(DEFAULT_VIEWPORT_WIDTH if raw_viewport_width is None else raw_viewport_width))
    height = max(1, int(DEFAULT_VIEWPORT_HEIGHT if raw_viewport_height is None else raw_viewport_height))
    is_mobile_narrow = width <= 768
    inferred_name = "mobile_narrow" if is_mobile_narrow else "desktop_standard"
    return {
        "name": inferred_name,
        "width": width,
        "height": height,
        "is_mobile_narrow": is_mobile_narrow,
        "source": "explicit_dimensions",
    }


def _build_browser_matrix_contract(args: argparse.Namespace) -> Dict[str, Any]:
    environment = _normalize_environment(getattr(args, "environment", ""))
    raw_headless = getattr(args, "headless", DEFAULT_HEADLESS)
    headless = bool(int(DEFAULT_HEADLESS if raw_headless is None else raw_headless))
    viewport_profile = _resolve_viewport_profile(args)
    viewport_width = int(viewport_profile["width"])
    viewport_height = int(viewport_profile["height"])
    is_mobile_narrow = bool(viewport_profile["is_mobile_narrow"])
    runtime_mode = "playwright_headless" if headless else "playwright_windowed"
    viewport_profile_name = str(viewport_profile["name"])
    matrix_key = f"chromium:{runtime_mode}:{viewport_profile_name}:{viewport_width}x{viewport_height}:{environment}"
    return {
        "browser_family": "chromium",
        "runtime_mode": runtime_mode,
        "headless": headless,
        "viewport": {
            "width": viewport_width,
            "height": viewport_height,
            "is_mobile_narrow": is_mobile_narrow,
        },
        "viewport_profile": viewport_profile_name,
        "viewport_profile_source": str(viewport_profile["source"]),
        "viewport_profile_registry": sorted(VIEWPORT_PROFILES.keys()),
        "environment": environment,
        "matrix_key": matrix_key,
    }


def _build_preflight(args: argparse.Namespace, base_url: str) -> Dict[str, Any]:
    recorder_python = _resolve_recorder_python()
    region_python = _resolve_region_grow_python()
    main_python = Path(args.main_python)
    browser_matrix = _build_browser_matrix_contract(args)
    return {
        "server": {
            "base_url": base_url,
            "http_alive": bool(_probe_http_alive(base_url, timeout_s=0.5)),
            "port_alive": bool(_port_alive(args.port)),
        },
        "control_agent": {
            "port": CONTROL_AGENT_PORT,
            "alive": bool(_port_alive(CONTROL_AGENT_PORT)),
        },
        "interpreters": {
            "main": {
                "path": str(main_python),
                "playwright": _python_has_module(main_python, "playwright"),
                "paddleocr": _python_has_module(main_python, "paddleocr"),
                "cupy": _python_has_module(main_python, "cupy"),
            },
            "recorder": {
                "path": str(recorder_python),
                "playwright": _python_has_module(recorder_python, "playwright"),
                "paddleocr": _python_has_module(recorder_python, "paddleocr"),
                "cupy": _python_has_module(recorder_python, "cupy"),
            },
            "region_grow": {
                "path": str(region_python),
                "playwright": _python_has_module(region_python, "playwright"),
                "paddleocr": _python_has_module(region_python, "paddleocr"),
                "cupy": _python_has_module(region_python, "cupy"),
            },
        },
        "artifacts": {
            "dom_current_page": str(DOM_CURRENT_PAGE_PATH),
            "current_controls": str(CURRENT_CONTROLS_PATH),
            "current_question": str(CURRENT_QUESTION_PATH),
            "screenshot": str(RAW_SCREENSHOT_PATH),
        },
        "browser_runtime": {
            "browser_family": browser_matrix["browser_family"],
            "runtime_mode": browser_matrix["runtime_mode"],
            "headless": browser_matrix["headless"],
            "viewport": browser_matrix["viewport"],
            "viewport_profile": browser_matrix["viewport_profile"],
            "matrix_key": browser_matrix["matrix_key"],
        },
        "environment_profile": {
            "environment": browser_matrix["environment"],
        },
    }


def _print_preflight(preflight: Dict[str, Any]) -> None:
    server = preflight.get("server") or {}
    control = preflight.get("control_agent") or {}
    interps = preflight.get("interpreters") or {}
    browser_runtime = preflight.get("browser_runtime") or {}
    environment_profile = preflight.get("environment_profile") or {}
    _print(
        "preflight "
        f"server_http={int(bool(server.get('http_alive')))} "
        f"server_port={int(bool(server.get('port_alive')))} "
        f"control_agent={int(bool(control.get('alive')))}"
    )
    for key in ("main", "recorder", "region_grow"):
        row = interps.get(key) or {}
        _print(
            f"python[{key}] path={row.get('path')} "
            f"playwright={int(bool(row.get('playwright')))} "
            f"paddleocr={int(bool(row.get('paddleocr')))} "
            f"cupy={int(bool(row.get('cupy')))}"
        )
    viewport = browser_runtime.get("viewport") or {}
    _print(
        "browser_runtime "
        f"family={browser_runtime.get('browser_family')} "
        f"mode={browser_runtime.get('runtime_mode')} "
        f"viewport={int(viewport.get('width') or 0)}x{int(viewport.get('height') or 0)} "
        f"profile={browser_runtime.get('viewport_profile')} "
        f"environment={environment_profile.get('environment')}"
    )


def _detect_screen_has_quiz(path: Path) -> bool:
    if not path.exists() or Image is None:
        return False
    try:
        with Image.open(path) as img:
            width = int(img.width)
            height = int(img.height)
        if width < 600 or height < 300:
            return False
    except Exception:
        return False
    region = _safe_read_json(REGION_GROW_CURRENT_PATH) or {}
    results = region.get("results") or []
    if isinstance(results, list) and len(results) >= 3:
        return True
    parse_payload = _safe_read_json(CURRENT_RUN_PARSE_PATH) or {}
    screen_state = parse_payload.get("screen_state") or {}
    if str(screen_state.get("question_text") or "").strip():
        return True
    return bool(path.exists() and path.stat().st_size > 80000)


def _compute_stop_reason(
    *,
    session_start: float,
    main_returncode: Optional[int],
    main_log_path: Path,
    scenario_url: str,
    timed_out: bool,
) -> str:
    if main_returncode is not None and int(main_returncode) != 0:
        log_text = _safe_read_text(main_log_path)
        if "KeyError" in log_text or "Traceback" in log_text:
            return "executor_failed"
        if not timed_out:
            return "main_exited_nonzero"

    controls = _safe_read_json(CURRENT_CONTROLS_PATH) or {}
    question = _safe_read_json(CURRENT_QUESTION_PATH) or {}
    dom_page = _safe_read_json(DOM_CURRENT_PAGE_PATH) or {}
    region = _safe_read_json(REGION_GROW_CURRENT_PATH) or {}
    brain = _safe_read_json(BRAIN_STATE_PATH) or {}

    controls_fresh = _artifact_freshness(CURRENT_CONTROLS_PATH, session_start)
    screenshot_fresh = _artifact_freshness(RAW_SCREENSHOT_PATH, session_start)
    region_fresh = _artifact_freshness(REGION_GROW_CURRENT_PATH, session_start)

    dom_qid = str(((controls.get("meta") or {}).get("qid")) or "").strip()
    dom_url = str(((controls.get("page") or {}).get("url")) or str(dom_page.get("url") or "")).strip()
    dom_has_quiz = bool((controls.get("question_blocks") or []) or dom_qid)
    screen_has_quiz = bool(screenshot_fresh == "fresh_for_session" and _detect_screen_has_quiz(RAW_SCREENSHOT_PATH))
    screen_dom_consistent = bool(dom_has_quiz and screen_has_quiz)

    if controls_fresh == "missing":
        return "recorder_not_ready"
    if controls_fresh == "fresh_for_session" and dom_url:
        if scenario_url.rstrip("/") != dom_url.rstrip("/") and screen_has_quiz:
            return "screen_wrong_tab"
    if dom_has_quiz and not screen_has_quiz and screenshot_fresh == "fresh_for_session":
        return "screen_wrong_tab"
    if dom_has_quiz and question == {}:
        return "question_projection_missing"
    if screen_has_quiz and region_fresh == "fresh_for_session" and not (region.get("results") or []):
        return "region_grow_empty"
    if screen_dom_consistent:
        actions = brain.get("actions") or []
        recommended = str(brain.get("recommended_action") or "").strip()
        if not actions and not recommended:
            return "brain_no_action"
    if dom_page == {} and controls_fresh == "fresh_for_session":
        return "ready_by_controls"
    return "time_budget_exceeded"


def _artifact_snapshot(session_start: float) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for name, path in {
        "page_current": PAGE_CURRENT_PATH,
        "dom_current_page": DOM_CURRENT_PAGE_PATH,
        "current_question": CURRENT_QUESTION_PATH,
        "current_controls": CURRENT_CONTROLS_PATH,
        "region_grow": REGION_GROW_CURRENT_PATH,
        "fast_summary": FAST_SUMMARY_PATH,
        "brain_state": BRAIN_STATE_PATH,
        "screen_quiz_parse": CURRENT_RUN_PARSE_PATH,
        "screenshot": RAW_SCREENSHOT_PATH,
    }.items():
        freshness = _artifact_freshness(path, session_start)
        entry: Dict[str, Any] = {"path": str(path), "freshness": freshness}
        if path.exists():
            try:
                entry["mtime"] = float(path.stat().st_mtime)
            except Exception:
                pass
        snapshot[name] = entry
    return snapshot


def _summarize_question(payload: Dict[str, Any]) -> str:
    question = _clip(payload.get("question") or payload.get("text") or payload.get("question_text") or "", 120)
    answers = payload.get("answers") or payload.get("options") or []
    answer_count = len(answers) if isinstance(answers, list) else 0
    return f"question text={question} answers={answer_count}"


def _summarize_controls(payload: Dict[str, Any]) -> str:
    controls = payload.get("controls") or []
    counts: Counter[str] = Counter()
    for item in controls:
        if isinstance(item, dict):
            counts[str(item.get("kind") or "unknown")] += 1
    kinds = ", ".join(f"{kind}:{count}" for kind, count in sorted(counts.items())) or "none"
    next_id = payload.get("next_control_id") or ""
    meta = payload.get("meta") or {}
    return f"controls total={len(controls)} kinds={kinds} next={next_id or '-'} qid={meta.get('qid') or '-'}"


def _summarize_region_grow(payload: Dict[str, Any]) -> str:
    results = payload.get("results") or []
    texts: List[str] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        text = _clip(row.get("text") or row.get("box_text") or "", 24)
        if text:
            texts.append(text)
        if len(texts) >= 4:
            break
    return f"region_grow results={len(results)} sample={texts or ['-']}"


def _summarize_fast_summary(payload: Dict[str, Any]) -> str:
    conf = payload.get("confidence") or {}
    top_labels = payload.get("top_labels") or {}
    top_parts: List[str] = []
    for key in ("answer_single", "answer_multi", "dropdown", "next_active"):
        row = top_labels.get(key)
        if isinstance(row, dict):
            top_parts.append(f"{key}={_clip(row.get('text') or '', 28)}")
    return (
        "fast_summary "
        f"answer={float(conf.get('answer') or 0.0):.3f} "
        f"next={float(conf.get('next') or 0.0):.3f} "
        f"dropdown={float(conf.get('dropdown') or 0.0):.3f} "
        f"tops={top_parts or ['-']}"
    )


def _summarize_brain(payload: Dict[str, Any]) -> str:
    actions = payload.get("actions") or []
    action_kinds: List[str] = []
    for row in actions[:4]:
        if isinstance(row, dict):
            action_kinds.append(f"{row.get('kind')}:{_clip(row.get('reason') or '', 32)}")
    return (
        "brain "
        f"recommended={payload.get('recommended_action') or '-'} "
        f"question={_clip(payload.get('question_text') or '', 90)} "
        f"fallback={int(bool(payload.get('fallback_used')))} "
        f"actions={action_kinds or ['-']}"
    )


def _summarize_quiz_parse(payload: Dict[str, Any]) -> str:
    screen_state = payload.get("screen_state") or {}
    resolved = payload.get("resolved_answer") or {}
    trace = payload.get("trace") or {}
    actions = trace.get("actions") or payload.get("actions") or []
    action_kinds: List[str] = []
    if isinstance(actions, list):
        for row in actions[:4]:
            if isinstance(row, dict):
                action_kinds.append(f"{row.get('kind')}:{_clip(row.get('reason') or '', 28)}")
    answers = resolved.get("correct_answers") or []
    return (
        "quiz_parse "
        f"q={_clip(screen_state.get('question_text') or '', 90)} "
        f"kind={resolved.get('question_type') or screen_state.get('control_kind') or '-'} "
        f"resolved={answers or ['-']} "
        f"source={resolved.get('source') or '-'} "
        f"conf={float(resolved.get('confidence') or 0.0):.3f} "
        f"actions={action_kinds or ['-']}"
    )


def _summarize_trace_line(line: str) -> str:
    try:
        payload = json.loads(line)
    except Exception:
        return f"quiz_trace raw={_clip(line, 120)}"
    trace = payload.get("trace") or {}
    resolved = payload.get("resolved_answer") or {}
    return (
        "quiz_trace "
        f"q={_clip(payload.get('question_text') or '', 80)} "
        f"stage={trace.get('stage') or '-'} "
        f"answer_ready={int(bool(trace.get('answer_ready')))} "
        f"source={resolved.get('source') or '-'}"
    )


def _summarize_screenshot(path: Path) -> str:
    size_text = "unknown"
    if Image is not None:
        try:
            with Image.open(path) as img:
                size_text = f"{img.width}x{img.height}"
        except Exception:
            size_text = "unknown"
    stat = path.stat()
    return f"screenshot path={path.name} size={size_text} bytes={stat.st_size}"


def _ensure_server(host: str, port: int) -> Optional[subprocess.Popen]:
    base_url = f"http://{host}:{port}/"
    try:
        import urllib.request

        with urllib.request.urlopen(base_url, timeout=0.6) as response:
            status = int(getattr(response, "status", 0) or 0)
            if 200 <= status < 500:
                return None
    except Exception:
        pass
    return subprocess.Popen(
        [sys.executable, str(QUIZ_SERVER_SCRIPT)],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_for_server(host: str, port: int, timeout_s: float = 15.0) -> None:
    start = time.time()
    base_url = f"http://{host}:{port}/"
    while time.time() - start < timeout_s:
        try:
            import urllib.request

            with urllib.request.urlopen(base_url, timeout=0.6) as response:
                status = int(getattr(response, "status", 0) or 0)
                if 200 <= status < 500:
                    return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"Quiz server did not respond at {base_url} within {timeout_s:.1f}s")


def _terminate(proc: Optional[subprocess.Popen], timeout: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


def _default_main_python() -> Path:
    env = str(os.environ.get("FULLBOT_MAIN_PYTHON", "") or "").strip()
    if env:
        return Path(env)
    local = ROOT.parent / ".conda" / "fullbot312" / "python.exe"
    if local.exists():
        return local
    return Path(sys.executable)


def _default_harness_python() -> Path:
    env = str(os.environ.get("FULLBOT_HARNESS_PYTHON", "") or "").strip()
    if env:
        return Path(env)
    return Path(sys.executable)


def _default_direct_scenario_url(host: str, port: int, scenario_type: Optional[int]) -> str:
    scenario_idx = int(scenario_type or 1)
    return f"http://{host}:{port}/t/{scenario_idx}/1?reset=1"


def _normalize_direct_scenario_url(host: str, port: int, scenario_type: Optional[int], scenario_url: Optional[str]) -> str:
    fallback = _default_direct_scenario_url(host, port, scenario_type)
    raw = str(scenario_url or "").strip()
    if not raw:
        return fallback
    try:
        parsed = urlparse(raw)
    except Exception:
        return fallback
    path_parts = [part for part in str(parsed.path or "").split("/") if part]
    if len(path_parts) >= 3 and path_parts[0] == "t":
        query = parse_qs(parsed.query or "", keep_blank_values=True)
        if not query.get("reset"):
            query["reset"] = ["1"]
        normalized_query = urlencode(query, doseq=True)
        scheme = parsed.scheme or "http"
        netloc = parsed.netloc or f"{host}:{port}"
        return urlunparse((scheme, netloc, f"/t/{path_parts[1]}/{path_parts[2]}", "", normalized_query, ""))
    return fallback


def _scenario_url(host: str, port: int, scenario_type: Optional[int], scenario_url: Optional[str]) -> str:
    return _normalize_direct_scenario_url(host, port, scenario_type, scenario_url)


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/"


def _quiz_lock_prefix(scenario_url: str, fallback_base_url: str) -> str:
    raw = str(scenario_url or "").strip()
    try:
        parsed = urlparse(raw)
    except Exception:
        return fallback_base_url
    path_parts = [part for part in str(parsed.path or "").split("/") if part]
    if len(path_parts) >= 2 and path_parts[0] == "t":
        base = f"{parsed.scheme or 'http'}://{parsed.netloc or urlparse(fallback_base_url).netloc}/t/{path_parts[1]}/"
        return base
    return fallback_base_url


def _copy_latest(path: Path, latest_dir: Path, archive_dir: Optional[Path]) -> None:
    latest_dir.mkdir(parents=True, exist_ok=True)
    target = latest_dir / path.name
    try:
        shutil.copy2(path, target)
    except Exception:
        return
    if archive_dir is None:
        return
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archived = archive_dir / f"{stamp}_{path.name}"
    try:
        shutil.copy2(path, archived)
    except Exception:
        pass


class ArtifactWatcher:
    def __init__(self, *, session_dir: Path, poll_s: float, archive_artifacts: bool, session_start: float):
        self.session_dir = session_dir
        self.poll_s = poll_s
        self.archive_artifacts = archive_artifacts
        self.session_start = session_start
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.state: Dict[str, float] = {}
        self.events_path = session_dir / "artifact_events.jsonl"
        self.latest_dir = session_dir / "latest"
        self.archive_dir = session_dir / "artifacts" if archive_artifacts else None
        self.watch_specs: List[Dict[str, Any]] = [
            {"name": "page_current", "path": PAGE_CURRENT_PATH, "handler": self._handle_json, "summary": _summarize_page},
            {"name": "dom_current_page", "path": DOM_CURRENT_PAGE_PATH, "handler": self._handle_json, "summary": _summarize_page},
            {"name": "current_question", "path": CURRENT_QUESTION_PATH, "handler": self._handle_json, "summary": _summarize_question},
            {"name": "current_controls", "path": CURRENT_CONTROLS_PATH, "handler": self._handle_json, "summary": _summarize_controls},
            {"name": "region_grow", "path": REGION_GROW_CURRENT_PATH, "handler": self._handle_json, "summary": _summarize_region_grow},
            {"name": "fast_summary", "path": FAST_SUMMARY_PATH, "handler": self._handle_json, "summary": _summarize_fast_summary},
            {"name": "brain_state", "path": BRAIN_STATE_PATH, "handler": self._handle_json, "summary": _summarize_brain},
            {"name": "screen_quiz_parse", "path": CURRENT_RUN_PARSE_PATH, "handler": self._handle_json, "summary": _summarize_quiz_parse},
            {"name": "screenshot", "path": RAW_SCREENSHOT_PATH, "handler": self._handle_file, "summary": _summarize_screenshot},
            {"name": "quiz_trace", "path": QUIZ_TRACE_PATH, "handler": self._handle_trace},
            {"name": "rate_summary_latest", "path_fn": lambda: _latest_json(RATE_SUMMARY_CURRENT_DIR), "handler": self._handle_json, "summary": self._summarize_rate_summary},
        ]

    def _summarize_rate_summary(self, payload: Dict[str, Any]) -> str:
        return (
            "rate_summary "
            f"question={_clip(((payload.get('question') or {}).get('text') if isinstance(payload.get('question'), dict) else payload.get('question_text')) or '', 90)} "
            f"top_labels={list((payload.get('top_labels') or {}).keys())[:5]}"
        )

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._loop, name="artifact-watcher", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)

    def _log_event(self, name: str, summary: str, path: Optional[Path]) -> None:
        payload = {
            "ts": time.time(),
            "name": name,
            "summary": summary,
            "path": str(path) if path else None,
            "freshness": _artifact_freshness(path, self.session_start) if path else "missing",
        }
        try:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            for spec in self.watch_specs:
                self._check_spec(spec)
            time.sleep(self.poll_s)

    def _check_spec(self, spec: Dict[str, Any]) -> None:
        name = str(spec["name"])
        path = spec.get("path")
        if path is None and callable(spec.get("path_fn")):
            path = spec["path_fn"]()
        if not isinstance(path, Path):
            return
        key = f"{name}:{path}"
        if not path.exists():
            return
        try:
            mtime = path.stat().st_mtime
        except Exception:
            return
        if float(self.state.get(key) or 0.0) == float(mtime):
            return
        self.state[key] = float(mtime)
        handler = spec.get("handler")
        if callable(handler):
            handler(name, path, spec)

    def _handle_json(self, name: str, path: Path, spec: Dict[str, Any]) -> None:
        payload = _safe_read_json(path)
        if payload is None:
            return
        summary_fn = spec.get("summary")
        summary = summary_fn(payload) if callable(summary_fn) else f"{name} updated"
        _print(f"{summary} freshness={_artifact_freshness(path, self.session_start)}")
        self._log_event(name, summary, path)
        _copy_latest(path, self.latest_dir, self.archive_dir)

    def _handle_file(self, name: str, path: Path, spec: Dict[str, Any]) -> None:
        summary_fn = spec.get("summary")
        summary = summary_fn(path) if callable(summary_fn) else f"{name} updated"
        _print(f"{summary} freshness={_artifact_freshness(path, self.session_start)}")
        self._log_event(name, summary, path)
        _copy_latest(path, self.latest_dir, self.archive_dir)

    def _handle_trace(self, name: str, path: Path, spec: Dict[str, Any]) -> None:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return
        summary = _summarize_trace_line(lines[-1])
        _print(f"{summary} freshness={_artifact_freshness(path, self.session_start)}")
        self._log_event(name, summary, path)
        _copy_latest(path, self.latest_dir, None)


class ProcessStreamer:
    def __init__(self, *, proc: subprocess.Popen[str], label: str, session_dir: Path):
        self.proc = proc
        self.label = label
        self.session_dir = session_dir
        self.thread: Optional[threading.Thread] = None
        self.log_path = session_dir / f"{label}.log"

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, name=f"{self.label}-stream", daemon=True)
        self.thread.start()

    def join(self) -> None:
        if self.thread is not None:
            self.thread.join(timeout=5.0)

    def _run(self) -> None:
        if self.proc.stdout is None:
            return
        with self.log_path.open("a", encoding="utf-8") as handle:
            for raw in self.proc.stdout:
                line = raw.rstrip("\r\n")
                text = f"[{self.label}] {line}"
                print(text, flush=True)
                handle.write(text + "\n")
                handle.flush()


def _build_main_cmd(args: argparse.Namespace, scenario_url: str, base_url: str) -> List[str]:
    lock_prefix = _quiz_lock_prefix(scenario_url, base_url)
    cmd = [
        str(args.main_python),
        str(MAIN_SCRIPT),
        "--auto",
        "--interval",
        str(args.interval),
        "--headless",
        str(int(args.headless)),
        "--width",
        str(int(args.viewport_width)),
        "--height",
        str(int(args.viewport_height)),
        "--quiz-mode",
        "--quiz-answer-cache",
        str(QA_CACHE),
        "--quiz-lock-url-prefix",
        lock_prefix,
    ]
    if args.quiz_suite:
        cmd.append("--quiz-suite")
    if args.disable_recorder:
        cmd.append("--disable-recorder")
    recorder_args = [
        "--url",
        scenario_url,
        "--quiz-mode",
        "--quiz-lock-url-prefix",
        lock_prefix,
    ]
    if not args.disable_recorder:
        cmd.append("--recorder-args")
        cmd.extend(recorder_args)
    for item in args.main_arg:
        cmd.append(str(item))
    return cmd


def _build_main_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env["FULLBOT_ENVIRONMENT"] = _normalize_environment(getattr(args, "environment", ""))
    if args.disable_console_overlay:
        env["FULLBOT_DISABLE_CONSOLE_OVERLAY"] = "1"
        env["FULLBOT_ENABLE_TK_OVERLAY"] = "0"
        env["FULLBOT_OVERLAY_MIRROR_STDIO"] = "0"
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live automation harness for quiz solver debugging.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--type-id", type=int, default=1, help="Quiz scenario type id (1..13).")
    parser.add_argument("--scenario-url", type=str, default=None, help="Direct /t/{type}/{question}?reset=1 URL. Non-direct values are normalized back to test_quiz_server scenario URLs.")
    parser.add_argument("--interval", type=float, default=1.0, help="main.py loop interval.")
    parser.add_argument("--poll", type=float, default=0.5, help="Artifact monitor poll interval in seconds.")
    parser.add_argument("--main-python", type=Path, default=_default_main_python(), help="Python used to run main.py.")
    parser.add_argument("--harness-python", type=Path, default=_default_harness_python(), help="Unused marker for documentation/debug.")
    parser.add_argument("--start-server", action="store_true", help="Start test_quiz_server.py if it is not already running.")
    parser.add_argument("--disable-recorder", action="store_true", help="Run pipeline without launching ai_recorder_live.")
    parser.add_argument("--monitor-only", action="store_true", help="Do not start main.py; only stream live artifacts.")
    parser.add_argument("--archive-artifacts", action="store_true", help="Archive every observed artifact version in the session directory.")
    parser.add_argument("--quiz-suite", action="store_true", help="Pass --quiz-suite through to main.py.")
    parser.add_argument("--disable-console-overlay", action="store_true", default=True, help="Run main.py with FULLBOT_DISABLE_CONSOLE_OVERLAY=1.")
    parser.add_argument("--headless", type=int, default=DEFAULT_HEADLESS, help="Pass headless mode through to auto_main.")
    parser.add_argument("--viewport-profile", type=str, default="", help="Named viewport preset: desktop_standard, tablet_portrait, mobile_narrow.")
    parser.add_argument("--viewport-width", type=int, default=DEFAULT_VIEWPORT_WIDTH, help="Viewport width passed through to auto_main.")
    parser.add_argument("--viewport-height", type=int, default=DEFAULT_VIEWPORT_HEIGHT, help="Viewport height passed through to auto_main.")
    parser.add_argument("--environment", type=str, default=os.environ.get("FULLBOT_ENVIRONMENT", "test"), help="Session environment profile marker: test, staging, production.")
    parser.add_argument("--max-seconds", type=float, default=20.0, help="Hard-stop the session after this many seconds.")
    parser.add_argument("--session-name", type=str, default="", help="Optional suffix for the session directory.")
    parser.add_argument("--main-arg", action="append", default=[], help="Extra argument passed through to main.py. Repeat for multiple values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_start = time.time()
    scenario_url = _scenario_url(args.host, args.port, args.type_id, args.scenario_url)
    base_url = _base_url(args.host, args.port)
    lock_prefix = _quiz_lock_prefix(scenario_url, base_url)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{stamp}_{args.session_name}" if args.session_name else stamp
    session_dir = SESSIONS_DIR / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_start_ts": session_start,
        "scenario_url": scenario_url,
        "scenario_is_direct_test_url": True,
        "base_url": base_url,
        "quiz_lock_prefix": lock_prefix,
        "main_python": str(args.main_python),
        "harness_python": str(args.harness_python),
        "archive_artifacts": bool(args.archive_artifacts),
        "monitor_only": bool(args.monitor_only),
        "disable_recorder": bool(args.disable_recorder),
        "disable_console_overlay": bool(args.disable_console_overlay),
        "headless": bool(int(args.headless)),
        "viewport": {
            "width": int(args.viewport_width),
            "height": int(args.viewport_height),
        },
        "environment": _normalize_environment(args.environment),
        "max_seconds": float(args.max_seconds),
        "main_args": list(args.main_arg),
    }
    _print(f"session={session_dir}")
    _print(f"scenario={scenario_url}")
    _print(f"quiz_lock_prefix={lock_prefix}")
    _print(f"main_python={args.main_python}")
    preflight = _build_preflight(args, base_url)
    manifest["preflight"] = preflight
    manifest["browser_matrix_contract"] = _build_browser_matrix_contract(args)
    (session_dir / "session.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_preflight(preflight)

    server_proc: Optional[subprocess.Popen] = None
    if args.start_server:
        server_proc = _ensure_server(args.host, args.port)
        _wait_for_server(args.host, args.port)
        _print(f"server ready at {base_url}")

    watcher = ArtifactWatcher(
        session_dir=session_dir,
        poll_s=args.poll,
        archive_artifacts=bool(args.archive_artifacts),
        session_start=session_start,
    )
    watcher.start()

    main_proc: Optional[subprocess.Popen[str]] = None
    streamer: Optional[ProcessStreamer] = None
    stop_reason = "time_budget_exceeded"
    timed_out = False
    try:
        if not args.monitor_only:
            cmd = _build_main_cmd(args, scenario_url, base_url)
            (session_dir / "main_cmd.txt").write_text(" ".join(cmd), encoding="utf-8")
            _print(f"launching main: {' '.join(cmd)}")
            main_proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=_build_main_env(args),
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            streamer = ProcessStreamer(proc=main_proc, label="main", session_dir=session_dir)
            streamer.start()
        else:
            _print("monitor-only mode active")

        start = time.time()
        while True:
            if (time.time() - start) >= float(args.max_seconds):
                _print(f"max runtime reached ({args.max_seconds:.1f}s), stopping session")
                stop_reason = "time_budget_exceeded"
                timed_out = True
                break
            if main_proc is not None and main_proc.poll() is not None:
                _print(f"main exited with code {main_proc.returncode}")
                stop_reason = _compute_stop_reason(
                    session_start=session_start,
                    main_returncode=main_proc.returncode,
                    main_log_path=session_dir / "main.log",
                    scenario_url=scenario_url,
                    timed_out=False,
                )
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        _print("stopping session")
        stop_reason = "interrupted"
    finally:
        watcher.stop()
        _terminate(main_proc)
        if streamer is not None:
            streamer.join()
        _terminate(server_proc)
        if stop_reason == "time_budget_exceeded":
            stop_reason = _compute_stop_reason(
                session_start=session_start,
                main_returncode=main_proc.returncode if main_proc is not None else None,
                main_log_path=session_dir / "main.log",
                scenario_url=scenario_url,
                timed_out=timed_out,
            )
        summary = {
            "stop_reason": stop_reason,
            "artifacts": _artifact_snapshot(session_start),
            "dom_has_quiz": bool(
                ((_safe_read_json(CURRENT_CONTROLS_PATH) or {}).get("question_blocks") or [])
                or (((_safe_read_json(CURRENT_CONTROLS_PATH) or {}).get("meta") or {}).get("qid"))
            ),
            "screen_has_quiz": bool(
                _artifact_freshness(RAW_SCREENSHOT_PATH, session_start) == "fresh_for_session"
                and _detect_screen_has_quiz(RAW_SCREENSHOT_PATH)
            ),
        }
        summary["screen_dom_consistent"] = bool(summary["dom_has_quiz"] and summary["screen_has_quiz"])
        summary["failure_summary"] = build_failure_summary_contract(
            stop_reason=stop_reason,
            dom_has_quiz=bool(summary["dom_has_quiz"]),
            screen_has_quiz=bool(summary["screen_has_quiz"]),
        )
        try:
            manifest_live = json.loads((session_dir / "session.json").read_text(encoding="utf-8", errors="replace"))
        except Exception:
            manifest_live = dict(manifest)
        manifest_live.update(summary)
        (session_dir / "session.json").write_text(json.dumps(manifest_live, ensure_ascii=False, indent=2), encoding="utf-8")
        _print(
            f"session_result stop_reason={stop_reason} "
            f"dom_has_quiz={int(bool(summary['dom_has_quiz']))} "
            f"screen_has_quiz={int(bool(summary['screen_has_quiz']))} "
            f"screen_dom_consistent={int(bool(summary['screen_dom_consistent']))}"
        )


if __name__ == "__main__":
    main()
