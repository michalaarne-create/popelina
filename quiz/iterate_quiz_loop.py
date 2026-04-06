from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEV_AUTOMATION = ROOT / "quiz" / "dev_automation.py"
SESSIONS_DIR = ROOT / "quiz" / "logs" / "dev_sessions"
REPORTS_DIR = ROOT / "quiz" / "logs"

PAGE_CURRENT_PATH = ROOT / "data" / "screen" / "page_current" / "page_current.json"
CURRENT_QUESTION_PATH = ROOT / "scripts" / "dom" / "dom_live" / "current_question.json"
REGION_GROW_CURRENT_PATH = ROOT / "data" / "screen" / "region_grow" / "region_grow_current" / "region_grow.json"
FAST_SUMMARY_PATH = ROOT / "data" / "screen" / "region_grow" / "region_grow_current" / "fast_summary.json"
SCREENSHOT_PATH = ROOT / "data" / "screen" / "current_run" / "screenshot.png"
CLICKS_ON_SCREEN_DIR = ROOT / "data" / "screen" / "clicks_on_screen"
VIEWPORT_PROFILE_DIMENSIONS: Dict[str, Tuple[int, int]] = {
    "desktop_standard": (1440, 1400),
    "tablet_portrait": (1024, 1366),
    "mobile_narrow": (430, 932),
}


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
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


def _quiz_count(default_count: int = 13) -> int:
    try:
        # Local import from quiz package file
        import importlib.util

        spec = importlib.util.spec_from_file_location("quiz_server_dyn", str(ROOT / "quiz" / "test_quiz_server.py"))
        if spec is None or spec.loader is None:
            return int(default_count)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[call-arg]
        bank = getattr(mod, "BANK", None)
        if isinstance(bank, list) and bank:
            return int(len(bank))
        build_bank = getattr(mod, "build_bank", None)
        if callable(build_bank):
            built = build_bank()
            if isinstance(built, list) and built:
                return int(len(built))
    except Exception:
        pass
    return int(default_count)


def _quiet_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(base or os.environ.copy())
    env["FULLBOT_RUNTIME_PROFILE"] = "ultra_fast"
    env["FULLBOT_REGION_GROW_LOG_VERBOSITY"] = "summary"
    env["FULLBOT_OCR_BOXES_DEBUG"] = "0"
    env["FULLBOT_HOVER_DEBUG_ARTIFACTS"] = "0"
    env["REGION_GROW_ENABLE_REGIONS_EXPORT"] = "0"
    env["FULLBOT_REGION_USE_DOWNSCALED_IMAGE"] = "0"
    env["RG_TARGET_SIDE_TURBO"] = "1280"
    env["FULLBOT_STAGE_REGION_RATING_BUDGET_MS"] = "1550"
    env["FULLBOT_ITERATION_BUDGET_MS"] = "2000"
    return env


def _diagnostic_env(base: Dict[str, str]) -> Dict[str, str]:
    env = dict(base)
    env["FULLBOT_REGION_GROW_LOG_VERBOSITY"] = "detailed"
    env["FULLBOT_OCR_BOXES_DEBUG"] = "1"
    env["FULLBOT_HOVER_DEBUG_ARTIFACTS"] = "1"
    env["REGION_GROW_ENABLE_REGIONS_EXPORT"] = "1"
    return env


def _extract_session_dir(stdout: str) -> Optional[Path]:
    for line in stdout.splitlines():
        if "session=" in line:
            m = re.search(r"session=(.+)$", line.strip())
            if m:
                cand = Path(m.group(1).strip())
                if cand.exists():
                    return cand
    return None


def _collect_question_signatures(session_dir: Path) -> Set[str]:
    sigs: Set[str] = set()
    events_path = session_dir / "artifact_events.jsonl"
    if events_path.exists():
        for raw in _safe_read_text(events_path).splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except Exception:
                continue
            if not isinstance(event, dict):
                continue
            name = str(event.get("name") or "")
            summary = str(event.get("summary") or "")
            if name in {"current_question", "page_current", "dom_current_page"} and summary:
                sigs.add(summary)
    # fallback from latest artifacts
    q = _safe_read_json(CURRENT_QUESTION_PATH) or {}
    p = _safe_read_json(PAGE_CURRENT_PATH) or {}
    if q:
        sigs.add(f"q:{json.dumps(q, ensure_ascii=False, sort_keys=True)[:240]}")
    if p:
        sigs.add(f"p:{json.dumps(p, ensure_ascii=False, sort_keys=True)[:240]}")
    return sigs


def _session_ok(session_dir: Path, *, min_signatures: int = 2) -> Tuple[bool, str]:
    sess = _safe_read_json(session_dir / "session.json") or {}
    stop_reason = str(sess.get("stop_reason") or "")
    bad_reasons = {
        "executor_failed",
        "main_exited_nonzero",
        "recorder_not_ready",
        "screen_wrong_tab",
        "question_projection_missing",
        "region_grow_empty",
    }
    if stop_reason in bad_reasons:
        return False, f"bad stop_reason={stop_reason}"

    if not SCREENSHOT_PATH.exists():
        return False, "missing screenshot"
    if not REGION_GROW_CURRENT_PATH.exists():
        return False, "missing region_grow_current"
    if not FAST_SUMMARY_PATH.exists():
        return False, "missing fast_summary"

    sigs = _collect_question_signatures(session_dir)
    if len(sigs) < max(1, int(min_signatures)):
        return False, f"no progress signatures={len(sigs)}"

    return True, f"ok stop_reason={stop_reason} signatures={len(sigs)}"


def _run_one(
    *,
    python_bin: Path,
    host: str,
    port: int,
    type_id: int,
    interval: float,
    max_seconds: float,
    pass_idx: int,
    quiet: bool,
    session_tag: str,
    headless: int,
    viewport_profile: str,
    viewport_width: int,
    viewport_height: int,
    environment: str,
) -> Dict[str, Any]:
    env = _quiet_env()
    if not quiet:
        env = _diagnostic_env(env)

    cmd = [
        str(python_bin),
        str(DEV_AUTOMATION),
        "--host",
        str(host),
        "--port",
        str(port),
        "--type-id",
        str(type_id),
        "--start-server",
        "--max-seconds",
        str(max_seconds),
        "--headless",
        str(int(headless)),
        "--viewport-profile",
        str(viewport_profile),
        "--viewport-width",
        str(int(viewport_width)),
        "--viewport-height",
        str(int(viewport_height)),
        "--environment",
        str(environment),
        "--interval",
        str(interval),
        "--disable-console-overlay",
        "--session-name",
        session_tag,
    ]
    _log(f"run pass={pass_idx} type={type_id} quiet={int(quiet)}")
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=env,
        check=False,
    )
    stdout = str(proc.stdout or "")
    stderr = str(proc.stderr or "")
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n")

    session_dir = _extract_session_dir(stdout)
    if session_dir is None:
        # fallback to newest
        latest = sorted((p for p in SESSIONS_DIR.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime) if SESSIONS_DIR.exists() else []
        session_dir = latest[-1] if latest else None

    ok = False
    reason = "missing session dir"
    if session_dir is not None and session_dir.exists():
        ok, reason = _session_ok(session_dir)

    # hard fail on non-zero process
    if int(proc.returncode) != 0:
        ok = False
        reason = f"dev_automation exit={proc.returncode}; {reason}"

    return {
        "pass_idx": int(pass_idx),
        "type_id": int(type_id),
        "quiet": bool(quiet),
        "viewport_profile": str(viewport_profile or ""),
        "environment": str(environment or ""),
        "ok": bool(ok),
        "reason": reason,
        "returncode": int(proc.returncode),
        "session_dir": str(session_dir) if session_dir else "",
    }


def _resolve_viewport_profiles(args: argparse.Namespace) -> List[Dict[str, Any]]:
    raw_profiles = str(getattr(args, "viewport_profiles", "") or "").strip()
    if raw_profiles:
        resolved: List[Dict[str, Any]] = []
        for item in raw_profiles.split(","):
            name = str(item or "").strip()
            if not name:
                continue
            dims = VIEWPORT_PROFILE_DIMENSIONS.get(name)
            if dims is None:
                raise SystemExit(f"Unsupported viewport profile: {name}")
            resolved.append(
                {
                    "name": name,
                    "width": int(dims[0]),
                    "height": int(dims[1]),
                }
            )
        if resolved:
            return resolved
    single_name = str(getattr(args, "viewport_profile", "") or "").strip()
    return [
        {
            "name": single_name,
            "width": int(getattr(args, "viewport_width", 1440)),
            "height": int(getattr(args, "viewport_height", 1400)),
        }
    ]


def _resolve_environments(args: argparse.Namespace) -> List[str]:
    raw = str(getattr(args, "environments", "") or "").strip()
    if raw:
        values = [str(item or "").strip().lower() for item in raw.split(",")]
        resolved = [item for item in values if item]
        if resolved:
            return resolved
    fallback = str(getattr(args, "environment", "test") or "test").strip().lower()
    return [fallback or "test"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-pass quiz loop: 1..N then 1..N with progress validation.")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=20.0)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--start-type", type=int, default=1)
    p.add_argument("--end-type", type=int, default=0, help="0 means auto-detect from test_quiz_server.py")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--headless", type=int, default=1)
    p.add_argument("--viewport-profile", type=str, default="")
    p.add_argument("--viewport-profiles", type=str, default="", help="Comma-separated viewport presets for one report run.")
    p.add_argument("--viewport-width", type=int, default=1440)
    p.add_argument("--viewport-height", type=int, default=1400)
    p.add_argument("--environment", type=str, default=os.environ.get("FULLBOT_ENVIRONMENT", "test"))
    p.add_argument("--environments", type=str, default="", help="Comma-separated environment names for one report run.")
    p.add_argument("--no-diagnostic-retry", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    quiz_n = int(args.end_type) if int(args.end_type) > 0 else _quiz_count(13)
    start_type = max(1, int(args.start_type))
    end_type = max(start_type, int(quiz_n))
    type_ids = list(range(start_type, end_type + 1))

    if not DEV_AUTOMATION.exists():
        raise SystemExit(f"Missing script: {DEV_AUTOMATION}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CLICKS_ON_SCREEN_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"iterate_quiz_loop_{stamp}.json"
    results: List[Dict[str, Any]] = []
    viewport_profiles = _resolve_viewport_profiles(args)
    environments = _resolve_environments(args)

    _log(f"types={type_ids[0]}..{type_ids[-1]} count={len(type_ids)} passes={int(args.passes)}")
    _log(f"max_seconds_per_run={float(args.max_seconds):.1f}")
    _log("viewport_profiles=" + ",".join(str(row["name"] or "custom") for row in viewport_profiles))
    _log("environments=" + ",".join(environments))

    t0 = time.time()
    for environment_name in environments:
        for viewport_row in viewport_profiles:
            profile_name = str(viewport_row["name"] or "")
            for pass_idx in range(1, int(args.passes) + 1):
                for type_id in type_ids:
                    env_tag = f"_env_{environment_name}" if environment_name else ""
                    tag_suffix = f"_vp_{profile_name}" if profile_name else ""
                    base_tag = f"p{pass_idx}_t{type_id}{env_tag}{tag_suffix}"
                    run = _run_one(
                        python_bin=Path(args.python),
                        host=args.host,
                        port=args.port,
                        type_id=type_id,
                        interval=float(args.interval),
                        max_seconds=float(args.max_seconds),
                        pass_idx=pass_idx,
                        quiet=True,
                        session_tag=base_tag,
                        headless=int(args.headless),
                        viewport_profile=profile_name,
                        viewport_width=int(viewport_row["width"]),
                        viewport_height=int(viewport_row["height"]),
                        environment=environment_name,
                    )
                    results.append(run)
                    if run["ok"]:
                        continue

                    _log(
                        f"FAIL pass={pass_idx} type={type_id} environment={environment_name} "
                        f"viewport={profile_name or 'custom'}: {run['reason']}"
                    )
                    if args.no_diagnostic_retry:
                        report = {
                            "ok": False,
                            "failed": run,
                            "results": results,
                            "viewport_profiles": [str(row["name"] or "custom") for row in viewport_profiles],
                            "environments": environments,
                            "elapsed_s": round(time.time() - t0, 3),
                        }
                        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                        raise SystemExit(1)

                    diag = _run_one(
                        python_bin=Path(args.python),
                        host=args.host,
                        port=args.port,
                        type_id=type_id,
                        interval=float(args.interval),
                        max_seconds=float(args.max_seconds),
                        pass_idx=pass_idx,
                        quiet=False,
                        session_tag=f"{base_tag}_diag",
                        headless=int(args.headless),
                        viewport_profile=profile_name,
                        viewport_width=int(viewport_row["width"]),
                        viewport_height=int(viewport_row["height"]),
                        environment=environment_name,
                    )
                    results.append(diag)
                    if not diag["ok"]:
                        report = {
                            "ok": False,
                            "failed": diag,
                            "results": results,
                            "viewport_profiles": [str(row["name"] or "custom") for row in viewport_profiles],
                            "environments": environments,
                            "elapsed_s": round(time.time() - t0, 3),
                        }
                        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                        raise SystemExit(1)

    report = {
        "ok": True,
        "type_ids": type_ids,
        "passes": int(args.passes),
        "viewport_profiles": [str(row["name"] or "custom") for row in viewport_profiles],
        "environments": environments,
        "results": results,
        "elapsed_s": round(time.time() - t0, 3),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"ALL OK report={report_path}")


if __name__ == "__main__":
    main()
