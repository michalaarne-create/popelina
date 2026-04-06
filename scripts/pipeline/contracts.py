from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class PipelineConfig:
    interval_s: float = 3.0
    auto: bool = False
    loop_count: Optional[int] = None
    fast_skip: bool = False
    disable_recorder: bool = False
    safe_test: bool = False


@dataclass
class IterationInput:
    loop_idx: int
    screenshot_prefix: str = "screen"
    input_image: Optional[Path] = None
    fast_skip: bool = False


@dataclass
class IterationResult:
    loop_idx: int
    screenshot_path: Optional[Path]
    region_json_path: Optional[Path]
    summary_path: Optional[Path]
    decision_action: Optional[str]
    rating_ok: bool
    elapsed_s: float
    metadata: dict[str, Any]


SCHEMA_VERSION = "2026-03-25.iteration_contracts.v1"


def _sha1_text(*parts: Any) -> str:
    payload = "|".join(str(part or "") for part in parts)
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def _safe_stat(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    try:
        stat = path.stat()
    except Exception:
        return {
            "path": str(path),
            "exists": False,
        }
    return {
        "path": str(path),
        "exists": True,
        "mtime": float(stat.st_mtime),
        "size": int(stat.st_size),
    }


def build_expectation_id(
    *,
    question_signature: str,
    stage: str,
    first_action_kind: str,
    expected_values: Iterable[Any] = (),
) -> str:
    normalized_values = [str(value or "").strip() for value in expected_values if str(value or "").strip()]
    digest = _sha1_text(question_signature, stage, first_action_kind, json.dumps(normalized_values, ensure_ascii=False))
    return f"exp_{digest[:16]}"


def build_action_id(
    *,
    screen_signature: str,
    question_signature: str,
    first_action_kind: str,
    reason: str,
    expected_values: Iterable[Any] = (),
) -> str:
    normalized_values = [str(value or "").strip() for value in expected_values if str(value or "").strip()]
    digest = _sha1_text(
        screen_signature,
        question_signature,
        first_action_kind,
        reason,
        json.dumps(normalized_values, ensure_ascii=False),
    )
    return f"act_{digest[:16]}"


def build_target_instance_id(
    *,
    question_signature: str,
    action_kind: str,
    target_text: str = "",
    target_bbox: Iterable[Any] = (),
) -> str:
    normalized_bbox: list[float] = []
    for value in target_bbox or ():
        try:
            normalized_bbox.append(round(float(value), 3))
        except Exception:
            continue
    digest = _sha1_text(
        question_signature,
        action_kind,
        str(target_text or "").strip(),
        json.dumps(normalized_bbox, ensure_ascii=False),
    )
    return f"target_{digest[:16]}"


def build_before_after_pair_id(
    *,
    action_id: str,
    previous_screen_signature: str,
    current_screen_signature: str,
    current_page_signature: str = "",
) -> str:
    digest = _sha1_text(action_id, previous_screen_signature, current_screen_signature, current_page_signature)
    return f"pair_{digest[:16]}"


def classify_transition_kind(
    *,
    previous_action_kind: str,
    question_changed: bool,
    page_changed: bool,
    values_match: bool,
    qa_changed_30: bool,
    same_screen: bool,
    failure: bool,
) -> str:
    prev_kind = str(previous_action_kind or "").strip().lower()
    if not prev_kind:
        return "initial_state"
    if page_changed or question_changed:
        return "new_question"
    if values_match and prev_kind in {"answer", "dropdown", "type"}:
        return "same_question_answered"
    if failure or same_screen:
        return "no_progress"
    if qa_changed_30:
        return "screen_changed"
    return "state_changed"


def summary_is_fresh(
    *,
    summary_path: Path,
    reference_path: Path,
    max_age_s: float = 240.0,
    max_negative_skew_s: float = 1.5,
) -> bool:
    try:
        summary_mtime = float(summary_path.stat().st_mtime)
        reference_mtime = float(reference_path.stat().st_mtime)
    except Exception:
        return False
    delta = summary_mtime - reference_mtime
    if delta < (-1.0 * float(max_negative_skew_s)):
        return False
    return abs(delta) <= float(max_age_s)


def build_iteration_manifest(
    *,
    screenshot_path: Path,
    region_json_path: Optional[Path],
    summary_path: Optional[Path],
    question_path: Optional[Path],
    controls_path: Optional[Path],
    page_path: Optional[Path],
    extra_artifacts: Optional[Dict[str, Optional[Path]]] = None,
    coherency_window_s: float = 2.5,
) -> dict[str, Any]:
    screenshot_stat = _safe_stat(screenshot_path) or {"path": str(screenshot_path), "exists": False}
    screenshot_mtime = float(screenshot_stat.get("mtime") or 0.0)
    recorder_stats = {
        "question": _safe_stat(question_path),
        "controls": _safe_stat(controls_path),
        "page": _safe_stat(page_path),
    }
    recorder_skews: dict[str, Optional[float]] = {}
    recorder_ok = True
    for key, meta in recorder_stats.items():
        if not isinstance(meta, dict) or not meta.get("exists"):
            recorder_ok = False
            recorder_skews[key] = None
            continue
        skew = float(meta.get("mtime") or 0.0) - screenshot_mtime
        recorder_skews[key] = round(skew, 4)
        if abs(skew) > float(coherency_window_s):
            recorder_ok = False
    artifact_id = f"iter_{_sha1_text(screenshot_stat.get('path'), screenshot_stat.get('mtime'), screenshot_stat.get('size'))[:16]}"
    artifacts: dict[str, Any] = {
        "screenshot": screenshot_stat,
        "region_json": _safe_stat(region_json_path),
        "summary": _safe_stat(summary_path),
        "question": recorder_stats["question"],
        "controls": recorder_stats["controls"],
        "page": recorder_stats["page"],
    }
    for name, path in (extra_artifacts or {}).items():
        artifacts[name] = _safe_stat(path)
    summary_fresh = False
    if summary_path is not None and summary_path.exists():
        summary_fresh = summary_is_fresh(
            summary_path=summary_path,
            reference_path=region_json_path if region_json_path is not None and region_json_path.exists() else screenshot_path,
            max_age_s=300.0,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "created_at": time.time(),
        "artifacts": artifacts,
        "coherency_check": {
            "window_s": float(coherency_window_s),
            "recorder_skews_s": recorder_skews,
            "recorder_coherent": recorder_ok,
            "summary_fresh": bool(summary_fresh),
        },
    }


def write_iteration_manifest(
    *,
    current_run_dir: Path,
    screenshot_path: Path,
    region_json_path: Optional[Path],
    summary_path: Optional[Path],
    question_path: Optional[Path],
    controls_path: Optional[Path],
    page_path: Optional[Path],
    extra_artifacts: Optional[Dict[str, Optional[Path]]] = None,
) -> Optional[Path]:
    try:
        current_run_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_iteration_manifest(
            screenshot_path=screenshot_path,
            region_json_path=region_json_path,
            summary_path=summary_path,
            question_path=question_path,
            controls_path=controls_path,
            page_path=page_path,
            extra_artifacts=extra_artifacts,
        )
        out_path = current_run_dir / "iteration_manifest.json"
        out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path
    except Exception:
        return None


def build_reproducibility_report(
    *,
    first_run: dict[str, Any],
    second_run: dict[str, Any],
) -> dict[str, Any]:
    first = dict(first_run or {})
    second = dict(second_run or {})
    comparable_keys = ("artifact_id", "global_seed", "deterministic_run_config", "model_fingerprint")
    matches = {}
    mismatches = []
    for key in comparable_keys:
        first_value = first.get(key)
        second_value = second.get(key)
        same = first_value == second_value
        matches[key] = same
        if not same:
            mismatches.append(key)
    return {
        "schema_version": SCHEMA_VERSION,
        "is_reproducible": not mismatches,
        "matches": matches,
        "mismatches": mismatches,
        "first_run": first,
        "second_run": second,
    }
