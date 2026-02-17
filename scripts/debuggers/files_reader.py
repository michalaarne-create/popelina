from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class BrainDispatchResult:
    summary_path: Path
    decision: Any
    is_new_question: bool


def collect_and_dispatch_to_brain(
    *,
    screenshot_path: Path,
    json_path: Path,
    rate_results_dir: Path,
    rate_results_debug_dir: Path,
    rate_results_current_dir: Path,
    rate_results_debug_current_dir: Path,
    rate_summary_dir: Path,
    rate_summary_current_dir: Path,
    write_current_artifact: Callable[[Path, Path, Optional[str]], Optional[Path]],
    brain_agent: Any,
    log: Callable[[str], None],
) -> Optional[BrainDispatchResult]:
    """
    Collect rating outputs, resolve summary path and dispatch it to brain agent.
    """
    try:
        stem = screenshot_path.stem
        rated_path = rate_results_dir / f"{stem}_rated.json"
        rated_debug_path = rate_results_debug_dir / f"{stem}_rated_debug.json"
        if rated_path.exists():
            write_current_artifact(rated_path, rate_results_current_dir, rated_path.name)
        if rated_debug_path.exists():
            write_current_artifact(rated_debug_path, rate_results_debug_current_dir, rated_debug_path.name)
    except Exception:
        pass

    summary_candidates = [
        rate_summary_dir / f"{screenshot_path.stem}_summary.json",
        rate_summary_dir / f"{json_path.stem}_summary.json",
    ]
    if json_path.stem.endswith("_rg_small"):
        alt_stem = json_path.stem[: -len("_rg_small")]
        summary_candidates.append(rate_summary_dir / f"{alt_stem}_summary.json")

    summary_path: Optional[Path] = None
    for cand in summary_candidates:
        if cand.exists():
            summary_path = cand
            break

    turbo_mode = str(os.environ.get("FULLBOT_TURBO_MODE", "1") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if summary_path is None and turbo_mode:
        # In turbo mode prefer deterministic fast summary over "latest *_summary.json".
        try:
            fast_candidates = [
                json_path.with_name(f"{json_path.stem}_fast_summary.json"),
                json_path.parent.parent / "region_grow_current" / "fast_summary.json",
            ]
            for cand in fast_candidates:
                if cand.exists():
                    summary_path = cand
                    log(f"[INFO] Using fast summary fallback: {cand.name}")
                    break
        except Exception:
            summary_path = None

    if summary_path is None and (not turbo_mode):
        # Fallback: rating can emit summary keyed by `image` basename (often `screenshot.png`)
        # rather than current screenshot stem (`screen_YYYY...`).
        try:
            recent = sorted(
                (p for p in rate_summary_dir.glob("*_summary.json") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if recent:
                summary_path = recent[0]
                log(
                    f"[WARN] Summary name mismatch; using latest summary: {summary_path.name}"
                )
        except Exception:
            summary_path = None

    if summary_path is None:
        log(f"[WARN] Summary missing at {summary_candidates[0]}; skipping brain decision.")
        return None

    try:
        write_current_artifact(summary_path, rate_summary_current_dir, summary_path.name)
    except Exception:
        pass

    decision = brain_agent.decide(summary_path)
    is_new_question = bool((getattr(decision, "brain_state", {}) or {}).get("question_changed"))
    return BrainDispatchResult(
        summary_path=summary_path,
        decision=decision,
        is_new_question=is_new_question,
    )
