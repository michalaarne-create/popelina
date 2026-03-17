from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

from PIL import Image

try:
    from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright
except Exception:  # pragma: no cover - runtime dependency
    Browser = Any  # type: ignore[assignment]
    BrowserContext = Any  # type: ignore[assignment]
    Page = Any  # type: ignore[assignment]
    sync_playwright = None  # type: ignore[assignment]


WORKSPACE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WORKSPACE_ROOT / "popelina_github"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.brain.runtime.action_planner import plan_actions
from scripts.brain.runtime.answer_resolver import resolve_answer
from scripts.brain.runtime.screen_quiz_parser import parse_screen_quiz_state

QUIZ_DIR = PROJECT_ROOT / "quiz"
QUIZ_SERVER_SCRIPT = QUIZ_DIR / "test_quiz_server.py"
REGION_GROW_SCRIPT = PROJECT_ROOT / "scripts" / "region_grow" / "region_grow" / "region_grow.py"
REGION_GROW_JSON_DIR = PROJECT_ROOT / "data" / "screen" / "region_grow" / "region_grow"
QA_CACHE_PATH = PROJECT_ROOT / "data" / "answers" / "qa_cache.json"
AUTO_DATA_DIR = PROJECT_ROOT / "data" / "auto_main"
CURRENT_RUN_DIR = AUTO_DATA_DIR / "current_run"
RUNS_DIR = AUTO_DATA_DIR / "runs"
AUTO_DEBUG_DIR = PROJECT_ROOT / "data" / "screen" / "auto_debug"
_RG_MODULE: Optional[Any] = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_current(path: Path, target_name: str) -> None:
    if not path.exists():
        return
    target = CURRENT_RUN_DIR / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())


def _slug(value: str) -> str:
    raw = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").strip())
    collapsed = "_".join(part for part in raw.split("_") if part)
    return (collapsed or "item")[:60]


def _probe(url: str, timeout: float = 0.75) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = int(getattr(response, "status", 0) or 0)
            return 200 <= status < 500
    except Exception:
        return False


def _ensure_server(host: str, port: int) -> Optional[subprocess.Popen[Any]]:
    base_url = f"http://{host}:{port}/"
    if _probe(base_url):
        return None
    return subprocess.Popen(
        [sys.executable, str(QUIZ_SERVER_SCRIPT)],
        cwd=str(PROJECT_ROOT),
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
        time.sleep(0.2)
    raise RuntimeError(f"Quiz server did not start at {base_url} within {timeout_s:.1f}s")


def _terminate(proc: Optional[subprocess.Popen[Any]], timeout: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def _sha1_text(*parts: str) -> str:
    joined = "\n".join(parts)
    return hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()


def _load_qa_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"QA cache missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    items = payload.get("items")
    if not isinstance(items, dict):
        raise RuntimeError(f"Invalid QA cache format at {path}")
    return items


@dataclass
class ScreenPlan:
    source: str
    action_kind: str
    target: Dict[str, Any]
    reason: str


def _screen_model_placeholder(state: Dict[str, Any], screenshot_path: Path) -> Optional[ScreenPlan]:
    # Reserved for future screen classifier / button model.
    # Current harness always falls back to DOM-driven interactions, but keeps
    # the interface stable so we can swap in screen inference later.
    _ = (state, screenshot_path)
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _run_region_grow_for_image(image_path: Path) -> Dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("FULLBOT_OCR_BOXES_DEBUG", "0")
    env.setdefault("FULLBOT_DEBUG_DEFERRED_RENDER", "1")
    env.setdefault("FULLBOT_REGION_GROW_ANNOTATE_INLINE", "0")
    env.setdefault("FULLBOT_REGION_GROW_LOG_VERBOSITY", "summary")
    env.setdefault("FULLBOT_TURBO_MODE", "1")
    attempts = 2
    last_payload: Optional[Dict[str, Any]] = None
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(
            [sys.executable, str(REGION_GROW_SCRIPT), str(image_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"region_grow failed for {image_path.name} with exit code {proc.returncode}")
        json_path = REGION_GROW_JSON_DIR / f"{image_path.stem}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"region_grow output missing: {json_path}")
        payload = _load_json(json_path)
        if not isinstance(payload, dict):
            raise RuntimeError(f"region_grow returned non-dict payload: {json_path}")
        last_payload = payload
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        if results or attempt >= attempts:
            return payload
        time.sleep(0.2)
    return last_payload or {}


def _get_region_grow_module() -> Any:
    global _RG_MODULE
    if _RG_MODULE is None:
        from scripts.region_grow.region_grow import region_grow as rg  # type: ignore

        _RG_MODULE = rg
    return _RG_MODULE


def _bbox_from_quad(quad: Sequence[Sequence[int]]) -> Optional[List[int]]:
    if not quad:
        return None
    xs: List[int] = []
    ys: List[int] = []
    for pt in quad:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        xs.append(int(round(float(pt[0]))))
        ys.append(int(round(float(pt[1]))))
    if not xs or not ys:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _build_ocr_rows_fallback_payload(image_path: Path, base_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rg = _get_region_grow_module()
    rows = None
    with Image.open(image_path).convert("RGB") as img_pil:
        silent = io.StringIO()
        with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
            rows = rg.read_ocr_wrapper(img_pil)
    if not isinstance(rows, list) or not rows:
        return None
    results: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        quad, text, conf = row[0], row[1], row[2]
        bbox = _bbox_from_quad(quad if isinstance(quad, (list, tuple)) else [])
        if bbox is None or not str(text or "").strip():
            continue
        results.append(
            {
                "id": f"ocr_{idx}",
                "text": str(text).strip(),
                "box_text": str(text).strip(),
                "conf": float(conf or 0.0),
                "bbox": bbox,
                "text_box": bbox,
                "has_frame": False,
            }
        )
    if not results:
        return None
    payload = dict(base_payload or {})
    payload["results"] = results
    fast_summary = payload.get("fast_summary") if isinstance(payload.get("fast_summary"), dict) else {}
    fast_summary = dict(fast_summary)
    fast_summary["total_elements"] = len(results)
    payload["fast_summary"] = fast_summary
    return payload


def _build_screen_decision(*, screenshot_path: Path) -> Dict[str, Any]:
    region_payload = _run_region_grow_for_image(screenshot_path)
    raw_results = region_payload.get("results") if isinstance(region_payload.get("results"), list) else []
    if not raw_results:
        fallback_payload = _build_ocr_rows_fallback_payload(screenshot_path, region_payload)
        if fallback_payload is not None:
            region_payload = fallback_payload
    summary_data = region_payload.get("fast_summary") if isinstance(region_payload.get("fast_summary"), dict) else None
    screen_state = parse_screen_quiz_state(
        region_payload=region_payload,
        summary_data=summary_data,
        page_data=None,
        rated_data=None,
    )
    resolved = resolve_answer(cache_path=QA_CACHE_PATH, screen_state=screen_state, controls_data=None)
    actions_objs, trace, fallback_used = plan_actions(
        screen_state=screen_state,
        resolved_answer=resolved,
        brain_state={},
        controls_data=None,
        transition=None,
    )
    return {
        "region_payload": region_payload,
        "summary_data": summary_data or {},
        "screen_state": screen_state,
        "resolved_answer": resolved.to_dict(),
        "actions": [action.to_dict() for action in actions_objs],
        "trace": trace,
        "fallback_used": bool(fallback_used),
    }


def _is_home_screen(screen_state: Dict[str, Any], region_payload: Dict[str, Any]) -> bool:
    texts = []
    for row in region_payload.get("results") or []:
        if not isinstance(row, dict):
            continue
        txt = str(row.get("text") or row.get("box_text") or "").strip().lower()
        if txt:
            texts.append(txt)
    question_text = str(screen_state.get("question_text") or "").strip().lower()
    joined = "\n".join(texts)
    start_count = sum(1 for txt in texts if "start" in txt)
    reset_count = sum(1 for txt in texts if "reset" in txt)
    numbered_type_count = sum(
        1
        for txt in texts
        if txt[:2].isdigit() or any(marker in txt for marker in ("jedna strona", "dropdown", "input", "ankieta"))
    )
    if (
        "test quiz server" in joined
        or "podziat na typy pyta" in question_text
        or "podział na typy pyta" in question_text
    ) and ((start_count + reset_count) >= 4 or numbered_type_count >= 4):
        return True
    if question_text and any(marker in question_text for marker in ("podziat na typy pyta", "podział na typy pyta")):
        return True
    return False


def _extract_page_state(page: Page) -> Dict[str, Any]:
    return page.evaluate(
        """
        () => {
          const qid = (document.getElementById('qid')?.textContent || '').trim();
          const question = (document.querySelector('.q')?.textContent || '').trim();
          const nextVisible = !!document.getElementById('next');
          const select = document.getElementById('sel');
          const textbox = document.getElementById('txt');
          const options = Array.from(document.querySelectorAll('label.opt')).map((label, idx) => {
            const input = label.querySelector('input');
            const text = (label.querySelector('.opt-text')?.textContent || label.textContent || '').trim();
            return {
              index: idx,
              text,
              kind: input ? input.type : 'label',
              id: input ? input.id : null,
              checked: !!(input && input.checked),
            };
          });
          const selectOptions = select ? Array.from(select.options).map((opt, idx) => ({
            index: idx,
            text: (opt.textContent || '').trim(),
            value: opt.value,
            selected: !!opt.selected,
          })) : [];
          const textBlocks = Array.from(document.querySelectorAll('.title,.desc,.q,.opt,.btn')).map((el, idx) => {
            const r = el.getBoundingClientRect();
            return {
              index: idx,
              tag: el.tagName.toLowerCase(),
              text: (el.textContent || '').trim(),
              bbox: [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)],
            };
          });
          const pageText = [question]
            .concat(options.map(o => o.text))
            .concat(selectOptions.map(o => o.text))
            .join('\\n');
          return {
            url: window.location.href,
            title: document.title,
            qid,
            question,
            nextVisible,
            hasSelect: !!select,
            hasTextbox: !!textbox,
            textboxValue: textbox ? textbox.value : '',
            options,
            selectOptions,
            textBlocks,
            pageText,
          };
        }
        """
    )


def _safe_extract_page_state(page: Page, retries: int = 8, delay_ms: int = 120) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for _ in range(max(1, retries)):
        try:
            return _extract_page_state(page)
        except Exception as exc:
            last_error = exc
            page.wait_for_timeout(delay_ms)
    if last_error is not None:
        raise last_error
    return {}


def _safe_bbox(locator: Any) -> Dict[str, float]:
    try:
        bbox = locator.bounding_box(timeout=600) or {}
    except Exception:
        bbox = {}
    return {str(k): float(v) for k, v in bbox.items()}


def _build_signature(state: Dict[str, Any]) -> str:
    question = str(state.get("question") or "")
    qid = str(state.get("qid") or "")
    option_texts = [str((row or {}).get("text") or "") for row in (state.get("options") or []) if isinstance(row, dict)]
    select_texts = [str((row or {}).get("text") or "") for row in (state.get("selectOptions") or []) if isinstance(row, dict)]
    return _sha1_text(qid, question, *option_texts, *select_texts)


def _locate_cache_item(cache: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    qid = str(state.get("qid") or "").strip()
    if qid and qid in cache:
        item = cache[qid]
        if isinstance(item, dict):
            return item
    question_norm = _normalize_text(state.get("question") or "")
    option_norms = sorted(_normalize_text((row or {}).get("text") or "") for row in (state.get("options") or []) if isinstance(row, dict))
    for item in cache.values():
        if not isinstance(item, dict):
            continue
        if _normalize_text(item.get("question_text") or "") != question_norm:
            continue
        item_opts = sorted(_normalize_text(value) for value in (item.get("options_text") or {}).values())
        if option_norms == item_opts:
            return item
    raise KeyError(f"Question not found in QA cache: {state.get('qid') or state.get('question')}")


def _find_option_index(select_options: Sequence[Dict[str, Any]], answer_text: str) -> Optional[int]:
    target = _normalize_text(answer_text)
    for row in select_options:
        if _normalize_text(row.get("text") or row.get("value") or "") == target:
            return int(row.get("index") or 0)
    return None


def _dom_plan(state: Dict[str, Any], cache_item: Dict[str, Any]) -> ScreenPlan:
    qtype = str(cache_item.get("question_type") or "").strip().lower()
    correct_answer = str(cache_item.get("text_answer") or cache_item.get("correct_answer") or "").strip()
    selected = [str(v) for v in (cache_item.get("selected_options") or [])]
    options_text = cache_item.get("options_text") if isinstance(cache_item.get("options_text"), dict) else {}
    target_answers: List[str] = []
    if qtype == "text":
        target_answers = [correct_answer]
    elif selected and options_text:
        for key in selected:
            value = options_text.get(key)
            if value:
                target_answers.append(str(value))
    elif correct_answer:
        target_answers = [correct_answer]

    if state.get("hasTextbox"):
        return ScreenPlan(
            source="dom_fallback",
            action_kind="text",
            target={"answers": target_answers[:1]},
            reason="textbox_answer",
        )
    if state.get("hasSelect"):
        return ScreenPlan(
            source="dom_fallback",
            action_kind="select",
            target={"answers": target_answers[:1]},
            reason="select_answer",
        )
    return ScreenPlan(
        source="dom_fallback",
        action_kind="choice",
        target={"answers": target_answers},
        reason="choice_answer",
    )


def _save_iteration_artifacts(run_dir: Path, iteration: int, page: Page, state: Dict[str, Any]) -> Dict[str, Path]:
    iter_dir = run_dir / f"iter_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = iter_dir / f"{run_dir.name}_iter_{iteration:03d}.png"
    html_path = iter_dir / "page.html"
    state_path = iter_dir / "state.json"
    page.screenshot(path=str(screenshot_path), full_page=True)
    html_path.write_text(page.content(), encoding="utf-8")
    state_payload = dict(state)
    state_payload["signature"] = _build_signature(state_payload)
    _json_dump(state_path, state_payload)
    _copy_current(screenshot_path, "page.png")
    _copy_current(html_path, "page.html")
    _copy_current(state_path, "state.json")
    return {"iter_dir": iter_dir, "screenshot": screenshot_path, "html": html_path, "state": state_path}


def _save_action_debug(
    *,
    page: Page,
    debug_dir: Path,
    iteration: int,
    action_index: int,
    action_kind: str,
    action_target: str,
    action_value: str,
    bbox: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    iter_dir = debug_dir / f"iter_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{action_index:02d}_{_slug(action_kind)}_{_slug(action_value or action_target)}"
    screenshot_path = iter_dir / f"{suffix}.png"
    meta_path = iter_dir / f"{suffix}.json"
    try:
        page.screenshot(path=str(screenshot_path), full_page=True)
    except Exception:
        page.wait_for_timeout(180)
        try:
            page.screenshot(path=str(screenshot_path), full_page=True)
        except Exception:
            pass
    meta_payload = {
        "ts_ms": _now_ms(),
        "iteration": iteration,
        "action_index": action_index,
        "kind": action_kind,
        "target": action_target,
        "value": action_value,
        "bbox": bbox or {},
        "url": page.url,
    }
    _json_dump(meta_path, meta_payload)
    return {"screenshot": str(screenshot_path), "meta": str(meta_path)}


def _playwright_key(combo: str) -> str:
    raw = str(combo or "").strip().lower()
    mapping = {
        "ctrl+a": "Control+A",
        "backspace": "Backspace",
        "enter": "Enter",
        "home": "Home",
        "down": "ArrowDown",
        "pagedown": "PageDown",
    }
    return mapping.get(raw, combo)


def _ensure_bbox_visible(page: Page, bbox: Sequence[float]) -> tuple[float, float]:
    viewport = page.viewport_size or {"width": 1440, "height": 1400}
    cx = (float(bbox[0]) + float(bbox[2])) / 2.0
    cy = (float(bbox[1]) + float(bbox[3])) / 2.0
    scroll_y = float(page.evaluate("() => window.scrollY"))
    target_scroll = max(0.0, cy - (float(viewport["height"]) * 0.35))
    if abs(target_scroll - scroll_y) > 6.0:
        page.evaluate("(y) => window.scrollTo(0, y)", target_scroll)
        page.wait_for_timeout(120)
        scroll_y = float(page.evaluate("() => window.scrollY"))
    return cx, max(2.0, cy - scroll_y)


def _resolve_click_point(reason: str, bbox: Sequence[float], default_x: float, default_y: float) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    reason_norm = str(reason or "").strip().lower()
    if reason_norm in {"focus_textbox", "focus_select"}:
        click_x = x1 + min(max(18.0, width * 0.12), width - 12.0)
        click_y = y1 + min(max(16.0, height * 0.32), height - 10.0)
        return click_x, click_y
    return default_x, default_y


def _apply_screen_actions(
    page: Page,
    actions_plan: Sequence[Dict[str, Any]],
    *,
    debug_dir: Path,
    iteration: int,
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    action_index = 0
    for planned in actions_plan:
        if not isinstance(planned, dict):
            continue
        kind = str(planned.get("kind") or "")
        reason = str(planned.get("reason") or "")
        bbox_list = planned.get("bbox") if isinstance(planned.get("bbox"), list) else None
        bbox: Dict[str, float] = {}
        if bbox_list and len(bbox_list) == 4:
            x, y = _ensure_bbox_visible(page, [float(v) for v in bbox_list])
            x, y = _resolve_click_point(reason, [float(v) for v in bbox_list], x, y)
            bbox = {
                "x1": float(bbox_list[0]),
                "y1": float(bbox_list[1]),
                "x2": float(bbox_list[2]),
                "y2": float(bbox_list[3]),
                "click_x": x,
                "click_y": y,
            }
        if kind == "screen_click" and bbox_list and len(bbox_list) == 4:
            page.mouse.click(bbox["click_x"], bbox["click_y"])
        elif kind == "screen_scroll":
            amount = int(planned.get("amount") or 1)
            direction = str(planned.get("direction") or "down").lower()
            delta = 220 * max(1, amount)
            page.mouse.wheel(0, delta if direction == "down" else -delta)
        elif kind == "key_press":
            page.keyboard.press(_playwright_key(str(planned.get("combo") or "")))
        elif kind == "key_repeat":
            combo = _playwright_key(str(planned.get("combo") or ""))
            repeat = max(1, int(planned.get("repeat") or 1))
            for _ in range(repeat):
                page.keyboard.press(combo)
                page.wait_for_timeout(30)
        elif kind == "type_text":
            page.keyboard.type(str(planned.get("text") or ""), delay=12)
        elif kind == "wait":
            page.wait_for_timeout(int(planned.get("amount") or 80))
        elif kind == "dom_select_option":
            answer_text = str(planned.get("text") or "")
            applied = bool(
                page.evaluate(
                    """
                    (answerText) => {
                      const norm = (value) => String(value || '').trim().toLowerCase();
                      const sel = document.querySelector('select');
                      if (!sel) return false;
                      const target = norm(answerText);
                      let matchedValue = null;
                      for (const opt of Array.from(sel.options || [])) {
                        if (norm(opt.textContent) === target || norm(opt.value) === target) {
                          matchedValue = opt.value;
                          break;
                        }
                      }
                      if (matchedValue === null) return false;
                      sel.value = matchedValue;
                      sel.dispatchEvent(new Event('input', { bubbles: true }));
                      sel.dispatchEvent(new Event('change', { bubbles: true }));
                      return true;
                    }
                    """,
                    answer_text,
                )
            )
            if not applied:
                raise RuntimeError(f"DOM execution fallback could not set select option: {answer_text}")
        elif kind == "noop":
            pass
        else:
            raise RuntimeError(f"Unsupported action kind: {kind}")
        action_index += 1
        debug_artifacts = _save_action_debug(
            page=page,
            debug_dir=debug_dir,
            iteration=iteration,
            action_index=action_index,
            action_kind=kind or "unknown",
            action_target=reason or kind or "unknown",
            action_value=str(planned.get("text") or planned.get("combo") or planned.get("amount") or ""),
            bbox=bbox,
        )
        actions.append({**planned, "debug": debug_artifacts, "bbox_debug": bbox})
    return actions


def _build_execution_override_actions(decision: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    resolved = decision.get("resolved_answer") if isinstance(decision.get("resolved_answer"), dict) else {}
    trace = decision.get("trace") if isinstance(decision.get("trace"), dict) else {}
    screen_state = decision.get("screen_state") if isinstance(decision.get("screen_state"), dict) else {}
    qtype = str(resolved.get("question_type") or "").strip().lower()
    answers = [str(v) for v in (resolved.get("correct_answers") or []) if str(v or "").strip()]
    expectation = trace.get("post_action_expectation") if isinstance(trace.get("post_action_expectation"), dict) else {}
    if qtype not in {"dropdown", "dropdown_scroll"} or not answers:
        return None
    visible_options = screen_state.get("options") if isinstance(screen_state.get("options"), list) else []
    control_kind = str(screen_state.get("control_kind") or "").strip().lower()
    has_select_bbox = isinstance(screen_state.get("select_bbox"), list) and len(screen_state.get("select_bbox")) == 4
    dropdown_like_screen = has_select_bbox or control_kind in {"dropdown", "text"}
    sparse_visible_options = len(visible_options) <= 1
    if not (dropdown_like_screen and sparse_visible_options):
        return None
    actions: List[Dict[str, Any]] = [
        {
            "kind": "dom_select_option",
            "text": answers[0],
            "reason": "screen_decision_dom_execute_dropdown",
        }
    ]
    next_bbox = None
    maybe_next = screen_state.get("next_bbox")
    if isinstance(maybe_next, list) and len(maybe_next) == 4 and bool(expectation.get("has_next_target")):
        actions.append({"kind": "wait", "amount": 120, "reason": "wait_before_next"})
        actions.append({"kind": "screen_click", "bbox": maybe_next, "reason": "click_next_after_answer"})
    return actions


def _apply_execution_fallback(
    page: Page,
    *,
    decision: Dict[str, Any],
    debug_dir: Path,
    iteration: int,
    start_index: int,
) -> List[Dict[str, Any]]:
    resolved = decision.get("resolved_answer") if isinstance(decision.get("resolved_answer"), dict) else {}
    trace = decision.get("trace") if isinstance(decision.get("trace"), dict) else {}
    qtype = str(resolved.get("question_type") or "").strip().lower()
    answers = [str(v) for v in (resolved.get("correct_answers") or []) if str(v or "").strip()]
    if qtype not in {"dropdown", "dropdown_scroll"} or not answers:
        return []
    if bool((trace.get("post_action_expectation") or {}).get("has_next_target")):
        return []

    action_index = start_index
    answer_text = answers[0]
    try:
        applied = bool(
            page.evaluate(
                """
                (answerText) => {
                  const norm = (value) => String(value || '').trim().toLowerCase();
                  const sel = document.querySelector('select');
                  if (!sel) return false;
                  const target = norm(answerText);
                  let matchedValue = null;
                  for (const opt of Array.from(sel.options || [])) {
                    if (norm(opt.textContent) === target || norm(opt.value) === target) {
                      matchedValue = opt.value;
                      break;
                    }
                  }
                  if (matchedValue === null) return false;
                  sel.value = matchedValue;
                  sel.dispatchEvent(new Event('input', { bubbles: true }));
                  sel.dispatchEvent(new Event('change', { bubbles: true }));
                  return true;
                }
                """,
                answer_text,
            )
        )
    except Exception:
        return []
    if not applied:
        return []
    action_index += 1
    debug_artifacts = _save_action_debug(
        page=page,
        debug_dir=debug_dir,
        iteration=iteration,
        action_index=action_index,
        action_kind="dom_select_option",
        action_target="execution_fallback_dropdown",
        action_value=answer_text,
        bbox=None,
    )
    return [
        {
            "kind": "dom_select_option",
            "reason": "execution_fallback_dropdown",
            "text": answer_text,
            "debug": debug_artifacts,
            "bbox_debug": {},
        }
    ]


def _wait_for_progress(page: Page, previous_signature: str, previous_url: str, timeout_s: float = 4.0) -> Dict[str, Any]:
    start = time.time()
    last_state = _safe_extract_page_state(page)
    while (time.time() - start) < timeout_s:
        page.wait_for_timeout(120)
        try:
            current = _safe_extract_page_state(page, retries=2, delay_ms=80)
        except Exception:
            continue
        current_sig = _build_signature(current)
        current_url = str(current.get("url") or "")
        if current_url.rstrip("/") == previous_url.rstrip("/") and current_sig == previous_signature:
            last_state = current
            continue
        return {
            "progressed": True,
            "state": current,
            "signature": current_sig,
            "url": current_url,
        }
    return {
        "progressed": False,
        "state": last_state,
        "signature": _build_signature(last_state),
        "url": str(last_state.get("url") or ""),
    }


def _write_trace(run_dir: Path, line: Dict[str, Any]) -> None:
    trace_path = run_dir / "trace.jsonl"
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    _json_dump(CURRENT_RUN_DIR / "trace_current.json", line)


def _run_quiz(
    *,
    page: Page,
    start_url: str,
    qa_cache: Dict[str, Any],
    run_dir: Path,
    max_steps: int,
) -> Dict[str, Any]:
    debug_dir = AUTO_DEBUG_DIR / run_dir.name
    debug_dir.mkdir(parents=True, exist_ok=True)
    page.goto(start_url, wait_until="domcontentloaded")
    page.wait_for_load_state("networkidle")
    summary: Dict[str, Any] = {
        "start_url": start_url,
        "steps": [],
        "completed": False,
        "final_url": "",
    }
    previous_signature = ""
    previous_url = ""
    previous_actions: List[Dict[str, Any]] = []

    for iteration in range(1, max_steps + 1):
        state = {"url": page.url, "title": page.title()}
        artifacts = _save_iteration_artifacts(run_dir, iteration, page, state)
        decision = _build_screen_decision(screenshot_path=artifacts["screenshot"])
        screen_state = decision["screen_state"]
        region_payload = decision["region_payload"]
        screen_signature = str(screen_state.get("screen_signature") or "")
        iter_dir = artifacts["iter_dir"]
        _json_dump(iter_dir / "screen_state.json", screen_state)
        _json_dump(iter_dir / "decision.json", decision)
        _copy_current(iter_dir / "screen_state.json", "screen_state.json")
        _copy_current(iter_dir / "decision.json", "decision.json")

        current_url = str(page.url)
        if previous_signature and previous_actions and screen_signature == previous_signature and current_url.rstrip("/") == previous_url.rstrip("/"):
            summary["final_url"] = str(page.url)
            summary["error"] = f"No screenshot progress after previous actions at iteration {iteration}"
            break

        if _is_home_screen(screen_state, region_payload):
            summary["completed"] = True
            summary["final_url"] = str(page.url)
            _write_trace(
                run_dir,
                {
                    "ts_ms": _now_ms(),
                    "iteration": iteration,
                    "event": "completed",
                    "url": page.url,
                    "screen_state": screen_state,
                },
            )
            break
        actions_plan = decision["actions"] if isinstance(decision.get("actions"), list) else []
        override_actions = _build_execution_override_actions(decision)
        if override_actions:
            actions_plan = override_actions
        url_before = str(page.url)
        if not actions_plan or (actions_plan and str((actions_plan[0] or {}).get("kind") or "") == "noop"):
            summary["final_url"] = str(page.url)
            summary["error"] = f"Screen-only planner returned no actionable steps at iteration {iteration}"
            _write_trace(
                run_dir,
                {
                    "ts_ms": _now_ms(),
                    "iteration": iteration,
                    "event": "blocked",
                    "url": page.url,
                    "screen_state": screen_state,
                    "resolved_answer": decision.get("resolved_answer"),
                    "trace": decision.get("trace"),
                },
            )
            break
        actions = _apply_screen_actions(page, actions_plan, debug_dir=debug_dir, iteration=iteration)
        resolved_for_progress = decision.get("resolved_answer") if isinstance(decision.get("resolved_answer"), dict) else {}
        auto_text_screen = (
            str(screen_state.get("control_kind") or "").strip().lower() == "text"
            and (not screen_state.get("next_bbox"))
            and str(resolved_for_progress.get("question_type") or "").strip().lower() == "text"
        )
        initial_wait_ms = 420 if auto_text_screen else 250
        progress_timeout_s = 3.2 if auto_text_screen else 1.4
        page.wait_for_timeout(initial_wait_ms)
        progress = _wait_for_progress(page, screen_signature, url_before, timeout_s=progress_timeout_s)
        fallback_actions: List[Dict[str, Any]] = []
        if not bool(progress.get("progressed")):
            fallback_actions = _apply_execution_fallback(
                page,
                decision=decision,
                debug_dir=debug_dir,
                iteration=iteration,
                start_index=len(actions),
            )
            if fallback_actions:
                actions.extend(fallback_actions)
                page.wait_for_timeout(260 if auto_text_screen else 180)
                progress = _wait_for_progress(page, screen_signature, url_before, timeout_s=progress_timeout_s)

        step_payload = {
            "iteration": iteration,
            "url": url_before,
            "question": screen_state.get("question_text"),
            "signature_before": screen_signature,
            "plan_source": "screen_only",
            "plan_kind": str((((actions_plan or [{}])[0]) or {}).get("kind") or ""),
            "plan_reason": str((((actions_plan or [{}])[0]) or {}).get("reason") or ""),
            "resolved_answer": decision.get("resolved_answer"),
            "actions": actions,
            "screen_state": screen_state,
            "trace": decision.get("trace"),
            "url_after": str(progress.get("url") or page.url),
            "artifacts": {
                "screenshot": str(artifacts["screenshot"]),
                "html": str(artifacts["html"]),
                "state": str(artifacts["state"]),
                "screen_state": str(iter_dir / "screen_state.json"),
                "decision": str(iter_dir / "decision.json"),
                "auto_debug_dir": str(debug_dir / f"iter_{iteration:03d}"),
            },
        }
        summary["steps"].append(step_payload)
        _write_trace(run_dir, {"ts_ms": _now_ms(), **step_payload})
        previous_signature = screen_signature
        previous_url = url_before
        previous_actions = actions
    else:
        summary["final_url"] = str(page.url)
        summary["error"] = f"Max steps reached ({max_steps})"

    if not summary.get("final_url"):
        summary["final_url"] = str(page.url)
    return summary


def _open_browser(*, headless: bool, width: int, height: int) -> tuple[Any, Browser, BrowserContext, Page]:
    if sync_playwright is None:
        raise RuntimeError("playwright is not installed in this interpreter")
    try:
        pw = sync_playwright().start()
    except PermissionError as exc:
        raise RuntimeError(
            "Playwright could not start its browser transport in this environment "
            f"(PermissionError/WinError 5: {exc}). Run auto_main.py directly in your local shell."
        ) from exc
    browser = pw.chromium.launch(headless=headless)
    context = browser.new_context(viewport={"width": width, "height": height}, locale="pl-PL")
    page = context.new_page()
    return pw, browser, context, page


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless Playwright harness for iterating quiz flows with screenshots and artifacts.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--type-id", type=int, default=1)
    parser.add_argument("--question", type=int, default=1)
    parser.add_argument("--headless", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument("--keep-server", action="store_true")
    args = parser.parse_args()

    AUTO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    AUTO_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = RUNS_DIR / f"{run_stamp}_t{int(args.type_id):02d}_q{int(args.question):02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    server_proc = _ensure_server(args.host, args.port)
    try:
        _wait_for_server(args.host, args.port)
        qa_cache = _load_qa_cache(QA_CACHE_PATH)
        start_url = f"http://{args.host}:{args.port}/t/{int(args.type_id)}/{int(args.question)}?reset=1"
        pw, browser, context, page = _open_browser(
            headless=bool(int(args.headless)),
            width=int(args.width),
            height=int(args.height),
        )
        try:
            summary = _run_quiz(
                page=page,
                start_url=start_url,
                qa_cache=qa_cache,
                run_dir=run_dir,
                max_steps=int(args.max_steps),
            )
        finally:
            context.close()
            browser.close()
            pw.stop()

        summary["run_dir"] = str(run_dir)
        summary["current_run_dir"] = str(CURRENT_RUN_DIR)
        _json_dump(run_dir / "summary.json", summary)
        _copy_current(run_dir / "summary.json", "summary.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not summary.get("completed"):
            raise SystemExit(1)
    finally:
        if not args.keep_server:
            _terminate(server_proc)


if __name__ == "__main__":
    main()
