from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ADVANCED_DEBUG = str(os.environ.get("FULLBOT_ADVANCED_DEBUG", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_FAST = ADVANCED_DEBUG or str(os.environ.get("FULLBOT_RATING_FAST_DEBUG", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_FAST_TIMERS = ADVANCED_DEBUG or str(os.environ.get("FULLBOT_RATING_FAST_DEBUG_TIMERS", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}


def _dbg(msg: str) -> None:
    if DEBUG_FAST:
        print(f"[RATING_FAST DEBUG] {msg}")


def _timer(label: str, t0: float) -> None:
    if DEBUG_FAST_TIMERS:
        print(f"[RATING_FAST TIMER] {label}: {(time.perf_counter() - t0)*1000.0:.1f} ms")


def _norm_text(s: Any) -> str:
    return " ".join(str(s or "").strip().lower().split())


def _is_next_text(s: str) -> bool:
    t = _norm_text(s)
    if not t:
        return False
    keys = ("dalej", "nast", "next", "continue", "kontynuuj", "wyślij", "wyslij", "submit", "finish", "done")
    return any(k in t for k in keys)


def _is_dropdown_text(s: str) -> bool:
    t = _norm_text(s)
    if not t:
        return False
    keys = ("wybierz", "select", "choose", "option", "lista", "rozwij")
    return any(k in t for k in keys)


def _extract_bbox(row: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    try:
        tb = row.get("text_box")
        if isinstance(tb, (list, tuple)) and len(tb) == 4:
            x1, y1, x2, y2 = [int(v) for v in tb]
            if x2 > x1 and y2 > y1:
                return (x1, y1, x2, y2)
        bb = row.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            x1, y1, x2, y2 = [int(v) for v in bb]
            if x2 > x1 and y2 > y1:
                return (x1, y1, x2, y2)
        if isinstance(bb, dict):
            x = int(bb.get("x"))
            y = int(bb.get("y"))
            w = int(bb.get("width"))
            h = int(bb.get("height"))
            if w > 0 and h > 0:
                return (x, y, x + w, y + h)
    except Exception:
        return None
    return None


def _score(row: Dict[str, Any], bonus: float = 0.0) -> float:
    try:
        conf = float(row.get("conf") or 0.0)
    except Exception:
        conf = 0.0
    return max(0.0, min(1.0, conf + bonus))


def _build_summary(data: Dict[str, Any], image_hint: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rows = data.get("results") if isinstance(data, dict) else []
    if not isinstance(rows, list):
        rows = []
    _dbg(f"_build_summary rows={len(rows)} image_hint={image_hint}")

    answer: List[Dict[str, Any]] = []
    nexts: List[Dict[str, Any]] = []
    dropdowns: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        bbox = _extract_bbox(row)
        if bbox is None:
            continue
        txt = str(row.get("text") or row.get("box_text") or "").strip()
        if DEBUG_FAST:
            try:
                conf_dbg = float(row.get("conf") or 0.0)
            except Exception:
                conf_dbg = 0.0
            _dbg(
                f"row#{i} conf={conf_dbg:.3f} has_frame={bool(row.get('has_frame'))} "
                f"text='{txt[:80]}'"
            )
        item = {
            "id": str(row.get("id") or f"rg_{i}"),
            "text": txt,
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            "score": round(_score(row), 4),
        }
        if _is_next_text(txt):
            nexts.append(dict(item, score=round(_score(row, 0.25), 4)))
            continue
        if bool(row.get("has_frame")) or _is_dropdown_text(txt):
            dropdowns.append(dict(item, score=round(_score(row, 0.2 if row.get("has_frame") else 0.1), 4)))
            continue
        if txt:
            answer.append(item)

    answer.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    nexts.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    dropdowns.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    top_labels: Dict[str, Dict[str, Any]] = {}
    if dropdowns:
        top_labels["dropdown"] = dict(dropdowns[0], label="dropdown")
    if nexts:
        top_labels["next_active"] = dict(nexts[0], label="next_active")
    if answer:
        top_labels["answer_single"] = dict(answer[0], label="answer_single")
        top_labels["answer_multi"] = dict(answer[0], label="answer_multi")
    _dbg(
        f"summary candidates answer={len(answer)} next={len(nexts)} dropdown={len(dropdowns)} "
        f"top={list(top_labels.keys())}"
    )
    _timer("_build_summary total", t0)

    return {
        "image": str(data.get("image") or image_hint),
        "total_elements": int(len(rows)),
        "background_layout": data.get("background_layout") if isinstance(data, dict) else {},
        "top_labels": top_labels,
        "question_like_boxes": [],
        "answer_candidate_boxes": answer[:8],
        "next_candidate_boxes": nexts[:5],
        "dropdown_candidate_boxes": dropdowns[:5],
        "confidence": {
            "answer": float(answer[0]["score"]) if answer else 0.0,
            "next": float(nexts[0]["score"]) if nexts else 0.0,
            "dropdown": float(dropdowns[0]["score"]) if dropdowns else 0.0,
        },
        "reasons": {"mode": "rating_fast", "source": "region_grow.results"},
    }


def process_file(in_json_path: str) -> Optional[str]:
    t_total = time.perf_counter()
    _dbg(f"process_file input={in_json_path}")
    in_path = Path(in_json_path)
    if not in_path.exists():
        _dbg("input does not exist")
        return None
    try:
        t_load = time.perf_counter()
        data = json.loads(in_path.read_text(encoding="utf-8", errors="replace"))
        _timer("load_json", t_load)
    except Exception:
        _dbg("load_json failed")
        return None

    t_summary = time.perf_counter()
    summary = _build_summary(data if isinstance(data, dict) else {}, image_hint=str(in_path))
    _timer("build_summary", t_summary)
    screen_dir = in_path.resolve().parents[2]
    out_dir = screen_dir / "rate" / "rate_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}_summary.json"
    t_write = time.perf_counter()
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _timer("write_summary", t_write)
    _timer("process_file total", t_total)
    _dbg(f"saved summary={out_path}")
    return str(out_path)
