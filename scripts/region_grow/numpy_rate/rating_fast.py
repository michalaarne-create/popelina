from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    rows = data.get("results") if isinstance(data, dict) else []
    if not isinstance(rows, list):
        rows = []

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
    in_path = Path(in_json_path)
    if not in_path.exists():
        return None
    try:
        data = json.loads(in_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None

    summary = _build_summary(data if isinstance(data, dict) else {}, image_hint=str(in_path))
    screen_dir = in_path.resolve().parents[2]
    out_dir = screen_dir / "rate" / "rate_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)

