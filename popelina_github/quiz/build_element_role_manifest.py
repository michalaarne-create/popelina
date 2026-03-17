from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
ROLES = ["question", "answer", "next", "noise"]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if isinstance(obj, dict):
                rows.append(obj)
        return rows
    payload = _load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [r for r in payload["rows"] if isinstance(r, dict)]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def _resolve(path: str, manifest_path: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (manifest_path.parent.parent / path).resolve()


def _safe_bbox(row: Dict[str, Any]) -> Tuple[int, int, int, int] | None:
    bbox = row.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float((ax2 - ax1) * (ay2 - ay1))
    area_b = float((bx2 - bx1) * (by2 - by1))
    denom = max(1.0, area_a + area_b - inter)
    return inter / denom


def _append_rows(
    out: List[Dict[str, Any]],
    dom_rows: List[Dict[str, Any]],
    role: str,
    label: Dict[str, Any],
    manifest_row: Dict[str, Any],
) -> None:
    viewport = label.get("dom", {}).get("viewport") if isinstance(label.get("dom"), dict) else {}
    vw = int(viewport.get("width") or label.get("viewport", {}).get("width") or 1)
    vh = int(viewport.get("height") or label.get("viewport", {}).get("height") or 1)
    for row in dom_rows:
        if not isinstance(row, dict):
            continue
        bbox = _safe_bbox(row)
        if bbox is None:
            continue
        attrs = row.get("attrs") if isinstance(row.get("attrs"), dict) else {}
        out.append(
            {
                "sample_id": label.get("sample_id"),
                "view_id": label.get("view_id"),
                "role": role,
                "text": str(row.get("text") or "").strip(),
                "bbox": list(bbox),
                "attrs": attrs,
                "global_type": str(label.get("global_type") or ""),
                "block_types": label.get("block_types") if isinstance(label.get("block_types"), list) else [],
                "has_next": bool(label.get("has_next")),
                "require_scroll": bool(label.get("require_scroll")),
                "viewport": {"width": vw, "height": vh},
                "image_path": manifest_row.get("image_path") or "",
                "label_path": manifest_row.get("label_path") or "",
                "source_kind": str(row.get("kind") or ""),
            }
        )


def _noise_dom_groups(dom: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    for key in ("hero", "hero_titles", "hero_descs", "nav", "nav_items", "hints", "noise", "secondary_cta"):
        value = dom.get(key)
        if isinstance(value, list) and value:
            groups.append(value)
    return groups


def _build_noise_rows(
    label: Dict[str, Any],
    manifest_row: Dict[str, Any],
    positives: List[Tuple[int, int, int, int]],
    rng: random.Random,
    per_view: int,
) -> List[Dict[str, Any]]:
    viewport = label.get("dom", {}).get("viewport") if isinstance(label.get("dom"), dict) else {}
    vw = int(viewport.get("width") or label.get("viewport", {}).get("width") or 0)
    vh = int(viewport.get("height") or label.get("viewport", {}).get("height") or 0)
    if vw < 50 or vh < 50:
        return []

    out: List[Dict[str, Any]] = []
    attempts = 0
    target = max(1, int(per_view))
    while len(out) < target and attempts < target * 40:
        attempts += 1
        bw = rng.randint(max(40, vw // 10), max(60, vw // 3))
        bh = rng.randint(max(24, vh // 28), max(32, vh // 8))
        x1 = rng.randint(0, max(0, vw - bw))
        y1 = rng.randint(0, max(0, vh - bh))
        box = (x1, y1, x1 + bw, y1 + bh)
        if any(_overlap(box, pos) > 0.05 for pos in positives):
            continue
        out.append(
            {
                "sample_id": label.get("sample_id"),
                "view_id": label.get("view_id"),
                "role": "noise",
                "text": "",
                "bbox": list(box),
                "attrs": {},
                "global_type": str(label.get("global_type") or ""),
                "block_types": label.get("block_types") if isinstance(label.get("block_types"), list) else [],
                "has_next": bool(label.get("has_next")),
                "require_scroll": bool(label.get("require_scroll")),
                "viewport": {"width": vw, "height": vh},
                "image_path": manifest_row.get("image_path") or "",
                "label_path": manifest_row.get("label_path") or "",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build element-role manifest from existing rendered quiz dataset labels.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "data" / "benchmarks" / "random_www_quiz_TEST_ONLY_IMG" / "manifests" / "dataset_manifest.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "data" / "benchmarks" / "element_role_manifest.jsonl"),
    )
    parser.add_argument("--noise-per-view", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260314)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = _read_manifest(manifest_path)
    rng = random.Random(int(args.seed))
    out_rows: List[Dict[str, Any]] = []
    skipped = 0

    for row in rows:
        label_path_raw = str(row.get("label_path") or "").strip()
        if not label_path_raw:
            skipped += 1
            continue
        label_path = _resolve(label_path_raw, manifest_path)
        if not label_path.exists():
            skipped += 1
            continue
        try:
            label = _load_json(label_path)
        except Exception:
            skipped += 1
            continue
        dom = label.get("dom") if isinstance(label.get("dom"), dict) else {}
        questions = dom.get("questions") if isinstance(dom.get("questions"), list) else []
        answers = dom.get("answers") if isinstance(dom.get("answers"), list) else []
        next_buttons = dom.get("next_buttons") if isinstance(dom.get("next_buttons"), list) else []
        selects = dom.get("selects") if isinstance(dom.get("selects"), list) else []
        text_inputs = dom.get("text_inputs") if isinstance(dom.get("text_inputs"), list) else []

        _append_rows(out_rows, questions, "question", label, row)
        _append_rows(out_rows, answers, "answer", label, row)
        _append_rows(out_rows, next_buttons, "next", label, row)
        _append_rows(out_rows, selects, "answer", label, row)
        _append_rows(out_rows, text_inputs, "answer", label, row)
        for group in _noise_dom_groups(dom):
            _append_rows(out_rows, group, "noise", label, row)

        positives: List[Tuple[int, int, int, int]] = []
        for group in (questions, answers, next_buttons, selects, text_inputs):
            for item in group:
                if not isinstance(item, dict):
                    continue
                bbox = _safe_bbox(item)
                if bbox is not None:
                    positives.append(bbox)
        real_noise_total = sum(len(group) for group in _noise_dom_groups(dom))
        if real_noise_total < int(args.noise_per_view):
            out_rows.extend(_build_noise_rows(label, row, positives, rng, int(args.noise_per_view) - real_noise_total))

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "rows_total": len(out_rows),
        "rows_skipped": skipped,
        "roles": {role: sum(1 for r in out_rows if r.get("role") == role) for role in ROLES},
        "out_path": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
