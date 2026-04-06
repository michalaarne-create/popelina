from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(payload, dict):
        rows = payload.get("samples")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        rows = payload.get("rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _bbox(v: Any) -> Optional[List[int]]:
    if not isinstance(v, list) or len(v) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(x))) for x in v]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _iou(a: List[int], b: List[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter) / float(aa + bb - inter)


def _norm_text(v: Any) -> str:
    return " ".join(str(v or "").strip().lower().split())


def _text_sim(a: str, b: str) -> float:
    aa = _norm_text(a)
    bb = _norm_text(b)
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 1.0
    sa = set(aa.split())
    sb = set(bb.split())
    inter = len(sa & sb)
    return float(inter) / float(max(1, len(sa | sb)))


def _center(box: List[int]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _build_dom_elements(label_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    dom = label_data.get("dom") if isinstance(label_data.get("dom"), dict) else {}
    out: List[Dict[str, Any]] = []
    for key in ("questions", "answers", "selects", "text_inputs", "next_buttons"):
        rows = dom.get(key)
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            bbox = _bbox(row.get("bbox"))
            if bbox is None:
                continue
            attrs = row.get("attrs") if isinstance(row.get("attrs"), dict) else {}
            block_id = str(attrs.get("block_id") or "").strip()
            if not block_id:
                continue
            out.append(
                {
                    "id": f"{key}_{idx}",
                    "kind": str(row.get("kind") or key[:-1]),
                    "text": str(row.get("text") or ""),
                    "bbox": bbox,
                    "block_id": block_id,
                }
            )
    return out


def _build_region_items(region_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = region_data.get("results")
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        bbox = _bbox(row.get("bbox") or row.get("text_box") or row.get("dropdown_box"))
        if bbox is None:
            continue
        text = str(row.get("text") or row.get("box_text") or "")
        out.append(
            {
                "id": str(row.get("id") or f"rg_{idx}"),
                "text": text,
                "bbox": bbox,
                "conf": float(row.get("conf") or 0.0),
                "has_frame": 1 if row.get("has_frame") else 0,
            }
        )
    return out


def _match_region_to_dom(item: Dict[str, Any], dom_elements: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ibox = item["bbox"]
    itext = item["text"]
    best = None
    best_score = 0.0
    for dom in dom_elements:
        dbox = dom["bbox"]
        iou = _iou(ibox, dbox)
        tsim = _text_sim(itext, dom["text"])
        cx1, cy1 = _center(ibox)
        cx2, cy2 = _center(dbox)
        dist = abs(cx1 - cx2) + abs(cy1 - cy2)
        dist_penalty = min(1.0, dist / 1200.0)
        score = max(iou, tsim * 0.85) + (0.35 * min(iou, tsim)) - (0.10 * dist_penalty)
        if iou >= 0.2 or tsim >= 0.5:
            if score > best_score:
                best = dom
                best_score = score
    return best


def _feature_row(sample_id: str, prompt: Dict[str, Any], item: Dict[str, Any], label: int, screen_h: int) -> Dict[str, Any]:
    pbox = prompt["bbox"]
    ibox = item["bbox"]
    pcx = (pbox[0] + pbox[2]) / 2.0
    icx = (ibox[0] + ibox[2]) / 2.0
    dy = float(ibox[1] - pbox[3])
    dx = float(icx - pcx)
    overlap_x = max(0.0, min(float(pbox[2]), float(ibox[2])) - max(float(pbox[0]), float(ibox[0])))
    overlap_x_ratio = overlap_x / float(max(1.0, min(pbox[2] - pbox[0], ibox[2] - ibox[0])))
    text = str(item.get("text") or "")
    prompt_text = str(prompt.get("text") or "")
    norm = _norm_text(text)
    pnorm = _norm_text(prompt_text)
    return {
        "sample_id": sample_id,
        "prompt_id": str(prompt.get("id") or ""),
        "item_id": str(item.get("id") or ""),
        "label": int(label),
        "prompt_text": prompt_text,
        "item_text": text,
        "prompt_control_kind": str(prompt.get("kind") or "unknown"),
        "item_candidate_type": "dropdown_trigger" if int(item.get("has_frame") or 0) else "answer_option",
        "item_role_pred": "unknown",
        "prompt_y": float(pbox[1]),
        "item_y": float(ibox[1]),
        "dy": dy,
        "dx": dx,
        "vertical_gap_norm": dy / float(max(1, screen_h)),
        "center_dx_norm": dx / float(max(1.0, pbox[2] - pbox[0])),
        "x_overlap_ratio": overlap_x_ratio,
        "iou_prompt": _iou(pbox, ibox),
        "item_width_norm": (ibox[2] - ibox[0]) / float(max(1.0, pbox[2] - pbox[0])),
        "item_height_norm": (ibox[3] - ibox[1]) / float(max(1.0, pbox[3] - pbox[1])),
        "prompt_item_text_sim": _text_sim(pnorm, norm),
        "item_has_frame": int(item.get("has_frame") or 0),
        "item_conf": float(item.get("conf") or 0.0),
        "item_text_len": float(len(norm)),
        "item_word_count": float(len([w for w in norm.split() if w])),
        "item_is_below_prompt": 1 if ibox[1] >= pbox[3] else 0,
        "item_is_far_below": 1 if dy > max(90, (pbox[3] - pbox[1]) * 2.2) else 0,
        "item_has_digit": 1 if any(ch.isdigit() for ch in norm) else 0,
        "item_has_scroll_hint": 1 if any(tok in norm for tok in ("scroll", "przewin", "lista")) else 0,
        "item_has_expand_hint": 1 if "expand" in norm or "rozwin" in norm or "rozwiń" in norm else 0,
        "item_has_input_hint": 1 if any(tok in norm for tok in ("wpisz", "type", "input", "odpowiedz", "odpowiedź")) else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build question grouping dataset from label DOM block_id + OCR region results.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    bench_rows = _load_manifest(manifest_path)
    if args.max_samples > 0:
        bench_rows = bench_rows[: int(args.max_samples)]
    if not bench_rows:
        print("[ERROR] no rows")
        return 2

    source_cache: Dict[str, Dict[str, Any]] = {}
    out_path = Path(args.out_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    matched_items = 0
    total_items = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in bench_rows:
            source_manifest_path = Path(str(row.get("source_manifest") or "")).resolve()
            if not source_manifest_path.exists():
                continue
            source_payload = source_cache.get(str(source_manifest_path))
            if source_payload is None:
                source_payload = _load_json(source_manifest_path)
                source_cache[str(source_manifest_path)] = source_payload or {}
            source_rows = []
            if isinstance(source_payload, dict):
                source_rows = source_payload.get("rows") if isinstance(source_payload.get("rows"), list) else []
            source_view_id = str(row.get("source_view_id") or row.get("id") or "")
            source_row = next((r for r in source_rows if isinstance(r, dict) and str(r.get("view_id") or "") == source_view_id), None)
            if not isinstance(source_row, dict):
                continue
            label_path = Path(str(source_row.get("label_path") or "")).resolve()
            region_path = Path(str(row.get("region_json") or "")).resolve()
            if not label_path.exists() or not region_path.exists():
                continue
            label_data = _load_json(label_path)
            region_data = _load_json(region_path)
            if not isinstance(label_data, dict) or not isinstance(region_data, dict):
                continue
            dom_elements = _build_dom_elements(label_data)
            region_items = _build_region_items(region_data)
            if not dom_elements or not region_items:
                continue
            prompts = [d for d in dom_elements if d.get("kind") == "question"]
            if not prompts:
                continue
            screen_h = int((((label_data.get("viewport") or {}).get("height")) or (((label_data.get("dom") or {}).get("viewport") or {}).get("height")) or 1080))
            item_block_map: Dict[str, str] = {}
            for item in region_items:
                total_items += 1
                best = _match_region_to_dom(item, dom_elements)
                if best and str(best.get("block_id") or ""):
                    item_block_map[str(item["id"])] = str(best["block_id"])
                    matched_items += 1
            for prompt in prompts:
                pblock = str(prompt.get("block_id") or "")
                if not pblock:
                    continue
                for item in region_items:
                    iblock = item_block_map.get(str(item["id"]) or "", "")
                    label = 1 if iblock and iblock == pblock else 0
                    feat = _feature_row(str(row.get("id") or ""), prompt, item, label, screen_h)
                    fh.write(json.dumps(feat, ensure_ascii=False) + "\n")
                    written += 1
    print(json.dumps({"manifest": str(manifest_path), "out_jsonl": str(out_path), "written": written, "matched_items": matched_items, "total_items": total_items}, ensure_ascii=False))
    return 0 if written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
