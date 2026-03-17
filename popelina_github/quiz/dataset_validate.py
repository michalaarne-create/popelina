from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set


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
    return []


def _resolve(base: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated dataset consistency.")
    parser.add_argument("--out-dir", required=True, type=str)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    manifest_path = out_dir / "manifests" / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = _read_manifest(manifest_path)
    manifest_dir = manifest_path.parent.parent
    errors: List[Dict[str, Any]] = []
    seen_view_ids: Set[str] = set()

    for row in rows:
        view_id = str(row.get("view_id") or "")
        if view_id in seen_view_ids:
            errors.append({"view_id": view_id, "error": "duplicate_view_id"})
        seen_view_ids.add(view_id)

        label_path_raw = str(row.get("label_path") or "")
        if not label_path_raw:
            errors.append({"view_id": view_id, "error": "missing_label_path"})
            continue
        label_path = _resolve(manifest_dir, label_path_raw)
        if not label_path.exists():
            errors.append({"view_id": view_id, "error": "missing_label_file"})
            continue
        try:
            label = _load_json(label_path)
        except Exception:
            errors.append({"view_id": view_id, "error": "invalid_label_json"})
            continue

        image_path_raw = str(row.get("image_path") or "")
        if image_path_raw:
            image_path = _resolve(manifest_dir, image_path_raw)
            if not image_path.exists():
                errors.append({"view_id": view_id, "error": "missing_image_file"})

        for block in label.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type") or "")
            options = list(block.get("options") or [])
            correct = list(block.get("correct") or [])
            if btype in {"single", "multi", "dropdown", "dropdown_scroll"}:
                if not set(correct).issubset(set(options)):
                    errors.append({"view_id": view_id, "error": "correct_not_subset_options", "block_id": block.get("block_id")})

    out = {
        "out_dir": str(out_dir),
        "rows_total": int(len(rows)),
        "errors_total": int(len(errors)),
        "ok": len(errors) == 0,
        "errors": errors[:200],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
