from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiz.gpu_ocr_runtime import ensure_local_ocr_cache_dirs, normalize_ocr_backend


def _default_python() -> str:
    preferred = ROOT.parent / ".venv312" / "Scripts" / "python.exe"
    if preferred.exists():
        return str(preferred.resolve())
    return sys.executable


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(payload, dict):
        rows = payload.get("rows") or payload.get("samples")
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def _resolve_from_source(base: Path, raw: Any) -> str:
    if not raw:
        return ""
    p = Path(str(raw))
    if not p.is_absolute():
        p = (base / p).resolve()
    return str(p)


def _normalize_rows_from_manifest(path: Path) -> List[Dict[str, Any]]:
    source_base = path.parent.parent
    normalized: List[Dict[str, Any]] = []
    for row in _read_manifest(path):
        item = dict(row)
        for key in ("image_path", "html_path", "label_path"):
            if key in item:
                item[key] = _resolve_from_source(source_base, item.get(key))
        item["_source_manifest"] = str(path)
        normalized.append(item)
    return normalized


def _write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.pop("_source_manifest", None)
        normalized.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": normalized}, ensure_ascii=False, indent=2), encoding="utf-8")


def _group_key(row: Dict[str, Any]) -> str:
    return str(row.get("expected_global_type") or "unknown").strip().lower() or "unknown"


def _balanced_sample(rows: List[Dict[str, Any]], per_class: int, seed: int) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(_group_key(row), []).append(row)
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for key in sorted(buckets.keys()):
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        take = bucket[: max(0, int(per_class))] if per_class > 0 else bucket
        out.extend(take)
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build larger screen benchmark and control_kind JSONL from rendered dataset manifest.")
    ap.add_argument("--dataset-manifest", required=True, action="append", help="Rendered dataset manifest JSON/JSONL. Can be provided multiple times.")
    ap.add_argument("--out-dir", required=True, help="Output directory for temporary and final artifacts.")
    ap.add_argument("--per-class", type=int, default=50, help="Max sampled rows per expected_global_type.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--python", default="", help="Python interpreter; defaults to current.")
    ap.add_argument("--region-grow-timeout-s", type=float, default=90.0)
    ap.add_argument("--rating-timeout-s", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers for screen benchmark build.")
    ap.add_argument("--resume", type=int, default=1, help="Resume/reuse existing screen benchmark artifacts.")
    ap.add_argument("--ocr-backend", default="cuda_fp16", help="Requested OCR backend for child benchmark.")
    ap.add_argument("--require-gpu", type=int, default=0, help="Require GPU OCR runtime in child benchmark.")
    ap.add_argument("--exec-mode", default="daemon", help="Execution mode hint for child region_grow benchmark.")
    args = ap.parse_args()

    manifest_paths = [Path(raw).resolve() for raw in (args.dataset_manifest or [])]
    missing = [str(path) for path in manifest_paths if not path.exists()]
    if missing:
        print(f"[ERROR] dataset manifest not found: {missing[0]}")
        return 2
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_local_ocr_cache_dirs()

    rows: List[Dict[str, Any]] = []
    for manifest_path in manifest_paths:
        rows.extend(_normalize_rows_from_manifest(manifest_path))
    if not rows:
        print(f"[ERROR] no rows in manifests: {', '.join(str(p) for p in manifest_paths)}")
        return 2
    sampled = _balanced_sample(rows, per_class=int(args.per_class), seed=int(args.seed))
    if not sampled:
        print("[ERROR] empty sampled set")
        return 2

    sampled_manifest = out_dir / "sampled_dataset_manifest.json"
    _write_manifest(sampled_manifest, sampled)

    py = str(Path(args.python).resolve()) if str(args.python or "").strip() else _default_python()
    screen_bench_dir = out_dir / "screen_benchmark"
    screen_builder = ROOT / "quiz" / "build_quiz_type_screen_benchmark.py"
    control_builder = ROOT / "quiz" / "build_control_kind_dataset.py"

    cmd_screen = [
        py,
        str(screen_builder),
        "--dataset-manifest",
        str(sampled_manifest),
        "--out-dir",
        str(screen_bench_dir),
        "--region-grow-timeout-s",
        str(args.region_grow_timeout_s),
        "--rating-timeout-s",
        str(args.rating_timeout_s),
        "--workers",
        str(args.workers),
        "--resume",
        str(args.resume),
        "--ocr-backend",
        normalize_ocr_backend(str(args.ocr_backend or "cuda_fp16")),
        "--require-gpu",
        str(int(args.require_gpu or 0)),
        "--exec-mode",
        str(args.exec_mode or "daemon"),
    ]
    screen = subprocess.run(cmd_screen, cwd=str(ROOT))
    if screen.returncode != 0:
        return int(screen.returncode)

    bench_manifest = screen_bench_dir / "benchmark_manifest.json"
    out_jsonl = out_dir / "control_kind_dataset.jsonl"
    cmd_control = [
        py,
        str(control_builder),
        "--manifest",
        str(bench_manifest),
        "--out-jsonl",
        str(out_jsonl),
    ]
    control = subprocess.run(cmd_control, cwd=str(ROOT))
    if control.returncode != 0:
        return int(control.returncode)

    print(json.dumps({
        "source_manifests": [str(p) for p in manifest_paths],
        "sampled_manifest": str(sampled_manifest),
        "sampled_rows": len(sampled),
        "screen_benchmark_manifest": str(bench_manifest),
        "control_kind_jsonl": str(out_jsonl),
        "ocr_backend_requested": str(args.ocr_backend),
        "gpu_required": bool(int(args.require_gpu or 0)),
        "exec_mode": str(args.exec_mode or "daemon"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
