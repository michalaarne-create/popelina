from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.brain.runtime.screen_quiz_parser import parse_screen_quiz_state


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line in text.splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
        return rows
    payload = json.loads(text)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("samples")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _resolve_path(manifest_dir: Path, raw: Any) -> Optional[Path]:
    if not raw:
        return None
    p = Path(str(raw))
    if not p.is_absolute():
        p = (manifest_dir / p).resolve()
    return p


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(float(v) for v in values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = max(0.0, min(1.0, q)) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _coerce_block_types(screen_state: Dict[str, Any]) -> List[str]:
    direct = screen_state.get("block_types")
    if isinstance(direct, list) and direct:
        return [str(v or "") for v in direct]
    out: List[str] = []
    for q in screen_state.get("questions") or []:
        if not isinstance(q, dict):
            continue
        bt = str(q.get("block_type") or "")
        if bt:
            out.append(bt)
    return out


def _evaluate_sample(
    *,
    sample: Dict[str, Any],
    manifest_dir: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    region_path = _resolve_path(manifest_dir, sample.get("region_json"))
    if not isinstance(region_path, Path) or not region_path.exists():
        return None, "missing_region_json"
    summary_path = _resolve_path(manifest_dir, sample.get("summary_json"))
    rated_path = _resolve_path(manifest_dir, sample.get("rated_json"))
    page_path = _resolve_path(manifest_dir, sample.get("page_json"))

    region_payload = _load_json(region_path)
    if not isinstance(region_payload, dict):
        return None, "invalid_region_json"
    summary_data = _load_json(summary_path) if isinstance(summary_path, Path) and summary_path.exists() else None
    rated_data = _load_json(rated_path) if isinstance(rated_path, Path) and rated_path.exists() else None
    page_data = _load_json(page_path) if isinstance(page_path, Path) and page_path.exists() else None

    t0 = time.perf_counter()
    screen_state = parse_screen_quiz_state(
        region_payload=region_payload,
        summary_data=summary_data,
        page_data=page_data,
        rated_data=rated_data,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    result = {
        "id": str(sample.get("id") or region_path.stem),
        "expected_global_type": str(sample.get("expected_global_type") or "").strip(),
        "pred_global_type": str(screen_state.get("detected_quiz_type") or "unknown"),
        "expected_block_types": [str(v or "") for v in (sample.get("expected_block_types") or []) if str(v or "").strip()],
        "pred_block_types": _coerce_block_types(screen_state),
        "confidence": float(screen_state.get("type_confidence") or 0.0),
        "margin": float(screen_state.get("decision_margin") or 0.0),
        "latency_ms": float(elapsed_ms),
    }
    return result, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark screen-only quiz type classifier.")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON/JSONL with benchmark samples.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional sample cap (0 = all).")
    parser.add_argument("--target-global-acc", type=float, default=0.99, help="Global type accuracy target.")
    parser.add_argument("--target-block-acc", type=float, default=0.97, help="Block type accuracy target.")
    parser.add_argument("--target-p95-ms", type=float, default=300.0, help="Latency p95 target in ms.")
    parser.add_argument("--enforce", action="store_true", help="Exit non-zero when targets are not met.")
    parser.add_argument("--out-json", default="", help="Optional output JSON path with detailed report.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return 2

    rows = _load_manifest(manifest_path)
    if args.max_samples > 0:
        rows = rows[: int(args.max_samples)]
    if not rows:
        print(f"[ERROR] No valid samples in manifest: {manifest_path}")
        return 2

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for row in rows:
        sample_res, err = _evaluate_sample(sample=row, manifest_dir=manifest_path.parent)
        if err:
            errors.append({"id": str(row.get("id") or row.get("region_json") or "?"), "error": err})
            continue
        if isinstance(sample_res, dict):
            results.append(sample_res)

    latencies = [float(r["latency_ms"]) for r in results]
    expected_global = [r for r in results if str(r.get("expected_global_type") or "").strip()]
    global_correct = sum(1 for r in expected_global if r["pred_global_type"] == r["expected_global_type"])
    global_acc = (float(global_correct) / float(len(expected_global))) if expected_global else 0.0

    expected_block_rows = [r for r in results if r.get("expected_block_types")]
    block_correct_rows = 0
    block_total_items = 0
    block_correct_items = 0
    for r in expected_block_rows:
        exp = [str(v) for v in r.get("expected_block_types") or []]
        pred = [str(v) for v in r.get("pred_block_types") or []]
        if exp == pred:
            block_correct_rows += 1
        n = min(len(exp), len(pred))
        block_total_items += len(exp)
        block_correct_items += sum(1 for i in range(n) if exp[i] == pred[i])
    block_acc_row = (float(block_correct_rows) / float(len(expected_block_rows))) if expected_block_rows else 0.0
    block_acc_item = (float(block_correct_items) / float(block_total_items)) if block_total_items > 0 else 0.0

    p50 = _percentile(latencies, 0.50)
    p95 = _percentile(latencies, 0.95)
    avg = float(statistics.mean(latencies)) if latencies else 0.0

    print(f"[BENCH] samples={len(rows)} ok={len(results)} errors={len(errors)}")
    print(f"[BENCH] global_type_acc={global_acc:.4f} ({global_correct}/{len(expected_global)}) target={args.target_global_acc:.4f}")
    print(
        "[BENCH] block_type_acc_row="
        f"{block_acc_row:.4f} ({block_correct_rows}/{len(expected_block_rows)}) "
        f"block_type_acc_item={block_acc_item:.4f} ({block_correct_items}/{block_total_items}) "
        f"target={args.target_block_acc:.4f}"
    )
    print(f"[BENCH] latency_ms avg={avg:.2f} p50={p50:.2f} p95={p95:.2f} target_p95={args.target_p95_ms:.2f}")

    report = {
        "manifest": str(manifest_path),
        "samples_total": len(rows),
        "samples_ok": len(results),
        "errors": errors,
        "metrics": {
            "global_type_accuracy": global_acc,
            "global_type_n": len(expected_global),
            "block_type_accuracy_row": block_acc_row,
            "block_type_accuracy_item": block_acc_item,
            "block_type_n_rows": len(expected_block_rows),
            "latency_ms_avg": avg,
            "latency_ms_p50": p50,
            "latency_ms_p95": p95,
        },
        "targets": {
            "global_type_accuracy": float(args.target_global_acc),
            "block_type_accuracy": float(args.target_block_acc),
            "latency_ms_p95": float(args.target_p95_ms),
        },
        "results": results,
    }

    if args.out_json:
        out_path = Path(args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[BENCH] report={out_path}")

    failed = (
        global_acc < float(args.target_global_acc)
        or block_acc_item < float(args.target_block_acc)
        or p95 > float(args.target_p95_ms)
    )
    if args.enforce and failed:
        print("[BENCH] FAIL target gate not met.")
        return 1
    print("[BENCH] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
