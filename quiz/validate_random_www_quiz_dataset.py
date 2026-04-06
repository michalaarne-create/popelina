from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from random_quiz_sandbox import DEFAULT_PROFILE_MIX, GLOBAL_TYPES, parse_profile_mix


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def _normalize(value: str) -> str:
    return str(value or "").replace("\\", "/")


def _image_files(images_dir: Path) -> List[Path]:
    out: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        out.extend(images_dir.glob(pattern))
    return sorted(set(out))


def _counter_to_ratio(counter: Counter[str]) -> Dict[str, float]:
    total = sum(counter.values()) or 1
    return {k: float(v) / float(total) for k, v in sorted(counter.items())}


def _ratio_diff(actual: Dict[str, float], expected: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(set(actual.keys()) | set(expected.keys()))
    return {k: float(actual.get(k, 0.0) - expected.get(k, 0.0)) for k in keys}


def _parse_targets(value: str) -> Dict[str, float]:
    if not str(value or "").strip():
        return {}
    out: Dict[str, float] = {}
    for token in str(value).split(","):
        part = token.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        key, raw = part.split(":", 1)
        out[str(key).strip()] = float(raw.strip())
    return out


def validate_dataset(
    *,
    out_dir: Path,
    profile_mix: str,
    distribution_tolerance_pp: float,
    min_edge_coverage: Dict[str, float],
    sample_review_count: int,
) -> Dict[str, Any]:
    manifests_dir = out_dir / "manifests"
    manifest_path = manifests_dir / "dataset_manifest.json"
    manifest_jsonl_path = manifests_dir / "dataset_manifest.jsonl"
    if (not manifest_path.exists()) and (not manifest_jsonl_path.exists()):
        return {
            "ok": False,
            "error": "manifest_missing",
            "manifest_path": str(manifest_path.resolve()),
            "manifest_jsonl_path": str(manifest_jsonl_path.resolve()),
        }
    rows: List[Dict[str, Any]] = []
    if manifest_path.exists():
        manifest_obj = _read_json(manifest_path)
        rows = list(manifest_obj.get("rows") or [])
    elif manifest_jsonl_path.exists():
        for line in manifest_jsonl_path.read_text(encoding="utf-8").splitlines():
            raw = str(line or "").strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    html_dir = out_dir / "html"

    missing_path_count = 0
    missing_paths: List[Dict[str, str]] = []
    view_id_counter: Counter[str] = Counter()
    sample_signature: Dict[str, Tuple[str, str, str]] = {}
    sample_conflicts: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    global_counter: Counter[str] = Counter()
    block_counter: Counter[str] = Counter()
    profile_counter: Counter[str] = Counter()
    next_variant_counter: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()
    labels_seen = 0
    sample_global_seen: Dict[str, str] = {}
    sample_profile_seen: Dict[str, str] = {}

    for row in rows:
        sid = str(row.get("sample_id") or "")
        vid = str(row.get("view_id") or "")
        if vid:
            view_id_counter[vid] += 1
        if sid:
            sig = (
                str(row.get("expected_global_type") or ""),
                json.dumps(row.get("expected_block_types") or [], ensure_ascii=False, sort_keys=True),
                str(row.get("profile") or ""),
            )
            prev = sample_signature.get(sid)
            if prev is None:
                sample_signature[sid] = sig
            elif prev != sig:
                sample_conflicts[sid].append(sig)
        gt = str(row.get("expected_global_type") or "")
        if sid and gt:
            if sid not in sample_global_seen:
                sample_global_seen[sid] = gt
                global_counter[gt] += 1
        elif (not sid) and gt:
            global_counter[gt] += 1
        for key in ("image_path", "label_path", "html_path"):
            value = str(row.get(key) or "")
            if not value:
                continue
            full = out_dir / _normalize(value)
            if not full.exists():
                missing_path_count += 1
                if len(missing_paths) < 50:
                    missing_paths.append({"sample_id": sid, "key": key, "path": value})

        label_path = str(row.get("label_path") or "")
        if not label_path:
            continue
        lp = out_dir / _normalize(label_path)
        if not lp.exists():
            continue
        labels_seen += 1
        try:
            label = _read_json(lp)
        except Exception:
            continue
        profile = str(label.get("profile") or (label.get("ui_flags") or {}).get("profile") or "")
        if profile:
            if sid:
                if sid not in sample_profile_seen:
                    sample_profile_seen[sid] = profile
                    profile_counter[profile] += 1
            else:
                profile_counter[profile] += 1
        ui_flags = label.get("ui_flags") or {}
        nv = str(ui_flags.get("next_variant") or "")
        if nv:
            next_variant_counter[nv] += 1
        if bool(label.get("partial_next_question_visible")):
            edge_counts["partial_next_question_visible"] += 1
        for k in ("mobile_narrow", "sticky_header", "sticky_footer", "modal_overlay", "floating_help", "sidebar_noise", "loading_stub"):
            if bool(ui_flags.get(k)):
                edge_counts[k] += 1
        for block in list(label.get("blocks") or []):
            btype = str(block.get("type") or "")
            if btype:
                block_counter[btype] += 1
            if block.get("question_meta"):
                edge_counts["question_meta"] += 1
            if bool(block.get("show_validation_error")):
                edge_counts["validation_error"] += 1
            if block.get("disabled_indices"):
                edge_counts["disabled_options"] += 1
            if block.get("preselected_indices"):
                edge_counts["preselected_options"] += 1
            if any(block.get("option_subtitles") or []):
                edge_counts["option_subtitles"] += 1
            if any(block.get("option_icons") or []):
                edge_counts["option_icons"] += 1
            if bool(block.get("textarea_variant")):
                edge_counts["textarea_variant"] += 1
            if bool(block.get("placeholder_trap")):
                edge_counts["placeholder_trap"] += 1
            if block.get("field_helper"):
                edge_counts["field_helper"] += 1

    duplicate_view_ids = {k: v for k, v in view_id_counter.items() if v > 1}
    actual_global = _counter_to_ratio(global_counter)
    target_global = _counter_to_ratio(Counter({k: 1 for k in GLOBAL_TYPES}))
    global_diff = _ratio_diff(actual_global, target_global)
    target_blocks = _counter_to_ratio(Counter({k: 1 for k in ("single", "multi", "dropdown", "dropdown_scroll", "slider", "text")}))
    actual_blocks = _counter_to_ratio(block_counter)
    block_diff = _ratio_diff(actual_blocks, target_blocks)

    mix_rows = parse_profile_mix(profile_mix if str(profile_mix or "").strip() else DEFAULT_PROFILE_MIX)
    target_profile = {k: w for k, w in mix_rows}
    actual_profile = _counter_to_ratio(profile_counter) if profile_counter else {}
    profile_diff = _ratio_diff(actual_profile, target_profile)

    total_labels = max(1, labels_seen)
    edge_ratio = {k: float(v) / float(total_labels) for k, v in sorted(edge_counts.items())}

    tolerance = float(distribution_tolerance_pp) / 100.0
    failed_global = {k: d for k, d in global_diff.items() if abs(d) > tolerance}
    failed_blocks = {k: d for k, d in block_diff.items() if abs(d) > tolerance}
    failed_profile = {k: d for k, d in profile_diff.items() if abs(d) > tolerance}

    failed_edge = {}
    for key, min_ratio in (min_edge_coverage or {}).items():
        got = float(edge_ratio.get(key, 0.0))
        if got < float(min_ratio):
            failed_edge[key] = {"min": float(min_ratio), "got": got}

    next_variant_ratio = _counter_to_ratio(next_variant_counter)
    next_variant_anomaly = {}
    for key, ratio in next_variant_ratio.items():
        if ratio > 0.85:
            next_variant_anomaly[key] = ratio

    review_candidates: List[str] = []
    if sample_review_count > 0:
        rng = random.Random(1337)
        image_paths = _image_files(images_dir)
        if image_paths:
            take = min(int(sample_review_count), len(image_paths))
            review_candidates = [_safe_rel(p, out_dir) for p in rng.sample(image_paths, take)]

    quality_gates = {
        "missing_paths": missing_path_count == 0,
        "unique_view_ids": len(duplicate_view_ids) == 0,
        "unique_sample_ids": len(sample_conflicts) == 0,
        "global_distribution": len(failed_global) == 0,
        "block_distribution": len(failed_blocks) == 0,
        "profile_distribution": (len(failed_profile) == 0) if target_profile else True,
        "edge_coverage": len(failed_edge) == 0,
        "next_variant_anomaly": len(next_variant_anomaly) == 0,
    }

    report = {
        "ok": all(bool(v) for v in quality_gates.values()),
        "out_dir": str(out_dir.resolve()),
        "counts": {
            "rows": len(rows),
            "labels_seen": labels_seen,
            "missing_path_count": missing_path_count,
            "duplicate_view_id_count": len(duplicate_view_ids),
            "sample_id_conflict_count": len(sample_conflicts),
            "images_count": len(_image_files(images_dir)),
            "labels_count": len(list(labels_dir.glob("*.json"))),
            "html_count": len(list(html_dir.glob("*.html"))),
        },
        "quality_gates": quality_gates,
        "distribution": {
            "global_actual": actual_global,
            "global_target_uniform": target_global,
            "global_diff": global_diff,
            "global_failed": failed_global,
            "block_actual": actual_blocks,
            "block_target_uniform": target_blocks,
            "block_diff": block_diff,
            "block_failed": failed_blocks,
            "profile_actual": actual_profile,
            "profile_target": target_profile,
            "profile_diff": profile_diff,
            "profile_failed": failed_profile,
            "next_variant_ratio": next_variant_ratio,
            "next_variant_anomaly": next_variant_anomaly,
        },
        "edge_coverage": {
            "actual_ratio": edge_ratio,
            "min_required": min_edge_coverage,
            "failed": failed_edge,
        },
        "issues": {
            "missing_paths": missing_paths,
            "duplicate_view_ids": duplicate_view_ids,
            "sample_id_conflicts": {k: v for k, v in sample_conflicts.items()},
        },
        "review_samples": review_candidates,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated random WWW quiz dataset.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--profile-mix", type=str, default=DEFAULT_PROFILE_MIX)
    parser.add_argument("--distribution-tolerance-pp", type=float, default=1.5)
    parser.add_argument("--min-edge-coverage", type=str, default="")
    parser.add_argument("--sample-review-count", type=int, default=40)
    parser.add_argument("--report-path", type=str, default="")
    args = parser.parse_args()

    report = validate_dataset(
        out_dir=Path(args.out_dir).resolve(),
        profile_mix=str(args.profile_mix or DEFAULT_PROFILE_MIX),
        distribution_tolerance_pp=float(args.distribution_tolerance_pp),
        min_edge_coverage=_parse_targets(str(args.min_edge_coverage or "")),
        sample_review_count=max(0, int(args.sample_review_count)),
    )
    if args.report_path:
        out = Path(args.report_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(out))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if bool(report.get("ok")) else 2)


if __name__ == "__main__":
    main()
