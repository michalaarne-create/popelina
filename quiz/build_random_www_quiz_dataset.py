from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from random_quiz_sandbox import DEFAULT_PROFILE_MIX, build_samples, parse_profile_mix
from validate_random_www_quiz_dataset import validate_dataset


ROOT = Path(__file__).resolve().parents[1]


DOM_ANNOTATIONS_SCRIPT = """
() => {
  const rect = (el) => {
    const r = el.getBoundingClientRect();
    return [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)];
  };
  const rows = (sel, kind) => Array.from(document.querySelectorAll(sel)).map((el) => ({
    kind,
    text: (el.textContent || '').trim(),
    bbox: rect(el),
    attrs: {
      block_id: el.getAttribute('data-block-id') || '',
      block_type: el.getAttribute('data-block-type') || '',
      answer_index: el.getAttribute('data-answer-index') || '',
      answer_text: el.getAttribute('data-answer-text') || '',
    },
  }));
  return {
    viewport: {
      width: window.innerWidth,
      height: window.innerHeight,
      scroll_x: Math.round(window.scrollX || 0),
      scroll_y: Math.round(window.scrollY || 0),
      doc_w: Math.max(document.documentElement.scrollWidth, document.body ? document.body.scrollWidth : 0),
      doc_h: Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0),
    },
    blocks: rows('[data-role="quiz-block"]', 'block'),
    questions: rows('[data-role="question"]', 'question'),
    answers: rows('[data-role="answer"]', 'answer'),
    selects: rows('[data-role="select"]', 'select'),
    text_inputs: rows('[data-role="text-input"]', 'text_input'),
    next_buttons: rows('[data-role="next"]', 'next'),
    hero: rows('[data-role="hero"]', 'hero'),
    hero_titles: rows('[data-role="hero-title"]', 'hero_title'),
    hero_descs: rows('[data-role="hero-desc"]', 'hero_desc'),
    nav: rows('[data-role="nav"]', 'nav'),
    nav_items: rows('[data-role="nav-item"]', 'nav_item'),
    hints: rows('[data-role="hint"]', 'hint'),
    noise: rows('[data-role="noise"]', 'noise'),
    secondary_cta: rows('[data-role="secondary-cta"]', 'secondary_cta'),
  };
}
"""


def _collect_dom_annotations(page: Any) -> Dict[str, Any]:
    return page.evaluate(DOM_ANNOTATIONS_SCRIPT)


def _max_scroll_y(page: Any) -> int:
    return int(
        page.evaluate(
            "() => Math.max(0, Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0) - window.innerHeight)"
        )
        or 0
    )


def _wait_render_stable(page: Any, include_fonts: bool = False) -> None:
    if include_fonts:
        try:
            page.evaluate(
                """() => {
                    if (document.fonts && document.fonts.ready) {
                        return document.fonts.ready.then(() => true);
                    }
                    return true;
                }"""
            )
        except Exception:
            pass
    try:
        page.evaluate("() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
    except Exception:
        pass


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve())


def _try_import_playwright() -> Optional[Any]:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore

        return sync_playwright
    except Exception:
        return None


def _write_json(path: Path, payload: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        body = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    path.write_text(body, encoding="utf-8")


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    total_safe = max(1, int(total))
    ratio = max(0.0, min(1.0, float(done) / float(total_safe)))
    filled = int(round(ratio * width))
    return "[" + ("#" * filled) + ("-" * max(0, width - filled)) + "]"


def _print_progress(prefix: str, done: int, total: int, started_at: float, extra: str = "") -> None:
    total_safe = max(1, int(total))
    done_safe = max(0, min(int(done), total_safe))
    ratio = float(done_safe) / float(total_safe)
    elapsed = max(0.001, time.time() - float(started_at))
    rate = float(done_safe) / elapsed
    eta = max(0.0, float(total_safe - done_safe) / max(0.001, rate))
    pct = ratio * 100.0
    line = (
        f"\r{prefix} {_progress_bar(done_safe, total_safe)} "
        f"{done_safe}/{total_safe} {pct:6.2f}% {rate:6.2f} it/s ETA {eta:7.1f}s"
    )
    if extra:
        line += f" | {extra}"
    print(line, end="", flush=True)
    if done_safe >= total_safe:
        print("", flush=True)


def _read_manifest_rows(manifests_dir: Path) -> List[Dict[str, Any]]:
    manifest_json = manifests_dir / "dataset_manifest.json"
    manifest_jsonl = manifests_dir / "dataset_manifest.jsonl"
    rows: List[Dict[str, Any]] = []
    if manifest_json.exists():
        try:
            obj = json.loads(manifest_json.read_text(encoding="utf-8"))
            raw = obj.get("rows") if isinstance(obj, dict) else []
            if isinstance(raw, list):
                rows = [r for r in raw if isinstance(r, dict)]
        except Exception:
            rows = []
    elif manifest_jsonl.exists():
        try:
            for line in manifest_jsonl.read_text(encoding="utf-8").splitlines():
                raw = str(line or "").strip()
                if not raw:
                    continue
                item = json.loads(raw)
                if isinstance(item, dict):
                    rows.append(item)
        except Exception:
            rows = []
    return rows


def _get_existing_run_id(out_dir: Path) -> str:
    summary_path = out_dir / "manifests" / "dataset_summary.json"
    if not summary_path.exists():
        return ""
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    value = str(summary.get("run_id") or "").strip()
    return value


def _is_completed_part(part_dir: Path, expected_count: int) -> bool:
    summary_path = part_dir / "manifests" / "dataset_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if int(summary.get("count_requested") or 0) != int(expected_count):
        return False
    if int(summary.get("total_samples") or 0) < int(expected_count):
        return False
    if summary.get("errors"):
        return False
    return True


def _estimate_default_workers() -> int:
    cpu = max(1, int(os.cpu_count() or 1))
    by_cpu = max(1, min(12, cpu // 2 if cpu >= 4 else 1))
    try:
        import psutil  # type: ignore

        ram_gb = float(psutil.virtual_memory().total) / (1024.0**3)
        by_ram = max(1, int(ram_gb // 3.5))
        return max(1, min(by_cpu, by_ram))
    except Exception:
        return by_cpu


def _normalize_profile_mix(value: str) -> str:
    parsed = parse_profile_mix(value if str(value or "").strip() else DEFAULT_PROFILE_MIX)
    return ",".join(f"{name}:{weight:.6f}" for name, weight in parsed)


def _parse_ratio_map(value: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for token in str(value or "").split(","):
        part = token.strip()
        if not part or ":" not in part:
            continue
        key, raw = part.split(":", 1)
        try:
            out[str(key).strip()] = float(raw.strip())
        except Exception:
            continue
    return out


def build_dataset(
    *,
    count: int,
    seed: int,
    out_dir: Path,
    headless: bool,
    timeout_ms: int,
    html_only: bool,
    cdp_endpoint: str,
    balanced: bool,
    difficulty: str = "hard",
    profile_mix: str = "",
    start_index: int = 0,
    run_id: str = "",
    shard_id: str = "",
    shard_count: int = 1,
    retry_count: int = 1,
    heartbeat_every: int = 200,
    resume: bool = True,
) -> Dict[str, Any]:
    profile_mix_norm = _normalize_profile_mix(profile_mix)
    samples = build_samples(
        count=max(1, int(count)),
        seed=int(seed),
        balanced=bool(balanced),
        start_index=int(start_index),
        difficulty=str(difficulty or "hard"),
        profile_mix=profile_mix_norm,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    html_dir = out_dir / "html"
    manifests_dir = out_dir / "manifests"
    for p in (images_dir, labels_dir, html_dir, manifests_dir):
        p.mkdir(parents=True, exist_ok=True)

    existing_rows = _read_manifest_rows(manifests_dir) if bool(resume) else []
    manifest_by_view_id: Dict[str, Dict[str, Any]] = {}
    for row in existing_rows:
        vid = str(row.get("view_id") or "").strip()
        if vid:
            manifest_by_view_id[vid] = dict(row)
    manifest_rows: List[Dict[str, Any]] = list(manifest_by_view_id.values())
    errors: List[Dict[str, Any]] = []
    per_type: Dict[str, int] = {}
    per_profile: Dict[str, int] = {}
    total_views = len(manifest_rows)
    processed_samples = 0
    started_at = time.time()
    existing_labels: Set[str] = {p.stem for p in labels_dir.glob("*.json")} if bool(resume) else set()
    existing_images: Set[str] = {p.stem for p in images_dir.glob("*.png")} if bool(resume) else set()
    existing_html: Set[str] = {p.stem for p in html_dir.glob("*.html")} if bool(resume) else set()

    for row in samples:
        gt = str(row.get("global_type") or "unknown")
        per_type[gt] = int(per_type.get(gt) or 0) + 1
        pf = str(row.get("profile") or "unknown")
        per_profile[pf] = int(per_profile.get(pf) or 0) + 1

    sync_playwright = _try_import_playwright()
    if (not html_only) and sync_playwright is None:
        raise RuntimeError(
            "Brak playwright w interpreterze. Uzyj .venv312 albo uruchom z --html-only. "
            "Instalacja: python -m pip install playwright. "
            "Jesli masz juz otwarta przegladarke CDP, mozesz podac --cdp-endpoint."
        )

    if html_only:
        for sample in samples:
            sid = str(sample["sample_id"])
            view_id = f"{sid}__html"
            if bool(resume):
                row0 = manifest_by_view_id.get(view_id)
                label_ok = sid in existing_labels
                html_ok = sid in existing_html
                if row0 is not None and label_ok and html_ok:
                    processed_samples += 1
                    if int(heartbeat_every) > 0 and (
                        (processed_samples % int(heartbeat_every) == 0) or (processed_samples == len(samples))
                    ):
                        _print_progress(
                            f"[build_dataset][{shard_id or 'single'}]",
                            processed_samples,
                            len(samples),
                            started_at,
                            extra=f"views={total_views} errors={len(errors)}",
                        )
                    continue
            html_path = html_dir / f"{sid}.html"
            html_path.write_text(str(sample["html"]), encoding="utf-8")
            label = {
                "sample_id": sid,
                "run_id": run_id,
                "shard_id": shard_id,
                "shard_count": int(shard_count),
                "profile": sample.get("profile") or "hard",
                "global_type": sample["global_type"],
                "block_types": sample["block_types"],
                "partial_next_question_visible": bool(sample.get("partial_next_question_visible")),
                "ui_flags": sample.get("ui_flags") or {},
                "has_next": sample["has_next"],
                "auto_next": sample["auto_next"],
                "require_scroll": sample["require_scroll"],
                "viewport": sample["viewport"],
                "blocks": sample["blocks"],
                "style": sample["style"],
                "render": {"mode": "html_only"},
            }
            label_path = labels_dir / f"{sid}.json"
            _write_json(label_path, label, pretty=False)
            row = {
                "sample_id": sid,
                "view_id": view_id,
                "expected_global_type": sample["global_type"],
                "expected_block_types": sample["block_types"],
                "image_path": "",
                "html_path": _relative(html_path, out_dir),
                "label_path": _relative(label_path, out_dir),
            }
            if view_id not in manifest_by_view_id:
                total_views += 1
            manifest_by_view_id[view_id] = row
            existing_labels.add(sid)
            existing_html.add(sid)
            processed_samples += 1
            if int(heartbeat_every) > 0 and (
                (processed_samples % int(heartbeat_every) == 0) or (processed_samples == len(samples))
            ):
                _print_progress(
                    f"[build_dataset][{shard_id or 'single'}]",
                    processed_samples,
                    len(samples),
                    started_at,
                    extra=f"views={total_views} errors={len(errors)}",
                )
    else:
        try:
            with sync_playwright() as pw:
                cdp = str(cdp_endpoint or "").strip()
                shared_context = None
                local_pages: Dict[str, Tuple[Any, Any]] = {}
                if cdp:
                    browser = pw.chromium.connect_over_cdp(cdp, timeout=int(timeout_ms))
                    shared_context = browser.contexts[0] if browser.contexts else browser.new_context()
                    cdp_page = shared_context.new_page()
                    close_browser = False
                else:
                    browser = pw.chromium.launch(headless=bool(headless))
                    close_browser = True
                try:
                    for sample in samples:
                        sid = str(sample["sample_id"])
                        rng = random.Random(int(sample["seed"]) * 97 + int(sample["index"]) * 37)
                        html = str(sample["html"])
                        html_path = html_dir / f"{sid}.html"
                        html_path.write_text(html, encoding="utf-8")
                        viewport = sample.get("viewport") if isinstance(sample.get("viewport"), dict) else {}
                        vw = int(viewport.get("width") or 1366)
                        vh = int(viewport.get("height") or 900)

                        if shared_context is None:
                            color_scheme = "dark" if str(sample["style"].get("bg_a", "")).startswith("#0") else "light"
                            if color_scheme not in local_pages:
                                ctx = browser.new_context(
                                    viewport={"width": vw, "height": vh},
                                    color_scheme=color_scheme,
                                    device_scale_factor=1.0,
                                )
                                pg = ctx.new_page()
                                local_pages[color_scheme] = (ctx, pg)
                            context, page = local_pages[color_scheme]
                            page.set_viewport_size({"width": vw, "height": vh})
                        else:
                            context = shared_context
                            page = cdp_page
                            page.set_viewport_size({"width": vw, "height": vh})
                        try:
                            sample_done = False
                            last_error = ""
                            for attempt in range(max(1, int(retry_count)) + 1):
                                try:
                                    page.set_content(html, wait_until="domcontentloaded", timeout=int(timeout_ms))
                                    _wait_render_stable(page, include_fonts=True)
                                    max_scroll = _max_scroll_y(page)
                                    states: List[Tuple[str, int]] = [("top", 0)]
                                    if bool(sample.get("require_scroll")) and max_scroll > 120:
                                        states.append(("mid", int(max_scroll * rng.uniform(0.35, 0.6))))
                                        states.append(("bottom", int(max_scroll * rng.uniform(0.78, 0.98))))
                                    last_scroll_y = int(page.evaluate("() => Math.round(window.scrollY || 0)") or 0)
                                    for state_name, y in states:
                                        view_id = f"{sid}__{state_name}"
                                        if bool(resume):
                                            row0 = manifest_by_view_id.get(view_id)
                                            label_ok = view_id in existing_labels
                                            image_ok = view_id in existing_images
                                            html_ok = sid in existing_html
                                            if row0 is not None and label_ok and image_ok and html_ok:
                                                continue
                                        y_clamped = max(0, min(max_scroll, int(y)))
                                        if y_clamped != last_scroll_y:
                                            page.evaluate(f"window.scrollTo(0, {y_clamped});")
                                            _wait_render_stable(page, include_fonts=False)
                                            last_scroll_y = y_clamped

                                        image_path = images_dir / f"{view_id}.jpg"
                                        page.screenshot(path=str(image_path), type="jpeg", quality=80, full_page=False, animations="disabled")
                                        dom_info = _collect_dom_annotations(page)

                                        label_payload = {
                                            "sample_id": sid,
                                            "view_id": view_id,
                                            "run_id": run_id,
                                            "shard_id": shard_id,
                                            "shard_count": int(shard_count),
                                            "profile": sample.get("profile") or "hard",
                                            "global_type": sample["global_type"],
                                            "block_types": sample["block_types"],
                                            "partial_next_question_visible": bool(sample.get("partial_next_question_visible")),
                                            "ui_flags": sample.get("ui_flags") or {},
                                            "has_next": sample["has_next"],
                                            "auto_next": sample["auto_next"],
                                            "require_scroll": sample["require_scroll"],
                                            "blocks": sample["blocks"],
                                            "style": sample["style"],
                                            "viewport": sample["viewport"],
                                            "render": {
                                                "state": state_name,
                                                "scroll_y": int(dom_info.get("viewport", {}).get("scroll_y", y_clamped)),
                                                "max_scroll": max_scroll,
                                                "attempt": attempt,
                                            },
                                            "dom": dom_info,
                                        }
                                        label_path = labels_dir / f"{view_id}.json"
                                        _write_json(label_path, label_payload, pretty=False)

                                        row = {
                                            "sample_id": sid,
                                            "view_id": view_id,
                                            "run_id": run_id,
                                            "shard_id": shard_id,
                                            "profile": sample.get("profile") or "hard",
                                            "expected_global_type": sample["global_type"],
                                            "expected_block_types": sample["block_types"],
                                            "image_path": _relative(image_path, out_dir),
                                            "html_path": _relative(html_path, out_dir),
                                            "label_path": _relative(label_path, out_dir),
                                        }
                                        if view_id not in manifest_by_view_id:
                                            total_views += 1
                                        manifest_by_view_id[view_id] = row
                                        existing_labels.add(view_id)
                                        existing_images.add(view_id)
                                        existing_html.add(sid)
                                    sample_done = True
                                    break
                                except Exception as exc:
                                    last_error = str(exc)
                                    page.wait_for_timeout(20)
                            if not sample_done:
                                errors.append({"sample_id": sid, "error": last_error, "retries": int(retry_count)})
                            processed_samples += 1
                            if int(heartbeat_every) > 0 and (
                                (processed_samples % int(heartbeat_every) == 0) or (processed_samples == len(samples))
                            ):
                                _print_progress(
                                    f"[build_dataset][{shard_id or 'single'}]",
                                    processed_samples,
                                    len(samples),
                                    started_at,
                                    extra=f"views={total_views} errors={len(errors)}",
                                )
                        finally:
                            pass
                finally:
                    if shared_context is not None:
                        try:
                            cdp_page.close()
                        except Exception:
                            pass
                    for ctx, pg in local_pages.values():
                        try:
                            pg.close()
                        except Exception:
                            pass
                        try:
                            ctx.close()
                        except Exception:
                            pass
                    if close_browser:
                        browser.close()
        except Exception as exc:
            raise RuntimeError(
                "Playwright render failed. Probable local policy/sandbox issue. "
                "Try: --html-only OR --cdp-endpoint http://127.0.0.1:9222 in normal user session."
            ) from exc

    manifest_rows = list(manifest_by_view_id.values())
    manifest_rows.sort(key=lambda x: (str(x.get("sample_id") or ""), str(x.get("view_id") or "")))
    manifest_json = manifests_dir / "dataset_manifest.json"
    _write_json(manifest_json, {"rows": manifest_rows}, pretty=False)
    manifest_jsonl = manifests_dir / "dataset_manifest.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "created_at_unix": int(time.time()),
        "count_requested": int(count),
        "seed": int(seed),
        "run_id": str(run_id or ""),
        "shard_id": str(shard_id or ""),
        "shard_count": int(shard_count),
        "profile_mix": profile_mix_norm,
        "total_samples": len(samples),
        "total_views": int(total_views),
        "per_global_type": per_type,
        "per_profile": per_profile,
        "errors": errors,
        "out_dir": str(out_dir.resolve()),
        "manifests": {
            "json": str(manifest_json.resolve()),
            "jsonl": str(manifest_jsonl.resolve()),
        },
    }
    _write_json(manifests_dir / "dataset_summary.json", summary)
    return summary


def _merge_parts(
    out_dir: Path,
    part_dirs: List[Path],
    count_requested: int,
    seed: int,
    run_id: str,
    profile_mix: str,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    html_dir = out_dir / "html"
    manifests_dir = out_dir / "manifests"
    for p in (images_dir, labels_dir, html_dir, manifests_dir):
        p.mkdir(parents=True, exist_ok=True)

    merged_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    per_type: Dict[str, int] = {}
    per_profile: Dict[str, int] = {}
    total_samples = 0
    total_views = 0

    for part_dir in part_dirs:
        path_remap: Dict[str, str] = {}
        summary_path = part_dir / "manifests" / "dataset_summary.json"
        manifest_path = part_dir / "manifests" / "dataset_manifest.json"
        if not summary_path.exists() or not manifest_path.exists():
            errors.append({"part_dir": str(part_dir), "error": "missing_manifest_or_summary"})
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows = manifest_obj.get("rows") if isinstance(manifest_obj, dict) else []
            if not isinstance(rows, list):
                rows = []
        except Exception as exc:
            errors.append({"part_dir": str(part_dir), "error": str(exc)})
            continue

        total_samples += int(summary.get("total_samples") or 0)
        total_views += int(summary.get("total_views") or 0)
        for k, v in (summary.get("per_global_type") or {}).items():
            per_type[str(k)] = int(per_type.get(str(k), 0) + int(v))
        for k, v in (summary.get("per_profile") or {}).items():
            per_profile[str(k)] = int(per_profile.get(str(k), 0) + int(v))
        if isinstance(summary.get("errors"), list):
            for e in summary["errors"]:
                errors.append({"part_dir": str(part_dir), "part_error": e})

        for sub_name in ("images", "labels", "html"):
            src = part_dir / sub_name
            dst = out_dir / sub_name
            if not src.exists():
                continue
            for item in src.iterdir():
                if not item.is_file():
                    continue
                target = dst / item.name
                if target.exists():
                    stem = target.stem
                    suffix = target.suffix
                    target = dst / f"{stem}__dup{int(time.time()*1000)}{suffix}"
                shutil.copy2(item, target)
                path_remap[f"{sub_name}/{item.name}".replace("\\", "/")] = f"{sub_name}/{target.name}".replace("\\", "/")

        for row in rows:
            if not isinstance(row, dict):
                continue
            merged_row = dict(row)
            for key in ("image_path", "label_path", "html_path"):
                value = merged_row.get(key)
                if not value:
                    continue
                normalized = str(value).replace("\\", "/")
                merged_row[key] = path_remap.get(normalized, normalized)
            merged_rows.append(merged_row)

    view_id_seen: Dict[str, int] = {}
    sample_signature: Dict[str, Tuple[str, str, str]] = {}
    sample_conflicts: Dict[str, List[Tuple[str, str, str]]] = {}
    for row in merged_rows:
        vid = str(row.get("view_id") or "")
        if vid:
            view_id_seen[vid] = int(view_id_seen.get(vid, 0) + 1)
        sid = str(row.get("sample_id") or "")
        if not sid:
            continue
        sig = (
            str(row.get("expected_global_type") or ""),
            json.dumps(row.get("expected_block_types") or [], ensure_ascii=False, sort_keys=True),
            str(row.get("profile") or ""),
        )
        prev = sample_signature.get(sid)
        if prev is None:
            sample_signature[sid] = sig
        elif prev != sig:
            if sid not in sample_conflicts:
                sample_conflicts[sid] = [prev]
            sample_conflicts[sid].append(sig)
    duplicate_view_ids = {k: v for k, v in view_id_seen.items() if v > 1}

    manifest_json = manifests_dir / "dataset_manifest.json"
    _write_json(manifest_json, {"rows": merged_rows}, pretty=False)
    manifest_jsonl = manifests_dir / "dataset_manifest.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "created_at_unix": int(time.time()),
        "count_requested": int(count_requested),
        "seed": int(seed),
        "run_id": str(run_id or ""),
        "profile_mix": str(profile_mix or ""),
        "shard_count": int(len(part_dirs)),
        "total_samples": int(total_samples),
        "total_views": int(total_views),
        "per_global_type": per_type,
        "per_profile": per_profile,
        "errors": errors,
        "quality_gates": {
            "unique_view_ids": len(duplicate_view_ids) == 0,
            "unique_sample_ids": len(sample_conflicts) == 0,
        },
        "duplicate_view_ids": duplicate_view_ids,
        "sample_id_conflicts": sample_conflicts,
        "out_dir": str(out_dir.resolve()),
        "manifests": {
            "json": str(manifest_json.resolve()),
            "jsonl": str(manifest_jsonl.resolve()),
        },
    }
    _write_json(manifests_dir / "dataset_summary.json", summary)
    return summary


def _prune_html(out_dir: Path) -> Dict[str, Any]:
    html_dir = out_dir / "html"
    if not html_dir.exists():
        return {"pruned_html": False, "removed_html_files": 0}
    removed = 0
    for fp in html_dir.glob("*.html"):
        try:
            fp.unlink(missing_ok=True)
            removed += 1
        except Exception:
            continue
    return {"pruned_html": True, "removed_html_files": int(removed)}


def _default_min_edge_coverage() -> Dict[str, float]:
    return {
        "mobile_narrow": 0.03,
        "sticky_header": 0.03,
        "sticky_footer": 0.03,
        "modal_overlay": 0.02,
        "floating_help": 0.03,
        "sidebar_noise": 0.03,
        "loading_stub": 0.02,
        "question_meta": 0.06,
        "validation_error": 0.015,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fully automatic randomized WWW-like quiz dataset.")
    parser.add_argument("--count", type=int, default=1000, help="How many base samples to generate.")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic random seed.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "data" / "benchmarks" / "random_www_quiz"),
        help="Output directory for images, labels and manifests.",
    )
    parser.add_argument("--headless", type=int, default=1, help="1=headless browser screenshots, 0=visible browser.")
    parser.add_argument("--timeout-ms", type=int, default=8000, help="Per-page render timeout.")
    parser.add_argument("--html-only", action="store_true", help="Generate html+labels only (no screenshots).")
    parser.add_argument(
        "--cdp-endpoint",
        type=str,
        default="",
        help="Optional existing Chromium CDP endpoint, e.g. http://127.0.0.1:9222",
    )
    parser.add_argument("--workers", type=int, default=0, help="Parallel shard workers; 0=auto by CPU/RAM.")
    parser.add_argument("--shard-size", type=int, default=0, help="Optional fixed shard size, e.g. 10000.")
    parser.add_argument("--balanced", type=int, default=1, help="1=balanced global classes, 0=weighted distribution.")
    parser.add_argument("--difficulty", type=str, default="hard", help="normal|hard|very_hard")
    parser.add_argument("--profile-mix", type=str, default=DEFAULT_PROFILE_MIX, help="Profile mix, e.g. normal:0.70,hard:0.25,very_hard:0.05")
    parser.add_argument("--run-id", type=str, default="", help="Optional run identifier for manifests/summaries.")
    parser.add_argument("--retry-count", type=int, default=1, help="Render retries per sample on transient failure.")
    parser.add_argument("--heartbeat-every", type=int, default=200, help="Progress heartbeat interval in samples.")
    parser.add_argument("--validate-only", type=int, default=0, help="1=only run validator on existing output directory.")
    parser.add_argument("--distribution-tolerance-pp", type=float, default=1.5, help="Allowed distribution drift in percentage points.")
    parser.add_argument("--sample-review-count", type=int, default=40, help="How many random samples to output for manual review.")
    parser.add_argument("--min-edge-coverage", type=str, default="", help="CSV map e.g. mobile_narrow:0.03,modal_overlay:0.02")
    parser.add_argument("--prune-html-after-validate", type=int, default=1, help="1=remove html files after successful validation.")
    parser.add_argument("--start-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--shard-id", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--resume",
        type=int,
        default=1,
        help="1=resume after restart: skip completed shards and continue partially generated shard views, 0=rerun all.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    requested_run_id = str(args.run_id or "").strip()
    if requested_run_id:
        run_id = requested_run_id
    elif bool(int(args.resume)):
        run_id = _get_existing_run_id(out_dir) or f"run_{int(time.time())}"
    else:
        run_id = f"run_{int(time.time())}"
    profile_mix_norm = _normalize_profile_mix(str(args.profile_mix or DEFAULT_PROFILE_MIX))
    min_edge = _parse_ratio_map(str(args.min_edge_coverage or "")) if str(args.min_edge_coverage or "").strip() else _default_min_edge_coverage()
    workers = int(args.workers)
    if workers <= 0:
        workers = _estimate_default_workers()
    workers = max(1, workers)

    if bool(int(args.validate_only)):
        report = validate_dataset(
            out_dir=out_dir,
            profile_mix=profile_mix_norm,
            distribution_tolerance_pp=float(args.distribution_tolerance_pp),
            min_edge_coverage=min_edge,
            sample_review_count=max(0, int(args.sample_review_count)),
        )
        report_path = out_dir / "manifests" / "validation_report.json"
        _write_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(0 if bool(report.get("ok")) else 2)

    shard_size = max(0, int(args.shard_size))
    total_count = max(1, int(args.count))
    if shard_size > 0:
        part_counts: List[int] = []
        left = int(total_count)
        while left > 0:
            part_counts.append(min(left, shard_size))
            left -= shard_size
    elif workers == 1:
        part_counts = [int(total_count)]
    else:
        base = int(total_count) // workers
        rem = int(total_count) % workers
        part_counts = [base + (1 if i < rem else 0) for i in range(workers)]

    # Single shard path without subprocess overhead.
    if len([pc for pc in part_counts if pc > 0]) == 1:
        final_count = [pc for pc in part_counts if pc > 0][0]
        summary = build_dataset(
            count=int(final_count),
            seed=int(args.seed),
            out_dir=out_dir,
            headless=bool(int(args.headless)),
            timeout_ms=int(args.timeout_ms),
            html_only=bool(args.html_only),
            cdp_endpoint=str(args.cdp_endpoint or ""),
            balanced=bool(int(args.balanced)),
            difficulty=str(args.difficulty or "hard"),
            start_index=int(args.start_index),
            profile_mix=profile_mix_norm,
            run_id=run_id,
            shard_id=str(args.shard_id or "part_01"),
            shard_count=max(1, int(args.shard_count)),
            retry_count=max(0, int(args.retry_count)),
            heartbeat_every=max(0, int(args.heartbeat_every)),
            resume=bool(int(args.resume)),
        )
    else:
        parts_root = out_dir / "_parts"
        parts_root.mkdir(parents=True, exist_ok=True)
        part_dirs: List[Path] = []
        part_specs: List[Dict[str, Any]] = []
        offset = int(args.start_index)
        for idx, pc in enumerate(part_counts):
            if pc <= 0:
                continue
            part_seed = int(args.seed) + (idx * 10_000_019)
            part_start_index = int(offset)
            offset += int(pc)
            part_id = f"part_{idx+1:03d}"
            part_out = parts_root / part_id
            part_dirs.append(part_out)
            part_specs.append(
                {
                    "count": int(pc),
                    "seed": int(part_seed),
                    "start_index": int(part_start_index),
                    "part_id": part_id,
                    "part_out": part_out,
                }
            )

        running: List[Tuple[subprocess.Popen[Any], Path, str]] = []
        queue = list(part_specs)
        total_parts = len(part_specs)
        worker_errors: List[Dict[str, Any]] = []
        shards_started = 0
        shards_done = 0
        shards_started_at = time.time()
        while queue or running:
            while queue and len(running) < workers:
                spec = queue.pop(0)
                part_out = Path(spec["part_out"])
                expected_count = int(spec["count"])
                if bool(int(args.resume)) and _is_completed_part(part_out, expected_count=expected_count):
                    shards_done += 1
                    _print_progress(
                        "[shards]",
                        shards_done,
                        total_parts,
                        shards_started_at,
                        extra=f"running={len(running)} queued={len(queue)} started={shards_started}",
                    )
                    continue
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--count",
                    str(expected_count),
                    "--seed",
                    str(int(spec["seed"])),
                    "--out-dir",
                    str(part_out),
                    "--headless",
                    str(int(args.headless)),
                    "--timeout-ms",
                    str(int(args.timeout_ms)),
                    "--balanced",
                    str(int(args.balanced)),
                    "--difficulty",
                    str(args.difficulty or "hard"),
                    "--profile-mix",
                    str(profile_mix_norm),
                    "--run-id",
                    str(run_id),
                    "--shard-id",
                    str(spec["part_id"]),
                    "--shard-count",
                    str(total_parts),
                    "--retry-count",
                    str(int(args.retry_count)),
                    "--heartbeat-every",
                    str(int(args.heartbeat_every)),
                    "--start-index",
                    str(int(spec["start_index"])),
                ]
                if bool(args.html_only):
                    cmd.append("--html-only")
                if str(args.cdp_endpoint or "").strip():
                    cmd.extend(["--cdp-endpoint", str(args.cdp_endpoint)])
                cmd.extend(["--workers", "1"])
                cmd.extend(["--shard-size", "0"])
                cmd.extend(["--validate-only", "0"])
                cmd.extend(["--prune-html-after-validate", "0"])
                cmd.extend(["--resume", str(int(args.resume))])
                cmd.extend(["--sample-review-count", str(int(args.sample_review_count))])
                cmd.extend(["--distribution-tolerance-pp", str(float(args.distribution_tolerance_pp))])
                cmd.extend(["--min-edge-coverage", str(args.min_edge_coverage or "")])
                proc = subprocess.Popen(cmd)
                running.append((proc, part_out, str(spec["part_id"])))
                shards_started += 1
                _print_progress(
                    "[shards]",
                    shards_done,
                    total_parts,
                    shards_started_at,
                    extra=f"running={len(running)} queued={len(queue)} started={shards_started}",
                )

            if not running:
                time.sleep(0.1)
                continue
            next_running: List[Tuple[subprocess.Popen[Any], Path, str]] = []
            for proc, part_out, part_id in running:
                code = proc.poll()
                if code is None:
                    next_running.append((proc, part_out, part_id))
                    continue
                if int(code) != 0:
                    worker_errors.append({"part_id": part_id, "part_dir": str(part_out), "exit_code": int(code)})
                shards_done += 1
                _print_progress(
                    "[shards]",
                    shards_done,
                    total_parts,
                    shards_started_at,
                    extra=f"running={len(next_running)} queued={len(queue)} started={shards_started}",
                )
            running = next_running
            time.sleep(0.15)

        summary = _merge_parts(
            out_dir=out_dir,
            part_dirs=part_dirs,
            count_requested=int(total_count),
            seed=int(args.seed),
            run_id=run_id,
            profile_mix=profile_mix_norm,
        )
        if worker_errors:
            summary["errors"] = list(summary.get("errors") or []) + worker_errors
            _write_json(out_dir / "manifests" / "dataset_summary.json", summary)

    report = validate_dataset(
        out_dir=out_dir,
        profile_mix=profile_mix_norm,
        distribution_tolerance_pp=float(args.distribution_tolerance_pp),
        min_edge_coverage=min_edge,
        sample_review_count=max(0, int(args.sample_review_count)),
    )
    report_path = out_dir / "manifests" / "validation_report.json"
    _write_json(report_path, report)
    summary["quality_gates"] = dict(report.get("quality_gates") or {})
    summary["validation_ok"] = bool(report.get("ok"))
    summary["validation_report"] = str(report_path.resolve())

    if bool(int(args.prune_html_after_validate)) and bool(report.get("ok")):
        prune = _prune_html(out_dir)
        summary.update(prune)
    else:
        summary["pruned_html"] = False

    _write_json(out_dir / "manifests" / "dataset_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
