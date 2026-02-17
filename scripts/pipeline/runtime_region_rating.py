from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from PIL import Image, ImageDraw


def _env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.environ.get(name, default) or default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except Exception:
        return int(default)


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


def _is_question_like(s: str) -> bool:
    return "?" in str(s or "")


def _build_fast_summary(payload: dict, image_path: Path) -> dict:
    results = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(results, list):
        results = []

    def _score(item: dict, base: float = 0.0) -> float:
        try:
            conf = float(item.get("conf") or 0.0)
        except Exception:
            conf = 0.0
        return max(0.0, min(1.0, base + conf))

    question_like_boxes: List[dict] = []
    answer_candidates: List[dict] = []
    next_candidates: List[dict] = []
    dropdown_candidates: List[dict] = []

    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        box = _extract_box_xyxy(row)
        if box is None:
            continue
        txt = str(row.get("text") or row.get("box_text") or "").strip()
        item = {
            "id": str(row.get("id") or f"rg_{idx}"),
            "text": txt,
            "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            "score": round(_score(row), 4),
        }
        if _is_question_like(txt):
            question_like_boxes.append(dict(item))
        if _is_next_text(txt):
            next_candidates.append(dict(item, score=round(_score(row, 0.25), 4)))
            continue
        if bool(row.get("has_frame")) or _is_dropdown_text(txt):
            bonus = 0.2 if row.get("has_frame") else 0.1
            dropdown_candidates.append(dict(item, score=round(_score(row, bonus), 4)))
            continue
        if txt:
            answer_candidates.append(dict(item))

    question_like_boxes.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    answer_candidates.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    next_candidates.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    dropdown_candidates.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    top_labels: dict = {}
    if dropdown_candidates:
        top_labels["dropdown"] = dict(dropdown_candidates[0], label="dropdown")
    if next_candidates:
        top_labels["next_active"] = dict(next_candidates[0], label="next_active")
    if answer_candidates:
        top_labels["answer_single"] = dict(answer_candidates[0], label="answer_single")
        top_labels["answer_multi"] = dict(answer_candidates[0], label="answer_multi")

    return {
        "image": str(image_path),
        "total_elements": int(len(results)),
        "background_layout": payload.get("background_layout") if isinstance(payload, dict) else {},
        "top_labels": top_labels,
        "question_like_boxes": question_like_boxes[:5],
        "answer_candidate_boxes": answer_candidates[:8],
        "next_candidate_boxes": next_candidates[:5],
        "dropdown_candidate_boxes": dropdown_candidates[:5],
        "confidence": {
            "answer": float(answer_candidates[0]["score"]) if answer_candidates else 0.0,
            "next": float(next_candidates[0]["score"]) if next_candidates else 0.0,
            "dropdown": float(dropdown_candidates[0]["score"]) if dropdown_candidates else 0.0,
        },
        "reasons": {"mode": "turbo_fast_summary", "source": "region_grow.results"},
    }


def _write_fast_summary_files(
    *,
    json_path: Path,
    image_path: Path,
    payload: dict,
    region_grow_current_dir: Path,
    log,
) -> Optional[Path]:
    try:
        fast = _build_fast_summary(payload, image_path)
        per_image = json_path.with_name(f"{json_path.stem}_fast_summary.json")
        per_image.write_text(json.dumps(fast, ensure_ascii=False, indent=2), encoding="utf-8")
        region_grow_current_dir.mkdir(parents=True, exist_ok=True)
        current = region_grow_current_dir / "fast_summary.json"
        current.write_text(json.dumps(fast, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[INFO] fast_summary saved: {per_image.name}")
        return per_image
    except Exception as exc:
        log(f"[WARN] fast_summary save failed: {exc}")
        return None


def _extract_box_xyxy(item: dict) -> Optional[Tuple[int, int, int, int]]:
    try:
        tb = item.get("text_box")
        if isinstance(tb, (list, tuple)) and len(tb) == 4:
            x1, y1, x2, y2 = [int(v) for v in tb]
            if x2 > x1 and y2 > y1:
                return (x1, y1, x2, y2)
        bb = item.get("bbox")
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


def _extract_box_xyxy_from_ocr_row(row: Any) -> Optional[Tuple[int, int, int, int]]:
    try:
        if not (isinstance(row, (list, tuple)) and len(row) >= 1):
            return None
        quad = row[0]
        if not (isinstance(quad, (list, tuple)) and len(quad) >= 4):
            return None
        xs = [int(p[0]) for p in quad if isinstance(p, (list, tuple)) and len(p) >= 2]
        ys = [int(p[1]) for p in quad if isinstance(p, (list, tuple)) and len(p) >= 2]
        if len(xs) < 4 or len(ys) < 4:
            return None
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)
    except Exception:
        return None


def _write_ocr_boxes_debug_image(image_path: Path, payload: dict, root: Path, log) -> None:
    try:
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            return
        ocr_dir = root / "data" / "screen" / "OCR boxes png"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        canvas = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        count = 0
        for row in results:
            if not isinstance(row, dict):
                continue
            box = _extract_box_xyxy(row)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
            txt = str(row.get("text") or "").strip()
            if txt:
                if len(txt) > 30:
                    txt = txt[:30] + "..."
                draw.text((x1 + 2, max(0, y1 - 14)), txt, fill=(0, 255, 0))
            count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = ocr_dir / f"{image_path.stem}_{ts}_ocr_boxes.png"
        canvas.save(out_path)
        log(f"[INFO] OCR boxes debug saved: {out_path.name} (boxes={count})")
    except Exception as exc:
        log(f"[WARN] OCR boxes debug save failed: {exc}")


def _write_ocr_boxes_debug_from_rows(
    image_path: Path,
    rows: List[Any],
    *,
    root: Path,
    log,
) -> None:
    try:
        ocr_dir = root / "data" / "screen" / "OCR boxes png"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        canvas = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        count = 0
        for row in rows:
            box = _extract_box_xyxy_from_ocr_row(row)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
            txt = ""
            conf = 0.0
            try:
                txt = str(row[1] or "").strip() if len(row) > 1 else ""
            except Exception:
                txt = ""
            try:
                conf = float(row[2] or 0.0) if len(row) > 2 else 0.0
            except Exception:
                conf = 0.0
            label = f"{conf:.2f}"
            if txt:
                if len(txt) > 30:
                    txt = txt[:30] + "..."
                label = f"{label} {txt}"
            draw.text((x1 + 2, max(0, y1 - 14)), label, fill=(0, 255, 0))
            count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = ocr_dir / f"{image_path.stem}_{ts}_ocr_boxes.png"
        canvas.save(out_path)
        cache_path = ocr_dir / f"{image_path.stem}_ocr_cache.json"
        try:
            cache_payload = {
                "image": str(image_path),
                "created_ts": ts,
                "rows": [
                    {
                        "quad": row[0] if isinstance(row, (list, tuple)) and len(row) > 0 else None,
                        "text": (row[1] if isinstance(row, (list, tuple)) and len(row) > 1 else ""),
                        "conf": float(row[2] if isinstance(row, (list, tuple)) and len(row) > 2 else 0.0),
                    }
                    for row in rows
                ],
            }
            cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        log(f"[INFO] OCR single-pass cache saved: {out_path.name} (boxes={count})")
    except Exception as exc:
        log(f"[WARN] OCR single-pass cache save failed: {exc}")


def run_region_grow(
    image_path: Path,
    *,
    get_region_grow_module,
    data_screen_dir: Path,
    region_grow_current_dir: Path,
    region_grow_annot_dir: Path,
    region_grow_json_dir: Path,
    screen_boxes_dir: Path,
    region_grow_script: Path,
    root: Path,
    region_grow_timeout: float,
    debug_mode: bool,
    debug,
    log,
    write_current_artifact,
    subprocess_kw: dict,
    sys_executable: str,
) -> Optional[Path]:
    turbo_mode = _env_flag("FULLBOT_TURBO_MODE", "1")
    if turbo_mode:
        os.environ.setdefault("REGION_GROW_TURBO", "1")
        os.environ.setdefault("REGION_GROW_REQUIRE_GPU", "1")
        os.environ.setdefault("REGION_GROW_LASER_REQUIRE_GPU", "1")
        os.environ.setdefault("REGION_GROW_EXTRA_OCR_REQUIRE_GPU", "1")
        os.environ.setdefault("REGION_GROW_BG_LAYOUT_REQUIRE_GPU", "1")
        os.environ.setdefault("REGION_GROW_ENABLE_EXTRA_OCR", "0")
        os.environ.setdefault("REGION_GROW_ENABLE_BACKGROUND_LAYOUT", "0")
        os.environ.setdefault("REGION_GROW_MAX_DETECTIONS_FOR_LASER", str(_env_int("REGION_GROW_MAX_DETECTIONS_TURBO", 60)))
        os.environ.setdefault("RG_TARGET_SIDE", str(_env_int("REGION_GROW_MAX_SIDE_TURBO", 960)))
    # Keep OCR debug boxes enabled by default in every mode unless explicitly disabled.
    ocr_debug_enabled = _env_flag("FULLBOT_OCR_BOXES_DEBUG", "1")
    allow_subprocess_fallback = str(os.environ.get("FULLBOT_REGION_GROW_SUBPROCESS_FALLBACK", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    log(f"[INFO] Running region_grow on {image_path.name}")
    if not ocr_debug_enabled:
        log("[INFO] OCR boxes debug disabled (FULLBOT_OCR_BOXES_DEBUG=0).")
    rg = get_region_grow_module()
    if rg is not None and hasattr(rg, "run_dropdown_detection"):
        try:
            t_rg_total = time.perf_counter()
            # Single OCR pass per iteration: compute once and reuse inside run_dropdown_detection.
            ocr_cached_rows: Optional[List[Any]] = None
            if (not turbo_mode) and hasattr(rg, "read_ocr_wrapper"):
                try:
                    with Image.open(image_path).convert("RGB") as im:
                        pre = rg.read_ocr_wrapper(im, timer=None)  # type: ignore[attr-defined]
                    if isinstance(pre, list):
                        ocr_cached_rows = list(pre)
                        if ocr_debug_enabled:
                            _write_ocr_boxes_debug_from_rows(image_path, ocr_cached_rows, root=root, log=log)
                except Exception as exc:
                    log(f"[WARN] Single-pass OCR cache build failed: {exc}")

            orig_read_ocr = getattr(rg, "read_ocr_wrapper", None)
            if ocr_cached_rows is not None and callable(orig_read_ocr):
                def _cached_read_ocr_wrapper(*_a, **_k):
                    return list(ocr_cached_rows or [])
                setattr(rg, "read_ocr_wrapper", _cached_read_ocr_wrapper)
            try:
                out = rg.run_dropdown_detection(str(image_path))  # type: ignore[attr-defined]
            finally:
                if ocr_cached_rows is not None and callable(orig_read_ocr):
                    setattr(rg, "read_ocr_wrapper", orig_read_ocr)
            json_dir = data_screen_dir / "region_grow" / "region_grow"
            json_dir.mkdir(parents=True, exist_ok=True)
            json_path = json_dir / f"{image_path.stem}.json"
            payload = out
            try:
                if hasattr(rg, "to_py"):
                    payload = rg.to_py(out)  # type: ignore[attr-defined]
            except Exception as exc:
                log(f"[WARN] region_grow to_py failed, saving raw output: {exc}")
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            try:
                write_current_artifact(json_path, region_grow_current_dir, "region_grow.json")
            except Exception:
                pass
            if ocr_debug_enabled:
                _write_ocr_boxes_debug_image(image_path=image_path, payload=payload, root=root, log=log)
            _write_fast_summary_files(
                json_path=json_path,
                image_path=image_path,
                payload=payload if isinstance(payload, dict) else {},
                region_grow_current_dir=region_grow_current_dir,
                log=log,
            )
            try:
                if hasattr(rg, "annotate_and_save"):
                    rg.annotate_and_save(  # type: ignore[attr-defined]
                        str(image_path),
                        payload.get("results", []),
                        payload.get("triangles"),
                        output_dir=str(region_grow_annot_dir),
                    )
            except Exception as exc:
                log(f"[WARN] region_grow annotate failed: {exc}")
            log(f"[TIMER] region_grow_inline {time.perf_counter() - t_rg_total:.3f}s (image={image_path.name})")
            return json_path
        except Exception as exc:
            if not allow_subprocess_fallback:
                log(f"[ERROR] Inline region_grow failed (subprocess fallback disabled): {exc}")
                return None
            log(f"[WARN] Inline region_grow failed, falling back to subprocess: {exc}")
    elif not allow_subprocess_fallback:
        log("[ERROR] Inline region_grow unavailable (subprocess fallback disabled).")
        return None
    t_rg_sub = time.perf_counter()
    cmd = [sys_executable, str(region_grow_script), str(image_path)]
    env = os.environ.copy()
    env.setdefault("RG_FAST", "1")
    env.setdefault("RAPID_OCR_MAX_SIDE", "800")
    env.setdefault("RAPID_OCR_AUTOCROP", "1")
    env.setdefault("RAPID_AUTOCROP_KEEP_RATIO", "0.9")
    env.setdefault("RAPID_OCR_AUTOCROP_DELTA", "8")
    env.setdefault("RAPID_OCR_AUTOCROP_PAD", "8")
    env.setdefault("REGION_GROW_USE_GPU", "1")
    env.setdefault("REGION_GROW_REQUIRE_GPU", "1")
    if turbo_mode:
        env.setdefault("REGION_GROW_TURBO", "1")
        env.setdefault("REGION_GROW_ENABLE_EXTRA_OCR", "0")
        env.setdefault("REGION_GROW_ENABLE_BACKGROUND_LAYOUT", "0")
        env.setdefault("REGION_GROW_MAX_DETECTIONS_FOR_LASER", str(_env_int("REGION_GROW_MAX_DETECTIONS_TURBO", 60)))
        env.setdefault("RG_TARGET_SIDE", str(_env_int("REGION_GROW_MAX_SIDE_TURBO", 960)))
    if debug_mode:
        debug(f"region_grow cmd: {cmd} timeout={region_grow_timeout}s env_fast=1")
    try:
        result = subprocess.run(cmd, cwd=str(root), timeout=region_grow_timeout, env=env, **subprocess_kw)
    except subprocess.TimeoutExpired:
        log(f"[ERROR] region_grow hung > {region_grow_timeout}s, killing.")
        return None
    if result.returncode != 0:
        log(f"[ERROR] region_grow failed with code {result.returncode}")
        return None
    json_path = region_grow_json_dir / f"{image_path.stem}.json"
    if not json_path.exists():
        legacy = screen_boxes_dir / f"{image_path.stem}.json"
        if legacy.exists():
            json_path = legacy
        else:
            log(f"[ERROR] Expected JSON missing: {legacy}")
            return None
    try:
        write_current_artifact(json_path, region_grow_current_dir, "region_grow.json")
    except Exception:
        pass
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        payload = {}
    if ocr_debug_enabled:
        _write_ocr_boxes_debug_image(image_path=image_path, payload=payload, root=root, log=log)
    _write_fast_summary_files(
        json_path=json_path,
        image_path=image_path,
        payload=payload if isinstance(payload, dict) else {},
        region_grow_current_dir=region_grow_current_dir,
        log=log,
    )
    log(f"[TIMER] region_grow_subprocess {time.perf_counter() - t_rg_sub:.3f}s (image={image_path.name})")
    return json_path


def run_arrow_post(json_path: Path, *, log) -> None:
    log(f"[INFO] Skipping arrow_post_region for performance (no-op for {json_path.name})")


def run_rating(
    json_path: Path,
    *,
    get_rating_module,
    rating_script: Path,
    root: Path,
    subprocess_kw: dict,
    sys_executable: str,
    log,
) -> bool:
    turbo_mode = _env_flag("FULLBOT_TURBO_MODE", "1")
    rating_mode = str(os.environ.get("RATING_MODE", "off" if turbo_mode else "heavy") or "heavy").strip().lower()
    run_fast_rating = _env_flag("FULLBOT_RUN_RATING_FAST", "1")
    async_heavy = _env_flag("FULLBOT_RATING_HEAVY_ASYNC", "0")

    if rating_mode == "off":
        if run_fast_rating:
            try:
                from scripts.region_grow.numpy_rate import rating_fast as rf  # type: ignore

                t_fast = time.perf_counter()
                out = rf.process_file(str(json_path))
                log(f"[TIMER] rating_fast {time.perf_counter() - t_fast:.3f}s ({json_path.name})")
                if out:
                    log(f"[INFO] rating_fast summary: {Path(out).name}")
            except Exception as exc:
                log(f"[WARN] rating_fast failed in RATING_MODE=off: {exc}")
        if async_heavy:
            try:
                cmd_bg = [sys_executable, str(rating_script), str(json_path)]
                subprocess.Popen(cmd_bg, cwd=str(root), **subprocess_kw)
                log(f"[INFO] rating_heavy async dispatched ({json_path.name})")
            except Exception as exc:
                log(f"[WARN] rating_heavy async dispatch failed: {exc}")
        log(f"[INFO] RATING_MODE=off -> skipping heavy rating ({json_path.name})")
        return True

    log(f"[INFO] Running rating on {json_path.name}")
    try:
        rating_timeout = float(os.environ.get("FULLBOT_RATING_TIMEOUT_S", "90") or 90.0)
    except Exception:
        rating_timeout = 90.0
    rating_timeout = max(10.0, min(600.0, rating_timeout))
    force_subprocess = str(os.environ.get("FULLBOT_RATING_FORCE_SUBPROCESS", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    rt = get_rating_module()
    if (not force_subprocess) and rt is not None and hasattr(rt, "process_file"):
        try:
            t0 = time.perf_counter()
            out_path = rt.process_file(str(json_path))  # type: ignore[attr-defined]
            if out_path is None:
                log("[ERROR] rating.process_file returned None")
                return False
            log(f"[TIMER] rating_inline {time.perf_counter() - t0:.3f}s ({json_path.name})")
            return True
        except Exception as exc:
            log(f"[WARN] Inline rating failed, falling back to subprocess: {exc}")
    cmd = [sys_executable, str(rating_script), str(json_path)]
    t_sub = time.perf_counter()
    try:
        result = subprocess.run(cmd, cwd=str(root), timeout=rating_timeout, **subprocess_kw)
    except subprocess.TimeoutExpired:
        log(f"[ERROR] rating hung > {rating_timeout:.0f}s, killing.")
        return False
    if result.returncode != 0:
        log(f"[ERROR] rating failed with code {result.returncode}")
        return False
    log(f"[TIMER] rating_subprocess {time.perf_counter() - t_sub:.3f}s ({json_path.name})")
    return True


def latest_file(directory: Path, suffixes: Tuple[str, ...]) -> Optional[Path]:
    try:
        candidates = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    except FileNotFoundError:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
