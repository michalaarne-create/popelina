from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from PIL import Image, ImageDraw


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
    allow_subprocess_fallback = str(os.environ.get("FULLBOT_REGION_GROW_SUBPROCESS_FALLBACK", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    log(f"[INFO] Running region_grow on {image_path.name}")
    rg = get_region_grow_module()
    if rg is not None and hasattr(rg, "run_dropdown_detection"):
        try:
            t_rg_total = time.perf_counter()
            # Single OCR pass per iteration: compute once and reuse inside run_dropdown_detection.
            ocr_cached_rows: Optional[List[Any]] = None
            if hasattr(rg, "read_ocr_wrapper"):
                try:
                    with Image.open(image_path).convert("RGB") as im:
                        pre = rg.read_ocr_wrapper(im, timer=None)  # type: ignore[attr-defined]
                    if isinstance(pre, list):
                        ocr_cached_rows = list(pre)
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
            _write_ocr_boxes_debug_image(image_path=image_path, payload=payload, root=root, log=log)
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
    _write_ocr_boxes_debug_image(image_path=image_path, payload=payload, root=root, log=log)
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
    log(f"[INFO] Running rating on {json_path.name}")
    rt = get_rating_module()
    if rt is not None and hasattr(rt, "process_file"):
        try:
            out_path = rt.process_file(str(json_path))  # type: ignore[attr-defined]
            if out_path is None:
                log("[ERROR] rating.process_file returned None")
                return False
            return True
        except Exception as exc:
            log(f"[WARN] Inline rating failed, falling back to subprocess: {exc}")
    cmd = [sys_executable, str(rating_script), str(json_path)]
    result = subprocess.run(cmd, cwd=str(root), **subprocess_kw)
    if result.returncode != 0:
        log(f"[ERROR] rating failed with code {result.returncode}")
        return False
    return True


def latest_file(directory: Path, suffixes: Tuple[str, ...]) -> Optional[Path]:
    try:
        candidates = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    except FileNotFoundError:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
