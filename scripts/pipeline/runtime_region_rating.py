from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw


def _env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.environ.get(name, default) or default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except Exception:
        return int(default)


_GPU_STACK_CACHE: Dict[str, Dict[str, bool]] = {}
_RG_WORKER_PROC: Optional[subprocess.Popen] = None
_RG_WORKER_PYTHON: Optional[Path] = None
_RG_WORKER_LOCK = threading.Lock()
_RG_PREFERRED_PYTHON_CACHE: Dict[str, Optional[Path]] = {}


def _same_path(left: Path, right: Path) -> bool:
    try:
        return str(left.resolve()).lower() == str(right.resolve()).lower()
    except Exception:
        return str(left).lower() == str(right).lower()


def _candidate_region_grow_pythons(root: Path, current_python: Path) -> List[Path]:
    candidates: List[Path] = []

    def _push(raw: Optional[str]) -> None:
        value = str(raw or "").strip()
        if not value:
            return
        path = Path(value)
        if not path.exists():
            return
        if any(_same_path(path, item) for item in candidates):
            return
        candidates.append(path)

    _push(os.environ.get("FULLBOT_REGION_GROW_PYTHON"))
    _push(os.environ.get("FULLBOT_GPU_PYTHON"))
    _push(os.environ.get("FULLBOT_RUNTIME_PYTHON"))
    _push(str(root.parent / ".venv312" / "Scripts" / "python.exe"))
    _push(str(root.parent / ".conda" / "fullbot312" / "python.exe"))
    _push(str(current_python))
    return candidates


def _python_gpu_region_grow_capabilities(python_path: Path) -> Dict[str, bool]:
    cache_key = str(python_path).lower()
    if cache_key in _GPU_STACK_CACHE:
        return dict(_GPU_STACK_CACHE[cache_key])

    if _same_path(python_path, Path(os.sys.executable)):
        try:
            import cupy  # type: ignore
            from cupyx.scipy import ndimage as _cupy_ndimage  # type: ignore
            import paddle  # type: ignore

            gpu_math_ready = cupy is not None and _cupy_ndimage is not None
            gpu_ocr_ready = bool(gpu_math_ready and paddle is not None and paddle.is_compiled_with_cuda())
        except Exception:
            gpu_math_ready = False
            gpu_ocr_ready = False
        _GPU_STACK_CACHE[cache_key] = {
            "gpu_math_ready": bool(gpu_math_ready),
            "gpu_ocr_ready": bool(gpu_ocr_ready),
        }
        return dict(_GPU_STACK_CACHE[cache_key])

    probe = [
        str(python_path),
        "-c",
        (
            "import cupy; from cupyx.scipy import ndimage; import paddle; "
            "ok_math = bool(cupy is not None and ndimage is not None); "
            "ok_ocr = bool(ok_math and paddle.is_compiled_with_cuda()); "
            "raise SystemExit(0 if ok_ocr else (2 if ok_math else 3))"
        ),
    ]
    try:
        result = subprocess.run(
            probe,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20.0,
            check=False,
        )
        code = int(result.returncode)
        gpu_ocr_ready = code == 0
        gpu_math_ready = code in {0, 2}
    except Exception:
        gpu_math_ready = False
        gpu_ocr_ready = False
    _GPU_STACK_CACHE[cache_key] = {
        "gpu_math_ready": bool(gpu_math_ready),
        "gpu_ocr_ready": bool(gpu_ocr_ready),
    }
    return dict(_GPU_STACK_CACHE[cache_key])


def _resolve_region_grow_python(root: Path, current_python: Path) -> Optional[Path]:
    cache_key = f"{str(root).lower()}|{str(current_python).lower()}"
    if cache_key in _RG_PREFERRED_PYTHON_CACHE:
        return _RG_PREFERRED_PYTHON_CACHE[cache_key]
    for candidate in _candidate_region_grow_pythons(root, current_python):
        caps = _python_gpu_region_grow_capabilities(candidate)
        if bool(caps.get("gpu_ocr_ready")):
            _RG_PREFERRED_PYTHON_CACHE[cache_key] = candidate
            return candidate
    _RG_PREFERRED_PYTHON_CACHE[cache_key] = None
    return None


def _region_grow_log_verbosity() -> str:
    raw = str(os.environ.get("FULLBOT_REGION_GROW_LOG_VERBOSITY", "summary") or "summary").strip().lower()
    return raw if raw in {"summary", "detailed"} else "summary"


def _should_forward_region_grow_line(line: str, *, advanced_debug: bool = False) -> bool:
    msg = str(line or "").strip()
    if not msg:
        return False
    verbosity = _region_grow_log_verbosity()
    if verbosity == "detailed" or advanced_debug:
        return True
    lo = msg.lower()
    if lo.startswith("[error]") or lo.startswith("[warn]"):
        return True
    if "pipeline total" in lo or "worker ready" in lo or "ocr preload" in lo:
        return True
    if msg.startswith("{") and "\"ok\"" in msg:
        return True
    return False


def _region_grow_worker_env(base_env: Dict[str, str], *, turbo_mode: bool, use_rapid_turbo: bool) -> Dict[str, str]:
    env = dict(base_env)
    env["RG_FAST"] = "1"
    env["REGION_GROW_USE_GPU"] = "1"
    env["REGION_GROW_REQUIRE_GPU"] = "1"
    env["REGION_GROW_RUNTIME_MODE"] = "1"
    env["REGION_GROW_PRELOAD_MODELS"] = "1"
    env["REGION_GROW_ANNOTATE_ENABLED"] = "0"
    env["REGION_GROW_RUN_CNN"] = "0"
    env["REGION_GROW_ENABLE_REGIONS_EXPORT"] = "0"
    env["FULLBOT_REGION_GROW_ANNOTATE"] = "0"
    env["FULLBOT_REGION_GROW_ANNOTATE_INLINE"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env["RAPID_OCR_MAX_SIDE"] = "1920"
    env["RAPID_OCR_AUTOCROP"] = "0"
    env["RAPID_AUTOCROP_KEEP_RATIO"] = "0.9"
    env["RAPID_OCR_AUTOCROP_DELTA"] = "8"
    env["RAPID_OCR_AUTOCROP_PAD"] = "8"
    if turbo_mode:
        env["REGION_GROW_TURBO"] = "1"
        env["REGION_GROW_USE_RAPID_OCR"] = "1" if use_rapid_turbo else "0"
        env["RAPID_OCR_MAX_SIDE"] = str(_env_int("RAPID_OCR_MAX_SIDE_TURBO", 1920))
        env["RAPID_OCR_AUTOCROP"] = str(_env_int("RAPID_OCR_AUTOCROP_TURBO", 0))
        env["REGION_GROW_ENABLE_EXTRA_OCR"] = "0"
        env["REGION_GROW_ENABLE_BACKGROUND_LAYOUT"] = "0"
        env["REGION_GROW_MAX_DETECTIONS_FOR_LASER"] = str(_env_int("REGION_GROW_MAX_DETECTIONS_TURBO", 60))
        env["RG_TARGET_SIDE"] = str(_env_int("RG_TARGET_SIDE_TURBO", 1280))
    return env


def _ensure_region_grow_worker(
    *,
    python_path: Path,
    region_grow_script: Path,
    root: Path,
    env: Dict[str, str],
    subprocess_kw: dict,
    region_grow_timeout: float,
    advanced_debug: bool,
    log,
) -> Optional[subprocess.Popen]:
    global _RG_WORKER_PROC, _RG_WORKER_PYTHON
    with _RG_WORKER_LOCK:
        if (
            _RG_WORKER_PROC is not None
            and _RG_WORKER_PROC.poll() is None
            and _RG_WORKER_PYTHON is not None
            and _same_path(_RG_WORKER_PYTHON, python_path)
        ):
            return _RG_WORKER_PROC

        run_kw = dict(subprocess_kw or {})
        run_kw["stdin"] = subprocess.PIPE
        run_kw["stdout"] = subprocess.PIPE
        run_kw["stderr"] = subprocess.STDOUT
        run_kw["text"] = True
        run_kw["encoding"] = "utf-8"
        run_kw["errors"] = "replace"
        cmd = [str(python_path), "-u", str(region_grow_script), "--worker"]
        proc = subprocess.Popen(cmd, cwd=str(root), env=env, **run_kw)
        deadline = time.perf_counter() + max(5.0, float(region_grow_timeout))
        ready = False
        if proc.stdout is not None:
            while time.perf_counter() < deadline:
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    continue
                msg = str(line).rstrip()
                if msg:
                    if _should_forward_region_grow_line(msg, advanced_debug=advanced_debug):
                        log(f"[region_grow] {msg}")
                if "worker ready" in msg.lower():
                    ready = True
                    break
        if (not ready) or proc.poll() is not None:
            with contextlib.suppress(Exception):
                proc.kill()
            return None
        _RG_WORKER_PROC = proc
        _RG_WORKER_PYTHON = Path(python_path)
        return proc


def _run_region_grow_worker_request(
    *,
    proc: subprocess.Popen,
    image_path: Path,
    region_grow_timeout: float,
    advanced_debug: bool,
    target_side: Optional[int],
    log,
) -> Optional[Dict[str, Any]]:
    if proc.stdin is None or proc.stdout is None:
        return None
    req = {"image_path": str(image_path)}
    if target_side is not None and int(target_side) > 0:
        req["target_side"] = int(target_side)
    try:
        proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        proc.stdin.flush()
    except Exception:
        return None

    deadline = time.perf_counter() + float(region_grow_timeout)
    while time.perf_counter() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                return None
            continue
        msg = str(line).rstrip()
        if not msg:
            continue
        if _should_forward_region_grow_line(msg, advanced_debug=advanced_debug):
            log(f"[region_grow] {msg}")
        try:
            payload = json.loads(msg)
        except Exception:
            continue
        if isinstance(payload, dict) and ("ok" in payload):
            return payload
    return None


def _is_weak_region_payload(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return True
    results = payload.get("results")
    if not isinstance(results, list):
        return True
    try:
        min_results = int(os.environ.get("FULLBOT_OCR_MIN_RESULTS_RETRY", "5") or 5)
    except Exception:
        min_results = 5
    if len(results) < max(1, min_results):
        return True
    confs: List[float] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        try:
            confs.append(float(row.get("conf") or 0.0))
        except Exception:
            continue
    if not confs:
        return True
    avg_conf = sum(confs) / float(max(1, len(confs)))
    try:
        min_avg = float(os.environ.get("FULLBOT_OCR_MIN_AVG_CONF_RETRY", "0.42") or 0.42)
    except Exception:
        min_avg = 0.42
    return avg_conf < min_avg


def _norm_text(s: Any) -> str:
    return " ".join(str(s or "").strip().lower().split())


def _is_next_text(s: str) -> bool:
    t = _norm_text(s)
    if not t:
        return False
    keys = ("dalej", "nast", "next", "continue", "kontynuuj", "wyślij", "wyslij", "submit", "finish", "done")
    return any(k in t for k in keys)


def _is_header_noise_text(s: str) -> bool:
    t = _norm_text(s)
    if not t:
        return False
    noise_tokens = (
        "home",
        "test quiz server",
        "radio +",
        "checkbox",
        "dropdown",
        "input + next",
        "jednokrotna odpowied",
        "wielokrotna odpowied",
        "pytanie ",
    )
    return any(token in t for token in noise_tokens)


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
    image_h = 1080
    try:
        with Image.open(image_path) as img:
            image_h = max(1, int(img.height))
    except Exception:
        image_h = 1080

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
        box_h = int(box[3]) - int(box[1])
        box_w = int(box[2]) - int(box[0])
        center_y = (int(box[1]) + int(box[3])) / 2.0
        lower_bias = min(0.22, max(0.0, (center_y / float(max(1, image_h))) - 0.45))
        if _is_question_like(txt):
            question_like_boxes.append(dict(item))
        if _is_next_text(txt):
            if (not _is_header_noise_text(txt)) and box_h >= 16 and box_w <= 280 and center_y >= image_h * 0.35:
                next_candidates.append(dict(item, score=round(_score(row, 0.25 + lower_bias), 4)))
            continue
        if bool(row.get("has_frame")) or _is_dropdown_text(txt):
            bonus = 0.2 if row.get("has_frame") else 0.1
            dropdown_candidates.append(dict(item, score=round(_score(row, bonus), 4)))
            continue
        if txt and (not _is_header_noise_text(txt)) and (not _is_question_like(txt)) and center_y >= image_h * 0.16:
            answer_candidates.append(dict(item, score=round(_score(row, lower_bias), 4)))

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
        xs = [int(round(float(p[0]))) for p in quad if isinstance(p, (list, tuple)) and len(p) >= 2]
        ys = [int(round(float(p[1]))) for p in quad if isinstance(p, (list, tuple)) and len(p) >= 2]
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


def _write_cached_ocr_rows(
    *,
    image_path: Path,
    rows: List[Any],
    data_screen_dir: Path,
    log,
) -> Optional[Path]:
    try:
        current_run_dir = data_screen_dir / "current_run"
        current_run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "image": str(image_path),
            "rows": [
                {
                    "quad": row[0] if isinstance(row, (list, tuple)) and len(row) > 0 else None,
                    "text": row[1] if isinstance(row, (list, tuple)) and len(row) > 1 else "",
                    "conf": float(row[2] if isinstance(row, (list, tuple)) and len(row) > 2 else 0.0),
                }
                for row in (rows or [])
            ],
        }
        path = current_run_dir / "ocr_rows.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except Exception as exc:
        log(f"[WARN] OCR rows cache save failed: {exc}")
        return None


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
    advanced_debug = _env_flag("FULLBOT_ADVANCED_DEBUG", "0")
    turbo_mode = _env_flag("FULLBOT_TURBO_MODE", "1")
    use_rapid_turbo = _env_flag("FULLBOT_USE_RAPID_OCR_TURBO", "0")
    current_python = Path(sys_executable)
    preferred_region_grow_python = _resolve_region_grow_python(root, current_python)
    inline_gpu_ready = preferred_region_grow_python is not None and _same_path(preferred_region_grow_python, current_python)
    prefer_gpu_subprocess = preferred_region_grow_python is not None and not inline_gpu_ready
    # Use the split OCR path (det -> crops -> rec -> global coords) by default.
    os.environ.setdefault("FULLBOT_OCR_PIPELINE", "two_stage")
    os.environ.setdefault("FULLBOT_OCR_STAGE1_BACKEND", "rapid_det")
    if turbo_mode:
        os.environ["REGION_GROW_TURBO"] = "1"
        # Accuracy-first default in turbo: PaddleOCR GPU for detection geometry.
        os.environ["REGION_GROW_USE_RAPID_OCR"] = "1" if use_rapid_turbo else "0"
        os.environ["RAPID_OCR_MAX_SIDE"] = str(_env_int("RAPID_OCR_MAX_SIDE_TURBO", 1920))
        os.environ["RAPID_OCR_AUTOCROP"] = str(_env_int("RAPID_OCR_AUTOCROP_TURBO", 0))
        os.environ["REGION_GROW_REQUIRE_GPU"] = "1"
        os.environ["REGION_GROW_LASER_REQUIRE_GPU"] = "1"
        os.environ["REGION_GROW_EXTRA_OCR_REQUIRE_GPU"] = "1"
        os.environ["REGION_GROW_BG_LAYOUT_REQUIRE_GPU"] = "1"
        os.environ["REGION_GROW_ENABLE_EXTRA_OCR"] = "0"
        os.environ["REGION_GROW_ENABLE_BACKGROUND_LAYOUT"] = "0"
        os.environ["REGION_GROW_MAX_DETECTIONS_FOR_LASER"] = str(_env_int("REGION_GROW_MAX_DETECTIONS_TURBO", 60))
        os.environ["RG_TARGET_SIDE"] = str(_env_int("RG_TARGET_SIDE_TURBO", 1280))
    # In turbo mode keep debug drawing off by default (it costs extra CPU work).
    ocr_debug_enabled = _env_flag("FULLBOT_OCR_BOXES_DEBUG", "0" if turbo_mode else "1")
    defer_debug_render = _env_flag("FULLBOT_DEBUG_DEFERRED_RENDER", "1")
    annotate_enabled = _env_flag("FULLBOT_REGION_GROW_ANNOTATE_INLINE", "0")
    allow_subprocess_fallback = str(os.environ.get("FULLBOT_REGION_GROW_SUBPROCESS_FALLBACK", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    log(f"[INFO] Running region_grow on {image_path.name}")
    if advanced_debug:
        log(
            "[DEBUG] region_grow runtime cfg "
            f"turbo={turbo_mode} ocr_debug={ocr_debug_enabled} "
            f"annotate={annotate_enabled} subprocess_fallback={allow_subprocess_fallback} "
            f"use_rapid_turbo={int(use_rapid_turbo)}"
        )
        log(
            "[DEBUG] region_grow rapid env "
            f"use_rapid={os.environ.get('REGION_GROW_USE_RAPID_OCR', '0')} "
            f"max_side={os.environ.get('RAPID_OCR_MAX_SIDE', '?')} "
            f"autocrop={os.environ.get('RAPID_OCR_AUTOCROP', '?')}"
        )
    if not ocr_debug_enabled:
        log("[INFO] OCR boxes debug disabled (FULLBOT_OCR_BOXES_DEBUG=0).")
    elif defer_debug_render:
        log("[INFO] OCR boxes debug deferred (FULLBOT_DEBUG_DEFERRED_RENDER=1).")
    if not annotate_enabled:
        log("[INFO] region_grow inline annotation disabled (deferred mode).")
    if preferred_region_grow_python is None:
        log("[ERROR] region_grow GPU OCR stack unavailable: no Python interpreter with working CUDA Paddle OCR was found.")
        return None
    if not inline_gpu_ready:
        log(
            "[INFO] region_grow inline disabled for current interpreter; "
            f"using GPU-capable subprocess: {preferred_region_grow_python}"
        )
    rg = get_region_grow_module() if inline_gpu_ready else None
    ocr_cached_rows: Optional[List[Any]] = None
    if rg is not None:
        try:
            desired_max_side = int(os.environ.get("RAPID_OCR_MAX_SIDE", "1920") or "1920")
        except Exception:
            desired_max_side = 1920
        desired_autocrop = str(os.environ.get("RAPID_OCR_AUTOCROP", "0") or "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            setattr(rg, "RAPID_OCR_MAX_SIDE", int(desired_max_side))
        except Exception:
            pass
        try:
            setattr(rg, "RAPID_OCR_AUTOCROP", bool(desired_autocrop))
        except Exception:
            pass
        if advanced_debug:
            log(
                "[DEBUG] region_grow module cfg "
                f"RAPID_OCR_MAX_SIDE={getattr(rg, 'RAPID_OCR_MAX_SIDE', '?')} "
                f"RAPID_OCR_AUTOCROP={int(bool(getattr(rg, 'RAPID_OCR_AUTOCROP', False)))}"
            )
    if rg is not None and hasattr(rg, "run_dropdown_detection"):
        try:
            t_rg_total = time.perf_counter()
            # Single OCR pass per iteration: compute once and reuse inside run_dropdown_detection.
            if hasattr(rg, "read_ocr_wrapper"):
                try:
                    t_ocr_cache = time.perf_counter()
                    with Image.open(image_path).convert("RGB") as im:
                        pre = rg.read_ocr_wrapper(im, timer=None)  # type: ignore[attr-defined]
                    if isinstance(pre, list):
                        ocr_cached_rows = list(pre)
                        if advanced_debug:
                            log(
                                f"[TIMER] region_grow_ocr_single_pass {time.perf_counter() - t_ocr_cache:.3f}s "
                                f"(rows={len(ocr_cached_rows)})"
                            )
                        if ocr_debug_enabled and (not defer_debug_render):
                            _write_ocr_boxes_debug_from_rows(image_path, ocr_cached_rows, root=root, log=log)
                except Exception as exc:
                    log(f"[WARN] Single-pass OCR cache build failed: {exc}")
            cached_ocr_path = None
            if ocr_cached_rows is not None:
                cached_ocr_path = _write_cached_ocr_rows(
                    image_path=image_path,
                    rows=ocr_cached_rows,
                    data_screen_dir=data_screen_dir,
                    log=log,
                )

            orig_read_ocr = getattr(rg, "read_ocr_wrapper", None)
            if ocr_cached_rows is not None and callable(orig_read_ocr):
                def _cached_read_ocr_wrapper(*_a, **_k):
                    return list(ocr_cached_rows or [])
                setattr(rg, "read_ocr_wrapper", _cached_read_ocr_wrapper)
            try:
                t_detect = time.perf_counter()
                out = rg.run_dropdown_detection(str(image_path))  # type: ignore[attr-defined]
                if advanced_debug:
                    out_results = out.get("results") if isinstance(out, dict) else []
                    out_count = len(out_results) if isinstance(out_results, list) else 0
                    log(f"[TIMER] region_grow_detection_only {time.perf_counter() - t_detect:.3f}s (results={out_count})")
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
            if ocr_debug_enabled and (not defer_debug_render):
                _write_ocr_boxes_debug_image(image_path=image_path, payload=payload, root=root, log=log)
            _write_fast_summary_files(
                json_path=json_path,
                image_path=image_path,
                payload=payload if isinstance(payload, dict) else {},
                region_grow_current_dir=region_grow_current_dir,
                log=log,
            )
            if annotate_enabled:
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
            if not allow_subprocess_fallback and not prefer_gpu_subprocess:
                log(f"[ERROR] Inline region_grow failed (subprocess fallback disabled): {exc}")
                return None
            log(f"[WARN] Inline region_grow failed, falling back to subprocess: {exc}")
    elif not allow_subprocess_fallback and not prefer_gpu_subprocess:
        log("[ERROR] Inline region_grow unavailable (subprocess fallback disabled).")
        return None
    t_rg_sub = time.perf_counter()
    base_env = os.environ.copy()
    env = _region_grow_worker_env(base_env, turbo_mode=turbo_mode, use_rapid_turbo=use_rapid_turbo)
    if ocr_cached_rows is not None:
        cached_ocr_path = _write_cached_ocr_rows(
            image_path=image_path,
            rows=ocr_cached_rows,
            data_screen_dir=data_screen_dir,
            log=log,
        )
        if cached_ocr_path is not None:
            env["FULLBOT_OCR_ROWS_PATH"] = str(cached_ocr_path)
    exec_mode = str(os.environ.get("FULLBOT_REGION_GROW_EXEC_MODE", "worker") or "worker").strip().lower()
    try:
        initial_target_side = int(os.environ.get("RG_TARGET_SIDE_TURBO", "1280") or 1280)
    except Exception:
        initial_target_side = 1280
    retry_target_sides: List[int] = [int(initial_target_side)]
    if int(initial_target_side) < 1408:
        retry_target_sides.append(1408)
    if 1920 not in retry_target_sides:
        retry_target_sides.append(1920)
    used_worker = False
    if exec_mode == "worker":
        if debug_mode:
            debug(f"region_grow worker start: {preferred_region_grow_python}")
        proc = _ensure_region_grow_worker(
            python_path=preferred_region_grow_python,
            region_grow_script=region_grow_script,
            root=root,
            env=env,
            subprocess_kw=subprocess_kw,
            region_grow_timeout=region_grow_timeout,
            advanced_debug=advanced_debug,
            log=log,
        )
        if proc is not None:
            for idx, side in enumerate(retry_target_sides):
                payload = _run_region_grow_worker_request(
                    proc=proc,
                    image_path=image_path,
                    region_grow_timeout=region_grow_timeout,
                    advanced_debug=advanced_debug,
                    target_side=int(side),
                    log=log,
                )
                if not (payload and bool(payload.get("ok"))):
                    payload = None
                    break
                json_path_try = region_grow_json_dir / f"{image_path.stem}.json"
                weak = True
                if json_path_try.exists():
                    try:
                        payload_json = json.loads(json_path_try.read_text(encoding="utf-8", errors="replace"))
                        weak = _is_weak_region_payload(payload_json if isinstance(payload_json, dict) else {})
                    except Exception:
                        weak = True
                if (not weak) or (idx >= len(retry_target_sides) - 1):
                    used_worker = True
                    break
                log(f"[WARN] Weak OCR payload detected; retrying region_grow with RG_TARGET_SIDE={retry_target_sides[idx+1]}.")
            if not used_worker:
                log("[WARN] region_grow worker request failed; falling back to one-shot subprocess.")

    if not used_worker:
        cmd = [str(preferred_region_grow_python), "-u", str(region_grow_script), "--runtime", str(image_path)]
        if debug_mode:
            debug(f"region_grow cmd: {cmd} timeout={region_grow_timeout}s env_fast=1")
        stream_region_grow_logs = _env_flag("FULLBOT_REGION_GROW_STREAM_LOGS", "1")
        run_kw = dict(subprocess_kw or {})
        try:
            if stream_region_grow_logs:
                run_kw["stdout"] = subprocess.PIPE
                run_kw["stderr"] = subprocess.STDOUT
                run_kw["text"] = True
                run_kw["encoding"] = "utf-8"
                run_kw["errors"] = "replace"
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(root),
                    env=env,
                    **run_kw,
                )
                deadline = time.perf_counter() + float(region_grow_timeout)
                stdout_lines: List[str] = []
                if proc.stdout is not None:
                    for raw_line in proc.stdout:
                        line = str(raw_line).rstrip()
                        if line:
                            stdout_lines.append(line)
                            if _should_forward_region_grow_line(line, advanced_debug=advanced_debug):
                                log(f"[region_grow] {line}")
                        if time.perf_counter() >= deadline:
                            proc.kill()
                            raise subprocess.TimeoutExpired(cmd, region_grow_timeout)
                result_code = proc.wait(timeout=max(0.1, deadline - time.perf_counter()))
                result = subprocess.CompletedProcess(cmd, returncode=int(result_code), stdout="\n".join(stdout_lines))
            else:
                result = subprocess.run(
                    cmd,
                    cwd=str(root),
                    timeout=region_grow_timeout,
                    env=env,
                    **run_kw,
                )
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
    if ocr_debug_enabled and (not defer_debug_render):
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


def bootstrap_region_grow_worker(
    *,
    root: Path,
    region_grow_script: Path,
    subprocess_kw: dict,
    sys_executable: str,
    region_grow_timeout: float,
    debug_mode: bool,
    debug,
    log,
) -> bool:
    t0 = time.perf_counter()
    turbo_mode = _env_flag("FULLBOT_TURBO_MODE", "1")
    use_rapid_turbo = _env_flag("FULLBOT_USE_RAPID_OCR_TURBO", "0")
    advanced_debug = _env_flag("FULLBOT_ADVANCED_DEBUG", "0")
    current_python = Path(sys_executable)
    preferred_region_grow_python = _resolve_region_grow_python(root, current_python)
    if preferred_region_grow_python is None:
        log("[WARN] region_grow bootstrap skipped: no GPU-capable interpreter.")
        return False
    exec_mode = str(os.environ.get("FULLBOT_REGION_GROW_EXEC_MODE", "worker") or "worker").strip().lower()
    if exec_mode != "worker":
        log(f"[INFO] region_grow bootstrap skipped (exec_mode={exec_mode}).")
        return True
    env = _region_grow_worker_env(os.environ.copy(), turbo_mode=turbo_mode, use_rapid_turbo=use_rapid_turbo)
    if debug_mode:
        debug(f"region_grow bootstrap worker: {preferred_region_grow_python}")
    proc = _ensure_region_grow_worker(
        python_path=preferred_region_grow_python,
        region_grow_script=region_grow_script,
        root=root,
        env=env,
        subprocess_kw=subprocess_kw,
        region_grow_timeout=region_grow_timeout,
        advanced_debug=advanced_debug,
        log=log,
    )
    ok = proc is not None and proc.poll() is None
    log(f"[TIMER] region_grow_bootstrap {time.perf_counter() - t0:.3f}s ok={int(ok)}")
    return bool(ok)


def run_arrow_post(json_path: Path, *, log) -> None:
    log(f"[INFO] Skipping arrow_post_region for performance (no-op for {json_path.name})")


def run_region_annotation(
    json_path: Path,
    *,
    image_path: Path,
    get_region_grow_module,
    region_grow_annot_dir: Path,
    region_grow_annot_current_dir: Path,
    write_current_artifact,
    log,
) -> Optional[Path]:
    enable = _env_flag("FULLBOT_REGION_GROW_ANNOTATE", "1")
    if not enable:
        log("[INFO] Deferred region_grow annotation disabled (FULLBOT_REGION_GROW_ANNOTATE=0).")
        return None
    if not json_path.exists():
        log(f"[WARN] Deferred annotation skipped, JSON missing: {json_path}")
        return None
    if not image_path.exists():
        log(f"[WARN] Deferred annotation skipped, image missing: {image_path}")
        return None
    if not _python_gpu_region_grow_capabilities(Path(os.sys.executable)).get("gpu_ocr_ready", False):
        log("[INFO] Deferred annotation skipped in current interpreter: region_grow GPU stack is unavailable here.")
        return None

    rg = get_region_grow_module()
    if rg is None or not hasattr(rg, "annotate_and_save"):
        log("[WARN] Deferred annotation skipped, region_grow.annotate_and_save unavailable.")
        return None

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        log(f"[WARN] Deferred annotation skipped, invalid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        log("[WARN] Deferred annotation skipped, JSON payload is not an object.")
        return None

    try:
        t0 = time.perf_counter()
        out_path_raw = rg.annotate_and_save(  # type: ignore[attr-defined]
            str(image_path),
            payload.get("results", []),
            payload.get("triangles"),
            output_dir=str(region_grow_annot_dir),
        )
        out_path = Path(out_path_raw) if out_path_raw else None
        if out_path and out_path.exists():
            try:
                write_current_artifact(out_path, region_grow_annot_current_dir, "region_grow_annot.png")
            except Exception:
                pass
            log(f"[TIMER] region_grow_annot_deferred {time.perf_counter() - t0:.3f}s ({out_path.name})")
            return out_path
        log(f"[WARN] Deferred annotation returned missing path: {out_path_raw}")
        return None
    except Exception as exc:
        log(f"[WARN] Deferred annotation failed: {exc}")
        return None


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
    advanced_debug = _env_flag("FULLBOT_ADVANCED_DEBUG", "0")
    rating_mode = str(os.environ.get("RATING_MODE", "heavy") or "heavy").strip().lower()
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
                if advanced_debug and out:
                    log(f"[DEBUG] rating_fast output path: {out}")
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
    if advanced_debug:
        log(
            "[DEBUG] rating runtime cfg "
            f"mode={rating_mode} force_subprocess={force_subprocess if 'force_subprocess' in locals() else 'n/a'} "
            f"run_fast_rating={run_fast_rating}"
        )
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
