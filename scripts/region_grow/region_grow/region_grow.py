# -*- coding: utf-8 -*-
"""
Dropdown / box detector (PaddleOCR 2.9.x) + lasery
WSZYSTKIE OCR wykrycia + informacja o obramowaniu (4 lasery)
"""

# ====== KONFIG =================================================================
import os
import sys
from pathlib import Path
import builtins

ROOT_PATH = Path(__file__).resolve().parents[1]
ROOT = str(ROOT_PATH)
DATA_SCREEN_DIR = ROOT_PATH / "data" / "screen"
RAW_SCREEN_DIR = DATA_SCREEN_DIR / "raw" / "raw screen"
CURRENT_RUN_DIR = DATA_SCREEN_DIR / "current_run"
DEFAULT_IMAGE_PATH = str(CURRENT_RUN_DIR / "screenshot.png")
REGION_GROW_BASE = DATA_SCREEN_DIR / "region_grow"
REGION_GROW_ANNOT_DIR = REGION_GROW_BASE / "region_grow_annot"
REGION_GROW_ANNOT_CURRENT_DIR = REGION_GROW_BASE / "region_grow_annot_current"
REGION_REGIONS_DIR = REGION_GROW_BASE / "regions"
REGION_REGIONS_CURRENT_DIR = REGION_GROW_BASE / "regions_current"
LEGACY_DEFAULT_IMAGE = str(RAW_SCREEN_DIR / "Zrzut ekranu 2025-10-25 163249.png")
PADDLE_GPU_ID = int(os.environ.get("PADDLEOCR_GPU_ID", "0"))
PADDLE_REC_BATCH = int(os.environ.get("PADDLEOCR_REC_BATCH", "16"))
PADDLE_LANG = str(os.environ.get("PADDLEOCR_LANG", "pl") or "pl").strip()

if hasattr(os, "add_dll_directory"):
    candidate_dirs = [
        Path(sys.prefix) / "Library" / "bin",
        Path(sys.prefix) / "DLLs",
    ]
    for dll_dir in candidate_dirs:
        if dll_dir.exists():
            try:
                os.add_dll_directory(str(dll_dir))
            except Exception:
                pass
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")

# === ŚCIEŻKI INTEGRACJI Z CNN ===

# gdzie zapisujemy boxy dla CNN (UWAGA: nowa lokalizacja!)
JSON_OUT_DIR = str(REGION_GROW_BASE / "region_grow")

# runner (plik 1) który wytnie cropy +10% i puści inferencję
CNN_RUNNER = str(ROOT_PATH / "utils" / "CNN" / "cnn_dropdown_runner.py")

# katalog na cropy z jednej rundy (czyścimy przed uruchomieniem)
CNN_CROPS_DIR = str(DATA_SCREEN_DIR / "temp" / "OCR_boxes+10%")

# Flood-fill / box
BASE_RADIUS = 600
RADIUS_SCALE = 3.0  # dodatkowy promień liczony względem rozmiaru boxa
MAX_RADIUS = 1600
TOL_RGB = 3
NEIGHBOR_8 = True
MIN_BOX = 5  # Obniżony próg
SEED_PAD = 10

# OCR
FAST_OCR = False              # używaj rozbudowanego pipeline'u
OCR_CONF_MIN = 0.01
OCR_MAX_SIDE = 2800
MAX_OCR_ITEMS = int(os.environ.get("REGION_GROW_MAX_OCR_ITEMS", "1200"))
MAX_DETECTIONS_FOR_LASER = int(os.environ.get("REGION_GROW_MAX_DETECTIONS_FOR_LASER", "120"))
OCR_NMS_IOU = 0.60
OCR_NMS_CONTAIN = float(os.environ.get("REGION_GROW_OCR_NMS_CONTAIN", "0.82"))
DB_THRESH = 0.12
DB_BOX_THRESH = 0.32
DB_UNCLIP = 1.45
OCR_SCALES = [0.90, 1.05]            # kilka skal, by łapać mały tekst (szybciej)
FORCE_DET_ONLY_IF_EMPTY_TEXT = True
MIN_OCR_RESULTS = 5  # jeśli mniej wyników, dołóż fallback det-only
OCR_MIN_BOX_W = int(os.environ.get("REGION_GROW_OCR_MIN_BOX_W", "22"))
OCR_MIN_BOX_H = int(os.environ.get("REGION_GROW_OCR_MIN_BOX_H", "10"))
OCR_MIN_BOX_AREA = int(os.environ.get("REGION_GROW_OCR_MIN_BOX_AREA", "240"))
OCR_MAX_ASPECT = float(os.environ.get("REGION_GROW_OCR_MAX_ASPECT", "45.0"))
OCR_LINE_MERGE = bool(int(os.environ.get("REGION_GROW_OCR_LINE_MERGE", "1")))
OCR_LINE_MERGE_X_GAP = float(os.environ.get("REGION_GROW_OCR_LINE_MERGE_X_GAP", "1.35"))
OCR_LINE_MERGE_Y_CENTER = float(os.environ.get("REGION_GROW_OCR_LINE_MERGE_Y_CENTER", "0.55"))
OCR_LINE_MERGE_Y_OVERLAP = float(os.environ.get("REGION_GROW_OCR_LINE_MERGE_Y_OVERLAP", "0.50"))
OCR_AUTOSHIFT = bool(int(os.environ.get("REGION_GROW_OCR_AUTOSHIFT", "1")))
OCR_AUTOSHIFT_MAX = int(os.environ.get("REGION_GROW_OCR_AUTOSHIFT_MAX", "24"))
OCR_AUTOSHIFT_MIN_GAIN = float(os.environ.get("REGION_GROW_OCR_AUTOSHIFT_MIN_GAIN", "1.01"))
# Global calibration offset (your environment shows stable left/up bias).
OCR_SHIFT_X = int(os.environ.get("REGION_GROW_OCR_SHIFT_X", "0"))
OCR_SHIFT_Y = int(os.environ.get("REGION_GROW_OCR_SHIFT_Y", "-50"))
RAPID_OCR_MAX_SIDE = int(os.environ.get("RAPID_OCR_MAX_SIDE", "960"))  # większa precyzja boxów
RAPID_OCR_AUTOCROP = bool(int(os.environ.get("RAPID_OCR_AUTOCROP", "0")))
RAPID_AUTOCROP_DELTA = int(os.environ.get("RAPID_OCR_AUTOCROP_DELTA", "12"))
RAPID_AUTOCROP_PADDING = int(os.environ.get("RAPID_OCR_AUTOCROP_PAD", "12"))
RAPID_AUTOCROP_KEEP_RATIO = float(os.environ.get("RAPID_OCR_AUTOCROP_KEEP_RATIO", "0.92"))
RAPID_AUTOCROP_MIN_FG = float(os.environ.get("RAPID_OCR_AUTOCROP_MIN_FG", "0.002"))

# Tło / histogram
HIST_BITS_PER_CH = 4
HIST_TOP_K = 8
GLOBAL_BG_OVER_PCT = 0.50
# Per-region background (layout) heuristics
BG_REGION_MARGIN = 2
BG_REGION_MIN_PIXELS = 32
BG_CLUSTER_TOL_RGB = int(os.environ.get("RG_BG_CLUSTER_TOL", "24"))

# ===================== REGIONS (FLOODFILL) =====================
# Parametry dla generowania `regions_current.png/json` (tło bez boxów).
RG_REGIONS_STEP = int(os.environ.get("RG_REGIONS_STEP", "10"))
RG_REGIONS_TOL_RGB = int(os.environ.get("RG_REGIONS_TOL_RGB", "18"))
RG_REGIONS_MAX = int(os.environ.get("RG_REGIONS_MAX", "12"))

# Filtry jakości (px^2)
RG_MIN_REGION_AREA = int(os.environ.get("RG_MIN_REGION_AREA", "50000"))
RG_MIN_BOX_AREA = int(os.environ.get("RG_MIN_BOX_AREA", "400"))

# Rysowanie
OCR_BOX_COLOR = (0, 128, 255, 255)
REGION_FILL_RGBA = (255, 0, 0, 150)
REGION_EDGE_RGBA = (255, 0, 0, 255)
LASER_ACCEPTED_RGBA = (255, 255, 0, 220)
LASER_REJECTED_RGBA = (255, 105, 180, 180)
LASER_WIDTH_ACCEPT = 3
LASER_WIDTH_REJECT = 2

# Lasery
EDGE_DELTA_E = 8.0
EDGE_CONSEC_N = 1
EDGE_MAX_LEN_PX = 2000
FRAME_SEARCH_MAX_PX = 500
TEXT_MASK_DILATE = 1

ADVANCED_DEBUG = bool(int(os.environ.get("FULLBOT_ADVANCED_DEBUG", "0")))
DEBUG_OCR = ADVANCED_DEBUG or bool(int(os.environ.get("REGION_GROW_DEBUG_OCR", "0")))
ENABLE_TIMINGS = ADVANCED_DEBUG or bool(int(os.environ.get("FULLBOT_ENABLE_TIMERS", "1")))
TIMING_PREFIX = "[TIMER] "
ENABLE_EXTRA_OCR = bool(int(os.environ.get("REGION_GROW_ENABLE_EXTRA_OCR", "0")))
ENABLE_BACKGROUND_LAYOUT = bool(int(os.environ.get("REGION_GROW_ENABLE_BACKGROUND_LAYOUT", "0")))
LASER_REQUIRE_GPU = bool(int(os.environ.get("REGION_GROW_LASER_REQUIRE_GPU", "1")))
EXTRA_OCR_REQUIRE_GPU = bool(int(os.environ.get("REGION_GROW_EXTRA_OCR_REQUIRE_GPU", "1")))
BACKGROUND_LAYOUT_REQUIRE_GPU = bool(int(os.environ.get("REGION_GROW_BG_LAYOUT_REQUIRE_GPU", "1")))
TURBO_DEFAULT = "1" if str(os.environ.get("FULLBOT_TURBO_MODE", "0") or "0").strip().lower() in {"1", "true", "yes", "on"} else "0"
REGION_GROW_TURBO = bool(int(os.environ.get("REGION_GROW_TURBO", TURBO_DEFAULT)))

# Tryb szybkiego uruchomienia (ustaw RG_FAST=1 w środowisku).
FAST_MODE = int(os.environ.get("RG_FAST", "0")) != 0
# W trybie fast używamy jednej skali (szybciej).
if FAST_MODE:
    OCR_SCALES = [1.0]
if REGION_GROW_TURBO:
    OCR_SCALES = [1.0]
    MAX_DETECTIONS_FOR_LASER = int(os.environ.get("REGION_GROW_MAX_DETECTIONS_TURBO", "60") or 60)
    ENABLE_EXTRA_OCR = False
    ENABLE_BACKGROUND_LAYOUT = False
    LASER_REQUIRE_GPU = True
    EXTRA_OCR_REQUIRE_GPU = True
    BACKGROUND_LAYOUT_REQUIRE_GPU = True
    os.environ.setdefault("REGION_GROW_REQUIRE_GPU", "1")
    RG_MIN_BOX_AREA = int(os.environ.get("RG_MIN_BOX_AREA_TURBO", str(RG_MIN_BOX_AREA)) or RG_MIN_BOX_AREA)
    os.environ.setdefault("RG_TARGET_SIDE", str(os.environ.get("REGION_GROW_MAX_SIDE_TURBO", "960") or "960"))

# ==============================================================================

import contextlib
import json
import logging
import os
import subprocess, shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, List, Tuple, Optional, Dict
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
try:
    import pytesseract
except Exception:
    pytesseract = None
import cv2
from paddleocr import PaddleOCR
try:
    import paddle
except Exception:
    paddle = None
try:
    from rapidocr_paddle import RapidOCR  # type: ignore
except Exception:
    RapidOCR = None
try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None
try:
    from cupyx.scipy import ndimage as cupy_ndimage  # type: ignore
except Exception:
    cupy_ndimage = None
USE_GPU_FLOOD = bool(int(os.environ.get("REGION_GROW_USE_GPU", "1")))
# Jeśli wyłączysz fallback, wymagamy CuPy + cupyx.scipy.
REQUIRE_GPU = bool(int(os.environ.get("REGION_GROW_REQUIRE_GPU", "1")))
GPU_ARRAY_AVAILABLE = bool(USE_GPU_FLOOD and cp is not None)
GPU_FLOOD_AVAILABLE = bool(USE_GPU_FLOOD and cp is not None and cupy_ndimage is not None)
if REQUIRE_GPU and not GPU_FLOOD_AVAILABLE:
    raise RuntimeError("REGION_GROW_REQUIRE_GPU=1, ale CuPy/cupyx.scipy.ndimage nie są dostępne – brak GPU flood fill")
try:
    import psutil  # type: ignore
except Exception:
    psutil = None
try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None

# OpenCV optymalizacja + środowisko (jednorazowo)
_env_initialized = False
_env_lock = threading.Lock()
VERBOSE = bool(int(os.environ.get("RG_VERBOSE", "0")))


def _filtered_print(*args, **kwargs):
    if (not VERBOSE) and args and isinstance(args[0], str) and args[0].startswith("[DEBUG"):
        return
    builtins.print(*args, **kwargs)


print = _filtered_print


class _PpocrWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "angle classifier" in msg and record.name.startswith("ppocr"):
            return False
        return True


def _init_environment_once():
    global _env_initialized
    if _env_initialized:
        return
    with _env_lock:
        if _env_initialized:
            return
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(PADDLE_GPU_ID))
        os.environ.setdefault("PP_OCR_SHOW_LOG", "0")
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        logging.getLogger("paddleocr").setLevel(logging.ERROR)
        logging.getLogger("ppocr").addFilter(_PpocrWarningFilter())
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))
        os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
        try:
            cv2.setUseOptimized(True)
            cv2.setNumThreads(max(1, os.cpu_count() or 1))
        except Exception:
            pass
        _env_initialized = True


_kernel_cache: Dict[Tuple[int, int], np.ndarray] = {}


def _get_rect_kernel(ksize: Tuple[int, int]) -> np.ndarray:
    key = (int(ksize[0]), int(ksize[1]))
    cached = _kernel_cache.get(key)
    if cached is not None:
        return cached
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, key)
    _kernel_cache[key] = ker
    return ker


_init_environment_once()
_K3 = _get_rect_kernel((3, 3))
_GPU_UNIFORM_KERNEL = None


def _get_gpu_uniform_kernel(block: int = 23):
    """
    Lazy init: NIE wolno odpalać CuPy kernel compilation w import-time,
    bo to potrafi wywalić cały pipeline zanim dojdziemy do region_grow.
    """
    if not GPU_ARRAY_AVAILABLE:
        raise RuntimeError("GPU array operations required, but CuPy is unavailable.")
    b = int(block)
    if b != 23:
        return cp.ones((b, b), dtype=cp.float32)
    global _GPU_UNIFORM_KERNEL
    if _GPU_UNIFORM_KERNEL is None:
        _GPU_UNIFORM_KERNEL = cp.ones((23, 23), dtype=cp.float32)
    return _GPU_UNIFORM_KERNEL

# =========== GPU utils (CupY) ===========
def _resize_gpu(arr: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return arr
    if GPU_ARRAY_AVAILABLE:
        try:
            arr_cp = cp.asarray(arr)
            res_cp = cupy_ndimage.zoom(arr_cp, (scale, scale, 1), order=1)
            return cp.asnumpy(res_cp)
        except Exception:
            if REQUIRE_GPU:
                raise
    return cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def _gaussian_gpu(arr: np.ndarray, sigma: float) -> np.ndarray:
    if GPU_ARRAY_AVAILABLE:
        try:
            arr_cp = cp.asarray(arr, dtype=cp.float32)
            blur_cp = cupy_ndimage.gaussian_filter(arr_cp, sigma=(sigma, sigma, 0))
            return cp.asnumpy(blur_cp.astype(cp.uint8))
        except Exception:
            if REQUIRE_GPU:
                raise
    return cv2.GaussianBlur(arr, (0, 0), sigma)


def _adaptive_mean_thresh_gpu(gray: np.ndarray, block: int = 23, C: int = 15) -> np.ndarray:
    """
    Przybliżenie cv2.adaptiveThreshold na GPU: lokalna średnia - C.
    """
    if GPU_ARRAY_AVAILABLE:
        try:
            g = cp.asarray(gray, dtype=cp.float32)
            k = _get_gpu_uniform_kernel(block)
            mean = cupy_ndimage.convolve(g, k / float(block * block), mode="reflect")
            bw = (g < (mean - float(C))).astype(cp.uint8) * 255
            return cp.asnumpy(bw)
        except Exception:
            if REQUIRE_GPU:
                raise
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, C)

# ===================== Timer =====================
class StageTimer:
    __slots__ = ("enabled", "prefix", "_last", "_start", "records")

    def __init__(self, enabled=False, prefix=TIMING_PREFIX):
        self.enabled = enabled
        self.prefix = prefix
        if enabled:
            now = time.perf_counter()
            self._last = now
            self._start = now
        else:
            self._last = 0.0
            self._start = 0.0
        self.records: list[dict] = []
    
    def mark(self, label: str):
        if not self.enabled:
            return 0.0
        now = time.perf_counter()
        dt = now - self._last
        print(f"{self.prefix}{label}: {dt*1000:.1f} ms")
        self._last = now
        self.records.append({"label": label, "ms": dt * 1000.0})
        return dt

    def add(self, label: str, dt_seconds: float):
        if not self.enabled:
            return
        ms = dt_seconds * 1000.0
        print(f"{self.prefix}{label}: {ms:.1f} ms")
        self.records.append({"label": label, "ms": ms})
    
    def total(self, label: str = "TOTAL"):
        if not self.enabled: return 0.0
        now = time.perf_counter()
        dt = now - self._start
        print(f"{self.prefix}{label}: {dt*1000:.1f} ms")
        self.records.append({"label": label, "ms": dt * 1000.0})
        return dt

    def dump_json(self, path: str | Path):
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "enabled": self.enabled,
                "records": self.records,
                "total_ms": sum(rec["ms"] for rec in self.records),
            }
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

# ===================== UTIL =====================
def clamp(v, lo, hi): 
    clipped = np.clip(v, lo, hi)
    if isinstance(clipped, np.ndarray):
        return clipped
    try:
        return clipped.item()
    except Exception:
        return clipped

def rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(img_rgb), cv2.COLOR_RGB2LAB)

def deltaE76(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.int16) - b.astype(np.int16)
    return float(np.sqrt((d * d).sum()))

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1)
    ub = (bx2 - bx1) * (by2 - by1)
    return inter / float(ua + ub - inter + 1e-9)

def suppress_ocr_overlaps(ocr_raw, iou_thr=OCR_NMS_IOU):
    if not ocr_raw:
        return ocr_raw

    boxes = []
    quads = []
    texts = []
    confs = []

    for q, t, c in ocr_raw:
        xs = [int(p[0]) for p in q]
        ys = [int(p[1]) for p in q]
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
        quads.append(q)
        texts.append(t)
        confs.append(float(c))

    boxes_np = np.asarray(boxes, dtype=np.float32)
    order = np.argsort(np.asarray(confs, dtype=np.float32))[::-1]
    boxes_np = boxes_np[order]
    quads_sorted = [quads[i] for i in order]
    texts_sorted = [texts[i] for i in order]
    confs_sorted = [float(confs[i]) for i in order]

    keep_idx: List[int] = []
    idxs = np.arange(len(quads_sorted))
    while idxs.size > 0:
        i = idxs[0]
        keep_idx.append(int(i))
        if idxs.size == 1:
            break
        rest = idxs[1:]

        xx1 = np.maximum(boxes_np[i, 0], boxes_np[rest, 0])
        yy1 = np.maximum(boxes_np[i, 1], boxes_np[rest, 1])
        xx2 = np.minimum(boxes_np[i, 2], boxes_np[rest, 2])
        yy2 = np.minimum(boxes_np[i, 3], boxes_np[rest, 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        area_i = (boxes_np[i, 2] - boxes_np[i, 0]) * (boxes_np[i, 3] - boxes_np[i, 1])
        area_rest = (boxes_np[rest, 2] - boxes_np[rest, 0]) * (boxes_np[rest, 3] - boxes_np[rest, 1])
        union = area_i + area_rest - inter + 1e-9
        iou = inter / union
        min_area = np.minimum(area_i, area_rest) + 1e-9
        contain = inter / min_area

        keep_mask = np.logical_and(iou < iou_thr, contain < OCR_NMS_CONTAIN)
        idxs = rest[keep_mask]

    return [(quads_sorted[i], texts_sorted[i], confs_sorted[i]) for i in keep_idx]

def _quad_to_xyxy(q: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    xs = [int(p[0]) for p in q]
    ys = [int(p[1]) for p in q]
    return (min(xs), min(ys), max(xs), max(ys))

def _xyxy_to_quad(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
    return [(int(x1), int(y1)), (int(x2), int(y1)), (int(x2), int(y2)), (int(x1), int(y2))]

def _is_valid_ocr_bbox(x1: int, y1: int, x2: int, y2: int) -> bool:
    w = max(0, int(x2) - int(x1))
    h = max(0, int(y2) - int(y1))
    if w < OCR_MIN_BOX_W or h < OCR_MIN_BOX_H:
        return False
    area = w * h
    if area < OCR_MIN_BOX_AREA:
        return False
    aspect = (w / float(max(1, h)))
    if aspect > OCR_MAX_ASPECT:
        return False
    return True

def _merge_ocr_lines(ocr_items: List[Tuple[List[Tuple[int, int]], str, float]]) -> List[Tuple[List[Tuple[int, int]], str, float]]:
    if not OCR_LINE_MERGE or not ocr_items:
        return ocr_items

    rows = []
    for q, t, c in ocr_items:
        try:
            x1, y1, x2, y2 = _quad_to_xyxy(q)
        except Exception:
            continue
        if not _is_valid_ocr_bbox(x1, y1, x2, y2):
            continue
        h = max(1, y2 - y1)
        rows.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "h": h, "txt": str(t or "").strip(), "conf": float(c or 0.0)})

    if not rows:
        return []

    rows.sort(key=lambda r: (r["y1"], r["x1"]))
    merged: List[dict] = []
    for r in rows:
        if not merged:
            merged.append(r)
            continue

        prev = merged[-1]
        prev_h = max(1, prev["y2"] - prev["y1"])
        cur_h = max(1, r["y2"] - r["y1"])
        y_center_prev = 0.5 * (prev["y1"] + prev["y2"])
        y_center_cur = 0.5 * (r["y1"] + r["y2"])
        y_center_ok = abs(y_center_cur - y_center_prev) <= OCR_LINE_MERGE_Y_CENTER * max(prev_h, cur_h)

        inter_h = max(0, min(prev["y2"], r["y2"]) - max(prev["y1"], r["y1"]))
        y_overlap_ok = (inter_h / float(max(1, min(prev_h, cur_h)))) >= OCR_LINE_MERGE_Y_OVERLAP

        x_gap = r["x1"] - prev["x2"]
        max_gap = OCR_LINE_MERGE_X_GAP * max(10.0, float(min(prev_h, cur_h)))
        x_ok = x_gap <= max_gap

        if y_center_ok and y_overlap_ok and x_ok:
            prev["x1"] = min(prev["x1"], r["x1"])
            prev["y1"] = min(prev["y1"], r["y1"])
            prev["x2"] = max(prev["x2"], r["x2"])
            prev["y2"] = max(prev["y2"], r["y2"])
            prev["conf"] = max(float(prev["conf"]), float(r["conf"]))
            if r["txt"]:
                if prev["txt"]:
                    prev["txt"] = f"{prev['txt']} {r['txt']}".strip()
                else:
                    prev["txt"] = r["txt"]
        else:
            merged.append(r)

    out: List[Tuple[List[Tuple[int, int]], str, float]] = []
    for m in merged:
        if not _is_valid_ocr_bbox(m["x1"], m["y1"], m["x2"], m["y2"]):
            continue
        out.append((_xyxy_to_quad(m["x1"], m["y1"], m["x2"], m["y2"]), str(m["txt"] or "").strip(), float(m["conf"])))
    return out

def _postprocess_ocr_items(ocr_items: List[Tuple[List[Tuple[int, int]], str, float]]) -> List[Tuple[List[Tuple[int, int]], str, float]]:
    if not ocr_items:
        return []

    cleaned: List[Tuple[List[Tuple[int, int]], str, float]] = []
    for q, t, c in ocr_items:
        try:
            x1, y1, x2, y2 = _quad_to_xyxy(q)
        except Exception:
            continue
        if not _is_valid_ocr_bbox(x1, y1, x2, y2):
            continue
        cleaned.append((_xyxy_to_quad(x1, y1, x2, y2), str(t or "").strip(), float(c or 0.0)))

    if not cleaned:
        return []

    merged = _merge_ocr_lines(cleaned)
    merged = suppress_ocr_overlaps(merged, OCR_NMS_IOU)
    merged.sort(key=lambda r: float(r[2] or 0.0), reverse=True)
    if len(merged) > MAX_OCR_ITEMS:
        merged = merged[:MAX_OCR_ITEMS]
    return merged


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w), dtype=mask.dtype)
    if h <= 0 or w <= 0:
        return out
    sx1 = max(0, -dx)
    sx2 = min(w, w - dx) if dx >= 0 else w
    sy1 = max(0, -dy)
    sy2 = min(h, h - dy) if dy >= 0 else h
    tx1 = max(0, dx)
    tx2 = tx1 + max(0, sx2 - sx1)
    ty1 = max(0, dy)
    ty2 = ty1 + max(0, sy2 - sy1)
    if sx2 > sx1 and sy2 > sy1 and tx2 > tx1 and ty2 > ty1:
        out[ty1:ty2, tx1:tx2] = mask[sy1:sy2, sx1:sx2]
    return out


def _estimate_ocr_global_shift(
    img_rgb: np.ndarray,
    ocr_items: List[Tuple[List[Tuple[int, int]], str, float]],
) -> Tuple[int, int]:
    if (not OCR_AUTOSHIFT) or img_rgb is None or img_rgb.size == 0 or (not ocr_items):
        return (0, 0)
    h, w = img_rgb.shape[:2]
    if h <= 8 or w <= 8:
        return (0, 0)

    # OCR mask from current boxes.
    mask_ocr = np.zeros((h, w), dtype=np.uint8)
    for q, _, _ in ocr_items:
        try:
            x1, y1, x2, y2 = _quad_to_xyxy(q)
        except Exception:
            continue
        x1 = clamp(int(x1), 0, w - 1)
        y1 = clamp(int(y1), 0, h - 1)
        x2 = clamp(int(x2), x1 + 1, w)
        y2 = clamp(int(y2), y1 + 1, h)
        mask_ocr[y1:y2, x1:x2] = 255
    if int(np.count_nonzero(mask_ocr)) < 64:
        return (0, 0)

    # Text-edge mask from image (contrast edges on dark UI).
    gray = cv2.cvtColor(np.ascontiguousarray(img_rgb), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0))
    _, edges = cv2.threshold(grad, 24, 255, cv2.THRESH_BINARY)
    edges = cv2.dilate(edges, _K3, iterations=1)
    if int(np.count_nonzero(edges)) < 64:
        return (0, 0)

    def _best_shift(mask_in: np.ndarray, max_s: int) -> Tuple[int, int, int, int]:
        base_local = int(np.count_nonzero(np.logical_and(mask_in > 0, edges > 0)))
        best_local = base_local
        best_dx_local, best_dy_local = 0, 0
        for dy in range(-max_s, max_s + 1):
            for dx in range(-max_s, max_s + 1):
                if dx == 0 and dy == 0:
                    continue
                shifted = _shift_mask(mask_in, dx, dy)
                score = int(np.count_nonzero(np.logical_and(shifted > 0, edges > 0)))
                if score > best_local:
                    best_local = score
                    best_dx_local, best_dy_local = int(dx), int(dy)
        return best_dx_local, best_dy_local, base_local, best_local

    max_s = max(0, int(OCR_AUTOSHIFT_MAX))
    dx1, dy1, base1, best1 = _best_shift(mask_ocr, max_s)
    gain1 = (float(best1) / float(max(1, base1))) if base1 > 0 else 0.0
    if (dx1 == 0 and dy1 == 0) or gain1 < float(OCR_AUTOSHIFT_MIN_GAIN):
        return (0, 0)

    # Fine-tune around the first shift to catch small residual bias (1-4 px).
    shifted_once = _shift_mask(mask_ocr, dx1, dy1)
    dx2, dy2, base2, best2 = _best_shift(shifted_once, 4)
    gain2 = (float(best2) / float(max(1, base2))) if base2 > 0 else 0.0
    if gain2 >= 1.003:
        return (int(dx1 + dx2), int(dy1 + dy2))
    return (dx1, dy1)


def _apply_ocr_global_shift(
    ocr_items: List[Tuple[List[Tuple[int, int]], str, float]],
    dx: int,
    dy: int,
    w: int,
    h: int,
) -> List[Tuple[List[Tuple[int, int]], str, float]]:
    if (dx == 0 and dy == 0) or not ocr_items:
        return ocr_items
    out: List[Tuple[List[Tuple[int, int]], str, float]] = []
    for q, t, c in ocr_items:
        qq: List[Tuple[int, int]] = []
        for px, py in q:
            nx = clamp(int(px) + int(dx), 0, max(0, int(w) - 1))
            ny = clamp(int(py) + int(dy), 0, max(0, int(h) - 1))
            qq.append((int(nx), int(ny)))
        out.append((qq, t, c))
    return out

def color_close_rgb(a: np.ndarray, b: np.ndarray, tol: int = TOL_RGB) -> bool:
    diff = np.subtract(a, b, dtype=np.int16)
    return bool(np.max(np.abs(diff)) <= tol)

# Zamień funkcję to_py() na bezpieczniejszą wersję:
def to_py(obj):
    """Konwersja numpy/Python do czystego Python (rekurencyjna, bezpieczna)"""
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        # Sprawdź NaN/Inf
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, (str, bytes)):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    # Fallback
    try:
        return obj.item()
    except (AttributeError, ValueError):
        return str(obj)

def _latest_image_in_dir(dir_path: Path) -> Optional[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if not dir_path.is_dir():
        return None
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

def _resolve_default_image() -> str:
    """Prefer the pipeline screenshot saved at start; fall back to legacy/raw images."""
    candidates: List[Optional[Path]] = [
        Path(DEFAULT_IMAGE_PATH),
        Path(LEGACY_DEFAULT_IMAGE) if LEGACY_DEFAULT_IMAGE else None,
        _latest_image_in_dir(RAW_SCREEN_DIR),
    ]
    for cand in candidates:
        if cand and cand.exists():
            return str(cand)
    return DEFAULT_IMAGE_PATH

# Debug entrypoint (single image with verbose diagnostics)
def main_debug():
    print("="*60)
    print("[DEBUG] SCRIPT START")
    print("="*60)
    
    import sys
    
    try:
        path = sys.argv[1] if len(sys.argv) > 1 else _resolve_default_image()
        print(f"[DEBUG] Image path: {path}")
        
        if not os.path.isfile(path):
            print(f"[ERROR] File not found: {path}")
            return
        
        print("[DEBUG] Running detection...")
        out = run_dropdown_detection(path)
        print(f"[DEBUG] Detection returned {len(out.get('results', []))} results")
        
        print("[DEBUG] Creating annotation...")
        out_path = annotate_and_save(path, out.get("results", []), out.get("triangles"), output_dir=str(REGION_GROW_ANNOT_DIR))
        print(f"[DEBUG] Annotation created: {out_path}")
        
        print("\n[DEBUG] Converting to Python types...")
        try:
            out_py = to_py(out)
            print(f"[DEBUG] Conversion OK, keys: {list(out_py.keys())}")
        except Exception as e:
            print(f"[ERROR] to_py() failed: {e}")
            import traceback
            traceback.print_exc()
            out_py = {"error": str(e)}
        
        print("\n[DEBUG] Generating JSON...")
        try:
            json_output = json.dumps(out_py, ensure_ascii=False, indent=2)
            print(f"[DEBUG] JSON length: {len(json_output)} chars")
        except Exception as e:
            print(f"[ERROR] json.dumps() failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback - prosta wersja
            json_output = json.dumps({
                "image": out_py.get("image"),
                "results_count": len(out_py.get("results", [])),
                "error": str(e)
            }, indent=2)
        
        print("\n" + "="*60)
        print("JSON OUTPUT:")
        print("="*60)
        print(json_output)
        print("\n" + "="*60)
        print(f"Annotation: {out_path}")
        print("="*60)
    
    except Exception as e:
        print(f"\n[ERROR] Script failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[DEBUG] SCRIPT END")

# ===================== OCR =====================
_paddle = None
_rapid = None
_last_ocr_debug = {}
_paddle_lock = threading.Lock()
_rapid_lock = threading.Lock()

def get_ocr():
    global _paddle
    if _paddle is not None:
        return _paddle

    if paddle is None:
        raise RuntimeError("Paddle is not available - cannot initialize PaddleOCR")

    _init_environment_once()

    with _paddle_lock:
        if _paddle is not None:
            return _paddle

        # Prefer GPU; fail loudly if CUDA build unavailable
        if paddle.is_compiled_with_cuda():
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(PADDLE_GPU_ID))
            target_device = f"gpu:{PADDLE_GPU_ID}"
            paddle.set_device(target_device)
            device_label = target_device
        else:
            raise RuntimeError("Paddle installed without CUDA support - GPU OCR unavailable")

        try:
            base_kwargs = dict(
                lang=PADDLE_LANG,
                use_textline_orientation=False,
                text_recognition_batch_size=PADDLE_REC_BATCH,
            )
            try:
                _paddle = PaddleOCR(
                    **base_kwargs,
                    use_gpu=True,
                    gpu_id=PADDLE_GPU_ID,
                )
            except Exception as exc:
                # PaddleOCR >=3 can reject legacy use_gpu/gpu_id kwargs.
                # Depending on version it may raise TypeError or ValueError-like exceptions.
                msg = str(exc).lower()
                legacy_arg_rejected = (
                    "use_gpu" in msg
                    or "gpu_id" in msg
                    or "unknown argument" in msg
                    or "unexpected keyword" in msg
                )
                if not legacy_arg_rejected:
                    raise
                _paddle = PaddleOCR(**base_kwargs)
            try:
                device_now = paddle.device.get_device()
            except Exception:
                device_now = device_label
            print(f"[DEBUG] PaddleOCR initialized on {device_now} (batch={PADDLE_REC_BATCH})")
        except Exception as exc:
            print(f"[ERROR] PaddleOCR initialization failed: {exc}")
            _paddle = None
            raise

    return _paddle

def get_rapid_ocr():
    global _rapid
    if _rapid is not None:
        return _rapid

    if RapidOCR is None:
        raise RuntimeError("RapidOCR (rapidocr_paddle) is not installed - cannot run OCR on GPU")
    if paddle is None:
        raise RuntimeError("Paddle is not available - RapidOCR paddle backend cannot use GPU")

    _init_environment_once()

    with _rapid_lock:
        if _rapid is not None:
            return _rapid

        if not paddle.is_compiled_with_cuda():
            raise RuntimeError(
                "Paddle used by RapidOCR is not compiled with CUDA - GPU OCR is not possible in this environment"
            )

        target_device = f"gpu:{PADDLE_GPU_ID}"
        try:
            paddle.device.set_device(target_device)
        except Exception:
            try:
                paddle.device.set_device("gpu")
                target_device = "gpu"
            except Exception as exc2:
                raise RuntimeError(f"Failed to set Paddle device for RapidOCR to GPU ({target_device}): {exc2}") from exc2

        try:
            _rapid = RapidOCR(
                det_use_cuda=True,
                cls_use_cuda=True,
                rec_use_cuda=True,
                det_gpu_id=PADDLE_GPU_ID,
                cls_gpu_id=PADDLE_GPU_ID,
                rec_gpu_id=PADDLE_GPU_ID,
            )
            try:
                device_now = paddle.device.get_device()
            except Exception:
                device_now = target_device
            print(f"[DEBUG] RapidOCR (paddle) initialized on {device_now} (gpu_id={PADDLE_GPU_ID})")
        except Exception as exc:
            _rapid = None
            raise RuntimeError(f"RapidOCR GPU initialization failed: {exc}") from exc

    return _rapid

def _preprocess_for_ocr(img_pil: Image.Image):
    w, h = img_pil.size
    base_scale = 1.0
    # Docelowy dłuższy bok dla OCR (można nadpisać RG_TARGET_SIDE)
    target_side = int(os.environ.get("RG_TARGET_SIDE", "1024"))
    if max(w, h) < target_side:
        base_scale = target_side / float(max(w, h))

    if base_scale != 1.0:
        tgt = img_pil.resize((int(w * base_scale), int(h * base_scale)), Image.BILINEAR)
    else:
        tgt = img_pil

    arr0 = np.ascontiguousarray(np.asarray(tgt.convert("RGB"), dtype=np.uint8))

    lab = cv2.cvtColor(arr0, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.createCLAHE(2.0, (8, 8)).apply(L)
    arr0 = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2RGB)
    arr0 = cv2.addWeighted(arr0, 1.5, cv2.GaussianBlur(arr0, (0, 0), 1.0), -0.5, 0)
    
    return arr0, base_scale

def _prepare_rapid_image(arr: np.ndarray):
    """
    Dla RapidOCR ogranicz rozdzielczość wejścia (dowolnie w dół),
    aby skrócić czas det+rec na dużych screenach.
    """
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    max_side = float(max(h, w))
    if max_side <= float(RAPID_OCR_MAX_SIDE):
        return arr, 1.0
    scale = float(RAPID_OCR_MAX_SIDE) / max_side
    resized = _resize_gpu(arr, scale)
    return resized, scale

def _auto_crop_for_rapid(arr: np.ndarray):
    if (not RAPID_OCR_AUTOCROP) or arr is None or arr.size == 0:
        return arr, (0, 0)
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    if h <= 0 or w <= 0:
        return arr, (0, 0)
    arr16 = arr.astype(np.int16, copy=False)
    bg = np.median(arr16.reshape(-1, 3), axis=0)
    diff = np.max(np.abs(arr16 - bg.reshape(1, 1, 3)), axis=2)
    mask = diff > RAPID_AUTOCROP_DELTA
    fg_ratio = float(mask.mean())
    if fg_ratio < RAPID_AUTOCROP_MIN_FG:
        return arr, (0, 0)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return arr, (0, 0)
    y1 = max(0, int(ys.min()) - RAPID_AUTOCROP_PADDING)
    y2 = min(h, int(ys.max()) + 1 + RAPID_AUTOCROP_PADDING)
    x1 = max(0, int(xs.min()) - RAPID_AUTOCROP_PADDING)
    x2 = min(w, int(xs.max()) + 1 + RAPID_AUTOCROP_PADDING)
    if y2 <= y1 or x2 <= x1:
        return arr, (0, 0)
    crop_area_ratio = float((y2 - y1) * (x2 - x1)) / float(h * w)
    if crop_area_ratio >= RAPID_AUTOCROP_KEEP_RATIO:
        return arr, (0, 0)
    cropped = arr[y1:y2, x1:x2]
    return cropped, (x1, y1)

def _parse_rec_result(res) -> Tuple[str, float]:
    try:
        # PaddleOCR >=3.x: list[OCRResult]-like dict objects
        if isinstance(res, list) and res and isinstance(res[0], dict):
            item0 = res[0]
            rec_texts = item0.get("rec_texts") if isinstance(item0, dict) else None
            rec_scores = item0.get("rec_scores") if isinstance(item0, dict) else None
            if isinstance(rec_texts, list) and rec_texts:
                txt = str(rec_texts[0] or "").strip()
                score = 0.0
                if isinstance(rec_scores, list) and rec_scores:
                    with contextlib.suppress(Exception):
                        score = float(rec_scores[0] or 0.0)
                return txt, float(score)

        if not isinstance(res, list) or not res:
            return "", 0.0
        entry = res[0]

        def _try_unpack(node):
            if isinstance(node, (list, tuple)) and len(node) >= 2 and isinstance(node[1], (int, float)):
                return (node[0] or "").strip(), float(node[1] or 0.0)
            if (
                isinstance(node, (list, tuple))
                and len(node) >= 2
                and isinstance(node[1], (list, tuple))
                and len(node[1]) >= 2
            ):
                return (node[1][0] or "").strip(), float(node[1][1] or 0.0)
            return None

        # Case 1: entry is already a (text, conf) tuple/list
        unpacked = _try_unpack(entry)
        if unpacked:
            return unpacked

        # Case 2: entry is a list whose first element is (text, conf)
        if isinstance(entry, (list, tuple)) and entry:
            unpacked = _try_unpack(entry[0])
            if unpacked:
                return unpacked
    except Exception:
        pass
    return "", 0.0


def _extract_ocr_lines(res: Any, inv: float = 1.0) -> List[Tuple[List[Tuple[int, int]], str, float]]:
    outs: List[Tuple[List[Tuple[int, int]], str, float]] = []
    try:
        if not isinstance(res, list) or not res:
            return outs

        # PaddleOCR >=3.x
        if isinstance(res[0], dict):
            for item in res:
                if not isinstance(item, dict):
                    continue
                polys = item.get("dt_polys") or item.get("rec_polys") or []
                texts = item.get("rec_texts") or []
                scores = item.get("rec_scores") or []
                if not isinstance(polys, list):
                    continue
                for i, poly in enumerate(polys):
                    try:
                        if hasattr(poly, "tolist"):
                            pts = poly.tolist()
                        else:
                            pts = poly
                        if not isinstance(pts, (list, tuple)) or len(pts) < 4:
                            continue
                        quad = []
                        for p in pts[:4]:
                            if not (isinstance(p, (list, tuple)) and len(p) >= 2):
                                quad = []
                                break
                            quad.append((int(round(float(p[0]) * inv)), int(round(float(p[1]) * inv))))
                        if len(quad) != 4:
                            continue
                        txt = str(texts[i] if i < len(texts) else "").strip() if isinstance(texts, list) else ""
                        conf = float(scores[i]) if (isinstance(scores, list) and i < len(scores)) else 0.0
                        outs.append((quad, txt, conf))
                    except Exception:
                        continue
            return outs

        # PaddleOCR <=2.x legacy list format
        lines = res[0] if isinstance(res, list) and len(res) > 0 else []
        for it in lines:
            try:
                quad = it[0]
                txt = (it[1][0] or "").strip()
                conf = float(it[1][1] or 0.0)
                quad = [(int(round(x * inv)), int(round(y * inv))) for (x, y) in quad]
                outs.append((quad, txt, conf))
            except Exception:
                continue
    except Exception:
        return outs
    return outs

def _tesseract_fallback(img_pil: Image.Image) -> List[Tuple[List[Tuple[int, int]], str, float]]:
    """Generuje (quad, text, conf) wykorzystując pytesseract, gdy Paddle nie widzi pól."""
    if pytesseract is None:
        return []
    try:
        data = pytesseract.image_to_data(
            img_pil,
            output_type=getattr(pytesseract, "Output", None).DICT if hasattr(pytesseract, "Output") else None,
        )
    except Exception:
        return []

    # Jeśli pytesseract.Output.DICT nie jest dostępny (stare wersje), spróbuj parsować ręcznie
    if not isinstance(data, dict) or "text" not in data:
        return []

    outs = []
    n = len(data.get("text", []))
    for i in range(n):
        try:
            txt = (data["text"][i] or "").strip()
            if len(txt) < 2:
                continue
            conf = float(data.get("conf", ["0"]*n)[i])
            if conf <= 0:
                continue
            x = int(data.get("left", [0]*n)[i])
            y = int(data.get("top", [0]*n)[i])
            w = int(data.get("width", [0]*n)[i])
            h = int(data.get("height", [0]*n)[i])
            if w <= 1 or h <= 1:
                continue
            quad = [
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h),
            ]
            outs.append((quad, txt, conf / 100.0))
        except Exception:
            continue

    return outs

def _cv_text_regions(img_rgb: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Wykrywa prostok?ty tekstu klasycznymi metodami CV (fallback gdy OCR nic nie widzi)."""
    # NOTE:
    # This helper is only a supplemental OCR-region proposal stage.
    # If CuPy kernel compilation fails (e.g. missing CUDA headers), do not abort
    # whole region_grow; fall back to robust OpenCV CPU path for this step only.
    try_gpu = bool(GPU_ARRAY_AVAILABLE)
    if try_gpu:
        try:
            img_cp = cp.asarray(img_rgb, dtype=cp.float32)
            gray_cp = (0.299 * img_cp[..., 0] + 0.587 * img_cp[..., 1] + 0.114 * img_cp[..., 2])
            blur_cp = cupy_ndimage.gaussian_filter(gray_cp, sigma=1.0)
            gray_np = cp.asnumpy(cp.clip(blur_cp, 0, 255).astype(cp.uint8))
            bw = _adaptive_mean_thresh_gpu(gray_np, 23, 15)
            kernel = _get_rect_kernel((9, 3))
            dil_gpu = _gpu_binary_dilate(bw, kernel, iterations=1)
            dil = dil_gpu if dil_gpu is not None else cv2.dilate(bw, kernel, iterations=1)
        except Exception as exc:
            print(f"[WARN] _cv_text_regions GPU path failed, using CPU OpenCV: {exc}")
            try_gpu = False

    if not try_gpu:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 23, 15)
        kernel = _get_rect_kernel((9, 3))
        dil = cv2.dilate(bw, kernel, iterations=1)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 30 or h < 12:
            continue
        if w * h < 400:
            continue
        aspect = w / float(h)
        if aspect < 1.5:
            continue
        x2, y2 = x + w, y + h
        quads.append([(x, y), (x2, y), (x2, y2), (x, y2)])
    return quads

def read_ocr_faster(img_pil: Image.Image, timer: Optional[StageTimer] = None):
    print("[DEBUG] OCR: Preprocessing...")
    arr0, base_scale = _preprocess_for_ocr(img_pil)
    if timer: timer.mark("OCR preprocess")
    
    t_init_start = time.perf_counter()
    ocr = get_ocr()
    t_init = time.perf_counter() - t_init_start
    if timer: timer.add("OCR init", t_init)
    det_quads = []
    
    # DET multi-scale
    print("[DEBUG] OCR: Detection phase...")
    def _det_scale(scale: float):
        try:
            arr = _resize_gpu(arr0, scale)
            det = ocr.ocr(arr)
            return scale, det, None
        except Exception as exc:
            return scale, None, exc

    scale_results = []
    max_workers = min(len(OCR_SCALES), max(1, min(4, (os.cpu_count() or 2))))
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for fut in as_completed([ex.submit(_det_scale, s) for s in OCR_SCALES]):
                scale_results.append(fut.result())
    else:
        for s in OCR_SCALES:
            scale_results.append(_det_scale(s))

    for s, det, err in sorted(scale_results, key=lambda x: x[0]):
        if err is not None:
            print(f"[ERROR] OCR detection scale {s}: {err}")
            continue
        inv = 1.0 / (base_scale * s)
        det_rows = _extract_ocr_lines(det, inv=inv)
        for q, _, _ in det_rows:
            det_quads.append(q)
        print(f"[DEBUG] OCR: Scale {s} -> {len(det_rows)} boxes")

    if not det_quads:
        print("[DEBUG] OCR: No detections found")
        if timer: timer.mark("OCR det (no boxes)")
        return []

    # NMS + limit
    det_boxes = suppress_ocr_overlaps([(q, "", 0.0) for q in det_quads], OCR_NMS_IOU)
    if len(det_boxes) > MAX_OCR_ITEMS:
        det_boxes = det_boxes[:MAX_OCR_ITEMS]
    
    print(f"[DEBUG] OCR: After NMS -> {len(det_boxes)} boxes")

    # REC batchem
    print("[DEBUG] OCR: Recognition phase...")
    arr_full = np.array(img_pil)
    crops, boxes = [], []
    
    for q, _, _ in det_boxes:
        xs = [int(p[0]) for p in q]
        ys = [int(p[1]) for p in q]
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(arr_full.shape[1], max(xs)), min(arr_full.shape[0], max(ys))
        if x2 - x1 <= 1 or y2 - y1 <= 1: 
            continue
        crops.append(arr_full[y1:y2, x1:x2])
        boxes.append(q)

    rec_outs = []
    if crops:
        res_list = []
        for c in crops:
            with contextlib.suppress(Exception):
                res_list.append(ocr.ocr(c))
        for q, res in zip(boxes, res_list):
            txt, conf = _parse_rec_result(res)
            rec_outs.append((q, txt, conf))

    outs = [r for r in rec_outs if float(r[2]) >= OCR_CONF_MIN]
    
    if (not outs or not any((t.strip() != "") for _, t, _ in outs)) and FORCE_DET_ONLY_IF_EMPTY_TEXT:
        print("[DEBUG] OCR: No text found, using det-only mode")
        outs = [(q, "", 0.5) for (q, _, _) in det_boxes]

    outs.sort(key=lambda r: float(r[2]), reverse=True)
    outs = suppress_ocr_overlaps(outs, OCR_NMS_IOU)
    
    if len(outs) > MAX_OCR_ITEMS:
        outs = outs[:MAX_OCR_ITEMS]
    
    print(f"[DEBUG] OCR: Final -> {len(outs)} results")
    if timer: timer.mark("OCR fast total")
    return outs

def read_ocr_full(img_pil: Image.Image, timer: Optional[StageTimer] = None):
    global _last_ocr_debug
    print("[DEBUG] OCR: Using FULL mode...")
    arr0, base_scale = _preprocess_for_ocr(img_pil)
    if timer: timer.mark("OCR preprocess")
    
    t_init_start = time.perf_counter()
    ocr = get_ocr()
    t_init = time.perf_counter() - t_init_start
    if timer: timer.add("OCR init", t_init)
    outs = []
    dbg = {
        "ppocr": "v4", 
        "base_scale": base_scale, 
        "scales": [],
        "boxes_rec_raw": 0, 
        "boxes_kept_after_thr_nms": 0,
        "det_only_used": False, 
        "det_only_boxes": 0, 
        "tesseract_boxes": 0,
        "cv_boxes": 0,
        "errors": []
    }

    def _rec_scale(scale: float):
        try:
            arr = _resize_gpu(arr0, scale)
            res = ocr.ocr(arr)
            lines = _extract_ocr_lines(res, inv=1.0)
            return scale, lines, None
        except Exception as exc:
            return scale, None, exc

    t_ocr_start = time.perf_counter()
    scale_results = []
    max_workers = min(len(OCR_SCALES), max(1, min(4, (os.cpu_count() or 2))))
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for fut in as_completed([ex.submit(_rec_scale, s) for s in OCR_SCALES]):
                scale_results.append(fut.result())
    else:
        for s in OCR_SCALES:
            scale_results.append(_rec_scale(s))

    for s, lines, err in sorted(scale_results, key=lambda x: x[0]):
        if err is not None:
            dbg["errors"].append(f"rec_scale{s}: {err}")
            continue
        dbg["scales"].append({"scale": s, "raw": len(lines)})
        inv = 1.0 / (base_scale * s)
        for it in lines:
            quad = [(int(round(x * inv)), int(round(y * inv))) for (x, y) in it[0]]
            txt = str(it[1] or "").strip()
            conf = float(it[2] or 0.0)
            outs.append((quad, txt, conf))

    if timer:
        timer.add("OCR infer", time.perf_counter() - t_ocr_start)
        timer.mark("OCR full total")

    dbg["boxes_rec_raw"] = len(outs)
    has_text = any((t.strip() != "") for (_, t, _) in outs)

    # Fallback: jeśli wyników jest mało lub brak tekstu, spróbuj
    # (a) wykryć boxy det-only na kilku skalach, a następnie
    # (b) uruchomić rozpoznawanie na każdym wykrytym boxie (bez kolejnej detekcji).
    need_det_only = (
        ((len(outs) == 0) or not has_text) and FORCE_DET_ONLY_IF_EMPTY_TEXT
    ) or (len(outs) < MIN_OCR_RESULTS)
    if need_det_only:
        det_quads = []
        try:
            arr_full = np.array(img_pil)
        except Exception:
            arr_full = None
        for s in OCR_SCALES:
            try:
                arr = cv2.resize(arr0, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                inv = 1.0 / (base_scale * s)
                det_rows = _extract_ocr_lines(ocr.ocr(arr), inv=inv)
                for quad, _, _ in det_rows:
                    det_quads.append(quad)
            except Exception as e:
                dbg["errors"].append(f"det_scale{s}: {e}")

        # OCR-only mode: do not use OpenCV text-region fallback here.
        dbg["cv_boxes"] = 0

        # Rozpoznanie per-box na oryginalnym obrazie (bez dodatkowej detekcji)
        rec_outs = []
        for quad in det_quads:
            try:
                xs = [int(p[0]) for p in quad]
                ys = [int(p[1]) for p in quad]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                if arr_full is None:
                    crop = None
                else:
                    x1c = max(0, x1); y1c = max(0, y1)
                    x2c = min(arr_full.shape[1], x2); y2c = min(arr_full.shape[0], y2)
                    if x2c - x1c > 1 and y2c - y1c > 1:
                        crop = arr_full[y1c:y2c, x1c:x2c]
                    else:
                        crop = None
                if crop is not None:
                    res = ocr.ocr(crop)
                    txt, c = _parse_rec_result(res)
                    rec_outs.append((quad, txt, float(c)))
                else:
                    rec_outs.append((quad, "", 0.5))
            except Exception as e:
                dbg["errors"].append(f"rec_box: {e}")
                rec_outs.append((quad, "", 0.5))

        outs.extend(rec_outs)
        dbg["det_only_used"] = True
        dbg["det_only_boxes"] = len(det_quads)

        # OCR-only mode: do not use Tesseract fallback here.
        dbg["tesseract_boxes"] = 0

    outs = [r for r in outs if float(r[2]) >= OCR_CONF_MIN]
    outs.sort(key=lambda r: float(r[2]), reverse=True)
    outs = suppress_ocr_overlaps(outs, OCR_NMS_IOU)
    
    if len(outs) > MAX_OCR_ITEMS:
        outs = outs[:MAX_OCR_ITEMS]
    
    dbg["boxes_kept_after_thr_nms"] = len(outs)
    _last_ocr_debug = dbg
    return outs

def _to_quad_from_points(points):
    if not points:
        return None
    if isinstance(points[0], (list, tuple)):
        if len(points) == 4:
            return [(int(round(p[0])), int(round(p[1]))) for p in points]
        if len(points) >= 8:
            pts = []
            for idx in range(0, len(points), 2):
                if idx + 1 >= len(points):
                    break
                pts.append((int(round(points[idx])), int(round(points[idx + 1]))))
            if pts:
                return pts[:4]
    elif isinstance(points, (list, tuple)) and len(points) == 8:
        return [(int(round(points[i])), int(round(points[i + 1]))) for i in range(0, 8, 2)]
    return None

def read_ocr_rapid(img_pil: Image.Image, rapid_engine, timer: Optional[StageTimer] = None):
    arr = np.ascontiguousarray(np.array(img_pil))
    if arr is None or arr.size == 0:
        return []
    arr_cropped, (offset_x, offset_y) = _auto_crop_for_rapid(arr)
    arr_proc, scale = _prepare_rapid_image(arr_cropped)
    if DEBUG_OCR and paddle is not None:
        try:
            print(f"[DEBUG] OCR(rapid) device: {paddle.device.get_device()}, scale={scale:.3f}")
        except Exception:
            pass
    result = rapid_engine(arr_proc)
    if isinstance(result, tuple):
        result = result[0]
    inv_scale = 1.0 / (scale if scale > 0 else 1.0)
    outs = []
    if isinstance(result, list):
        for item in result:
            quad = None
            text = ""
            conf = 0.0
            if isinstance(item, dict):
                quad = _to_quad_from_points(item.get("boxes") or item.get("box"))
                text = item.get("text") or ""
                conf = float(item.get("score") or item.get("conf") or 0.0)
            elif isinstance(item, (list, tuple)):
                if item:
                    quad = _to_quad_from_points(item[0])
                if len(item) >= 2:
                    payload = item[1]
                    if isinstance(payload, (list, tuple)):
                        text = str(payload[0] or "")
                        if len(payload) > 1:
                            conf = float(payload[1] or 0.0)
                    elif isinstance(payload, str):
                        text = payload
                        if len(item) > 2:
                            conf = float(item[2] or 0.0)
            if quad is None:
                continue
            # quad jest w przeskalowanej przestrzeni – przywracamy do oryginalnych wymiarów
            quad_rescaled = []
            for pt in quad:
                rx = int(round(pt[0] * inv_scale)) + int(offset_x)
                ry = int(round(pt[1] * inv_scale)) + int(offset_y)
                quad_rescaled.append((rx, ry))
            outs.append((quad_rescaled, (text or "").strip(), conf))
    if timer:
        timer.mark("OCR rapid total")
    return outs

def read_ocr_wrapper(img_pil: Image.Image, timer: Optional[StageTimer] = None):
    # Prefer PaddleOCR by default; RapidOCR only if explicitly enabled
    use_rapid = bool(int(os.environ.get("REGION_GROW_USE_RAPID_OCR", "0")))
    t_dispatch = time.perf_counter()
    mode = "rapid" if use_rapid else ("fast" if FAST_OCR else "full")
    if ADVANCED_DEBUG:
        print(f"[DEBUG] OCR wrapper dispatch mode={mode} image={img_pil.size[0]}x{img_pil.size[1]}")
    if use_rapid:
        try:
            rapid = get_rapid_ocr()
            out = read_ocr_rapid(img_pil, rapid_engine=rapid, timer=timer)
            out = _postprocess_ocr_items(out)
            if timer:
                timer.add("OCR wrapper dispatch", time.perf_counter() - t_dispatch)
            return out
        except Exception as exc:
            print(f"[WARN] RapidOCR failed or unavailable ({exc}), falling back to PaddleOCR")

    if FAST_OCR:
        out = read_ocr_faster(img_pil, timer=timer)
    else:
        out = read_ocr_full(img_pil, timer=timer)
    out = _postprocess_ocr_items(out)
    if timer:
        timer.add("OCR wrapper dispatch", time.perf_counter() - t_dispatch)
    if ADVANCED_DEBUG:
        print(f"[DEBUG] OCR wrapper done mode={mode} results={len(out)} dt={(time.perf_counter() - t_dispatch)*1000.0:.1f} ms")
    return out

def _ocr_text_for_bbox(img_rgb: np.ndarray, bbox: Optional[List[int]], pad: int = 4,
                       min_conf: float = 0.2) -> str:
    """
    Dodatkowe rozpoznanie tekstu wewnątrz zadanego bboxa (np. po flood fillu).
    Przydaje się, gdy początkowy OCR zwrócił pusty string, a chcemy zapisać
    tekst opisujący cały box odpowiedzi/dropdown.
    """
    if bbox is None:
        return ""
    rapid = None
    ocr = None
    use_rapid = bool(int(os.environ.get("REGION_GROW_USE_RAPID_OCR", "0")))
    if use_rapid:
        try:
            rapid = get_rapid_ocr()
        except Exception:
            rapid = None
        if rapid is None:
            return ""
    else:
        try:
            ocr = get_ocr()
        except Exception:
            ocr = None
        if ocr is None or img_rgb is None or img_rgb.size == 0:
            return ""

    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(W, int(x2) + pad)
    y2 = min(H, int(y2) + pad)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return ""

    crop_rgb = img_rgb[y1:y2, x1:x2]
    try:
        if use_rapid and rapid is not None:
            result = rapid(crop_rgb)
            if isinstance(result, tuple):
                result = result[0]
            if isinstance(result, list) and result:
                best = result[0]
                text = ""
                conf = 0.0
                if isinstance(best, dict):
                    text = best.get("text") or ""
                    conf = float(best.get("score") or 0.0)
                elif isinstance(best, (list, tuple)) and len(best) >= 2:
                    payload = best[1]
                    if isinstance(payload, (list, tuple)):
                        text = payload[0]
                        if len(payload) > 1:
                            conf = float(payload[1] or 0.0)
                    elif isinstance(payload, str):
                        text = payload
                if text and conf >= min_conf:
                    return text
        elif not use_rapid and ocr is not None:
            crop_pil = Image.fromarray(crop_rgb)
            arr_proc, _ = _preprocess_for_ocr(crop_pil)
            res = ocr.ocr(arr_proc)
            lines = _extract_ocr_lines(res, inv=1.0)
            texts = []
            for it in lines:
                t = str(it[1] or "").strip()
                c = float(it[2] or 0.0)
                if t and c >= min_conf:
                    texts.append(t)
            if texts:
                return " ".join(texts)
    except Exception:
        pass
    return ""

def _ocr_text_for_bbox_batch(
    img_rgb: np.ndarray,
    bboxes: List[List[int]],
    pad: int = 4,
    min_conf: float = 0.2,
    downscale_factor: float = 0.5,
) -> List[str]:
    """
    Zbiorcze rozpoznawanie tekstu w wielu bboxach jednocze�nie.
    - przyspiesza, bo wywo�ujemy PaddleOCR raz na batch listy crop�w
    - dodatkowo ka�dy crop jest zmniejszany multiplikatywnie (scale < 1),
      np. downscale_factor=0.5 zmniejsza szeroko�� i wysoko�� o po�ow�.
    """
    if not bboxes or img_rgb is None or img_rgb.size == 0:
        return []

    use_rapid = bool(int(os.environ.get("REGION_GROW_USE_RAPID_OCR", "0")))
    # Dla RapidOCR na razie zachowujemy prosty per-box fallback,
    # bo interfejs batchowy jest mniej stabilny ni� w PaddleOCR.
    if use_rapid:
        return [
            _ocr_text_for_bbox(img_rgb, bbox, pad=pad, min_conf=min_conf)
            for bbox in bboxes
        ]

    try:
        ocr = get_ocr()
    except Exception:
        ocr = None
    if EXTRA_OCR_REQUIRE_GPU:
        try:
            import paddle as _paddle  # type: ignore
            if not bool(_paddle.is_compiled_with_cuda()):
                raise RuntimeError("Extra OCR requires CUDA-enabled Paddle (REGION_GROW_EXTRA_OCR_REQUIRE_GPU=1).")
        except Exception as exc:
            raise RuntimeError(f"Extra OCR GPU requirement check failed: {exc}") from exc
    if ocr is None:
        return ["" for _ in bboxes]

    H, W = img_rgb.shape[:2]
    crops: List[np.ndarray] = []
    indices: List[int] = []

    for idx, bbox in enumerate(bboxes):
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(W, int(x2) + pad)
        y2 = min(H, int(y2) + pad)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        crop_rgb = img_rgb[y1:y2, x1:x2]
        if crop_rgb is None or crop_rgb.size == 0:
            continue

        ch, cw = crop_rgb.shape[:2]
        # Multiplikatywne zmniejszenie rozdzielczo�ci cropa
        if downscale_factor > 0.0 and downscale_factor < 1.0:
            new_w = max(2, int(round(cw * downscale_factor)))
            new_h = max(2, int(round(ch * downscale_factor)))
            if new_w < cw or new_h < ch:
                crop_rgb = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        crops.append(crop_rgb)
        indices.append(idx)

    texts_out: List[str] = ["" for _ in bboxes]
    if not crops:
        return texts_out

    try:
        # PaddleOCR >=3.x compatible call.
        res_batch = ocr.ocr(crops)
    except Exception as exc:
        if EXTRA_OCR_REQUIRE_GPU:
            raise RuntimeError(f"Extra OCR batch failed in GPU-only mode: {exc}") from exc
        return [_ocr_text_for_bbox(img_rgb, bbox, pad=pad, min_conf=min_conf) for bbox in bboxes]

    try:
        for img_idx, img_res in enumerate(res_batch):
            # wynik rec-only mo�e mie� kilka format�w, u�ywamy parsera pomocniczego
            txt, conf = _parse_rec_result(img_res)
            if txt and conf >= min_conf and img_idx < len(indices):
                texts_out[indices[img_idx]] = txt
    except Exception:
        # w razie dziwnego formatu wyniku nic nie zmieniamy
        return texts_out

    return texts_out

# ===================== MASKA TEKSTU =====================
def build_text_mask_cv(ocr_items, W, H):
    m = np.zeros((H, W), np.uint8)
    for q, _, _ in ocr_items:
        xs = [int(p[0]) for p in q]
        ys = [int(p[1]) for p in q]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        cv2.rectangle(m, (x1, y1), (min(W, x2+1), min(H, y2+1)), 1, thickness=-1)
    
    if TEXT_MASK_DILATE > 0:
        k = _get_rect_kernel((1 + 2 * TEXT_MASK_DILATE, 1 + 2 * TEXT_MASK_DILATE))
        dil = _gpu_binary_dilate(m, k, iterations=1)
        if dil is not None:
            m = dil.astype(np.uint8)
        else:
            m = cv2.dilate(m, k, iterations=1)
    
    return m.astype(bool)

# ===================== HISTOGRAM =====================
def quantize_rgb(img: np.ndarray, bits: int = HIST_BITS_PER_CH) -> np.ndarray:
    return (img >> (8 - bits)).astype(np.uint8)

def hist_color_percent(img: np.ndarray, bits: int = HIST_BITS_PER_CH, top_k: int = HIST_TOP_K):
    uniq = cnt = None
    max_bins = 1 << (3 * bits)
    step = 256 // (1 << bits)
    total = float(max(1, img.shape[0] * img.shape[1]))

    if GPU_ARRAY_AVAILABLE:
        try:
            qg = (cp.asarray(img, dtype=cp.uint8) >> (8 - bits)).astype(cp.uint8)
            keyg = (qg[..., 0].astype(cp.uint32) << (2 * bits)) | (qg[..., 1].astype(cp.uint32) << bits) | qg[..., 2].astype(cp.uint32)
            flat_g = keyg.reshape(-1)
            cnt_g = cp.bincount(flat_g, minlength=max_bins)
            idx_g = cp.nonzero(cnt_g)[0]
            uniq = cp.asnumpy(idx_g)
            cnt = cp.asnumpy(cnt_g[idx_g])
        except Exception:
            uniq = cnt = None

    if uniq is None or cnt is None:
        q = quantize_rgb(np.ascontiguousarray(img, dtype=np.uint8), bits)
        key = (q[..., 0].astype(np.uint32) << (2 * bits)) | (q[..., 1].astype(np.uint32) << bits) | q[..., 2].astype(np.uint32)
        flat = key.reshape(-1)
        cnt_arr = np.bincount(flat, minlength=max_bins)
        idx = np.nonzero(cnt_arr)[0]
        cnt = cnt_arr[idx]
        uniq = idx

    order = np.argsort(cnt)[::-1]
    uniq = uniq[order]
    cnt = cnt[order]

    items = []
    for k_, c in zip(uniq[:top_k], cnt[:top_k]):
        r = int(((k_ >> (2 * bits)) & ((1 << bits) - 1)) * step + step // 2)
        g = int(((k_ >> bits) & ((1 << bits) - 1)) * step + step // 2)
        b = int((k_ & ((1 << bits) - 1)) * step + step // 2)
        items.append({"rgb": [r, g, b], "pct": float(round(c / total, 6))})

    if len(cnt):
        dom_pct = float(cnt[0] / total)
        r0 = int(((uniq[0] >> (2 * bits)) & ((1 << bits) - 1)) * step + step // 2)
        g0 = int(((uniq[0] >> bits) & ((1 << bits) - 1)) * step + step // 2)
        b0 = int((uniq[0] & ((1 << bits) - 1)) * step + step // 2)
        dom = [r0, g0, b0]
    else:
        dom_pct = 0.0
        dom = [255, 255, 255]

    return {"top": items, "dominant_pct": dom_pct, "dominant_rgb": dom}


def _analyze_region_backgrounds(
    img_rgb: np.ndarray,
    text_mask: np.ndarray,
    results: List[dict],
    dominant_bg_rgb: np.ndarray,
) -> dict:
    """
    Szacuje kolor tła osobno dla każdego wyniku OCR
    i grupuje regiony w klastry tła (np. główna treść vs. pasek boczny).

    Zwraca słownik z meta-danymi layoutu oraz uzupełnia każdy element
    `results` o pola:
      - bg_mean_rgb: [r,g,b] oszacowanego tła
      - bg_dist_to_global: odległość koloru tła od globalnego tła
      - bg_cluster_id: ID klastra tła
      - bg_is_main_like: czy należy do klastra najbliższego globalnemu tle
    """
    H, W = img_rgb.shape[:2]
    if not results:
        return {"clusters": [], "main_cluster_id": None}
    if BACKGROUND_LAYOUT_REQUIRE_GPU and not GPU_ARRAY_AVAILABLE:
        raise RuntimeError("Background layout requires GPU (REGION_GROW_BG_LAYOUT_REQUIRE_GPU=1), but CuPy is unavailable.")

    dom = np.asarray(dominant_bg_rgb, dtype=np.float32)

    def _color_dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))

    cluster_colors: List[np.ndarray] = []
    per_result: List[dict] = []

    for idx, r in enumerate(results):
        box = r.get("dropdown_box") or r.get("text_box")
        if not box or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
        except Exception:
            continue
        x1 = clamp(x1 - BG_REGION_MARGIN, 0, W - 1)
        x2 = clamp(x2 + BG_REGION_MARGIN, 1, W)
        y1 = clamp(y1 - BG_REGION_MARGIN, 0, H - 1)
        y2 = clamp(y2 + BG_REGION_MARGIN, 1, H)
        if x2 <= x1 or y2 <= y1:
            continue

        region = img_rgb[y1:y2, x1:x2]
        if region.size == 0:
            continue

        # Usuń piksele tekstu z próbkowania tła
        text_roi = text_mask[y1:y2, x1:x2]
        bg_mask = ~text_roi
        if int(bg_mask.sum()) >= BG_REGION_MIN_PIXELS:
            samples = region[bg_mask]
        else:
            samples = region.reshape(-1, 3)

        if samples.size == 0:
            continue

        if GPU_ARRAY_AVAILABLE:
            try:
                mean_rgb = cp.asnumpy(cp.mean(cp.asarray(samples, dtype=cp.float32), axis=0)).astype(np.float32)
            except Exception:
                if BACKGROUND_LAYOUT_REQUIRE_GPU:
                    raise RuntimeError("Background layout GPU mean failed in GPU-only mode.")
                mean_rgb = samples.mean(axis=0).astype(np.float32)
        else:
            mean_rgb = samples.mean(axis=0).astype(np.float32)
        dist_dom = _color_dist(mean_rgb, dom)

        # Proste grupowanie po kolorze tła (maksymalna różnica w RGB)
        cluster_id = None
        for ci, ccol in enumerate(cluster_colors):
            if _color_dist(mean_rgb, ccol) <= float(BG_CLUSTER_TOL_RGB):
                cluster_id = ci
                break
        if cluster_id is None:
            cluster_id = len(cluster_colors)
            cluster_colors.append(mean_rgb)

        info = {
            "index": idx,
            "mean_rgb": [int(round(float(mean_rgb[0]))),
                         int(round(float(mean_rgb[1]))),
                         int(round(float(mean_rgb[2])))],
            "dist_to_global": float(round(dist_dom, 2)),
            "cluster_id": int(cluster_id),
        }
        per_result.append(info)

    clusters_meta: List[dict] = []
    for ci, ccol in enumerate(cluster_colors):
        members = [pr for pr in per_result if pr["cluster_id"] == ci]
        if not members:
            continue
        clusters_meta.append(
            {
                "id": int(ci),
                "mean_rgb": [
                    int(round(float(ccol[0]))),
                    int(round(float(ccol[1]))),
                    int(round(float(ccol[2]))),
                ],
                "count": len(members),
            }
        )

    main_cluster_id = None
    if clusters_meta:
        for cm in clusters_meta:
            ccol = np.asarray(cm["mean_rgb"], dtype=np.float32)
            cm["dist_to_global"] = float(round(_color_dist(ccol, dom), 2))
        clusters_meta.sort(key=lambda c: c.get("dist_to_global", 0.0))
        main_cluster_id = clusters_meta[0]["id"]

    # Uzupełnij poszczególne wyniki o meta-dane tła
    for info in per_result:
        r = results[info["index"]]
        r["bg_mean_rgb"] = info["mean_rgb"]
        r["bg_dist_to_global"] = info["dist_to_global"]
        r["bg_cluster_id"] = info["cluster_id"]
        if main_cluster_id is not None:
            r["bg_is_main_like"] = bool(info["cluster_id"] == main_cluster_id)

    return {"clusters": clusters_meta, "main_cluster_id": main_cluster_id}

# ===================== FLOOD i LASERY =====================
def pick_seed(img_rgb: np.ndarray, text_box_xyxy: Tuple[int, int, int, int], pad: int = SEED_PAD):
    H, W, _ = img_rgb.shape
    x1, y1, x2, y2 = text_box_xyxy
    cy = (y1 + y2)//2
    
    for dx in range(2, pad*3):
        x = x2 + dx
        if x >= W: break
        y0, y1b = clamp(cy-1, 0, H-1), clamp(cy+1, 0, H-1)
        x0, x1b = clamp(x-1, 0, W-1), clamp(x+1, 0, W-1)
        neigh = img_rgb[y0:y1b+1, x0:x1b+1].astype(np.int16)
        m = neigh.mean(axis=(0,1))
        if np.max(np.abs(neigh - m)) <= 6.0:
            return cy, x, img_rgb[cy, x]
    
    cx = (x1 + x2)//2
    return cy, cx, img_rgb[cy, cx]


def annotate_regions_floodfill_and_save_from_mask(image_path: str, img_rgb: np.ndarray, text_mask: np.ndarray) -> None:
    """
    Flood-fill regionów tła (bez boxów) z ignorowaniem maski tekstu.

    Zapisuje obok siebie:
    - `REGION_REGIONS_CURRENT_DIR/regions_current.png`
    - `REGION_REGIONS_CURRENT_DIR/regions_current.json`

    Regiony/boxy < `RG_MIN_REGION_AREA` są odrzucane.
    Brak GPU / błąd GPU => wyjątek (bez CPU fallback).
    """
    if not GPU_FLOOD_AVAILABLE:
        raise RuntimeError("GPU floodfill required for regions overlay, but CuPy/cupyx.scipy.ndimage is unavailable.")

    img_np = np.ascontiguousarray(np.asarray(img_rgb, dtype=np.uint8))
    H, W = img_np.shape[:2]

    tm = np.asarray(text_mask)
    if tm.ndim == 3:
        tm = (tm != 0).any(axis=2)
    tm = (tm != 0).astype(bool)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    from collections import deque

    step = max(2, int(RG_REGIONS_STEP))
    tol = int(max(2, RG_REGIONS_TOL_RGB))

    img_s = img_np[0::step, 0::step]
    tm_s = tm[0::step, 0::step]
    hs, ws = img_s.shape[:2]
    visited = np.zeros((hs, ws), dtype=bool)

    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def _find_seed_ref(sy: int, sx: int):
        if not tm_s[sy, sx]:
            return sy, sx
        max_r = 6
        for r in range(1, max_r + 1):
            for dy, dx in neigh:
                ny, nx = sy + dy * r, sx + dx * r
                if 0 <= ny < hs and 0 <= nx < ws and not tm_s[ny, nx]:
                    return ny, nx
        return None

    coarse: list[dict] = []
    for sy in range(hs):
        for sx in range(ws):
            if visited[sy, sx]:
                continue
            visited[sy, sx] = True
            ref_pos = _find_seed_ref(sy, sx)
            if ref_pos is None:
                continue
            ry, rx = ref_pos
            ref = img_s[ry, rx].astype(np.int16)

            q = deque([(sy, sx)])
            miny = maxy = sy
            minx = maxx = sx

            while q:
                y, x = q.popleft()
                miny = min(miny, y)
                maxy = max(maxy, y)
                minx = min(minx, x)
                maxx = max(maxx, x)
                for dy, dx in neigh:
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < hs and 0 <= nx < ws):
                        continue
                    if visited[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    if tm_s[ny, nx]:
                        q.append((ny, nx))
                        continue
                    col = img_s[ny, nx].astype(np.int16)
                    if int(np.max(np.abs(col - ref))) <= tol:
                        q.append((ny, nx))

            x1 = int(minx * step)
            y1 = int(miny * step)
            x2 = int(min((maxx + 1) * step, W))
            y2 = int(min((maxy + 1) * step, H))
            area = int(max(0, x2 - x1) * max(0, y2 - y1))
            if area < int(RG_MIN_REGION_AREA):
                continue
            coarse.append(
                {"seed_y": int(ry * step), "seed_x": int(rx * step), "bbox": (x1, y1, x2, y2), "area": area}
            )

    if not coarse:
        # Nadal zapisujemy pusty JSON (żeby brain widział brak regionów)
        REGION_REGIONS_CURRENT_DIR.mkdir(parents=True, exist_ok=True)
        empty_path = REGION_REGIONS_CURRENT_DIR / "regions_current.json"
        with empty_path.open("w", encoding="utf-8") as f:
            json.dump({"image": str(image_path), "params": {"step": step, "tol_rgb": tol}, "regions": []}, f, ensure_ascii=False, indent=2)
        return

    coarse.sort(key=lambda r: r.get("area", 0), reverse=True)
    coarse = coarse[: max(1, int(RG_REGIONS_MAX))]

    region_masks: list[np.ndarray] = []
    regions_out: list[dict] = []
    for r in coarse:
        x1, y1, x2, y2 = r["bbox"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        seed_y = clamp(int(r["seed_y"]), 0, H - 1)
        seed_x = clamp(int(r["seed_x"]), 0, W - 1)
        radius = int(min(MAX_RADIUS, max(120, 0.65 * max(w, h) + 80)))

        mask_bbox = flood_region_gpu_masked(img_np, tm, seed_y, seed_x, tol_rgb=tol, radius=radius, neighbor8=True)
        if mask_bbox is None:
            continue
        m_full, bbox_xyxy = mask_bbox
        if m_full is None or bbox_xyxy is None:
            continue

        bx1, by1, bx2, by2 = [int(v) for v in bbox_xyxy]
        refined_area = int(max(0, bx2 - bx1) * max(0, by2 - by1))
        if refined_area < int(RG_MIN_REGION_AREA):
            continue

        region_masks.append(m_full.astype(bool))
        regions_out.append(
            {
                "seed_xy": [int(seed_x), int(seed_y)],
                "coarse_bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "refined_bbox_xyxy": [int(bx1), int(by1), int(bx2), int(by2)],
                "area": refined_area,
                "radius": int(radius),
            }
        )

    if not region_masks:
        REGION_REGIONS_CURRENT_DIR.mkdir(parents=True, exist_ok=True)
        empty_path = REGION_REGIONS_CURRENT_DIR / "regions_current.json"
        with empty_path.open("w", encoding="utf-8") as f:
            json.dump({"image": str(image_path), "params": {"step": step, "tol_rgb": tol}, "regions": []}, f, ensure_ascii=False, indent=2)
        return

    out = img_np.copy()
    alpha = 0.22
    palette = [
        (0, 140, 255),
        (255, 0, 180),
        (0, 200, 0),
        (255, 165, 0),
        (180, 0, 255),
        (255, 0, 0),
        (0, 255, 200),
        (255, 255, 0),
    ]

    used = np.zeros((H, W), dtype=bool)
    for idx, m in enumerate(region_masks):
        m = m.astype(bool) & (~used)
        if not m.any():
            continue
        used |= m
        cr, cg, cb = palette[idx % len(palette)]
        col = np.array([cr, cg, cb], dtype=np.float32)
        out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + col * alpha).astype(np.uint8)

    REGION_REGIONS_DIR.mkdir(parents=True, exist_ok=True)
    REGION_REGIONS_CURRENT_DIR.mkdir(parents=True, exist_ok=True)

    hist_png = REGION_REGIONS_DIR / f"{base_name}_regions.png"
    current_png = REGION_REGIONS_CURRENT_DIR / "regions_current.png"
    Image.fromarray(out).save(hist_png)
    Image.fromarray(out).save(current_png)

    regions_json_path = REGION_REGIONS_CURRENT_DIR / "regions_current.json"
    payload = {
        "image": str(image_path),
        "regions_current_png": str(current_png),
        "params": {"step": int(step), "tol_rgb": int(tol), "max_regions": int(RG_REGIONS_MAX), "min_area": int(RG_MIN_REGION_AREA)},
        "regions": regions_out,
    }
    with regions_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def annotate_regions_and_save(image_path: str, results: List[dict]) -> None:
    """
    Generuje regiony tła floodfillem (bez boxów) z ignorowaniem maski tekstu.

    Zapisuje obok siebie:
    - `REGION_REGIONS_CURRENT_DIR/regions_current.png`
    - `REGION_REGIONS_CURRENT_DIR/regions_current.json`

    Regiony < `RG_MIN_REGION_AREA` są odrzucane.
    Brak GPU / błąd GPU => wyjątek (bez CPU fallback).
    """
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.ascontiguousarray(np.array(img_pil, dtype=np.uint8))
    H, W = img_np.shape[:2]
    text_mask = _build_text_mask_from_results(W, H, results)
    annotate_regions_floodfill_and_save_from_mask(image_path, img_np, text_mask)

def _extract_roi_patch(img_rgb: np.ndarray, seed_y: int, seed_x: int, radius: int):
    H, W = img_rgb.shape[:2]
    ry1, ry2 = max(0, seed_y-radius), min(H, seed_y+radius+1)
    rx1, rx2 = max(0, seed_x-radius), min(W, seed_x+radius+1)
    if ry2 <= ry1 or rx2 <= rx1:
        return None
    roi = img_rgb[ry1:ry2, rx1:rx2]
    return roi, ry1, rx1, seed_y - ry1, seed_x - rx1

def _run_cv_floodfill(roi: np.ndarray, rel_y: int, rel_x: int, tol_rgb: int, neighbor8: bool):
    roi_c = np.ascontiguousarray(roi)
    mask = np.zeros((roi_c.shape[0] + 2, roi_c.shape[1] + 2), np.uint8)
    flags = ((8 if neighbor8 else 4) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)
    try:
        _, _, _, rect = cv2.floodFill(
            roi_c, mask, (int(rel_x), int(rel_y)), newVal=(0, 0, 0),
            loDiff=(tol_rgb,)*3, upDiff=(tol_rgb,)*3, flags=flags
        )
    except Exception:
        return None, None
    if rect[2] < MIN_BOX or rect[3] < MIN_BOX:
        return None, None
    return mask[1:-1, 1:-1].astype(bool), rect

def _finalize_region_mask(region_small: np.ndarray, ry1: int, rx1: int, img_shape: Tuple[int, int, int]):
    """
    Szybkie wyprowadzenie maski regionu i bboxa:
    - pracujemy tylko na lokalnej masce region_small (ROI),
    - unikamy kosztownego domykania morfologicznego na całym obrazie.
    """
    h, w = region_small.shape
    ys, xs = np.where(region_small)
    if ys.size == 0 or xs.size == 0:
        return None
    y1b, y2b = int(ys.min()) + ry1, int(ys.max()) + ry1
    x1b, x2b = int(xs.min()) + rx1, int(xs.max()) + rx1
    bbox_xyxy = (x1b, y1b, x2b + 1, y2b + 1)

    region_big = np.zeros(img_shape[:2], np.uint8)
    region_big[ry1:ry1 + h, rx1:rx1 + w] = region_small.astype(np.uint8)
    return region_big.astype(bool), bbox_xyxy

def _gpu_binary_closing(mask: np.ndarray, kernel: np.ndarray, iterations: int = 1):
    if not GPU_FLOOD_AVAILABLE:
        return None
    try:
        arr = cp.asarray(mask, dtype=cp.uint8)
        ker = cp.asarray(kernel, dtype=cp.uint8)
        res = cupy_ndimage.binary_closing(arr, structure=ker, iterations=iterations)
        return cp.asnumpy(res)
    except Exception:
        return None

def _gpu_binary_dilate(mask: np.ndarray, kernel: np.ndarray, iterations: int = 1):
    if not GPU_FLOOD_AVAILABLE:
        return None
    try:
        arr = cp.asarray(mask, dtype=cp.uint8)
        ker = cp.asarray(kernel, dtype=cp.uint8)
        res = cupy_ndimage.binary_dilation(arr, structure=ker, iterations=iterations)
        return cp.asnumpy(res)
    except Exception:
        return None

def flood_region_cpu(img_rgb: np.ndarray, seed_y: int, seed_x: int,
                     tol_rgb: int = TOL_RGB, radius: int = BASE_RADIUS, neighbor8: bool = True):
    extracted = _extract_roi_patch(img_rgb, seed_y, seed_x, radius)
    if extracted is None:
        return None
    roi, ry1, rx1, rel_y, rel_x = extracted
    mask_local, _ = _run_cv_floodfill(roi, rel_y, rel_x, tol_rgb, neighbor8)
    if mask_local is None:
        return None
    return _finalize_region_mask(mask_local, ry1, rx1, img_rgb.shape)

def flood_region_gpu(img_rgb: np.ndarray, seed_y: int, seed_x: int,
                     tol_rgb: int = TOL_RGB, radius: int = BASE_RADIUS, neighbor8: bool = True):
    if not GPU_FLOOD_AVAILABLE:
        return None
    extracted = _extract_roi_patch(img_rgb, seed_y, seed_x, radius)
    if extracted is None:
        return None
    roi_np, ry1, rx1, rel_y, rel_x = extracted
    try:
        roi = cp.asarray(roi_np.astype(np.int16))
        seed_color = roi[rel_y, rel_x]
        diff = cp.abs(roi - seed_color.reshape(1, 1, 3))
        mask_candidates = cp.max(diff, axis=2) <= tol_rgb
        if neighbor8:
            struct = cp.ones((3, 3), dtype=cp.uint8)
        else:
            struct = cp.array([[0,1,0],[1,1,1],[0,1,0]], dtype=cp.uint8)
        labeled, num = cupy_ndimage.label(mask_candidates, structure=struct)
        if num == 0:
            return None
        seed_label = int(cp.asnumpy(labeled[rel_y, rel_x]))
        if seed_label == 0:
            return None
        region_small = cp.asnumpy(labeled == seed_label)
    except Exception:
        return None
    return _finalize_region_mask(region_small, ry1, rx1, img_rgb.shape)

def _estimate_radius(box_xyxy, base: int = BASE_RADIUS, scale: float = RADIUS_SCALE):
    x1, y1, x2, y2 = box_xyxy
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    extra = int(scale * max(w, h))
    rad = base + extra
    return min(MAX_RADIUS, max(100, rad))

def flood_same_color_bbox_cv(img_rgb: np.ndarray, seed_y: int, seed_x: int,
                             bbox_hint=None,
                             tol_rgb: int = TOL_RGB, radius: Optional[int] = None, neighbor8: bool = True):
    r = radius if radius is not None else BASE_RADIUS
    if bbox_hint is not None:
        r = _estimate_radius(bbox_hint)
    if REQUIRE_GPU and not GPU_FLOOD_AVAILABLE:
        raise RuntimeError("GPU flood fill wymagany (REGION_GROW_REQUIRE_GPU=1), ale CuPy niedostępny")
    if GPU_FLOOD_AVAILABLE:
        gpu_result = flood_region_gpu(img_rgb, seed_y, seed_x, tol_rgb, r, neighbor8)
        if gpu_result is not None:
            return gpu_result
    if REQUIRE_GPU:
        raise RuntimeError("GPU flood fill zwrócił None – brak CPU fallback (REQUIRE_GPU=1)")
    return flood_region_cpu(img_rgb, seed_y, seed_x, tol_rgb, r, neighbor8)


def flood_region_gpu_masked(
    img_rgb: np.ndarray,
    text_mask: np.ndarray,
    seed_y: int,
    seed_x: int,
    tol_rgb: int = TOL_RGB,
    radius: int = BASE_RADIUS,
    neighbor8: bool = True,
):
    """
    GPU-first region flood helper used by regions_current generation.
    Returns (full_mask, bbox_xyxy) or None when no region was produced.
    `text_mask` is accepted for API compatibility with existing call sites.
    """
    _ = text_mask
    mask_bbox = flood_same_color_bbox_cv(
        img_rgb,
        seed_y=seed_y,
        seed_x=seed_x,
        bbox_hint=None,
        tol_rgb=tol_rgb,
        radius=radius,
        neighbor8=neighbor8,
    )
    if mask_bbox is None:
        return None
    mask, bbox_xyxy = mask_bbox
    if mask is None or bbox_xyxy is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    if x2 <= x1 or y2 <= y1:
        return None
    return mask, (x1, y1, x2, y2)

def boundary_colors_from_region_fast(img_rgb: np.ndarray, region_mask: np.ndarray,
                                     text_mask: np.ndarray, tol_rgb: int = TOL_RGB) -> Dict[str, dict]:
    if REQUIRE_GPU and not GPU_ARRAY_AVAILABLE:
        raise RuntimeError("GPU boundary colors wymagany (REGION_GROW_REQUIRE_GPU=1), ale CuPy niedostępny")
    H, W = region_mask.shape
    ys, xs = np.where(region_mask)
    if ys.size == 0: return {}

    y1, y2 = max(0, ys.min()-1), min(H-1, ys.max()+1)
    x1, x2 = max(0, xs.min()-1), min(W-1, xs.max()+1)
    reg = region_mask[y1:y2+1, x1:x2+1]
    txt = text_mask[y1:y2+1, x1:x2+1]
    sub = img_rgb[y1:y2+1, x1:x2+1]

    gpu_res = _boundary_colors_gpu(sub, reg, txt, tol_rgb, y1, x1)
    if gpu_res is not None:
        return gpu_res
    if REQUIRE_GPU:
        raise RuntimeError("GPU boundary colors zwrócił None – brak CPU fallback (REQUIRE_GPU=1)")

    rim = cv2.dilate(reg.astype(np.uint8), _K3, iterations=1).astype(bool)
    rim = rim & (~reg) & (~txt)
    rys, rxs = np.where(rim)
    if rys.size == 0: return {}

    cols = sub[rys, rxs]
    posy = rys + y1
    posx = rxs + x1
    gpu_clusters = _cluster_boundary_colors_gpu(cols, posy, posx, tol_rgb)
    if gpu_clusters is not None:
        return gpu_clusters

    step = int(tol_rgb + 1)
    quant = (cols // step).astype(np.int16)
    key = (quant[:, 0].astype(np.int32) << 16) | (quant[:, 1].astype(np.int32) << 8) | quant[:, 2].astype(np.int32)
    uniq, idx, counts = np.unique(key, return_index=True, return_counts=True)
    order = np.argsort(counts)[::-1]
    uniq = uniq[order]
    idx = idx[order]
    counts = counts[order]

    out = {}
    for k_, i_, c_ in zip(uniq, idx, counts):
        rgb = cols[int(i_)]
        y = int(posy[int(i_)])
        x = int(posx[int(i_)])
        key_str = f"{int(rgb[0])},{int(rgb[1])},{int(rgb[2])}"
        out[key_str] = {
            "rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])],
            "count": int(c_),
            "sample_pos": [y, x],
        }
    return out

def _cluster_boundary_colors_gpu(cols: np.ndarray, posy: np.ndarray, posx: np.ndarray, tol_rgb: int):
    if not GPU_ARRAY_AVAILABLE or cols.size == 0:
        return None
    try:
        cg = cp.asarray(cols.astype(np.uint8))
        step = int(max(1, tol_rgb + 1))
        quant = (cg // step).astype(cp.int16)
        key = (quant[:, 0].astype(cp.int32) << 16) | (quant[:, 1].astype(cp.int32) << 8) | quant[:, 2].astype(cp.int32)
        uniq, idx, counts = cp.unique(key, return_index=True, return_counts=True)
        order = cp.argsort(counts)[::-1]
        uniq = uniq[order]
        idx = idx[order]
        counts = counts[order]
        idx_np = cp.asnumpy(idx)
        counts_np = cp.asnumpy(counts)
        cols_np = cols  # already CPU
        out = {}
        for i, cnt in zip(idx_np, counts_np):
            rgb = cols_np[int(i)]
            y = int(posy[int(i)])
            x = int(posx[int(i)])
            key_str = f"{int(rgb[0])},{int(rgb[1])},{int(rgb[2])}"
            out[key_str] = {
                "rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])],
                "count": int(cnt),
                "sample_pos": [y, x],
            }
        return out
    except Exception:
        return None

def _boundary_colors_gpu(sub: np.ndarray, reg: np.ndarray, txt: np.ndarray,
                         tol_rgb: int, offset_y: int, offset_x: int):
    if not GPU_ARRAY_AVAILABLE:
        return None
    try:
        reg_gpu = cp.asarray(reg.astype(bool))
        txt_gpu = cp.asarray(txt.astype(bool))
        struct = cp.asarray(_K3)
        rim_gpu = cupy_ndimage.binary_dilation(reg_gpu, structure=struct)
        rim_gpu = cp.logical_and(rim_gpu, cp.logical_not(reg_gpu))
        rim_gpu = cp.logical_and(rim_gpu, cp.logical_not(txt_gpu))
        coords = cp.argwhere(rim_gpu)
        if coords.size == 0:
            return {}

        sub_gpu = cp.asarray(sub.astype(np.uint8))
        cols_gpu = sub_gpu[coords[:, 0], coords[:, 1]]
        posy_gpu = coords[:, 0] + offset_y
        posx_gpu = coords[:, 1] + offset_x

        step = int(tol_rgb + 1)
        quant = (cols_gpu // step).astype(cp.int16)
        key = (quant[:, 0].astype(cp.int32) << 16) | (quant[:, 1].astype(cp.int32) << 8) | quant[:, 2].astype(cp.int32)
        uniq, idx, counts = cp.unique(key, return_index=True, return_counts=True)
        order = cp.argsort(counts)[::-1]
        idx = idx[order]
        counts = counts[order]

        cols_sel = cols_gpu[idx]
        posy_sel = posy_gpu[idx]
        posx_sel = posx_gpu[idx]
        counts_np = cp.asnumpy(counts)
        cols_np = cp.asnumpy(cols_sel)
        posy_np = cp.asnumpy(posy_sel)
        posx_np = cp.asnumpy(posx_sel)

        out = {}
        for rgb, cnt, yy, xx in zip(cols_np, counts_np, posy_np, posx_np):
            key_str = f"{int(rgb[0])},{int(rgb[1])},{int(rgb[2])}"
            out[key_str] = {
                "rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])],
                "count": int(cnt),
                "sample_pos": [int(yy), int(xx)],
            }
        return out
    except Exception:
        return None

def region_ref_lab(lab_img: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(region_mask)
    if len(ys) == 0: return np.array([0, 0, 0], dtype=np.uint8)
    return np.median(lab_img[ys, xs], axis=0).astype(np.uint8)

def _deltaE_map(lab_win: np.ndarray, ref_lab: np.ndarray) -> np.ndarray:
    if GPU_ARRAY_AVAILABLE:
        try:
            lw = cp.asarray(lab_win.astype(np.int16))
            ref = cp.asarray(ref_lab.reshape(1, 1, 3).astype(np.int16))
            diff = lw - ref
            return cp.asnumpy(cp.sqrt((diff*diff).sum(axis=2))).astype(np.float32)
        except Exception:
            pass
    d = lab_win.astype(np.int16) - ref_lab.reshape(1,1,3).astype(np.int16)
    return np.sqrt((d*d).sum(axis=2)).astype(np.float32)

def _laser_box_check_cpu(img_rgb: np.ndarray, lab_img: np.ndarray, bbox_text, 
                         region_mask, frame_rgb, text_mask, tol_rgb: int):
    x1, y1, x2, y2 = bbox_text
    cy, cx = (y1+y2)//2, (x1+x2)//2
    ys, xs = np.where(region_mask)
    
    if xs.size == 0: return 0, [(cx, cy)]*4
    
    ry1, ry2 = int(ys.min()), int(ys.max())
    rx1, rx2 = int(xs.min()), int(xs.max())
    pad = min(30, FRAME_SEARCH_MAX_PX)
    wy1, wy2 = max(0, ry1-pad), min(lab_img.shape[0], ry2+pad+1)
    wx1, wx2 = max(0, rx1-pad), min(lab_img.shape[1], rx2+pad+1)
    sub_lab = lab_img[wy1:wy2, wx1:wx2]
    ref_lab = region_ref_lab(lab_img, region_mask)
    dEmap = _deltaE_map(sub_lab, ref_lab)

    def cast_dir(dy, dx):
        y, x = cy - wy1, cx - wx1
        H, W = dEmap.shape
        steps = 0
        consec = 0
        searched = 0
        last = (x, y)
        
        while steps < EDGE_MAX_LEN_PX and 0 <= y < H and 0 <= x < W and text_mask[wy1+y, wx1+x]:
            y += dy
            x += dx
            steps += 1
        
        while steps < EDGE_MAX_LEN_PX and 0 <= y < H and 0 <= x < W and region_mask[wy1+y, wx1+x]:
            y += dy
            x += dx
            steps += 1
        
        while steps < EDGE_MAX_LEN_PX and 0 <= y < H and 0 <= x < W and searched < FRAME_SEARCH_MAX_PX:
            if not text_mask[wy1+y, wx1+x]:
                ok = False
                if frame_rgb is not None and color_close_rgb(img_rgb[wy1+y, wx1+x], frame_rgb, tol_rgb):
                    ok = True
                else:
                    if dEmap[y, x] > EDGE_DELTA_E:
                        consec += 1
                        if consec >= EDGE_CONSEC_N: 
                            ok = True
                    else:
                        consec = 0
                
                if ok: return True, (wx1+x, wy1+y)
            
            last = (x, y)
            y += dy
            x += dx
            steps += 1
            searched += 1
        
        return False, (wx1+last[0], wy1+last[1])

    hits = 0
    pts = []
    for dy, dx in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
        ok, pt = cast_dir(dy, dx)
        hits += int(ok)
        pts.append(pt)
    
    return hits, pts

def _gpu_first_true(arr: "cp.ndarray"):
    if arr.size == 0:
        return None
    idx = cp.where(arr)[0]
    if idx.size == 0:
        return None
    return int(idx[0].item())

def _gpu_count_leading_true(arr: "cp.ndarray"):
    if arr.size == 0:
        return 0
    false_idx = cp.where(~arr)[0]
    if false_idx.size == 0:
        return int(arr.size)
    return int(false_idx[0].item())

def _gpu_first_consecutive(arr: "cp.ndarray", needed: int):
    if arr.size == 0:
        return None
    if needed <= 1:
        return _gpu_first_true(arr)
    conv = cp.convolve(arr.astype(cp.int16), cp.ones((needed,), dtype=cp.int16), mode="valid")
    hit = _gpu_first_true(conv >= needed)
    if hit is None:
        return None
    return hit + needed - 1

def _laser_cast_gpu(dy: int, dx: int, start_y: int, start_x: int, region_gpu, text_gpu,
                    dEmap_gpu, rgb_gpu, frame_rgb_gpu, wy1: int, wx1: int,
                    tol_rgb: int) -> Tuple[bool, Tuple[int, int]]:
    H, W = region_gpu.shape
    steps = cp.arange(0, EDGE_MAX_LEN_PX, dtype=cp.int32)
    ys = start_y + steps * dy
    xs = start_x + steps * dx
    valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    if not bool(cp.any(valid).item()):
        return False, (wx1 + start_x, wy1 + start_y)
    ys = ys[valid]
    xs = xs[valid]
    last_global = (int(xs[-1].item()) + wx1, int(ys[-1].item()) + wy1)

    txt_vals = text_gpu[(ys, xs)]
    lead_txt = _gpu_count_leading_true(txt_vals)
    ys = ys[lead_txt:]
    xs = xs[lead_txt:]
    if ys.size == 0:
        return False, last_global

    reg_vals = region_gpu[(ys, xs)]
    lead_reg = _gpu_count_leading_true(reg_vals)
    ys = ys[lead_reg:]
    xs = xs[lead_reg:]
    if ys.size == 0:
        return False, last_global

    search_len = min(int(ys.size), FRAME_SEARCH_MAX_PX)
    ys = ys[:search_len]
    xs = xs[:search_len]
    if ys.size == 0:
        return False, last_global

    best_idx = None
    if frame_rgb_gpu is not None:
        cols = rgb_gpu[(ys, xs)].astype(cp.int16)
        diff = cp.abs(cols - frame_rgb_gpu.reshape(1, 3))
        color_hit = cp.max(diff, axis=1) <= tol_rgb
        hit_idx = _gpu_first_true(color_hit)
        if hit_idx is not None:
            best_idx = hit_idx

    if best_idx is None:
        delta_vals = dEmap_gpu[(ys, xs)]
        delta_hit = delta_vals > EDGE_DELTA_E
        hit_idx = _gpu_first_consecutive(delta_hit, EDGE_CONSEC_N)
        if hit_idx is not None:
            best_idx = hit_idx

    if best_idx is None:
        return False, last_global

    gx = int(xs[best_idx].item()) + wx1
    gy = int(ys[best_idx].item()) + wy1
    return True, (gx, gy)

def _laser_box_check_gpu(img_rgb: np.ndarray, lab_img: np.ndarray, bbox_text,
                         region_mask, frame_rgb, text_mask, tol_rgb: int):
    if not GPU_ARRAY_AVAILABLE:
        return None
    x1, y1, x2, y2 = bbox_text
    cy, cx = (y1+y2)//2, (x1+x2)//2
    ys, xs = np.where(region_mask)
    if xs.size == 0:
        return 0, [(cx, cy)]*4
    ry1, ry2 = int(ys.min()), int(ys.max())
    rx1, rx2 = int(xs.min()), int(xs.max())
    pad = min(30, FRAME_SEARCH_MAX_PX)
    wy1, wy2 = max(0, ry1-pad), min(lab_img.shape[0], ry2+pad+1)
    wx1, wx2 = max(0, rx1-pad), min(lab_img.shape[1], rx2+pad+1)
    sub_lab = lab_img[wy1:wy2, wx1:wx2]
    sub_rgb = img_rgb[wy1:wy2, wx1:wx2]
    region_roi = region_mask[wy1:wy2, wx1:wx2]
    text_roi = text_mask[wy1:wy2, wx1:wx2]
    ref_lab = region_ref_lab(lab_img, region_mask)

    try:
        region_gpu = cp.asarray(region_roi.astype(bool))
        text_gpu = cp.asarray(text_roi.astype(bool))
        lab_gpu = cp.asarray(sub_lab.astype(np.int16))
        ref_gpu = cp.asarray(ref_lab.reshape(1, 1, 3).astype(np.int16))
        dEmap_gpu = cp.sqrt(((lab_gpu - ref_gpu) ** 2).sum(axis=2)).astype(cp.float32)
        rgb_gpu = cp.asarray(sub_rgb.astype(np.int16))
        frame_rgb_gpu = None
        if frame_rgb is not None:
            frame_rgb_gpu = cp.asarray(np.array(frame_rgb, dtype=np.int16))
    except Exception:
        return None

    hits = 0
    pts = []
    for dy, dx in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
        ok, pt = _laser_cast_gpu(
            dy, dx, cy - wy1, cx - wx1,
            region_gpu, text_gpu, dEmap_gpu, rgb_gpu, frame_rgb_gpu,
            wy1, wx1, tol_rgb
        )
        hits += int(ok)
        pts.append(pt)
    return hits, pts

def laser_box_check_fast(img_rgb: np.ndarray, lab_img: np.ndarray, bbox_text, 
                        region_mask, frame_rgb, text_mask, tol_rgb: int):
    if GPU_FLOOD_AVAILABLE:
        gpu_result = _laser_box_check_gpu(img_rgb, lab_img, bbox_text, region_mask, frame_rgb, text_mask, tol_rgb)
        if gpu_result is not None:
            return gpu_result
    if LASER_REQUIRE_GPU:
        raise RuntimeError("Laser stage requires GPU (REGION_GROW_LASER_REQUIRE_GPU=1), but GPU path failed.")
    return _laser_box_check_cpu(img_rgb, lab_img, bbox_text, region_mask, frame_rgb, text_mask, tol_rgb)


def _build_fast_summary_from_results(results: List[dict], image_path: str, background_layout: Optional[dict], timer: Optional[StageTimer]) -> dict:
    def _norm(s: Any) -> str:
        return " ".join(str(s or "").strip().lower().split())

    def _is_next(s: str) -> bool:
        t = _norm(s)
        if not t:
            return False
        keys = ("dalej", "nast", "next", "continue", "kontynuuj", "wyślij", "wyslij", "submit", "finish", "done")
        return any(k in t for k in keys)

    def _is_dropdown(s: str) -> bool:
        t = _norm(s)
        if not t:
            return False
        keys = ("wybierz", "select", "choose", "option", "lista", "rozwij")
        return any(k in t for k in keys)

    def _score(row: dict, bonus: float = 0.0) -> float:
        try:
            conf = float(row.get("conf") or 0.0)
        except Exception:
            conf = 0.0
        return max(0.0, min(1.0, conf + bonus))

    question_like: List[dict] = []
    answers: List[dict] = []
    nexts: List[dict] = []
    dropdowns: List[dict] = []

    for idx, row in enumerate(results or []):
        if not isinstance(row, dict):
            continue
        box = row.get("dropdown_box") or row.get("text_box")
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        txt = str(row.get("text") or row.get("box_text") or "").strip()
        try:
            bx = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        except Exception:
            continue
        item = {
            "id": str(row.get("id") or f"rg_{idx}"),
            "text": txt,
            "bbox": bx,
            "score": round(_score(row), 4),
        }
        if "?" in txt:
            question_like.append(dict(item))
        if _is_next(txt):
            nexts.append(dict(item, score=round(_score(row, 0.25), 4)))
            continue
        if bool(row.get("has_frame")) or _is_dropdown(txt):
            bonus = 0.2 if row.get("has_frame") else 0.1
            dropdowns.append(dict(item, score=round(_score(row, bonus), 4)))
            continue
        if txt:
            answers.append(dict(item))

    question_like.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    answers.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    nexts.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    dropdowns.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    top_labels: Dict[str, dict] = {}
    if dropdowns:
        top_labels["dropdown"] = dict(dropdowns[0], label="dropdown")
    if nexts:
        top_labels["next_active"] = dict(nexts[0], label="next_active")
    if answers:
        top_labels["answer_single"] = dict(answers[0], label="answer_single")
        top_labels["answer_multi"] = dict(answers[0], label="answer_multi")

    timings = timer.records if isinstance(timer, StageTimer) else []
    return {
        "image": str(image_path),
        "total_elements": int(len(results or [])),
        "background_layout": background_layout or {},
        "top_labels": top_labels,
        "question_like_boxes": question_like[:5],
        "answer_candidate_boxes": answers[:8],
        "next_candidate_boxes": nexts[:5],
        "dropdown_candidate_boxes": dropdowns[:5],
        "confidence": {
            "answer": float(answers[0]["score"]) if answers else 0.0,
            "next": float(nexts[0]["score"]) if nexts else 0.0,
            "dropdown": float(dropdowns[0]["score"]) if dropdowns else 0.0,
        },
        "reasons": {"mode": "turbo_fast_summary", "source": "region_grow.results"},
        "timings": timings,
    }

# ===================== PIPELINE =================================================
def run_dropdown_detection(image_path: str) -> dict:
    t_pipeline_start = time.perf_counter()
    print(f"[DEBUG] Starting detection: {image_path}")
    if ADVANCED_DEBUG:
        print(
            "[DEBUG] region_grow cfg "
            f"fast={FAST_MODE} turbo={REGION_GROW_TURBO} "
            f"timings={ENABLE_TIMINGS} debug_ocr={DEBUG_OCR} "
            f"max_det={MAX_DETECTIONS_FOR_LASER} max_ocr_items={MAX_OCR_ITEMS}"
        )
        print(
            "[DEBUG] rapid cfg "
            f"use_rapid={int(os.environ.get('REGION_GROW_USE_RAPID_OCR', '0') or 0)} "
            f"max_side={RAPID_OCR_MAX_SIDE} autocrop={int(RAPID_OCR_AUTOCROP)} "
            f"delta={RAPID_AUTOCROP_DELTA} pad={RAPID_AUTOCROP_PADDING}"
        )
    timer = StageTimer(ENABLE_TIMINGS)

    def _step_add(label: str, started_at: float) -> None:
        if ADVANCED_DEBUG:
            timer.add(f"Step/{label}", time.perf_counter() - started_at)

    # Info o GPU/CPU na starcie
    try:
        import paddle  # type: ignore
        paddle_cuda = bool(paddle.is_compiled_with_cuda())
    except Exception:
        paddle_cuda = False
    print(
        f"[DEBUG] GPU flood available={GPU_FLOOD_AVAILABLE} "
        f"use_gpu_flood={USE_GPU_FLOOD} "
        f"paddle_cuda={paddle_cuda}"
    )

    print("[DEBUG] Loading image...")
    t_step = time.perf_counter()
    img_pil = Image.open(image_path).convert("RGB")
    _step_add("load_image_pil", t_step)
    t_step = time.perf_counter()
    img = np.ascontiguousarray(np.array(img_pil))
    _step_add("pil_to_numpy", t_step)
    H, W = img.shape[:2]
    print(f"[DEBUG] Image size: {W}x{H}")
    
    t_step = time.perf_counter()
    lab = np.ascontiguousarray(rgb_to_lab(img))
    _step_add("rgb_to_lab", t_step)
    timer.mark("Load + LAB")

    print("[DEBUG] Computing histogram...")
    t_step = time.perf_counter()
    hist = hist_color_percent(img, bits=HIST_BITS_PER_CH, top_k=HIST_TOP_K)
    _step_add("histogram", t_step)
    is_plain_bg_global = bool(hist["dominant_pct"] >= float(GLOBAL_BG_OVER_PCT))
    dominant_bg_rgb = np.array(hist["dominant_rgb"], dtype=np.uint8)
    timer.mark("Histogram")

    print("[DEBUG] Running OCR...")
    try:
        t_step = time.perf_counter()
        ocr_raw = read_ocr_wrapper(img_pil, timer=timer)
        _step_add("ocr_wrapper", t_step)
        t_step = time.perf_counter()
        dx_auto, dy_auto = _estimate_ocr_global_shift(img, ocr_raw)
        dx_total = int(dx_auto) + int(OCR_SHIFT_X)
        dy_total = int(dy_auto) + int(OCR_SHIFT_Y)
        if dx_total or dy_total:
            ocr_raw = _apply_ocr_global_shift(ocr_raw, dx_total, dy_total, W, H)
            print(
                f"[DEBUG] OCR shift applied: dx={dx_total}, dy={dy_total} "
                f"(auto={dx_auto},{dy_auto} manual={OCR_SHIFT_X},{OCR_SHIFT_Y})"
            )
        _step_add("ocr_autoshift", t_step)
        print(f"[DEBUG] OCR found {len(ocr_raw)} items")
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        import traceback
        traceback.print_exc()
        ocr_raw = []
    
    timer.mark("OCR wrapper")
    
    print("[DEBUG] Building text mask...")
    t_step = time.perf_counter()
    text_mask = build_text_mask_cv(ocr_raw, W, H)
    _step_add("build_text_mask", t_step)
    timer.mark("Text mask")

    detection_stats: Dict[str, float] = defaultdict(float)
    detection_counts: Dict[str, int] = defaultdict(int)
    stat_lock = threading.Lock()

    def _stat_add(label: str, duration: float):
        if not ENABLE_TIMINGS or duration <= 0:
            return
        with stat_lock:
            detection_stats[label] += duration
            detection_counts[label] += 1

    results: List[dict] = []

    print(f"[DEBUG] Processing {len(ocr_raw)} detections...")
    if len(ocr_raw) > max(1, MAX_DETECTIONS_FOR_LASER):
        t_step = time.perf_counter()
        try:
            ocr_raw = sorted(ocr_raw, key=lambda r: float(r[2] or 0.0), reverse=True)[: max(1, MAX_DETECTIONS_FOR_LASER)]
            print(f"[DEBUG] OCR detections capped to {len(ocr_raw)} (MAX_DETECTIONS_FOR_LASER)")
        except Exception:
            ocr_raw = ocr_raw[: max(1, MAX_DETECTIONS_FOR_LASER)]
        _step_add("cap_detections_for_laser", t_step)

    def _process_detection(idx: int, quad, txt, conf):
        xs = [int(p[0]) for p in quad]
        ys = [int(p[1]) for p in quad]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        result = {
            "text": txt,
            "conf": float(conf),
            "text_box": [x1, y1, x2, y2],
            "has_frame": False,
            "frame_hits": 0,
            "dropdown_box": None,
            "frame_rgb": None,
            "laser_endpoints": None,
        }

        try:
            # 1D lasery od środka boxa – BEZ floodfill
            t_laser = time.perf_counter()
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ends = [
                _ray_stop_on_edge(lab, text_mask, cx, cy, +1, 0),  # prawo
                _ray_stop_on_edge(lab, text_mask, cx, cy, -1, 0),  # lewo
                _ray_stop_on_edge(lab, text_mask, cx, cy, 0, -1),  # góra
                _ray_stop_on_edge(lab, text_mask, cx, cy, 0, +1),  # dół
            ]
            _stat_add("laser", time.perf_counter() - t_laser)

            # sprawdź kolor w 4 końcach
            colors = []
            for ex, ey in ends:
                ex = clamp(ex, 0, W - 1)
                ey = clamp(ey, 0, H - 1)
                colors.append(img[ey, ex].astype(np.int16))

            frame_rgb = None
            hits = 0
            if colors:
                arr = np.stack(colors, axis=0)
                ref = np.median(arr, axis=0).astype(np.int16)
                for c in arr:
                    if np.max(np.abs(c - ref)) <= TOL_RGB:
                        hits += 1
                if hits == 4:
                    frame_rgb = ref.astype(np.uint8)

            result["laser_endpoints"] = [[int(px), int(py)] for (px, py) in ends]

            # jeśli wszystkie 4 lasery trafiły w zbliżony kolor – robimy prostokąt
            if frame_rgb is not None:
                xs_all = [x1, x2] + [int(px) for (px, _) in ends]
                ys_all = [y1, y2] + [int(py) for (_, py) in ends]
                bx1, bx2 = int(min(xs_all)), int(max(xs_all))
                by1, by2 = int(min(ys_all)), int(max(ys_all))
                result["dropdown_box"] = [bx1, by1, bx2, by2]
                result["frame_rgb"] = [int(frame_rgb[0]), int(frame_rgb[1]), int(frame_rgb[2])]
                result["frame_hits"] = hits
                result["has_frame"] = True

        except Exception as e:
            print(f"[ERROR] Processing item {idx+1} failed: {e}")

        return idx, result


    max_workers = min(max(1, os.cpu_count() or 2), 8)
    t_step = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Regiony tła równolegle do per-box processingu (GPU-only; błąd ma przerwać pipeline).
        regions_future = ex.submit(annotate_regions_floodfill_and_save_from_mask, image_path, img, text_mask)

        t_det_submit = time.perf_counter()
        futures = [ex.submit(_process_detection, idx, quad, txt, conf) for idx, (quad, txt, conf) in enumerate(ocr_raw)]
        _step_add("submit_detection_jobs", t_det_submit)
        t_det_wait = time.perf_counter()
        for fut in as_completed(futures):
            i, res = fut.result()
            results.append((i, res))
        _step_add("wait_detection_jobs", t_det_wait)

        # Poczekaj na regiony – wyjątek ma przerwać pipeline zgodnie z wymaganiami (brak CPU fallback).
        t_regions_wait = time.perf_counter()
        regions_future.result()
        _step_add("wait_regions_future", t_regions_wait)
    _step_add("threadpool_total", t_step)

    t_step = time.perf_counter()
    results = [r for _, r in sorted(results, key=lambda x: x[0])]
    _step_add("sort_results", t_step)

    # Odrzuć małe boxy z głównego region_grow (text_box / dropdown_box) – ważne dla spójności pipeline.
    try:
        min_box_area = int(RG_MIN_BOX_AREA)
    except Exception:
        min_box_area = 50000

    if min_box_area > 0 and results:
        t_step = time.perf_counter()
        kept: List[dict] = []
        dropped = 0
        for r in results:
            box = r.get("dropdown_box") or r.get("text_box")
            if not box or len(box) != 4:
                kept.append(r)
                continue
            try:
                x1, y1, x2, y2 = [int(v) for v in box]
            except Exception:
                kept.append(r)
                continue
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area < min_box_area:
                dropped += 1
                continue
            kept.append(r)
        if dropped:
            print(f"[DEBUG] Dropped {dropped} boxes below min area {min_box_area}px^2")
        results = kept
        _step_add("filter_small_boxes", t_step)

    print(f"[DEBUG] Collected {len(results)} results")
    timer.mark("Process detections")
    if ENABLE_TIMINGS and detection_stats:
        print(f"{TIMING_PREFIX}Detections summary:")
        for label, total in detection_stats.items():
            calls = detection_counts.get(label, len(ocr_raw) or 1)
            avg = (total * 1000.0) / max(1, calls)
            print(f"{TIMING_PREFIX}Detections/{label}: {total*1000.0:.1f} ms total ({avg:.2f} ms avg over {calls} calls)")

    # Uzupełnij tekst opisujący cały box (np. całą odpowiedź) – jeżeli
    # OCR pierwotnie zwrócił pusty string, spróbujemy ponownie na flood-bboxie.
    t_extra_block_start = time.perf_counter()
    if ENABLE_EXTRA_OCR and results:
        t_step = time.perf_counter()
        # wybierz boxy, dla kt�rych faktycznie op�aca si� robi� extra OCR
        bboxes_needing_extra: list[tuple[int, list[int]]] = []
        for idx, result in enumerate(results):
            current_text = (result.get("text") or "").strip()
            bbox_for_text = result.get("dropdown_box") or result.get("text_box")
            if not bbox_for_text:
                continue
            # je�li ju� mamy do�� d�ugi tekst z sensown� konfidencj�, pomi� extra OCR
            if current_text and len(current_text) >= 6:
                continue
            bboxes_needing_extra.append((idx, bbox_for_text))
        _step_add("extra_ocr_prepare_boxes", t_step)

        if bboxes_needing_extra:
            box_indices = [i for (i, _) in bboxes_needing_extra]
            box_list = [b for (_, b) in bboxes_needing_extra]

            t_extra_call = time.perf_counter()
            extra_texts = _ocr_text_for_bbox_batch(
                img, box_list, pad=6, min_conf=0.15, downscale_factor=0.5
            )
            dt_extra = time.perf_counter() - t_extra_call
            _step_add("extra_ocr_batch_call", t_extra_call)

            if extra_texts:
                per_box = dt_extra / max(1, len(extra_texts))
                _stat_add("extra_text", per_box)

            for idx, txt_extra in zip(box_indices, extra_texts):
                if not txt_extra:
                    continue
                result = results[idx]
                current_text = (result.get("text") or "").strip()
                result["box_text"] = txt_extra
                if not current_text:
                    result["text"] = txt_extra
    timer.add("Extra OCR per box (batch)", time.perf_counter() - t_extra_block_start)

    # Analiza tła per region (layout: np. główna treść vs. pasek boczny)
    if ENABLE_BACKGROUND_LAYOUT:
        try:
            t_step = time.perf_counter()
            bg_layout = _analyze_region_backgrounds(img, text_mask, results, dominant_bg_rgb)
            _step_add("background_layout_analysis", t_step)
        except Exception as exc:
            print(f"[WARN] Background layout analysis failed: {exc}")
            bg_layout = {"clusters": [], "main_cluster_id": None}
    else:
        bg_layout = {"clusters": [], "main_cluster_id": None}
    timer.mark("Background layout")

    triangles: List[dict] = []
    timer.total("Pipeline TOTAL")
    if ADVANCED_DEBUG:
        print(f"{TIMING_PREFIX}region_grow TOTAL wall: {(time.perf_counter() - t_pipeline_start)*1000.0:.1f} ms")

    # Save timings next to the region_grow JSON (data/screen/region_grow/region_grow/<stem>_timings.json)
    try:
        t_step = time.perf_counter()
        img_path = Path(image_path)
        timings_path = img_path.parent / f"{img_path.stem}_timings.json"
        timer.dump_json(timings_path)
        _step_add("dump_timings_json", t_step)
    except Exception:
        pass

    return {
        "image": image_path,
        "color_histogram": hist,
        "dominant_bg_over_50": bool(is_plain_bg_global),
        "background_layout": bg_layout,
        "results": results,
        "triangles": triangles,
        "fast_summary": _build_fast_summary_from_results(results, image_path, bg_layout, timer),
        "ocr_debug": _last_ocr_debug if DEBUG_OCR else None,
    }

# ===================== ANOTACJA =================================================
def _build_text_mask_from_results(W: int, H: int, results: Optional[List[dict]]) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    if not results: return m
    
    for r in results:
        x1, y1, x2, y2 = r["text_box"]
        x1 = clamp(x1, 0, W-1)
        x2 = clamp(x2, 1, W)
        y1 = clamp(y1, 0, H-1)
        y2 = clamp(y2, 1, H)
        m[y1:y2, x1:x2] = True
    
    if TEXT_MASK_DILATE > 0:
        k = _get_rect_kernel((1 + 2 * TEXT_MASK_DILATE, 1 + 2 * TEXT_MASK_DILATE))
        m = cv2.dilate(m.astype(np.uint8), k, iterations=1).astype(bool)
    
    return m

def _ray_stop_on_edge(lab_img: np.ndarray, text_mask: np.ndarray, cx: int, cy: int, dx: int, dy: int,
                     max_len: int = EDGE_MAX_LEN_PX, thr_delta_e: float = EDGE_DELTA_E, consec_req: int = EDGE_CONSEC_N):
    """
    Szybka (wektoryzowana) wersja lasera 1D:
    - poruszamy się po linii od (cx, cy) w kierunku (dx, dy),
    - najpierw wychodzimy z obszaru tekstu (text_mask==True),
    - potem szukamy pierwszego piksela, gdzie deltaE względem koloru startowego
      przekracza próg (thr_delta_e).

    Uwaga: consec_req > 1 jest ignorowane (i tak zawsze mamy 1), ale API zostaje.
    """
    H, W = lab_img.shape[:2]
    if dx == 0 and dy == 0:
        return (clamp(cx, 0, W - 1), clamp(cy, 0, H - 1))

    # Wektory kroków wzdłuż promienia
    steps = np.arange(max_len, dtype=np.int32)
    xs = cx + dx * steps
    ys = cy + dy * steps

    # Ogranicz do wnętrza obrazu
    inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if not np.any(inside):
        return (clamp(cx, 0, W - 1), clamp(cy, 0, H - 1))

    xs = xs[inside]
    ys = ys[inside]

    # Faza 1: wyjście z obszaru tekstu
    text_vals = text_mask[ys, xs]
    non_text_idx = np.where(~text_vals)[0]
    if non_text_idx.size == 0:
        # Cały promień w tekście – zwróć ostatni ważny punkt
        return (clamp(int(xs[-1]), 0, W - 1), clamp(int(ys[-1]), 0, H - 1))

    start_idx = int(non_text_idx[0])
    base_x = int(xs[start_idx])
    base_y = int(ys[start_idx])
    base = lab_img[base_y, base_x].astype(np.int16, copy=False)

    # Faza 2: szukamy przejścia koloru powyżej progu dE
    xs_tail = xs[start_idx:]
    ys_tail = ys[start_idx:]
    if xs_tail.size == 0:
        return (clamp(base_x, 0, W - 1), clamp(base_y, 0, H - 1))

    # Ekstrahuj próbki LAB na ogonie promienia
    samples = lab_img[ys_tail, xs_tail].astype(np.int16, copy=False)
    diff = samples - base.reshape(1, 3)
    # Używamy kwadratu odległości zamiast sqrt dla szybkości
    dist2 = np.sum(diff * diff, axis=1)
    thr2 = float(thr_delta_e * thr_delta_e)

    hit_idx_rel_candidates = np.where(dist2 > thr2)[0]
    if hit_idx_rel_candidates.size == 0:
        # Nie znaleziono ostrej krawędzi – bierzemy ostatni punkt na promieniu
        end_x = int(xs_tail[-1])
        end_y = int(ys_tail[-1])
    else:
        hit_idx_rel = int(hit_idx_rel_candidates[0])
        end_x = int(xs_tail[hit_idx_rel])
        end_y = int(ys_tail[hit_idx_rel])

    return (clamp(end_x, 0, W - 1), clamp(end_y, 0, H - 1))

def annotate_and_save(
    image_path: str,
    results: List[dict],
    triangles: Optional[List[dict]] = None,
    output_dir: Optional[str] = None,
) -> str:
    print(f"[DEBUG ANNOT] Starting annotation for {len(results)} results")
    im = Image.open(image_path).convert("RGBA")
    W, H = im.size
    im_rgb = np.array(im.convert("RGB"))
    im_lab = rgb_to_lab(im_rgb)
    text_mask = _build_text_mask_from_results(W, H, results)

    base = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    # Regiony z obramowaniem
    # Najpierw zbierz wszystkie ramki, potem policz zagnieżdżenie,
    # żeby nadać różne odcienie czerwieni (do 5 poziomów).
    frame_boxes: List[Tuple[int, int, int, int]] = []
    for r in results:
        if r.get("has_frame") and r.get("dropdown_box"):
            bx1, by1, bx2, by2 = r["dropdown_box"]
            bx1 = clamp(bx1, 0, W - 1)
            by1 = clamp(by1, 0, H - 1)
            bx2 = clamp(bx2, 1, W)
            by2 = clamp(by2, 1, H)
            frame_boxes.append((bx1, by1, bx2, by2))

    def _contains(outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int], margin: int = 1) -> bool:
        ox1, oy1, ox2, oy2 = outer
        ix1, iy1, ix2, iy2 = inner
        # outer musi być realnie większy od inner (żeby nie liczyć tego samego poziomu)
        if (ox2 - ox1) <= (ix2 - ix1) or (oy2 - oy1) <= (iy2 - iy1):
            return False
        return (ox1 <= ix1 + margin and oy1 <= iy1 + margin and
                ox2 >= ix2 - margin and oy2 >= iy2 - margin)

    max_levels = 5
    depths: List[int] = []
    for i, inner in enumerate(frame_boxes):
        depth = 0
        for j, outer in enumerate(frame_boxes):
            if i == j:
                continue
            if _contains(outer, inner):
                depth += 1
        # 0 = brak zagnieżdżenia, 1 = w jednym boxie, ...
        # ucinamy na max_levels-1 (np. 4 dla 5 poziomów).
        depths.append(min(depth, max_levels - 1))

    # Przygotuj 5 odcieni czerwieni: im głębiej, tym jaśniej.
    # Kolor powstaje jako miks bazowego REGION_EDGE_RGBA z bielą.
    base_r, base_g, base_b, _ = REGION_EDGE_RGBA

    def _shade_for_level(level: int) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        # poziomy 0..4 -> rosnąca jasność
        factors = [0.0, 0.25, 0.5, 0.7, 0.85]
        f = factors[max(0, min(level, len(factors) - 1))]
        r = int(base_r * (1.0 - f) + 255 * f)
        g = int(base_g * (1.0 - f) + 255 * f)
        b = int(base_b * (1.0 - f) + 255 * f)
        # zachowujemy różne alpha dla wypełnienia i krawędzi
        fill = (r, g, b, REGION_FILL_RGBA[3])
        edge = (r, g, b, REGION_EDGE_RGBA[3])
        return fill, edge

    frame_count = 0
    for (bx1, by1, bx2, by2), depth in zip(frame_boxes, depths):
        fill_rgba, edge_rgba = _shade_for_level(depth)
        od.rectangle((bx1, by1, bx2, by2), fill=fill_rgba, outline=edge_rgba, width=3)
        frame_count += 1
    print(f"[DEBUG ANNOT] Drew {frame_count} frame regions")

    # Wszystkie OCR boxy
    box_count = 0
    for r in results:
        x1, y1, x2, y2 = r["text_box"]
        draw.rectangle((x1, y1, x2, y2), outline=OCR_BOX_COLOR, width=2)
        label = (r.get("text") or "")[:40]
        if label: 
            draw.text((x1+2, max(0, y1-14)), label, fill=OCR_BOX_COLOR)
        box_count += 1
    print(f"[DEBUG ANNOT] Drew {box_count} OCR boxes")

    # Lasery
    laser_count = 0
    for r in results:
        if r.get("laser_endpoints") is None:
            continue
            
        x1, y1, x2, y2 = r["text_box"]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        ends = [
            _ray_stop_on_edge(im_lab, text_mask, cx, cy, +1, 0),
            _ray_stop_on_edge(im_lab, text_mask, cx, cy, -1, 0),
            _ray_stop_on_edge(im_lab, text_mask, cx, cy, 0, -1),
            _ray_stop_on_edge(im_lab, text_mask, cx, cy, 0, +1),
        ]
        color = LASER_ACCEPTED_RGBA if r.get("has_frame") else LASER_REJECTED_RGBA
        width = LASER_WIDTH_ACCEPT if r.get("has_frame") else LASER_WIDTH_REJECT
        for ex, ey in ends:
            od.line([(cx, cy), (ex, ey)], fill=color, width=width)
        laser_count += 1
    print(f"[DEBUG ANNOT] Drew {laser_count} laser sets")

    # Trójkąty (wyłączone)
    print("[DEBUG ANNOT] Triangles skipped (feature removed)")

    print("[DEBUG ANNOT] Compositing layers...")
    merged = Image.alpha_composite(im, base)
    out = Image.alpha_composite(merged, overlay).convert("RGB")
    
    target_dir = output_dir or str(REGION_GROW_ANNOT_DIR)
    os.makedirs(target_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(target_dir, f"{base_name}_annot.png")
    out.save(out_path)
    # Dodatkowy zapis do pliku potrzebnego w pipeline UI.
    # Gdy output_dir jest podane przez orchestrator, trzymaj *_current obok tego katalogu,
    # aby uniknąć zapisu do błędnego roota po relokacjach projektu.
    try:
        if output_dir:
            target_path = Path(output_dir)
            current_dir = target_path.parent / "region_grow_annot_current"
            history_dir = target_path
        else:
            current_dir = REGION_GROW_ANNOT_CURRENT_DIR
            history_dir = REGION_GROW_ANNOT_DIR
        current_dir.mkdir(parents=True, exist_ok=True)
        current_path = current_dir / "region_grow_annot_current.png"
        out.save(current_path)
        # Archiwalna kopia (zawsze nowy plik) w region_grow_annot
        hist_ts = int(time.time() * 1000)
        hist_name = f"{base_name}_annot_{hist_ts}.png"
        hist_path = history_dir / hist_name
        try:
            history_dir.mkdir(parents=True, exist_ok=True)
            out.save(hist_path)
            print(f"[DEBUG ANNOT] Saved history copy: {hist_path}")
        except Exception as e_hist:
            print(f"[WARN] Failed to save history annotation: {e_hist}")
        print(f"[DEBUG ANNOT] Saved current: {current_path}")
    except Exception as e:
        print(f"[WARN] Failed to save current annotation: {e}")
    # Dodatkowa wizualizacja regionów tła (regions_current.*) – GPU-only, błąd ma przerwać.
    annotate_regions_and_save(image_path, results)
    print(f"[DEBUG ANNOT] Saved: {out_path}")
    return out_path

# ===================== CLI ======================================================
# ===================== CLI ======================================================
def main():
    print("="*60)
    print("[DEBUG] SCRIPT START")
    print("="*60)
    
    import sys
    
    try:
        path = sys.argv[1] if len(sys.argv) > 1 else _resolve_default_image()
        print(f"[DEBUG] Image path: {path}")
        
        if not os.path.isfile(path):
            print(f"[ERROR] File not found: {path}")
            return
        
        out = run_dropdown_detection(path)

        # ========== ZAPIS JSON DO PLIKU (NOWA ŚCIEŻKA) ==========
        # Najpierw zapisujemy JSON, żeby pipeline mógł go znaleźć nawet jeśli
        # anotacje/debug overlay polegną później.
        os.makedirs(JSON_OUT_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(path))[0]
        json_path = os.path.join(JSON_OUT_DIR, f"{base_name}.json")
        print(f"[DEBUG] Saving JSON to: {json_path}")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(to_py(out), f, ensure_ascii=False, indent=2)
        print("[DEBUG] JSON saved successfully!")

        out_path = annotate_and_save(
            path,
            out.get("results", []),
            out.get("triangles"),
            output_dir=str(REGION_GROW_ANNOT_DIR),
        )

        # ========== CZYŚCIMY CROPY Z POPRZEDNIEJ RUNDY ==========
        try:
            t_clean = time.perf_counter()
            if os.path.isdir(CNN_CROPS_DIR):
                shutil.rmtree(CNN_CROPS_DIR, ignore_errors=True)
            os.makedirs(CNN_CROPS_DIR, exist_ok=True)
            print(f"[DEBUG] Cleaned: {CNN_CROPS_DIR} ({(time.perf_counter()-t_clean)*1000:.1f} ms)")
        except Exception as e:
            print(f"[WARN] Could not clean {CNN_CROPS_DIR}: {e}")

        # ========== AUTO-START RUNNERA CNN ==========
        try:
            if os.path.isfile(CNN_RUNNER):
                print("[DEBUG] Launching CNN runner...")
                cmd = [
                    sys.executable, CNN_RUNNER,
                    "--json-dir", JSON_OUT_DIR,        # skąd czytać boxy (screen_boxes)
                    "--out-dir",  CNN_CROPS_DIR,       # dokąd zapisać cropy +10%
                    "--model",    fr"{ROOT}\tri_cnn.pt",
                    "--img-size", "128",
                    "--padding",  "0.10",              # +10% wokół textboxa
                    "--batch",    "256",
                    "--thresh",   "0.50"
                ]
                t0 = time.perf_counter()
                subprocess.run(cmd, check=False)
                print(f"[DEBUG] CNN runner done in {(time.perf_counter()-t0)*1000:.1f} ms")
            else:
                print(f"[WARN] CNN runner not found: {CNN_RUNNER}")
        except Exception as e:
            print(f"[WARN] CNN runner failed: {e}")
        
        print("\n" + "="*60)
        print("JSON OUTPUT (console):")
        print("="*60)
        print(json.dumps(to_py(out), ensure_ascii=False, indent=2))
        print("\n" + "="*60)
        print(f"Annotation: {out_path}")
        print(f"JSON file:  {json_path}")
        print("="*60)
    
    except Exception as e:
        print(f"\n[ERROR] Script failed: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
    
    print("\n[DEBUG] SCRIPT END")


# ===================== MONITORING ==============================================
class ResourceMonitor:
    def __init__(self, pid: int, interval: float, output_path: Path):
        self.pid = int(pid)
        self.interval = max(0.2, float(interval))
        self.output_path = Path(output_path)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional["psutil.Process"] = None
        self._samples: List[Dict[str, float]] = []
        self._started_at = 0.0
        self._gpu_handles = []
        self._gpu_ready = False

    def start(self) -> bool:
        if psutil is None:
            print("[WARN] Debug monitor wymaga biblioteki psutil (pomijam).")
            return False
        try:
            self._proc = psutil.Process(self.pid)
            self._proc.cpu_percent(None)
        except Exception as exc:
            print(f"[WARN] Debug monitor nie może połączyć się z procesem: {exc}")
            return False
        self._gpu_ready = self._init_gpu()
        self._started_at = time.time()
        self._thread = threading.Thread(
            target=self._run, name="RegionGrowMonitor", daemon=True
        )
        self._thread.start()
        print(
            f"[DEBUG] Resource monitor aktywny (dt={self.interval:.2f}s, output={self.output_path})"
        )
        return True

    def stop(self):
        if self._thread:
            self._stop_event.set()
            self._thread.join(timeout=2.0)
        if self._gpu_ready and pynvml is not None:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
        if self._samples:
            self._write_summary()

    def _init_gpu(self) -> bool:
        if pynvml is None:
            return False
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)
            ]
            return bool(self._gpu_handles)
        except Exception:
            return False

    def _gpu_usage_now(self) -> float:
        if not (self._gpu_ready and pynvml is not None):
            return 0.0
        total = 0.0
        for handle in self._gpu_handles:
            processes = []
            for getter_name in (
                "nvmlDeviceGetGraphicsRunningProcesses_v3",
                "nvmlDeviceGetGraphicsRunningProcesses",
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetComputeRunningProcesses",
            ):
                getter = getattr(pynvml, getter_name, None)
                if getter is None:
                    continue
                try:
                    processes = getter(handle)  # type: ignore
                    if processes:
                        break
                except Exception:
                    continue
            for proc in processes or []:
                pid = getattr(proc, "pid", None)
                mem = getattr(proc, "usedGpuMemory", 0)
                if pid == self.pid and mem:
                    total = max(total, float(mem) / (1024 * 1024))
        return total

    def _run(self):
        while not self._stop_event.is_set():
            try:
                cpu = self._proc.cpu_percent(None) if self._proc else 0.0
                rss = (
                    self._proc.memory_info().rss / (1024 * 1024)
                    if self._proc
                    else 0.0
                )
            except Exception:
                break
            gpu = self._gpu_usage_now()
            self._samples.append(
                {
                    "ts": time.time(),
                    "cpu_percent": float(cpu),
                    "rss_mb": float(rss),
                    "gpu_mem_mb": float(gpu),
                }
            )
            self._stop_event.wait(self.interval)

    def _write_summary(self):
        def _avg(key: str) -> float:
            if not self._samples:
                return 0.0
            return float(
                sum(sample.get(key, 0.0) for sample in self._samples) / len(self._samples)
            )

        def _max(key: str) -> float:
            if not self._samples:
                return 0.0
            return float(max(sample.get(key, 0.0) for sample in self._samples))

        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "pid": self.pid,
            "process_name": self._proc.name() if self._proc else "",
            "duration_sec": round(time.time() - self._started_at, 3),
            "interval_sec": self.interval,
            "sample_count": len(self._samples),
            "cpu_percent_avg": round(_avg("cpu_percent"), 3),
            "cpu_percent_max": round(_max("cpu_percent"), 3),
            "rss_mb_avg": round(_avg("rss_mb"), 3),
            "rss_mb_max": round(_max("rss_mb"), 3),
            "gpu_mem_mb_avg": round(_avg("gpu_mem_mb"), 3),
            "gpu_mem_mb_max": round(_max("gpu_mem_mb"), 3),
            "gpu_monitoring": bool(self._gpu_ready),
            "samples": [
                {
                    "ts": datetime.fromtimestamp(sample["ts"]).isoformat(timespec="seconds"),
                    "cpu_percent": round(sample["cpu_percent"], 3),
                    "rss_mb": round(sample["rss_mb"], 3),
                    "gpu_mem_mb": round(sample["gpu_mem_mb"], 3),
                }
                for sample in self._samples
            ],
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[DEBUG] Resource monitor zapisany do: {self.output_path}")


def main_cli():
    """
    Wariant CLI, który potrafi przetwarzać cały folder z RAW screenami.
    - bez argumentów: wszystkie obrazy z RAW_SCREEN_DIR
    - argument = plik: pojedynczy obraz
    - argument = folder: wszystkie obsługiwane obrazy z tego folderu
    """
    print("=" * 60)
    print("[DEBUG] CLI (batch) START")
    print("=" * 60)

    import sys
    from pathlib import Path as _Path

    def _iter_images_in_dir(dir_path: _Path):
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        if not dir_path.is_dir():
            return []
        return sorted(p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts)

    raw_args = sys.argv[1:]
    debug_enabled = False
    debug_interval = 1.0
    debug_output: Optional[_Path] = None
    cleaned_args: List[str] = []
    i = 0
    while i < len(raw_args):
        tok = raw_args[i]
        if tok == "--debug":
            debug_enabled = True
            i += 1
            continue
        if tok.startswith("--debug-interval="):
            debug_enabled = True
            try:
                debug_interval = float(tok.split("=", 1)[1])
            except ValueError:
                print("[ERROR] Invalid value for --debug-interval")
                return
            i += 1
            continue
        if tok == "--debug-interval":
            if i + 1 >= len(raw_args):
                print("[ERROR] Missing value for --debug-interval")
                return
            try:
                debug_interval = float(raw_args[i + 1])
            except ValueError:
                print("[ERROR] Invalid value for --debug-interval")
                return
            debug_enabled = True
            i += 2
            continue
        if tok.startswith("--debug-output="):
            debug_enabled = True
            debug_output = _Path(tok.split("=", 1)[1]).expanduser()
            i += 1
            continue
        if tok == "--debug-output":
            if i + 1 >= len(raw_args):
                print("[ERROR] Missing value for --debug-output")
                return
            debug_enabled = True
            debug_output = _Path(raw_args[i + 1]).expanduser()
            i += 2
            continue
        cleaned_args.append(tok)
        i += 1

    arg = cleaned_args[0] if cleaned_args else None

    monitor: Optional[ResourceMonitor] = None
    if debug_enabled:
        default_name = f"usage_debug_{int(time.time())}.json"
        out_path = debug_output or _Path(JSON_OUT_DIR) / default_name
        monitor = ResourceMonitor(os.getpid(), debug_interval, out_path)
        if not monitor.start():
            monitor = None

    try:

        if arg is None:
            default_img = _Path(_resolve_default_image())
            if default_img.exists():
                images = [default_img]
                print(f"[DEBUG] No CLI path -> using default image: {default_img}")
            else:
                images = _iter_images_in_dir(_Path(RAW_SCREEN_DIR))
                if images:
                    print(
                        f"[DEBUG] No CLI path -> processing folder: {RAW_SCREEN_DIR} "
                        f"({len(images)} images)"
                    )
                else:
                    print(f"[ERROR] No images available (checked {RAW_SCREEN_DIR})")
                    print("\n[DEBUG] CLI END")
                    return
        else:
            p = _Path(arg)
            if p.is_dir():
                images = _iter_images_in_dir(p)
                if not images:
                    print(f"[ERROR] Folder is empty or has no supported images: {p}")
                    print("\n[DEBUG] CLI END")
                    return
                print(f"[DEBUG] Processing folder: {p} ({len(images)} images)")
            else:
                images = [p]
                print(f"[DEBUG] Processing single image: {p}")

        os.makedirs(JSON_OUT_DIR, exist_ok=True)
        processed = 0

        for img_path in images:
            path_str = str(img_path)
            if not os.path.isfile(path_str):
                print(f"[WARN] File not found, skipping: {path_str}")
                continue

            try:
                CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
                current_target = CURRENT_RUN_DIR / "screenshot.png"
                shutil.copy2(path_str, current_target)
                print(f"[DEBUG] Current run screenshot updated: {current_target}")
            except Exception as exc:
                print(f"[WARN] Could not update CURRENT_RUN_DIR screenshot: {exc}")

            print("\n" + "=" * 60)
            print(f"[DEBUG] Image path: {path_str}")

            out = run_dropdown_detection(path_str)
            t_annot = time.perf_counter()
            out_path = annotate_and_save(path_str, out.get("results", []), out.get("triangles"), output_dir=str(REGION_GROW_ANNOT_DIR))
            print(f"[TIMER] Annotate + save overlay: {(time.perf_counter()-t_annot)*1000:.1f} ms")

            base_name = os.path.splitext(os.path.basename(path_str))[0]
            json_path = os.path.join(JSON_OUT_DIR, f"{base_name}.json")

            print(f"[DEBUG] Saving JSON to: {json_path}")
            try:
                t_json = time.perf_counter()
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(to_py(out), f, ensure_ascii=False, indent=2)
                print(f"[DEBUG] JSON saved successfully! ({(time.perf_counter()-t_json)*1000:.1f} ms)")
                print(f"[DEBUG] Annotation: {out_path}")
                processed += 1
            except Exception as e:
                print(f"[ERROR] Could not save JSON for {path_str}: {e}")

        if processed == 0:
            print("[WARN] No images processed, skipping CNN runner.")
            print("\n[DEBUG] CLI END")
            return

        # ========== CZYŚCIMY CROPY Z POPRZEDNIEJ RUNDY ==========
        try:
            if os.path.isdir(CNN_CROPS_DIR):
                shutil.rmtree(CNN_CROPS_DIR, ignore_errors=True)
            os.makedirs(CNN_CROPS_DIR, exist_ok=True)
            print(f"[DEBUG] Cleaned: {CNN_CROPS_DIR}")
        except Exception as e:
            print(f"[WARN] Could not clean {CNN_CROPS_DIR}: {e}")

        # ========== AUTO-START RUNNERA CNN ==========
        try:
            if os.path.isfile(CNN_RUNNER):  
                print("[DEBUG] Launching CNN runner...")
                cmd = [
                    sys.executable,
                    CNN_RUNNER,
                    "--json-dir",
                    JSON_OUT_DIR,
                    "--out-dir",
                    CNN_CROPS_DIR,
                    "--model",
                    fr"{ROOT}\tri_cnn.pt",
                    "--img-size",
                    "128",
                    "--padding",
                    "0.10",
                    "--batch",
                    "256",
                    "--thresh",
                    "0.50",
                ]
                t0 = time.perf_counter()
                subprocess.run(cmd, check=False)
                print(f"[DEBUG] CNN runner done in {(time.perf_counter()-t0)*1000:.1f} ms")
            else:
                print(f"[WARN] CNN runner not found: {CNN_RUNNER}")
        except Exception as e:
            print(f"[WARN] CNN runner failed: {e}")

        print("\n[DEBUG] CLI END")

    except Exception as e:
        print(f"\n[ERROR] CLI failed: {e}")
        import traceback

        traceback.print_exc()
        print("\n[DEBUG] CLI END")
    finally:
        if monitor:
            monitor.stop()


if __name__ == "__main__":
    main_cli()
