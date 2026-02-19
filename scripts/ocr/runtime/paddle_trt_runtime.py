from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

try:
    import paddle  # type: ignore
except Exception:
    paddle = None  # type: ignore[assignment]

try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:
    PaddleOCR = None  # type: ignore[assignment]
try:
    from rapidocr_paddle import RapidOCR  # type: ignore
except Exception:
    RapidOCR = None  # type: ignore[assignment]
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore[assignment]
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore[assignment]


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in _TRUE_VALUES


def _blue_timer_line(text: str) -> str:
    s = str(text or "")
    if "TIMER" not in s:
        return s
    return f"\x1b[94m{s}\x1b[0m"


@dataclass
class OcrRuntimeConfig:
    backend: str = "gpu_fp32"
    model_dir: str = ""
    trt_cache_dir: str = ""
    calib_cache: str = ""
    enable_cls: bool = False
    int8_min_conf: float = 0.01
    fallback_on_fail: bool = True
    lang: str = "pl"
    gpu_id: int = 0
    rec_batch_num: int = 16
    show_log: bool = False
    warmup_runs: int = 1
    use_textline_orientation: bool = False

    @classmethod
    def from_env(cls, *, lang: Optional[str] = None, rec_batch_num: Optional[int] = None) -> "OcrRuntimeConfig":
        cfg = cls(
            backend=str(os.environ.get("FULLBOT_OCR_BACKEND", "gpu_fp32") or "gpu_fp32").strip().lower(),
            model_dir=str(os.environ.get("FULLBOT_OCR_MODEL_DIR", "") or "").strip(),
            trt_cache_dir=str(os.environ.get("FULLBOT_OCR_TRT_CACHE_DIR", "") or "").strip(),
            calib_cache=str(os.environ.get("FULLBOT_OCR_CALIB_CACHE", "") or "").strip(),
            enable_cls=_as_bool(os.environ.get("FULLBOT_OCR_ENABLE_CLS"), False),
            int8_min_conf=float(os.environ.get("FULLBOT_OCR_INT8_MIN_CONF", "0.01") or "0.01"),
            fallback_on_fail=_as_bool(os.environ.get("FULLBOT_OCR_FALLBACK_ON_FAIL"), True),
            lang=str(os.environ.get("PADDLEOCR_LANG", "pl") or "pl").strip(),
            gpu_id=int(os.environ.get("PADDLEOCR_GPU_ID", "0") or 0),
            rec_batch_num=int(os.environ.get("PADDLEOCR_REC_BATCH", "16") or 16),
            show_log=_as_bool(os.environ.get("PP_OCR_SHOW_LOG"), False),
            warmup_runs=max(0, int(os.environ.get("FULLBOT_OCR_WARMUP_RUNS", "1") or 1)),
            use_textline_orientation=_as_bool(os.environ.get("FULLBOT_OCR_TEXTLINE_ORIENTATION"), False),
        )
        if lang:
            cfg.lang = lang
        if rec_batch_num is not None:
            cfg.rec_batch_num = int(rec_batch_num)
        return cfg


class OcrRuntimeError(RuntimeError):
    pass


class OcrRuntime:
    def __init__(self, config: OcrRuntimeConfig):
        self.config = config
        self._reader: Any = None
        self._rapid_detector: Any = None
        self.active_backend = ""
        self.active_precision = ""
        self._iter_counter = 0
        self._pipeline_mode = str(os.environ.get("FULLBOT_OCR_PIPELINE", "two_stage") or "two_stage").strip().lower()
        self._stage1_backend = str(os.environ.get("FULLBOT_OCR_STAGE1_BACKEND", "rapid_det") or "rapid_det").strip().lower()
        self._require_rapid_stage1 = _as_bool(os.environ.get("FULLBOT_OCR_STAGE1_REQUIRE_RAPID"), True)
        self._stage1_max_side = int(os.environ.get("FULLBOT_OCR_STAGE1_MAX_SIDE", "640") or 640)
        self._stage1_use_det = _as_bool(os.environ.get("FULLBOT_OCR_STAGE1_USE_DET"), True)
        self._stage1_use_cls = _as_bool(os.environ.get("FULLBOT_OCR_STAGE1_USE_CLS"), False)
        self._stage1_use_rec = _as_bool(os.environ.get("FULLBOT_OCR_STAGE1_USE_REC"), False)
        self._stage1_det_limit_type = str(
            os.environ.get("FULLBOT_OCR_STAGE1_DET_LIMIT_TYPE", "max") or "max"
        ).strip().lower()
        self._stage1_det_limit_side_len = int(
            os.environ.get("FULLBOT_OCR_STAGE1_DET_LIMIT_SIDE_LEN", str(self._stage1_max_side))
            or self._stage1_max_side
        )
        self._stage1_det_max_candidates = max(
            20, int(os.environ.get("FULLBOT_OCR_STAGE1_DET_MAX_CANDIDATES", "300") or 300)
        )
        self._stage1_warmup_runs = max(
            0, int(os.environ.get("FULLBOT_OCR_STAGE1_WARMUP_RUNS", "2") or 2)
        )
        self._stage2_batch = max(1, int(os.environ.get("FULLBOT_OCR_STAGE2_BATCH", "24") or 24))
        self._stage2_min_conf = float(os.environ.get("FULLBOT_OCR_STAGE2_MIN_CONF", "0.20") or 0.20)
        self._crop_margin = int(os.environ.get("FULLBOT_OCR_STAGE2_CROP_MARGIN_PX", "20") or 20)
        self._timers_enabled = _as_bool(
            os.environ.get("FULLBOT_OCR_ENABLE_TIMERS"),
            _as_bool(os.environ.get("FULLBOT_ADVANCED_DEBUG"), False),
        )
        if self._pipeline_mode == "two_stage":
            self._timers_enabled = True
        repo_root = Path(__file__).resolve().parents[3]
        data_root = repo_root / "data" / "screen"
        debug_default = (
            self._pipeline_mode == "two_stage"
            or _as_bool(os.environ.get("FULLBOT_ADVANCED_DEBUG"), False)
            or _as_bool(os.environ.get("FULLBOT_OCR_BOXES_DEBUG"), False)
        )
        # In two_stage mode always persist debug artifacts (user-required diagnostics).
        if self._pipeline_mode == "two_stage":
            self._debug_save = True
        else:
            self._debug_save = _as_bool(
                os.environ.get("FULLBOT_OCR_DEBUG_SAVE"),
                debug_default,
            )
        self._debug_det_dir = Path(os.environ.get("FULLBOT_OCR_DEBUG_DET_DIR", str(data_root / "OCR_det")))
        self._debug_crops_root = Path(os.environ.get("FULLBOT_OCR_DEBUG_CROPS_DIR", str(data_root / "OCR_crops")))
        if self._debug_save:
            try:
                self._debug_det_dir.mkdir(parents=True, exist_ok=True)
                self._debug_crops_root.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                print(f"[WARN] OCR debug dir init failed: {exc}")
        print(
            "[OCR] debug "
            f"save={int(self._debug_save)} "
            f"det_dir={self._debug_det_dir} "
            f"crops_root={self._debug_crops_root}"
        )
        self._init_runtime()

    def _backend_chain(self) -> list[str]:
        requested = self.config.backend
        if requested == "trt_int8":
            return ["trt_int8", "gpu_fp32", "cpu_fp32"] if self.config.fallback_on_fail else ["trt_int8"]
        if requested == "gpu_fp32":
            return ["gpu_fp32", "cpu_fp32"] if self.config.fallback_on_fail else ["gpu_fp32"]
        if requested == "cpu_fp32":
            return ["cpu_fp32"]
        return ["gpu_fp32", "cpu_fp32"] if self.config.fallback_on_fail else ["gpu_fp32"]

    def _init_runtime(self) -> None:
        errors: list[str] = []
        for backend in self._backend_chain():
            try:
                self._reader = self._build_reader(backend)
                self.active_backend = backend
                self.active_precision = "INT8" if backend == "trt_int8" else "FP32"
                self._warmup()
                return
            except Exception as exc:
                errors.append(f"{backend}: {exc}")
        joined = "; ".join(errors) if errors else "unknown runtime error"
        raise OcrRuntimeError(f"OCR runtime init failed ({self.config.backend}) -> {joined}")

    def _base_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "lang": self.config.lang,
            "show_log": self.config.show_log,
            "use_angle_cls": self.config.enable_cls,
            "use_textline_orientation": self.config.use_textline_orientation,
            "rec_batch_num": self.config.rec_batch_num,
            "text_recognition_batch_size": self.config.rec_batch_num,
        }

        model_root = Path(self.config.model_dir) if self.config.model_dir else None
        if model_root and model_root.exists():
            det_dir = model_root / "det"
            rec_dir = model_root / "rec"
            cls_dir = model_root / "cls"
            if det_dir.exists():
                kwargs["det_model_dir"] = str(det_dir)
            if rec_dir.exists():
                kwargs["rec_model_dir"] = str(rec_dir)
            if cls_dir.exists():
                kwargs["cls_model_dir"] = str(cls_dir)
        return kwargs

    def _prepare_gpu(self) -> None:
        if paddle is None:
            raise OcrRuntimeError("paddle is not installed")
        if not paddle.is_compiled_with_cuda():
            raise OcrRuntimeError("paddle is installed without CUDA support")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.config.gpu_id))
        try:
            paddle.set_device(f"gpu:{self.config.gpu_id}")
        except Exception:
            paddle.set_device("gpu")

    def _build_reader(self, backend: str) -> Any:
        if PaddleOCR is None:
            raise OcrRuntimeError("paddleocr package is not available")

        kwargs = self._base_kwargs()
        if backend == "cpu_fp32":
            kwargs.update({"use_gpu": False})
            return self._construct_paddle_reader(kwargs)

        self._prepare_gpu()
        kwargs.update({"use_gpu": True, "gpu_id": self.config.gpu_id})
        if backend == "trt_int8":
            kwargs.update(
                {
                    "use_tensorrt": True,
                    "precision": "int8",
                    "ir_optim": True,
                    "gpu_mem": 2000,
                    "min_subgraph_size": 3,
                }
            )
            if self.config.trt_cache_dir:
                cache_dir = Path(self.config.trt_cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("FULLBOT_OCR_TRT_CACHE_DIR", str(cache_dir))
            if self.config.calib_cache:
                os.environ.setdefault("FULLBOT_OCR_CALIB_CACHE", self.config.calib_cache)
        return self._construct_paddle_reader(kwargs)

    @staticmethod
    def _construct_paddle_reader(kwargs: dict[str, Any]) -> Any:
        attempts: list[dict[str, Any]] = [dict(kwargs)]
        # Compatibility for PaddleOCR 2.x/3.x argument differences.
        drop_keys = [
            "precision",
            "use_tensorrt",
            "gpu_id",
            "use_gpu",
            "gpu_mem",
            "ir_optim",
            "min_subgraph_size",
            "text_recognition_batch_size",
            "use_textline_orientation",
            "use_angle_cls",
            "rec_batch_num",
        ]
        for key in drop_keys:
            last = attempts[-1]
            if key in last:
                trimmed = dict(last)
                trimmed.pop(key, None)
                attempts.append(trimmed)

        last_exc: Optional[Exception] = None
        for params in attempts:
            try:
                return PaddleOCR(**params)
            except Exception as exc:
                last_exc = exc
                msg = str(exc or "")
                m = re.search(r"Unknown argument:\s*([A-Za-z0-9_]+)", msg)
                if m:
                    bad_key = m.group(1).strip()
                    if bad_key and bad_key in params:
                        trimmed = dict(params)
                        trimmed.pop(bad_key, None)
                        attempts.append(trimmed)
                continue
        raise OcrRuntimeError(f"PaddleOCR init failed: {last_exc}")

    def _warmup(self) -> None:
        runs = max(0, int(self.config.warmup_runs))
        if runs <= 0:
            return
        if np is None:
            return

        dummy = np.zeros((96, 320, 3), dtype=np.uint8)
        for _ in range(runs):
            try:
                self._reader_ocr_compat(dummy, det=True, rec=True, cls=self.config.enable_cls)
            except Exception:
                break

    def _reader_ocr_compat(self, image: Any, **kwargs: Any) -> Any:
        """Call PaddleOCR.ocr across API variants that may reject det/rec/cls kwargs."""
        call_kwargs = dict(kwargs)
        attempts = [dict(call_kwargs)]
        # Common fallback trims for newer PaddleOCR APIs.
        for key in ("det", "rec", "cls"):
            last = attempts[-1]
            if key in last:
                trimmed = dict(last)
                trimmed.pop(key, None)
                attempts.append(trimmed)
        attempts.append({})

        last_exc: Optional[Exception] = None
        for kw in attempts:
            try:
                return self._reader.ocr(image, **kw)
            except TypeError as exc:
                last_exc = exc
                msg = str(exc or "")
                m = re.search(r"unexpected keyword argument '([A-Za-z0-9_]+)'", msg)
                if m:
                    bad = m.group(1).strip()
                    if bad in kw:
                        kw2 = dict(kw)
                        kw2.pop(bad, None)
                        attempts.append(kw2)
                continue
            except Exception as exc:
                last_exc = exc
                msg = str(exc or "")
                m = re.search(r"unexpected keyword argument '([A-Za-z0-9_]+)'", msg)
                if m:
                    bad = m.group(1).strip()
                    if bad in kw:
                        kw2 = dict(kw)
                        kw2.pop(bad, None)
                        attempts.append(kw2)
                        continue
                raise
        if last_exc is not None:
            raise last_exc
        return self._reader.ocr(image)

    def _log_timer(self, key: str, started: float) -> None:
        if self._timers_enabled:
            line = f"[TIMER] {key} {(time.perf_counter() - started) * 1000.0:.1f}ms"
            print(_blue_timer_line(line), flush=True)

    def _next_iteration_id(self, explicit: Optional[str] = None) -> str:
        if explicit:
            return explicit
        env_id = str(os.environ.get("FULLBOT_OCR_ITERATION_ID", "") or "").strip()
        if env_id:
            return env_id
        self._iter_counter += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        return f"iter_{self._iter_counter}_{ts}"

    def _ensure_debug_dirs(self, iteration_id: str) -> Tuple[Path, Path]:
        self._debug_det_dir.mkdir(parents=True, exist_ok=True)
        iter_dir = self._debug_crops_root / iteration_id
        iter_dir.mkdir(parents=True, exist_ok=True)
        return self._debug_det_dir, iter_dir

    def _get_rapid_detector(self) -> Any:
        if self._rapid_detector is not None:
            return self._rapid_detector
        if RapidOCR is None:
            raise OcrRuntimeError("rapidocr_paddle is not available for stage1 det")
        self._rapid_detector = RapidOCR(
            det_use_cuda=True,
            cls_use_cuda=False,
            rec_use_cuda=False,
            use_det=self._stage1_use_det,
            use_cls=self._stage1_use_cls,
            use_rec=self._stage1_use_rec,
            det_limit_type=self._stage1_det_limit_type,
            det_limit_side_len=self._stage1_det_limit_side_len,
            det_max_candidates=self._stage1_det_max_candidates,
        )
        return self._rapid_detector

    def warmup_stage1(self, sample_image: Any | None = None) -> None:
        if np is None:
            return
        runs = int(self._stage1_warmup_runs)
        if runs <= 0:
            return
        rapid = self._get_rapid_detector()
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(runs):
            with_error = None
            try:
                with_error = rapid(
                    np.ascontiguousarray(dummy),
                    use_det=self._stage1_use_det,
                    use_cls=self._stage1_use_cls,
                    use_rec=self._stage1_use_rec,
                )
            except Exception:
                break
            _ = with_error
        if sample_image is not None:
            try:
                rapid(
                    np.ascontiguousarray(sample_image),
                    use_det=self._stage1_use_det,
                    use_cls=self._stage1_use_cls,
                    use_rec=self._stage1_use_rec,
                )
            except Exception:
                pass
        self._log_timer("ocr_stage1_warmup_ms", t0)

    @staticmethod
    def _to_quad(points: Any) -> Optional[list[list[float]]]:
        if points is None:
            return None
        if hasattr(points, "tolist"):
            points = points.tolist()
        if not isinstance(points, (list, tuple)):
            return None
        if len(points) == 4 and isinstance(points[0], (list, tuple)):
            out = []
            for p in points:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    return None
                out.append([float(p[0]), float(p[1])])
            return out
        if len(points) >= 8:
            out = []
            for idx in range(0, 8, 2):
                out.append([float(points[idx]), float(points[idx + 1])])
            return out
        return None

    @staticmethod
    def _bbox_from_quad(quad: list[list[float]]) -> tuple[int, int, int, int]:
        xs = [int(round(p[0])) for p in quad]
        ys = [int(round(p[1])) for p in quad]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
        area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
        return inter / max(1e-6, area_a + area_b - inter)

    def _dedup_boxes(self, boxes: list[dict[str, Any]], iou_thr: float = 0.7) -> list[dict[str, Any]]:
        if not boxes:
            return boxes
        out: list[dict[str, Any]] = []
        for item in boxes:
            keep = True
            for ex in out:
                if self._iou(item["bbox"], ex["bbox"]) >= iou_thr:
                    keep = False
                    break
            if keep:
                out.append(item)
        return out

    def _cap_and_dedup_stage1(self, boxes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        raw_count = len(boxes)
        if not boxes:
            return boxes, raw_count
        boxes_sorted = sorted(boxes, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        boxes_capped = boxes_sorted[: self._stage1_det_max_candidates]
        boxes_out = self._dedup_boxes(boxes_capped, iou_thr=0.7)
        return boxes_out, raw_count

    def _resize_for_stage1(self, image: Any) -> tuple[Any, float]:
        if np is None:
            return image, 1.0
        h, w = image.shape[:2]
        mx = max(h, w)
        if mx <= self._stage1_max_side:
            return image, 1.0
        scale = float(self._stage1_max_side) / float(mx)
        nw = max(2, int(round(w * scale)))
        nh = max(2, int(round(h * scale)))
        if cv2 is None:
            return image, 1.0
        return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA), scale

    def _parse_stage1_rapid(self, result: Any, scale_inv: float, w0: int, h0: int) -> list[dict[str, Any]]:
        boxes: list[dict[str, Any]] = []
        if isinstance(result, tuple):
            result = result[0]
        if hasattr(result, "tolist"):
            try:
                result = result.tolist()
            except Exception:
                pass
        if not isinstance(result, list):
            return boxes
        for item in result:
            quad = None
            score = 0.0
            if isinstance(item, dict):
                quad = self._to_quad(item.get("boxes") or item.get("box") or item.get("dt_boxes"))
                with_error = item.get("score") or item.get("conf") or 0.0
                try:
                    score = float(with_error)
                except Exception:
                    score = 0.0
            elif isinstance(item, (list, tuple)):
                # rapidocr_paddle det-only can return a plain quad list:
                #   [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                # Older/full outputs may return nested tuples/lists.
                quad = self._to_quad(item)
                if quad is None:
                    quad = self._to_quad(item[0] if item else None)
                    if len(item) >= 2:
                        payload = item[1]
                        if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                            try:
                                score = float(payload[1] or 0.0)
                            except Exception:
                                score = 0.0
                        elif len(item) >= 3:
                            try:
                                score = float(item[2] or 0.0)
                            except Exception:
                                score = 0.0
            if quad is None:
                continue
            q0: list[list[int]] = []
            for x, y in quad:
                gx = int(round(float(x) * scale_inv))
                gy = int(round(float(y) * scale_inv))
                gx = max(0, min(w0 - 1, gx))
                gy = max(0, min(h0 - 1, gy))
                q0.append([gx, gy])
            bbox = self._bbox_from_quad(q0)
            if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
                continue
            boxes.append({"quad": q0, "bbox": bbox, "score": float(score)})
        return boxes

    def _parse_stage1_paddle(self, result: Any) -> list[dict[str, Any]]:
        boxes: list[dict[str, Any]] = []
        lines = self._parse_legacy_crop_result(result)
        for quad, _txt, conf in lines:
            q0 = [[int(round(x)), int(round(y))] for x, y in quad]
            if len(q0) != 4:
                continue
            bbox = self._bbox_from_quad(q0)
            if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
                continue
            boxes.append({"quad": q0, "bbox": bbox, "score": float(conf)})
        return boxes

    @staticmethod
    def _parse_legacy_crop_result(res: Any) -> list[tuple[list[list[float]], str, float]]:
        lines: list[tuple[list[list[float]], str, float]] = []
        if res is None:
            return lines
        if isinstance(res, dict):
            res = [res]
        elif isinstance(res, tuple):
            res = list(res)
        elif not isinstance(res, list):
            # PaddleOCR 3.x can return a single OCRResult-like object.
            res = [res]
        if not res:
            return lines
        # PaddleOCR >=3 can return list of OCRResult objects.
        first = res[0]
        if not isinstance(first, (dict, list, tuple)):
            for item in res:
                try:
                    polys = getattr(item, "dt_polys", None) or getattr(item, "rec_polys", None) or []
                    texts = getattr(item, "rec_texts", None) or []
                    scores = getattr(item, "rec_scores", None) or []
                    if hasattr(polys, "tolist"):
                        polys = polys.tolist()
                    if not isinstance(polys, list):
                        continue
                    for idx, poly in enumerate(polys):
                        quad = OcrRuntime._to_quad(poly)
                        if quad is None:
                            continue
                        txt = str(texts[idx] if isinstance(texts, (list, tuple)) and idx < len(texts) else "").strip()
                        try:
                            conf = float(scores[idx]) if isinstance(scores, (list, tuple)) and idx < len(scores) else 0.0
                        except Exception:
                            conf = 0.0
                        lines.append((quad, txt, conf))
                except Exception:
                    continue
            return lines
        if isinstance(res[0], dict):
            for item in res:
                polys = item.get("dt_polys") or item.get("rec_polys") or []
                texts = item.get("rec_texts") or []
                scores = item.get("rec_scores") or []
                if hasattr(polys, "tolist"):
                    polys = polys.tolist()
                if hasattr(texts, "tolist"):
                    texts = texts.tolist()
                if hasattr(scores, "tolist"):
                    scores = scores.tolist()
                if not isinstance(polys, list):
                    continue
                for idx, poly in enumerate(polys):
                    quad = OcrRuntime._to_quad(poly)
                    if quad is None:
                        continue
                    txt = (
                        str(texts[idx] if idx < len(texts) else "").strip()
                        if isinstance(texts, (list, tuple))
                        else ""
                    )
                    try:
                        conf = float(scores[idx]) if isinstance(scores, (list, tuple)) and idx < len(scores) else 0.0
                    except Exception:
                        conf = 0.0
                    lines.append((quad, txt, conf))
            return lines
        block = res[0] if isinstance(res, list) else []
        if not isinstance(block, list):
            return lines
        for it in block:
            try:
                quad = OcrRuntime._to_quad(it[0])
                if quad is None:
                    continue
                txt = str(it[1][0] or "").strip()
                conf = float(it[1][1] or 0.0)
                lines.append((quad, txt, conf))
            except Exception:
                continue
        return lines

    def _save_stage1_debug(self, det_dir: Path, iteration_id: str, screen960: Any, boxes: list[dict[str, Any]], scale: float) -> None:
        if cv2 is None:
            return
        p960 = det_dir / f"{iteration_id}_screen960.png"
        pbox = det_dir / f"{iteration_id}_det_boxes.png"
        pjson = det_dir / f"{iteration_id}_det_boxes.json"
        cv2.imwrite(str(p960), screen960)
        overlay = screen960.copy()
        for b in boxes:
            pts = b["quad_960"] if "quad_960" in b else None
            if not pts:
                x1, y1, x2, y2 = b["bbox_960"]
                pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            arr = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [arr], True, (0, 255, 0), 2)
        cv2.imwrite(str(pbox), overlay)
        payload = {
            "iteration_id": iteration_id,
            "image_size_960": [int(screen960.shape[1]), int(screen960.shape[0])],
            "scale_to_original": float(1.0 / scale if scale > 0 else 1.0),
            "det_count": len(boxes),
            "boxes": boxes,
        }
        pjson.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_crop_debug(
        self,
        iter_dir: Path,
        crops: list[Any],
        crop_meta: list[dict[str, Any]],
        stage2_payload: list[dict[str, Any]],
        final_payload: list[dict[str, Any]],
    ) -> None:
        if cv2 is not None:
            for i, crop in enumerate(crops):
                cv2.imwrite(str(iter_dir / f"crop_{i+1:04d}.png"), crop)
        (iter_dir / "crops_manifest.json").write_text(json.dumps({"crops": crop_meta}, ensure_ascii=False, indent=2), encoding="utf-8")
        (iter_dir / "ocr_stage2_results.json").write_text(json.dumps({"results": stage2_payload}, ensure_ascii=False, indent=2), encoding="utf-8")
        (iter_dir / "final_ocr_items.json").write_text(json.dumps({"results": final_payload}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _ocr_two_stage(self, image: Any, iteration_id: str, **kwargs: Any) -> Any:
        if np is None or cv2 is None:
            raise OcrRuntimeError("two_stage OCR requires numpy and opencv")
        if image is None or not hasattr(image, "shape"):
            raise OcrRuntimeError("two_stage OCR expects ndarray image")
        h0, w0 = image.shape[:2]
        det_dir: Optional[Path] = None
        iter_dir: Optional[Path] = None
        if self._debug_save:
            det_dir, iter_dir = self._ensure_debug_dirs(iteration_id)
            try:
                (iter_dir / "_debug_status.json").write_text(
                    json.dumps({"iteration_id": iteration_id, "status": "started"}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

        t0 = time.perf_counter()
        det_elapsed_ms = 0.0
        crop_elapsed_ms = 0.0
        stage2_elapsed_ms = 0.0
        post_elapsed_ms = 0.0
        stage1_debug_build_ms = 0.0
        stage1_debug_save_ms = 0.0
        crops_manifest_io_ms = 0.0
        stage2_call_ms = 0.0
        stage2_parse_ms = 0.0
        stage2_fallback_ms = 0.0
        debug_write_final_ms = 0.0
        t1 = time.perf_counter()
        t_stage1_resize = time.perf_counter()
        stage1_img, scale = self._resize_for_stage1(image)
        scale_inv = 1.0 / (scale if scale > 0 else 1.0)
        self._log_timer("ocr_stage1_resize_ms", t_stage1_resize)

        stage1_backend_used = "rapid_det"
        stage1_raw_count = 0
        stage1_after_nms = 0
        try:
            t_stage1_init = time.perf_counter()
            rapid = self._get_rapid_detector()
            self._log_timer("ocr_stage1_init_ms", t_stage1_init)
            t_det_rapid = time.perf_counter()
            det_out = rapid(
                np.ascontiguousarray(stage1_img),
                use_det=self._stage1_use_det,
                use_cls=self._stage1_use_cls,
                use_rec=self._stage1_use_rec,
            )
            self._log_timer("ocr_stage1_rapid_call_ms", t_det_rapid)
            if isinstance(det_out, tuple) and len(det_out) >= 2 and isinstance(det_out[1], (list, tuple)) and det_out[1]:
                try:
                    native_ms = float(det_out[1][0]) * 1000.0
                    if self._timers_enabled:
                        print(_blue_timer_line(f"[TIMER] ocr_stage1_rapid_native_ms {native_ms:.1f}ms"), flush=True)
                except Exception:
                    pass
            det_payload = det_out[0] if isinstance(det_out, tuple) else det_out

            t_stage1_parse = time.perf_counter()
            stage1_boxes_raw = self._parse_stage1_rapid(det_payload, scale_inv=scale_inv, w0=w0, h0=h0)
            self._log_timer("ocr_stage1_parse_ms", t_stage1_parse)

            t_stage1_nms = time.perf_counter()
            stage1_boxes, stage1_raw_count = self._cap_and_dedup_stage1(stage1_boxes_raw)
            stage1_after_nms = len(stage1_boxes)
            self._log_timer("ocr_stage1_nms_ms", t_stage1_nms)
        except Exception as exc:
            if self._require_rapid_stage1:
                if self._debug_save and det_dir is not None:
                    err_path = det_dir / f"{iteration_id}_stage1_error.json"
                    err_path.write_text(
                        json.dumps(
                            {
                                "iteration_id": iteration_id,
                                "stage": "stage1_rapid_det",
                                "error": str(exc),
                                "fallback": "disabled_by_FULLBOT_OCR_STAGE1_REQUIRE_RAPID",
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                raise OcrRuntimeError(f"RapidOCR stage1 failed and fallback is disabled: {exc}")
            stage1_backend_used = "paddle_det_fallback"
            if self._debug_save and det_dir is not None:
                err_path = det_dir / f"{iteration_id}_stage1_error.json"
                err_path.write_text(
                    json.dumps(
                        {
                            "iteration_id": iteration_id,
                            "stage": "stage1_rapid_det",
                            "error": str(exc),
                            "fallback": "paddle_det_fallback",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                        encoding="utf-8",
                    )
            # Fallback: use Paddle det/rec output only to recover text boxes.
            t_det_paddle_fb = time.perf_counter()
            det_res = self._reader_ocr_compat(np.ascontiguousarray(stage1_img), det=True, rec=True, cls=False)
            self._log_timer("ocr_stage1_paddle_fallback_call_ms", t_det_paddle_fb)
            t_stage1_parse = time.perf_counter()
            stage1_small = self._parse_stage1_paddle(det_res)
            stage1_boxes_raw = []
            for b in stage1_small:
                q0 = []
                for x, y in b["quad"]:
                    gx = int(round(float(x) * scale_inv))
                    gy = int(round(float(y) * scale_inv))
                    gx = max(0, min(w0 - 1, gx))
                    gy = max(0, min(h0 - 1, gy))
                    q0.append([gx, gy])
                bbox = self._bbox_from_quad(q0)
                stage1_boxes_raw.append({"quad": q0, "bbox": bbox, "score": float(b.get("score", 0.0))})
            self._log_timer("ocr_stage1_parse_ms", t_stage1_parse)
            t_stage1_nms = time.perf_counter()
            stage1_boxes, stage1_raw_count = self._cap_and_dedup_stage1(stage1_boxes_raw)
            stage1_after_nms = len(stage1_boxes)
            self._log_timer("ocr_stage1_nms_ms", t_stage1_nms)
        self._log_timer("ocr_stage1_det_ms", t1)
        det_elapsed_ms = (time.perf_counter() - t1) * 1000.0
        if self._timers_enabled:
            print(
                _blue_timer_line(
                    f"[TIMER] ocr_stage1_counts raw={int(stage1_raw_count)} after_nms={int(stage1_after_nms)} "
                    f"max_candidates={int(self._stage1_det_max_candidates)}"
                ),
                flush=True,
            )

        t_dbg_build = time.perf_counter()
        stage1_dbg: list[dict[str, Any]] = []
        for b in stage1_boxes:
            x1, y1, x2, y2 = b["bbox"]
            x1s = int(round(x1 * scale))
            y1s = int(round(y1 * scale))
            x2s = int(round(x2 * scale))
            y2s = int(round(y2 * scale))
            stage1_dbg.append(
                {
                    "bbox_960": [x1s, y1s, x2s, y2s],
                    "bbox_original": [x1, y1, x2, y2],
                    "score": b["score"],
                    "backend": stage1_backend_used,
                    "quad_960": [[int(round(px * scale)), int(round(py * scale))] for px, py in b["quad"]],
                    "quad_original": b["quad"],
                }
            )
        stage1_debug_build_ms = (time.perf_counter() - t_dbg_build) * 1000.0
        self._log_timer("ocr_stage1_debug_build_ms", t_dbg_build)
        if self._debug_save and det_dir is not None:
            t_dbg_save = time.perf_counter()
            self._save_stage1_debug(det_dir, iteration_id, stage1_img, stage1_dbg, scale)
            stage1_debug_save_ms = (time.perf_counter() - t_dbg_save) * 1000.0
            self._log_timer("ocr_stage1_debug_save_ms", t_dbg_save)

        t_crop = time.perf_counter()
        crops: list[Any] = []
        crop_meta: list[dict[str, Any]] = []
        for idx, b in enumerate(stage1_boxes):
            x1, y1, x2, y2 = b["bbox"]
            mx = self._crop_margin
            my = self._crop_margin
            cx1 = max(0, x1 - mx)
            cy1 = max(0, y1 - my)
            cx2 = min(w0, x2 + mx)
            cy2 = min(h0, y2 + my)
            if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                continue
            crop = image[cy1:cy2, cx1:cx2]
            if crop is None or crop.size == 0:
                continue
            crops.append(crop)
            crop_meta.append(
                {
                    "crop_index": idx + 1,
                    "source_bbox_original": [x1, y1, x2, y2],
                    "bbox_with_margin": [cx1, cy1, cx2, cy2],
                    "margin_px": self._crop_margin,
                }
            )
        self._log_timer("ocr_crop_export_ms", t_crop)
        self._log_timer("ocr_crops_prepare_ms", t_crop)
        crop_elapsed_ms = (time.perf_counter() - t_crop) * 1000.0
        if self._debug_save and iter_dir is not None:
            t_manifest = time.perf_counter()
            try:
                # Save manifest immediately so we can diagnose failures before stage2.
                (iter_dir / "crops_manifest.json").write_text(
                    json.dumps({"crops": crop_meta}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (iter_dir / "_debug_status.json").write_text(
                    json.dumps(
                        {
                            "iteration_id": iteration_id,
                            "status": "crops_ready",
                            "crops_count": len(crops),
                            "crop_meta_count": len(crop_meta),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"[OCR] crops debug pre-save: iter_dir={iter_dir} count={len(crops)}", flush=True)
            except Exception as exc:
                print(f"[WARN] OCR crops pre-save failed: {exc}", flush=True)
            crops_manifest_io_ms = (time.perf_counter() - t_manifest) * 1000.0
            self._log_timer("ocr_crops_manifest_io_ms", t_manifest)

        stage2_started = time.perf_counter()
        stage2_payload: list[dict[str, Any]] = []
        final_items: list[Any] = []
        stage2_error: Optional[str] = None
        if crops:
            batch_size = self._stage2_batch
            idx = 0
            try:
                while idx < len(crops):
                    chunk = crops[idx : idx + batch_size]
                    metas = crop_meta[idx : idx + batch_size]
                    try:
                        t_stage2_call = time.perf_counter()
                        res_batch = self._reader_ocr_compat(
                            chunk,
                            det=True,
                            rec=True,
                            cls=kwargs.get("cls", self.config.enable_cls),
                        )
                        stage2_call_ms += (time.perf_counter() - t_stage2_call) * 1000.0
                    except Exception as exc:
                        lowered = str(exc).lower()
                        is_mem = any(k in lowered for k in ("out of memory", "oom", "cuda", "cublas", "alloc"))
                        if is_mem and batch_size > 1:
                            batch_size = max(1, batch_size // 2)
                            print(f"[WARN] OCR stage2 OOM-like failure, reducing batch to {batch_size}")
                            continue
                        raise
                    if not isinstance(res_batch, list):
                        res_batch = [res_batch]
                    t_stage2_parse = time.perf_counter()
                    for crop_i, crop_res in enumerate(res_batch):
                        if crop_i >= len(metas):
                            break
                        meta = metas[crop_i]
                        bx1, by1, _, _ = meta["bbox_with_margin"]
                        parsed = self._parse_legacy_crop_result(crop_res)
                        lines_payload = []
                        for quad_local, txt, conf in parsed:
                            if float(conf) < float(self._stage2_min_conf):
                                continue
                            quad_global = [
                                [int(round(qx + bx1)), int(round(qy + by1))]
                                for qx, qy in quad_local
                            ]
                            final_items.append((quad_global, (txt or "").strip(), float(conf)))
                            lines_payload.append(
                                {
                                    "text": (txt or "").strip(),
                                    "conf": float(conf),
                                    "quad_local": [[int(round(qx)), int(round(qy))] for qx, qy in quad_local],
                                    "quad_global": quad_global,
                                }
                            )
                        stage2_payload.append(
                            {
                                "crop_index": int(meta["crop_index"]),
                                "bbox_with_margin": meta["bbox_with_margin"],
                                "lines": lines_payload,
                            }
                        )
                    stage2_parse_ms += (time.perf_counter() - t_stage2_parse) * 1000.0
                    idx += len(chunk)
            except Exception as exc:
                stage2_error = str(exc)
                print(f"[WARN] OCR stage2 batch failed, fallback per-crop: {exc}")
                stage2_payload = []
                final_items = []
                t_stage2_fb = time.perf_counter()
                for crop, meta in zip(crops, crop_meta):
                    bx1, by1, _, _ = meta["bbox_with_margin"]
                    lines_payload = []
                    try:
                        crop_res = self._reader_ocr_compat(
                            crop,
                            det=True,
                            rec=True,
                            cls=kwargs.get("cls", self.config.enable_cls),
                        )
                        parsed = self._parse_legacy_crop_result(crop_res)
                    except Exception:
                        parsed = []
                    for quad_local, txt, conf in parsed:
                        if float(conf) < float(self._stage2_min_conf):
                            continue
                        quad_global = [
                            [int(round(qx + bx1)), int(round(qy + by1))]
                            for qx, qy in quad_local
                        ]
                        final_items.append((quad_global, (txt or "").strip(), float(conf)))
                        lines_payload.append(
                            {
                                "text": (txt or "").strip(),
                                "conf": float(conf),
                                "quad_local": [[int(round(qx)), int(round(qy))] for qx, qy in quad_local],
                                "quad_global": quad_global,
                            }
                        )
                    stage2_payload.append(
                        {
                            "crop_index": int(meta["crop_index"]),
                            "bbox_with_margin": meta["bbox_with_margin"],
                            "lines": lines_payload,
                        }
                    )
                stage2_fallback_ms = (time.perf_counter() - t_stage2_fb) * 1000.0
                self._log_timer("ocr_stage2_fallback_per_crop_ms", t_stage2_fb)
        self._log_timer("ocr_stage2_batch_ms", stage2_started)
        self._log_timer("ocr_crops_ocr_ms", stage2_started)
        stage2_elapsed_ms = (time.perf_counter() - stage2_started) * 1000.0
        if self._timers_enabled:
            print(
                _blue_timer_line(
                    f"[TIMER] ocr_stage2_split_ms call={stage2_call_ms:.1f} parse={stage2_parse_ms:.1f} "
                    f"fallback={stage2_fallback_ms:.1f}"
                ),
                flush=True,
            )

        t_post = time.perf_counter()
        final_items.sort(key=lambda x: float(x[2]), reverse=True)
        final_payload = [
            {"text": t, "conf": float(c), "quad_global": [[int(px), int(py)] for px, py in q]}
            for q, t, c in final_items
        ]
        self._log_timer("ocr_postprocess_ms", t_post)
        post_elapsed_ms = (time.perf_counter() - t_post) * 1000.0
        self._log_timer("ocr_total_ms", t0)
        if self._timers_enabled:
            print(f"[TIMER] ocr_saved_det_count {len(stage1_boxes)}")
            print(f"[TIMER] ocr_saved_crops_count {len(crops)}")

        if self._debug_save and iter_dir is not None:
            t_debug_final = time.perf_counter()
            self._save_crop_debug(iter_dir, crops, crop_meta, stage2_payload, final_payload)
            if stage2_error:
                (iter_dir / "stage2_error.json").write_text(
                    json.dumps({"iteration_id": iteration_id, "error": stage2_error}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            try:
                (iter_dir / "_debug_status.json").write_text(
                    json.dumps(
                        {
                            "iteration_id": iteration_id,
                            "status": "done",
                            "crops_count": len(crops),
                            "results_count": len(final_payload),
                            "stage2_error": stage2_error or "",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"[OCR] crops debug saved: iter_dir={iter_dir} files_written=1", flush=True)
            except Exception:
                pass
            debug_write_final_ms = (time.perf_counter() - t_debug_final) * 1000.0
            self._log_timer("ocr_debug_write_final_ms", t_debug_final)
        if self._timers_enabled:
            total_ms = (time.perf_counter() - t0) * 1000.0
            # Keep stable split log in one line for main console parser.
            print(
                _blue_timer_line(
                    "[TIMER] OCR split "
                    f"det_total={det_elapsed_ms:.1f}ms "
                    f"stage1_dbg_build={stage1_debug_build_ms:.1f}ms "
                    f"stage1_dbg_save={stage1_debug_save_ms:.1f}ms "
                    f"crops_prepare={crop_elapsed_ms:.1f}ms "
                    f"crops_manifest_io={crops_manifest_io_ms:.1f}ms "
                    f"crops_ocr={stage2_elapsed_ms:.1f}ms "
                    f"post={post_elapsed_ms:.1f}ms "
                    f"debug_write_final={debug_write_final_ms:.1f}ms "
                    f"total={total_ms:.1f}ms"
                ),
                flush=True,
            )
        # PaddleOCR legacy style: [ [ [quad, (text, conf)], ... ] ]
        block = []
        for q, t, c in final_items:
            block.append((q, (t, float(c))))
        return [block]

    def ocr(self, image: Any, *args: Any, **kwargs: Any) -> Any:
        iteration_id = self._next_iteration_id(kwargs.pop("iteration_id", None))
        if kwargs.get("det", True) and self._pipeline_mode == "two_stage" and self._stage1_backend == "rapid_det":
            # Only apply two-stage path for single image calls.
            if not isinstance(image, (list, tuple)):
                try:
                    return self._ocr_two_stage(image, iteration_id=iteration_id, **kwargs)
                except Exception as exc:
                    if self.config.fallback_on_fail:
                        print(f"[WARN] two_stage OCR failed, fallback to legacy PaddleOCR: {exc}")
                    else:
                        raise
        if "cls" not in kwargs:
            kwargs["cls"] = self.config.enable_cls
        # Keep compatibility for caller kwargs on heterogeneous PaddleOCR versions.
        if args:
            return self._reader.ocr(image, *args, **kwargs)
        return self._reader_ocr_compat(image, **kwargs)

    def describe(self) -> str:
        return f"backend={self.active_backend} precision={self.active_precision} lang={self.config.lang}"


def create_ocr_runtime(config: Optional[OcrRuntimeConfig] = None, **overrides: Any) -> OcrRuntime:
    cfg = config or OcrRuntimeConfig.from_env()
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    rt = OcrRuntime(cfg)
    print(f"[OCR] runtime ready: {rt.describe()} (requested={cfg.backend})")
    return rt


def benchmark_runtime(runtime: OcrRuntime, images: Iterable[Any]) -> dict[str, Any]:
    count = 0
    lat_ms: list[float] = []
    for img in images:
        start = time.perf_counter()
        runtime.ocr(img, det=True, rec=True, cls=runtime.config.enable_cls)
        lat_ms.append((time.perf_counter() - start) * 1000.0)
        count += 1
    lat_ms.sort()
    p50 = lat_ms[int(len(lat_ms) * 0.5)] if lat_ms else 0.0
    p95 = lat_ms[int(len(lat_ms) * 0.95)] if lat_ms else 0.0
    return {
        "samples": count,
        "backend": runtime.active_backend,
        "precision": runtime.active_precision,
        "latency_ms_p50": round(p50, 2),
        "latency_ms_p95": round(p95, 2),
        "latency_ms_mean": round(sum(lat_ms) / max(1, len(lat_ms)), 2),
    }
