from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ocr.runtime.paddle_trt_runtime import OcrRuntimeConfig, benchmark_runtime, create_ocr_runtime


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _load_images(images_dir: Path, limit: int) -> List:
    imgs = []
    for path in sorted(images_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        imgs.append(img)
        if limit > 0 and len(imgs) >= limit:
            break
    return imgs


def _artifact_layout(model_root: Path, trt_root: Path) -> None:
    for sub in ("det", "rec", "cls"):
        (model_root / "inference" / sub).mkdir(parents=True, exist_ok=True)
        (trt_root / sub).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/warmup PaddleOCR TensorRT INT8 runtime cache.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Calibration/validation screenshots directory.")
    parser.add_argument("--limit", type=int, default=200, help="Max number of images for warmup/benchmark.")
    parser.add_argument("--lang", type=str, default="pl")
    parser.add_argument("--model-dir", type=Path, default=Path("models/ocr/paddle"))
    parser.add_argument("--trt-cache-dir", type=Path, default=Path("models/ocr/paddle/trt_int8"))
    parser.add_argument("--calib-cache", type=Path, default=Path("models/ocr/paddle/trt_int8/calib_cache.bin"))
    parser.add_argument("--report", type=Path, default=Path("models/ocr/paddle/trt_int8/build_report.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.images_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.trt_cache_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _artifact_layout(args.model_dir, args.trt_cache_dir)

    images = _load_images(args.images_dir, limit=args.limit)
    if not images:
        print(f"[ERR] No images found in {args.images_dir}")
        return 2

    start = time.perf_counter()
    cfg = OcrRuntimeConfig.from_env(lang=args.lang)
    cfg.backend = "trt_int8"
    cfg.model_dir = str(args.model_dir / "inference")
    cfg.trt_cache_dir = str(args.trt_cache_dir)
    cfg.calib_cache = str(args.calib_cache)
    cfg.enable_cls = False
    cfg.fallback_on_fail = True
    cfg.warmup_runs = 2

    runtime = create_ocr_runtime(cfg)
    perf = benchmark_runtime(runtime, images)
    total_s = time.perf_counter() - start

    report = {
        "requested_backend": "trt_int8",
        "active_backend": runtime.active_backend,
        "precision": runtime.active_precision,
        "images": len(images),
        "model_dir": cfg.model_dir,
        "trt_cache_dir": cfg.trt_cache_dir,
        "calib_cache": cfg.calib_cache,
        "build_total_s": round(total_s, 3),
        "perf": perf,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] report saved: {args.report}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
