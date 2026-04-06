from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiz.gpu_ocr_runtime import select_ocr_runtime


def main() -> int:
    ap = argparse.ArgumentParser(description="Check GPU OCR runtime availability for benchmark/dataset builders.")
    ap.add_argument("--ocr-backend", default="cuda_fp16")
    ap.add_argument("--require-gpu", type=int, default=0)
    args = ap.parse_args()

    result = select_ocr_runtime(
        str(args.ocr_backend),
        require_gpu=bool(int(args.require_gpu or 0)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if bool(int(args.require_gpu or 0)) and not bool(result.get("gpu_ready")):
        return 1
    return 0 if bool(result.get("ocr_ready")) else 1


if __name__ == "__main__":
    raise SystemExit(main())

