from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PDX_CACHE = (ROOT.parent / ".runtime" / "paddlex_cache").resolve()
LOCAL_PADDLE_HOME = (ROOT.parent / ".runtime" / "paddle_home").resolve()


def ensure_local_ocr_cache_dirs() -> dict[str, str]:
    LOCAL_PDX_CACHE.mkdir(parents=True, exist_ok=True)
    LOCAL_PADDLE_HOME.mkdir(parents=True, exist_ok=True)
    return {
        "PADDLE_PDX_CACHE_HOME": str(LOCAL_PDX_CACHE),
        "PADDLE_HOME": str(LOCAL_PADDLE_HOME),
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "1",
        "PADDLE_PDX_MODEL_SOURCE": "huggingface",
    }


def normalize_ocr_backend(raw: str | None) -> str:
    token = str(raw or "").strip().lower()
    if token in {"", "auto", "gpu", "cuda", "cuda_fp16", "cuda_fp32"}:
        return "gpu_fp32"
    if token in {"gpu_fp32", "cpu_fp32", "trt_int8"}:
        return token
    return "gpu_fp32"


def backend_prefers_gpu(backend: str) -> bool:
    return normalize_ocr_backend(backend) in {"gpu_fp32", "trt_int8"}


def candidate_python_paths() -> List[Path]:
    candidates = [
        ROOT.parent / ".venv312" / "Scripts" / "python.exe",
        ROOT.parent / ".conda" / "fullbot312" / "python.exe",
        Path(sys.executable),
    ]
    out: List[Path] = []
    seen = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = cand
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def probe_gpu_ocr_runtime(python_path: Path, requested_backend: str) -> Dict[str, Any]:
    backend = normalize_ocr_backend(requested_backend)
    code = r"""
import json, os, sys
from pathlib import Path
repo = Path(sys.argv[1])
backend = sys.argv[2]
lang = sys.argv[3]
sys.path.insert(0, str(repo))
os.environ["FULLBOT_OCR_TEXTLINE_ORIENTATION"] = "0"
os.environ["FULLBOT_OCR_ENABLE_CLS"] = "0"
os.environ["PP_OCR_SHOW_LOG"] = "0"
os.environ["FULLBOT_OCR_MODEL_DIR"] = str((repo.parent / ".runtime" / "paddlex_cache" / "official_models").resolve())
out = {
    "python": sys.executable,
    "backend_requested": backend,
    "backend_used": backend,
    "gpu_ready": False,
    "ocr_ready": False,
    "cuda_compiled": False,
    "device_info": "",
    "error": "",
}
try:
    import paddle  # type: ignore
    out["cuda_compiled"] = bool(paddle.is_compiled_with_cuda())
    if out["cuda_compiled"]:
        try:
            out["device_info"] = str(paddle.device.cuda.get_device_name())
        except Exception:
            out["device_info"] = "cuda"
except Exception as exc:
    out["error"] = f"paddle_import:{exc}"
try:
    from scripts.ocr.runtime.paddle_trt_runtime import OcrRuntimeConfig, create_ocr_runtime
    os.environ["FULLBOT_OCR_BACKEND"] = backend
    os.environ["PADDLEOCR_LANG"] = lang
    cfg = OcrRuntimeConfig.from_env(lang=lang)
    runtime = create_ocr_runtime(cfg)
    out["ocr_ready"] = True
    out["backend_used"] = str(getattr(runtime, "active_backend", backend) or backend)
    out["gpu_ready"] = out["backend_used"] != "cpu_fp32"
except Exception as exc:
    out["error"] = str(exc)
print(json.dumps(out, ensure_ascii=False))
"""
    env = os.environ.copy()
    env.update(ensure_local_ocr_cache_dirs())
    proc = subprocess.run(
        [str(python_path), "-c", code, str(ROOT), backend, "en"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        env=env,
        timeout=90.0,
    )
    result: Dict[str, Any] = {
        "python": str(python_path),
        "backend_requested": backend,
        "backend_used": "cpu_fp32",
        "gpu_ready": False,
        "ocr_ready": False,
        "cuda_compiled": False,
        "device_info": "",
        "error": "",
        "returncode": int(proc.returncode),
    }
    if proc.returncode != 0:
        result["error"] = (proc.stderr or proc.stdout or "").strip()[-500:]
        return result
    last = ""
    for line in (proc.stdout or "").splitlines():
        raw = line.strip()
        if raw.startswith("{") and raw.endswith("}"):
            last = raw
    if not last:
        result["error"] = (proc.stdout or "").strip()[-500:]
        return result
    try:
        parsed = json.loads(last)
    except Exception as exc:
        result["error"] = f"json_decode:{exc}"
        return result
    if isinstance(parsed, dict):
        result.update(parsed)
    return result


def select_ocr_runtime(
    requested_backend: str,
    *,
    require_gpu: bool = False,
    candidates: Iterable[Path] | None = None,
) -> Dict[str, Any]:
    backend = normalize_ocr_backend(requested_backend)
    probes: List[Dict[str, Any]] = []
    best_cpu: Dict[str, Any] | None = None
    for candidate in list(candidates or candidate_python_paths()):
        if not candidate.exists():
            continue
        probe = probe_gpu_ocr_runtime(candidate, backend)
        probes.append(probe)
        if probe.get("gpu_ready") and probe.get("ocr_ready"):
            return {
                "python": str(candidate),
                "backend_requested": requested_backend,
                "backend_used": str(probe.get("backend_used") or backend),
                "gpu_ready": True,
                "ocr_ready": True,
                "device_info": str(probe.get("device_info") or ""),
                "error": "",
                "cache_home": str(LOCAL_PDX_CACHE),
                "probes": probes,
            }
        if probe.get("ocr_ready") and best_cpu is None:
            best_cpu = probe
    if require_gpu:
        return {
            "python": "",
            "backend_requested": requested_backend,
            "backend_used": "cpu_fp32",
            "gpu_ready": False,
            "ocr_ready": False,
            "device_info": "",
            "error": "gpu_runtime_not_ready",
            "cache_home": str(LOCAL_PDX_CACHE),
            "probes": probes,
        }
    fallback_python = ""
    for candidate in list(candidates or candidate_python_paths()):
        if candidate.exists():
            fallback_python = str(candidate)
            break
    if fallback_python:
        return {
            "python": fallback_python,
            "backend_requested": requested_backend,
            "backend_used": backend,
            "gpu_ready": False,
            "ocr_ready": True,
            "device_info": "",
            "error": "runtime_probe_failed_falling_back_to_script",
            "cache_home": str(LOCAL_PDX_CACHE),
            "probes": probes,
        }
    if best_cpu is not None:
        return {
            "python": str(best_cpu.get("python") or ""),
            "backend_requested": requested_backend,
            "backend_used": str(best_cpu.get("backend_used") or "cpu_fp32"),
            "gpu_ready": False,
            "ocr_ready": True,
            "device_info": str(best_cpu.get("device_info") or ""),
            "error": str(best_cpu.get("error") or ""),
            "cache_home": str(LOCAL_PDX_CACHE),
            "probes": probes,
        }
    return {
        "python": "",
        "backend_requested": requested_backend,
        "backend_used": "cpu_fp32",
        "gpu_ready": False,
        "ocr_ready": False,
        "device_info": "",
        "error": "no_ocr_runtime",
        "cache_home": str(LOCAL_PDX_CACHE),
        "probes": probes,
    }
