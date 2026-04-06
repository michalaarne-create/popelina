from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiz.gpu_ocr_runtime import ensure_local_ocr_cache_dirs, normalize_ocr_backend, select_ocr_runtime


def _default_python() -> str:
    preferred = ROOT.parent / ".venv312" / "Scripts" / "python.exe"
    if preferred.exists():
        return str(preferred.resolve())
    return sys.executable


def _low_priority_subprocess_kwargs() -> Dict[str, Any]:
    if os.name != "nt":
        return {}
    idle = getattr(subprocess, "IDLE_PRIORITY_CLASS", 0)
    if idle:
        return {"creationflags": idle}
    return {}


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _resolve_path(base: Path, raw: Any) -> Optional[Path]:
    if not raw:
        return None
    p = Path(str(raw))
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _copy_if_exists(src: Path, dst: Path) -> Optional[Path]:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _global_artifact_paths(root: Path, image_path: Path) -> Tuple[Path, Path, Path]:
    stem = image_path.stem
    global_rg_dir = root / "data" / "screen" / "region_grow" / "region_grow"
    global_rate_dir = root / "data" / "screen" / "rate"
    region_json = global_rg_dir / f"{stem}.json"
    rated_json = global_rate_dir / "rate_results" / f"{stem}_rated.json"
    summary_json = global_rate_dir / "rate_summary" / f"{stem}_summary.json"
    return region_json, rated_json, summary_json


def _copy_cached_global_artifacts(
    *,
    root: Path,
    image_path: Path,
    copied_image: Path,
    copied_region: Path,
    copied_rated: Path,
    copied_summary: Path,
) -> Optional[Tuple[Path, Path, Path, Path]]:
    region_json, rated_json, summary_json = _global_artifact_paths(root, image_path)
    if not region_json.exists() or not rated_json.exists() or not summary_json.exists():
        return None
    copied_image_final = _copy_if_exists(image_path, copied_image)
    copied_region_final = _copy_if_exists(region_json, copied_region)
    copied_rated_final = _copy_if_exists(rated_json, copied_rated)
    copied_summary_final = _copy_if_exists(summary_json, copied_summary)
    if copied_image_final is None or copied_region_final is None or copied_rated_final is None or copied_summary_final is None:
        return None
    return copied_image_final, copied_region_final, copied_rated_final, copied_summary_final


def _artifact_path(root_dir: Path, subdir: str, stem: str, suffix: str) -> Path:
    return root_dir / subdir / f"{stem}{suffix}"


def _print_progress(done: int, total: int, started_at: float, errors: int) -> None:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    elapsed = max(0.001, time.time() - float(started_at))
    rate = float(done) / elapsed
    eta = max(0.0, float(total - done) / max(0.001, rate))
    pct = 100.0 * float(done) / float(total)
    print(f"\r[screen_bench] {done}/{total} {pct:6.2f}% {rate:5.2f} it/s ETA {eta:6.1f}s | errors={errors}", end="", flush=True)
    if done >= total:
        print("", flush=True)


def _load_existing_manifest(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not path.exists():
        return [], []
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return [], []
    rows = payload.get("samples") if isinstance(payload, dict) and isinstance(payload.get("samples"), list) else []
    errors = payload.get("errors") if isinstance(payload, dict) and isinstance(payload.get("errors"), list) else []
    return [r for r in rows if isinstance(r, dict)], [e for e in errors if isinstance(e, dict)]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rg_server_request(port: int, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8", errors="ignore")
    with socket.create_connection(("127.0.0.1", int(port)), timeout=float(timeout_s)) as sock:
        sock.settimeout(float(timeout_s))
        sock.sendall(raw)
        chunks: List[bytes] = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    line = b"".join(chunks).split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
    if not line:
        raise RuntimeError("empty region_grow daemon response")
    data = json.loads(line)
    if not isinstance(data, dict):
        raise RuntimeError("invalid region_grow daemon response")
    return data


def _start_region_grow_server(
    py: str,
    region_grow_script: Path,
    root: Path,
    env: Dict[str, str],
    port: int,
    startup_timeout_s: float = 90.0,
) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [py, str(region_grow_script), "--server", "--port", str(int(port))],
        cwd=str(root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        **_low_priority_subprocess_kwargs(),
    )
    deadline = time.time() + float(startup_timeout_s)
    last_exc: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"region_grow server exited early with code {proc.returncode}")
        try:
            resp = _rg_server_request(int(port), {"cmd": "ping"}, timeout_s=3.0)
            if resp.get("ok") and resp.get("pong"):
                return proc
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5)
    try:
        proc.kill()
    except Exception:
        pass
    raise RuntimeError(f"region_grow server startup timeout on port {port}: {last_exc}")


def _stop_region_grow_server(proc: subprocess.Popen[str], port: int) -> None:
    try:
        _rg_server_request(int(port), {"cmd": "shutdown"}, timeout_s=5.0)
    except Exception:
        pass
    try:
        proc.wait(timeout=5.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _process_row(
    row: Dict[str, Any],
    manifest_path: Path,
    manifest_dir: Path,
    out_dir: Path,
    root: Path,
    py: str,
    region_grow_script: Path,
    rating_script: Path,
    env: Dict[str, str],
    region_grow_timeout_s: float,
    rating_timeout_s: float,
    daemon_port: int = 0,
) -> Dict[str, Any]:
    image_path = _resolve_path(manifest_dir, row.get("image_path"))
    stem = str(row.get("view_id") or (image_path.stem if isinstance(image_path, Path) else "unknown"))
    if image_path is None or not image_path.exists():
        return {"ok": False, "error": {"id": stem, "error": "missing_image"}}

    copied_image = _artifact_path(out_dir, "images", stem, image_path.suffix)
    copied_region = _artifact_path(out_dir, "region", stem, ".json")
    copied_rated = _artifact_path(out_dir, "rated", stem, ".json")
    copied_summary = _artifact_path(out_dir, "summary", stem, ".json")
    if copied_image.exists() and copied_region.exists() and copied_rated.exists() and copied_summary.exists():
        return {
            "ok": True,
            "row": {
                "id": stem,
                "image_path": str(copied_image),
                "region_json": str(copied_region),
                "summary_json": str(copied_summary),
                "rated_json": str(copied_rated),
                "expected_global_type": str(row.get("expected_global_type") or ""),
                "expected_block_types": row.get("expected_block_types") or [],
                "source_manifest": str(manifest_path),
                "source_view_id": str(row.get("view_id") or ""),
                "build_latency_ms": 0.0,
            },
        }
    cached = _copy_cached_global_artifacts(
        root=root,
        image_path=image_path,
        copied_image=copied_image,
        copied_region=copied_region,
        copied_rated=copied_rated,
        copied_summary=copied_summary,
    )
    if cached is not None:
        copied_image_final, copied_region_final, copied_rated_final, copied_summary_final = cached
        return {
            "ok": True,
            "row": {
                "id": stem,
                "image_path": str(copied_image_final),
                "region_json": str(copied_region_final),
                "summary_json": str(copied_summary_final),
                "rated_json": str(copied_rated_final),
                "expected_global_type": str(row.get("expected_global_type") or ""),
                "expected_block_types": row.get("expected_block_types") or [],
                "source_manifest": str(manifest_path),
                "source_view_id": str(row.get("view_id") or ""),
                "build_latency_ms": 0.0,
                "cache_hit": "global_artifacts",
            },
        }

    t0 = time.perf_counter()
    if int(daemon_port or 0) > 0:
        try:
            rg_resp = _rg_server_request(int(daemon_port), {"image_path": str(image_path)}, timeout_s=float(region_grow_timeout_s))
        except Exception as exc:
            return {"ok": False, "error": {"id": stem, "error": "region_grow_daemon_failed", "stderr": str(exc)[-400:]}}
        if not bool(rg_resp.get("ok")):
            return {"ok": False, "error": {"id": stem, "error": "region_grow_daemon_failed", "stderr": str(rg_resp.get("error") or "")[-400:]}}
    else:
        rg_cmd = [py, str(region_grow_script), "--runtime", str(image_path)]
        rg = subprocess.run(
            rg_cmd,
            cwd=str(root),
            env=env,
            timeout=float(region_grow_timeout_s),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_low_priority_subprocess_kwargs(),
        )
        if rg.returncode != 0:
            return {"ok": False, "error": {"id": stem, "error": "region_grow_failed", "returncode": int(rg.returncode), "stderr": (rg.stderr or "")[-400:]}}

    region_json, rated_json, summary_json = _global_artifact_paths(root, image_path)
    if not region_json.exists():
        return {"ok": False, "error": {"id": stem, "error": "missing_region_json"}}

    rt = subprocess.run(
        [py, str(rating_script), str(region_json)],
        cwd=str(root),
        env=env,
        timeout=float(rating_timeout_s),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_low_priority_subprocess_kwargs(),
    )
    if rt.returncode != 0:
        return {"ok": False, "error": {"id": stem, "error": "rating_failed", "returncode": int(rt.returncode), "stderr": (rt.stderr or "")[-400:]}}

    if not rated_json.exists() or not summary_json.exists():
        return {"ok": False, "error": {"id": stem, "error": "missing_rating_artifacts"}}

    copied_image_final = _copy_if_exists(image_path, copied_image)
    copied_region_final = _copy_if_exists(region_json, copied_region)
    copied_rated_final = _copy_if_exists(rated_json, copied_rated)
    copied_summary_final = _copy_if_exists(summary_json, copied_summary)
    if copied_image_final is None or copied_region_final is None or copied_rated_final is None or copied_summary_final is None:
        return {"ok": False, "error": {"id": stem, "error": "copy_artifacts_failed"}}

    return {
        "ok": True,
        "row": {
            "id": stem,
            "image_path": str(copied_image_final),
            "region_json": str(copied_region_final),
            "summary_json": str(copied_summary_final),
            "rated_json": str(copied_rated_final),
            "expected_global_type": str(row.get("expected_global_type") or ""),
            "expected_block_types": row.get("expected_block_types") or [],
            "source_manifest": str(manifest_path),
            "source_view_id": str(row.get("view_id") or ""),
            "build_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "cache_hit": "",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build rendered screen benchmark manifest for quiz_type runtime benchmark.")
    ap.add_argument("--dataset-manifest", required=True, help="Rendered dataset manifest JSON from build_random_www_quiz_dataset.py")
    ap.add_argument("--out-dir", required=True, help="Output directory for copied benchmark artifacts")
    ap.add_argument("--max-samples", type=int, default=0, help="Optional sample cap (0 = all)")
    ap.add_argument("--python", default="", help="Python interpreter for region_grow/rating; defaults to current")
    ap.add_argument("--region-grow-timeout-s", type=float, default=90.0)
    ap.add_argument("--rating-timeout-s", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers for region_grow/rating.")
    ap.add_argument("--resume", type=int, default=1, help="Reuse existing benchmark manifest/artifacts when present.")
    ap.add_argument("--ocr-backend", default="cuda_fp16", help="Requested OCR backend: cuda_fp16/gpu/cpu_fp32/trt_int8.")
    ap.add_argument("--require-gpu", type=int, default=0, help="Fail fast if GPU OCR runtime is not available.")
    ap.add_argument("--exec-mode", default="daemon", help="Execution mode hint for region_grow: daemon|subprocess.")
    args = ap.parse_args()

    manifest_path = Path(args.dataset_manifest).resolve()
    if not manifest_path.exists():
        print(f"[ERROR] dataset manifest not found: {manifest_path}")
        return 2
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = manifest_path.parent.parent
    rows = _load_manifest(manifest_path)
    if args.max_samples > 0:
        rows = rows[: int(args.max_samples)]
    if not rows:
        print(f"[ERROR] no rows in manifest: {manifest_path}")
        return 2

    requested_backend = normalize_ocr_backend(str(args.ocr_backend or "cuda_fp16"))
    runtime = select_ocr_runtime(
        requested_backend,
        require_gpu=bool(int(args.require_gpu or 0)),
        candidates=[Path(str(Path(args.python).resolve()))] if str(args.python or "").strip() else None,
    )
    if not bool(runtime.get("ocr_ready")):
        print(json.dumps({"error": "ocr_runtime_not_ready", "runtime": runtime}, ensure_ascii=False, indent=2))
        return 2
    if bool(int(args.require_gpu or 0)) and not bool(runtime.get("gpu_ready")):
        print(json.dumps({"error": "gpu_required_but_unavailable", "runtime": runtime}, ensure_ascii=False, indent=2))
        return 2
    py = str(runtime.get("python") or _default_python())
    region_grow_script = ROOT / "scripts" / "region_grow" / "region_grow" / "region_grow.py"
    rating_script = ROOT / "scripts" / "region_grow" / "numpy_rate" / "rating.py"
    if not region_grow_script.exists() or not rating_script.exists():
        print("[ERROR] region_grow or rating script missing.")
        return 2

    env = os.environ.copy()
    env.update(ensure_local_ocr_cache_dirs())
    env.setdefault("FULLBOT_TURBO_MODE", "1")
    env["FULLBOT_REGION_GROW_EXEC_MODE"] = str(args.exec_mode or "daemon").strip().lower() or "daemon"
    env.setdefault("FULLBOT_REGION_GROW_STREAM_LOGS", "0")
    env.setdefault("FULLBOT_OCR_BOXES_DEBUG", "0")
    env.setdefault("FULLBOT_REGION_GROW_ANNOTATE", "0")
    env.setdefault("FULLBOT_REGION_GROW_ANNOTATE_INLINE", "0")
    env.setdefault("FULLBOT_RUN_RATING_FAST", "1")
    env.setdefault("FULLBOT_RATING_FORCE_SUBPROCESS", "1")
    env["FULLBOT_OCR_BACKEND"] = str(runtime.get("backend_used") or requested_backend)
    env.setdefault("FULLBOT_OCR_MODEL_DIR", str((ROOT.parent / ".runtime" / "paddlex_cache" / "official_models").resolve()))
    env.setdefault("PADDLEOCR_LANG", "en")
    env.setdefault("FULLBOT_OCR_TEXTLINE_ORIENTATION", "0")
    env.setdefault("FULLBOT_OCR_ENABLE_CLS", "0")
    env.setdefault("PP_OCR_SHOW_LOG", "0")

    out_manifest = out_dir / "benchmark_manifest.json"
    benchmark_rows, errors = _load_existing_manifest(out_manifest) if int(args.resume or 0) else ([], [])
    existing_ok = {
        str(r.get("source_view_id") or r.get("id") or "").strip(): r
        for r in benchmark_rows
        if isinstance(r, dict)
    }
    pending: List[Dict[str, Any]] = []
    for row in rows:
        view_id = str(row.get("view_id") or "").strip()
        if view_id and view_id in existing_ok:
            continue
        pending.append(row)

    if pending and errors:
        pending_ids = {str(r.get("view_id") or "").strip() for r in pending}
        errors = [e for e in errors if str(e.get("id") or "").strip() not in pending_ids]

    started_at = time.time()
    total = len(rows)
    done = len(benchmark_rows) + len(errors)
    _print_progress(done, total, started_at, len(errors))
    workers = max(1, int(args.workers or 1))
    daemon_specs: List[Tuple[int, subprocess.Popen[str]]] = []
    use_daemon = str(args.exec_mode or "daemon").strip().lower() == "daemon"
    if pending and use_daemon:
        for _ in range(workers):
            port = _find_free_port()
            proc = _start_region_grow_server(py, region_grow_script, ROOT, env, port, startup_timeout_s=min(120.0, max(30.0, float(args.region_grow_timeout_s))))
            daemon_specs.append((port, proc))
    if pending:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = []
                for idx, row in enumerate(pending):
                    daemon_port = daemon_specs[idx % len(daemon_specs)][0] if daemon_specs else 0
                    futures.append(
                        ex.submit(
                            _process_row,
                            row,
                            manifest_path,
                            manifest_dir,
                            out_dir,
                            ROOT,
                            py,
                            region_grow_script,
                            rating_script,
                            env,
                            float(args.region_grow_timeout_s),
                            float(args.rating_timeout_s),
                            int(daemon_port),
                        )
                    )
                for fut in concurrent.futures.as_completed(futures):
                    result = fut.result()
                    if result.get("ok"):
                        benchmark_rows.append(result["row"])
                    else:
                        errors.append(result["error"])
                    done += 1
                    _print_progress(done, total, started_at, len(errors))
        finally:
            for port, proc in daemon_specs:
                _stop_region_grow_server(proc, port)

    benchmark_rows.sort(key=lambda r: str(r.get("id") or ""))
    errors.sort(key=lambda r: str(r.get("id") or ""))
    out_report = {
        "samples": benchmark_rows,
        "errors": errors,
        "created_at_unix": int(time.time()),
        "source_manifest": str(manifest_path),
        "samples_ok": len(benchmark_rows),
        "samples_total": len(rows),
        "cache_hits": int(sum(1 for r in benchmark_rows if str(r.get("cache_hit") or "").strip())),
        "ocr_backend_requested": str(args.ocr_backend),
        "ocr_backend_used": str(runtime.get("backend_used") or requested_backend),
        "python_used": py,
        "gpu_ready": bool(runtime.get("gpu_ready")),
        "gpu_required": bool(int(args.require_gpu or 0)),
        "device_info": str(runtime.get("device_info") or ""),
        "exec_mode": str(args.exec_mode or "daemon"),
        "daemon_count": len(daemon_specs),
        "low_priority_processes": True,
        "cache_home": str(runtime.get("cache_home") or env.get("PADDLE_PDX_CACHE_HOME") or ""),
        "runtime_probe": runtime,
    }
    out_manifest.write_text(json.dumps(out_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[screen_bench] manifest={out_manifest}")
    print(f"[screen_bench] ok={len(benchmark_rows)} total={len(rows)} errors={len(errors)}")
    return 0 if benchmark_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
