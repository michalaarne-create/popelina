from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple


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
            out = rg.run_dropdown_detection(str(image_path))  # type: ignore[attr-defined]
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
