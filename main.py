#!/usr/bin/env python3
"""
Main orchestrator for the live dropdown pipeline.

It launches the Chrome recorder (ai_recorder_live) once and then, every
PIPELINE_INTERVAL seconds, captures the full screen, runs scripts/region_grow/region_grow/region_grow.py
on the screenshot (region_grow already performs OCR internally) and finally
feeds the JSON emitted by region_grow into scripts/region_grow/numpy_rate/rating.py.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import os
import queue
import re
import concurrent.futures
import subprocess
import sys
import threading
import time
import math
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.brain.pipeline_brain_agent import BrainDecision, PipelineBrainAgent
from scripts.click.hover import hover_runtime as _hover_runtime
from scripts.debuggers.files_reader import collect_and_dispatch_to_brain as _collect_and_dispatch_to_brain
from scripts.overlay.console_overlay import colorize_log_message, init_console_overlay
from scripts.pipeline.cli_args import parse_args as _parse_args_mod
from scripts.pipeline.hotkeys import (
    start_hotkey_listener as _start_hotkey_listener_mod,
    wait_for_p_in_console as _wait_for_p_in_console_mod,
)
from scripts.pipeline.iteration_orchestrator import run_iteration as _run_iteration_orchestrator
from scripts.pipeline.loop_controller import run_loop as _run_loop_controller
from scripts.pipeline.manual_commands import (
    drain_manual_commands as _drain_manual_commands_mod,
    handle_manual_command as _handle_manual_command_mod,
)
from scripts.pipeline.pipeline import pipeline_iteration as _pipeline_iteration_external
from scripts.pipeline.process_control import (
    ensure_control_agent as _ensure_control_agent_mod,
    start_ai_recorder as _start_ai_recorder_mod,
    stop_process as _stop_process_mod,
)
from scripts.pipeline.runtime_capture import (
    capture_fullscreen as _capture_fullscreen_mod,
    prepare_hover_image as _prepare_hover_image_mod,
)
from scripts.pipeline.runtime_clicks import (
    scroll_on_box as _scroll_on_box_mod,
    send_click_from_bbox as _send_click_from_bbox_mod,
)
from scripts.pipeline.runtime_click_summaries import (
    find_screenshot_for_summary as _find_screenshot_for_summary_mod,
    send_best_click as _send_best_click_mod,
    send_random_click as _send_random_click_mod,
)
from scripts.pipeline.runtime_region_rating import (
    latest_file as _latest_file_mod,
    run_arrow_post as _run_arrow_post_mod,
    run_rating as _run_rating_mod,
    run_region_grow as _run_region_grow_mod,
)
from scripts.pipeline.runtime_control_agent import (
    send_control_agent as _send_control_agent_mod,
    send_udp_payload as _send_udp_payload_mod,
)
from scripts.pipeline.runtime_hover_dispatch import (
    dispatch_hover_to_control_agent as _dispatch_hover_to_control_agent_mod,
)


def _load_register_main_launch():
    repo_root = Path(__file__).resolve().parent
    mod_path = repo_root / "github" / "git pusher" / "launch_git_automation.py"
    if not mod_path.exists():
        raise ModuleNotFoundError(f"Missing launch_git_automation at {mod_path}")
    spec = importlib.util.spec_from_file_location("launch_git_automation_dynamic", str(mod_path))
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Cannot load module spec from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "register_main_launch", None)
    if not callable(fn):
        raise ModuleNotFoundError(f"register_main_launch not found in {mod_path}")
    return fn


register_main_launch = _load_register_main_launch()

ROOT = Path(__file__).resolve().parent
DATA_SCREEN_DIR = ROOT / "data" / "screen"
HOVER_DIR = ROOT / "data" / "screen" / "hover"
RAW_DIR = ROOT / "data" / "screen" / "raw"
AI_RECORDER_SCRIPT = ROOT / "scripts" / "dom" / "dom_renderer" / "ai_recorder_live.py"
REGION_GROW_SCRIPT = ROOT / "scripts" / "region_grow" / "region_grow" / "region_grow.py"
RATING_SCRIPT = ROOT / "scripts" / "region_grow" / "numpy_rate" / "rating.py"
ARROW_POST_SCRIPT = ROOT / "scripts" / "arrow_post_region.py"
HOVER_SINGLE_SCRIPT = ROOT / "scripts" / "click" / "hover" / "hover_single.py"
CONTROL_AGENT_SCRIPT = ROOT / "scripts" / "click" / "control_agent" / "control_agent.py"
SCREENSHOT_DIR = ROOT / "data" / "screen" / "raw" / "raw screen"
SCREEN_BOXES_DIR = ROOT / "data" / "screen" / "numpy_points" / "screen_boxes"
DOM_LIVE_DIR = ROOT / "scripts" / "dom" / "dom_live"
CURRENT_QUESTION_PATH = ROOT / "scripts" / "dom" / "dom_live" / "current_question.json"
CURRENT_RUN_DIR = ROOT / "data" / "screen" / "current_run"
REGION_GROW_BASE_DIR = ROOT / "data" / "screen" / "region_grow"
REGION_GROW_JSON_DIR = ROOT / "data" / "screen" / "region_grow" / "region_grow"
REGION_GROW_ANNOT_DIR = ROOT / "data" / "screen" / "region_grow" / "region_grow_annot"
REGION_GROW_ANNOT_CURRENT_DIR = ROOT / "data" / "screen" / "region_grow" / "region_grow_annot_current"
REGION_GROW_CURRENT_DIR = ROOT / "data" / "screen" / "region_grow" / "region_grow_current"
HOVER_INPUT_DIR = ROOT / "data" / "screen" / "hover" / "hover_input"
HOVER_OUTPUT_DIR = ROOT / "data" / "screen" / "hover" / "hover_output"
HOVER_INPUT_CURRENT_DIR = ROOT / "data" / "screen" / "hover" / "hover_input_current"
HOVER_OUTPUT_CURRENT_DIR = ROOT / "data" / "screen" / "hover" / "hover_output_current"
RAW_CURRENT_DIR = ROOT / "data" / "screen" / "raw" / "raw_screens_current"
HOVER_PATH_DIR = ROOT / "data" / "screen" / "hover" / "hover_path"
HOVER_PATH_CURRENT_DIR = ROOT / "data" / "screen" / "hover" / "hover_path_current"
HOVER_SPEED_DIR = ROOT / "data" / "screen" / "hover" / "hover_speed"
HOVER_SPEED_CURRENT_DIR = ROOT / "data" / "screen" / "hover" / "hover_speed_current"
HOVER_SPEED_RECORDER_SCRIPT = ROOT / "scripts" / "click" / "recorder" / "hover_speed_recorder.py"
HOVER_SIDE_CROP = 300
HOVER_TOP_CROP = int(os.environ.get("HOVER_TOP_CROP", "120"))
CONTROL_AGENT_PORT = int(os.environ.get("CONTROL_AGENT_PORT", "8765"))
HOVER_FALLBACK_SECONDS = float(os.environ.get("HOVER_FALLBACK_SECONDS", "10"))
REGION_GROW_TIMEOUT = float(os.environ.get("REGION_GROW_TIMEOUT", "45"))
HOVER_SPACING_X = float(os.environ.get("HOVER_SPACING_X", "1.0"))
RATE_RESULTS_DIR = ROOT / "data" / "screen" / "rate" / "rate_results"
RATE_RESULTS_DEBUG_DIR = ROOT / "data" / "screen" / "rate" / "rate_results_debug"
RATE_SUMMARY_DIR = ROOT / "data" / "screen" / "rate" / "rate_summary"
RATE_RESULTS_CURRENT_DIR = ROOT / "data" / "screen" / "rate" / "rate_results_current"
RATE_RESULTS_DEBUG_CURRENT_DIR = ROOT / "data" / "screen" / "rate" / "rate_results_debug_current"
RATE_SUMMARY_CURRENT_DIR = ROOT / "data" / "screen" / "rate" / "rate_summary_current"
BRAIN_STATE_FILE = ROOT / "data" / "brain_state.json"
# Global timer switch (1=ON, 0=OFF). Controls all "[TIMER]" logs.





ENABLE_TIMERS = str(os.environ.get("FULLBOT_ENABLE_TIMERS", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}

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
try:
    from scripts.click.hover import send_hover_path as _shp_defaults  # type: ignore

    HOVER_SPEED_DEFAULTS = {
        "speed": getattr(_shp_defaults, "DEFAULT_SPEED_MODE", "normal"),
        "speed_factor": float(getattr(_shp_defaults, "DEFAULT_SPEED_FACTOR", 1.0)),
        "min_dt": float(getattr(_shp_defaults, "DEFAULT_MIN_DT", 0.004)),
        "speed_px_per_s": float(getattr(_shp_defaults, "DEFAULT_SPEED_PX_PER_S", 740.0)),
        "gap_boost": float(getattr(_shp_defaults, "DEFAULT_GAP_BOOST", 1.4)),
        "line_jump_boost": float(getattr(_shp_defaults, "DEFAULT_LINE_JUMP_BOOST", 1.8)),
    }
except Exception:
    _shp_defaults = None
    HOVER_SPEED_DEFAULTS = {
        "speed": "normal",
        "speed_factor": 1.0,
        "min_dt": 0.004,
        "speed_px_per_s": 740.0,
        "gap_boost": 1.4,
        "line_jump_boost": 1.8,
    }
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
_JSON_EXTS = (".json",)
_turbo_default = str(os.environ.get("FULLBOT_TURBO_MODE", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
REGION_RESIZE_MAX_SIDE = int(os.environ.get("REGION_RESIZE_MAX_SIDE", "960" if _turbo_default else "1280"))

CREATE_NO_WINDOW = 0
if os.name == "nt":
    CREATE_NO_WINDOW = 0x08000000
SUBPROCESS_KW = {"creationflags": CREATE_NO_WINDOW} if os.name == "nt" else {}
_hover_fallback_timer: Optional[threading.Timer] = None
_hover_fallback_allowed = False
HOLD_LEFT_BUTTON = False
DEBUG_MODE = False
_hover_reader_cache = None
_region_grow_module = None
_ocr_warmed = False
_mss_singleton: Optional[object] = None
_rating_module = None


_SUPPRESSED_PADDLE_LOG_SNIPPETS = (
    "Checking connectivity to the model hosters",
    "Connectivity check to the model hoster has been skipped",
    "Model files already exist. Using cached files.",
    "Creating model:",
    "No ccache found",
    "Logging before InitGoogleLogging()",
    "gpu_resources.cc:",
)

# Silence native Paddle/GLOG noise as early as possible.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("FLAGS_minlog_level", "3")
os.environ.setdefault("FLAGS_logtostderr", "0")
os.environ.setdefault("FULLBOT_TURBO_MODE", "1")
os.environ.setdefault("REGION_GROW_TURBO", "1")
os.environ.setdefault("RATING_MODE", "off")
os.environ.setdefault("FULLBOT_REGION_RATING_BUDGET_MS", "1000")
os.environ.setdefault("REGION_GROW_MAX_SIDE_TURBO", "960")
os.environ.setdefault("REGION_GROW_MAX_DETECTIONS_TURBO", "60")
os.environ.setdefault("FULLBOT_OCR_BOXES_DEBUG", "1")


class _FilterLineWriter(io.TextIOBase):
    """Forward stream writes but drop known noisy Paddle init lines."""

    def __init__(self, target: Any, suppressed_snippets: Tuple[str, ...]) -> None:
        super().__init__()
        self._target = target
        self._suppressed = tuple(str(s) for s in suppressed_snippets if str(s))
        self._buf = ""

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        text = str(s or "")
        if not text:
            return 0
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            full = line + "\n"
            plain = re.sub(r"\x1b\[[0-9;]*m", "", full)
            if any(sn in plain for sn in self._suppressed):
                continue
            with contextlib.suppress(Exception):
                self._target.write(full)
        return len(text)

    def flush(self) -> None:
        if self._buf:
            full = self._buf
            self._buf = ""
            plain = re.sub(r"\x1b\[[0-9;]*m", "", full)
            if not any(sn in plain for sn in self._suppressed):
                with contextlib.suppress(Exception):
                    self._target.write(full)
        with contextlib.suppress(Exception):
            self._target.flush()


@contextlib.contextmanager
def _suppress_paddle_init_noise():
    out = _FilterLineWriter(sys.stdout, _SUPPRESSED_PADDLE_LOG_SNIPPETS)
    err = _FilterLineWriter(sys.stderr, _SUPPRESSED_PADDLE_LOG_SNIPPETS)
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield
    out.flush()
    err.flush()


def _set_hover_reader_cache(reader: Any) -> None:
    global _hover_reader_cache
    _hover_reader_cache = reader
    try:
        from scripts.click.hover import hover_bot as hb  # type: ignore

        setattr(hb, "_CACHED_READER", reader)
    except Exception:
        pass


def _get_shared_ocr_reader() -> Any:
    # Prefer region_grow OCR as the single shared Paddle reader.
    rg = _get_region_grow_module()
    if rg is not None and hasattr(rg, "get_ocr"):
        with _suppress_paddle_init_noise():
            return rg.get_ocr()  # type: ignore[attr-defined]
    raise RuntimeError("Shared OCR reader unavailable from region_grow.get_ocr()")


def preload_hover_reader():
    global _hover_reader_cache
    if _hover_reader_cache is not None:
        return
    try:
        reader = None
        with contextlib.suppress(Exception):
            reader = _get_shared_ocr_reader()
        if reader is not None:
            _set_hover_reader_cache(reader)
            log("[INFO] Preloaded shared OCR reader (region_grow -> hover).")
            return
        with _suppress_paddle_init_noise():
            from scripts.click.hover import hover_bot as hb  # type: ignore
            _set_hover_reader_cache(hb.create_paddleocr_reader(lang=hb.OCR_LANG))
        log("[INFO] Preloaded hover_bot OCR reader.")
    except Exception as exc:
        log(f"[WARN] Could not preload hover_bot reader: {exc}")


def _get_region_grow_module():
    """
    Import scripts.region_grow.region_grow once per process with the same fast settings
    that we used for the old subprocess-based call.
    """
    global _region_grow_module
    if _region_grow_module is not None:
        return _region_grow_module

    # Ustaw domyślne flagi środowiskowe tak jak dla wywołania CLI.
    os.environ.setdefault("RG_FAST", "1")
    os.environ.setdefault("RAPID_OCR_MAX_SIDE", "800")
    os.environ.setdefault("RAPID_OCR_AUTOCROP", "1")
    os.environ.setdefault("RAPID_AUTOCROP_KEEP_RATIO", "0.9")
    os.environ.setdefault("RAPID_OCR_AUTOCROP_DELTA", "8")
    os.environ.setdefault("RAPID_OCR_AUTOCROP_PAD", "8")
    # GPU floodfill/cupy required.
    os.environ.setdefault("REGION_GROW_USE_GPU", "1")
    os.environ.setdefault("REGION_GROW_REQUIRE_GPU", "0")

    try:
        with _suppress_paddle_init_noise():
            from scripts.region_grow.region_grow import region_grow as rg  # type: ignore
    except Exception as exc:
        log(f"[WARN] Inline region_grow import failed, falling back to subprocess: {exc}")
        _region_grow_module = None
        return None

    _region_grow_module = rg
    return rg


def _get_rating_module():
    """
    Import scripts.region_grow.numpy_rate.rating raz na proces, tak aby rating
    by‘' wykonywany inline (bez dodatkowego procesu Pythona).
    """
    global _rating_module
    if _rating_module is not None:
        return _rating_module
    try:
        from scripts.region_grow.numpy_rate import rating as rt  # type: ignore
    except Exception as exc:
        log(f"[WARN] Inline rating import failed, falling back to subprocess: {exc}")
        _rating_module = None
        return None
    _rating_module = rt
    return rt


def warm_ocr_once():
    """
    Preload all OCR models (hover + region_grow) and keep them in VRAM
    for subsequent iterations. Idempotent.
    """
    global _ocr_warmed
    if _ocr_warmed:
        return
    t_ocr_total = time.perf_counter()

    # Single shared OCR warmup used by both region_grow and hover.
    try:
        t_shared_ocr = time.perf_counter()
        reader = _get_shared_ocr_reader()
        _set_hover_reader_cache(reader)
        log(f"[TIMER] shared_ocr_warm {time.perf_counter() - t_shared_ocr:.3f}s")
        log("[INFO] Preloaded shared OCR (region_grow + hover).")
    except Exception as exc:
        log(f"[WARN] Could not preload shared OCR: {exc}")
        # Fallback: keep legacy hover preload if shared path is unavailable.
        with contextlib.suppress(Exception):
            t_hover_ocr = time.perf_counter()
            preload_hover_reader()
            log(f"[TIMER] hover_ocr_warm {time.perf_counter() - t_hover_ocr:.3f}s")

    log(f"[TIMER] ocr_warm_total {time.perf_counter() - t_ocr_total:.3f}s")
    _ocr_warmed = True

def update_overlay_status(message: str):
    if not message:
        return
    log(f"[INFO] {str(message)}")
def cancel_hover_fallback_timer():
    global _hover_fallback_timer
    if _hover_fallback_timer is not None:
        _hover_fallback_timer.cancel()
        _hover_fallback_timer = None


def start_hover_fallback_timer():
    global _hover_fallback_allowed
    if HOVER_FALLBACK_SECONDS <= 0:
        return
    if not _hover_fallback_allowed:
        return
    cancel_hover_fallback_timer()

    def _timeout():
        msg = f"Hover fallback! no response for {HOVER_FALLBACK_SECONDS:.0f}s. Stopping."
        log(f"[ERROR] {msg}")
        update_overlay_status(msg)
        os._exit(3)

    timer = threading.Timer(HOVER_FALLBACK_SECONDS, _timeout)
    timer.daemon = True
    timer.start()
    globals()["_hover_fallback_timer"] = timer
    update_overlay_status(f"Hover path sent. Awaiting response ({HOVER_FALLBACK_SECONDS:.0f}s timeout).")
    _hover_fallback_allowed = False


def log(message: str) -> None:
    if ("[TIMER]" in str(message)) and (not ENABLE_TIMERS):
        return
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[main {ts}] {str(message)}"
    print(colorize_log_message(line), flush=True)


def debug(message: str) -> None:
    if DEBUG_MODE:
        log(f"[DEBUG] {message}")


BRAIN_AGENT = PipelineBrainAgent(
    question_path=CURRENT_QUESTION_PATH,
    state_path=BRAIN_STATE_FILE,
    logger=log,
)


def _debug_hover_output_current() -> None:
    """
    Lightweight debug helper for hover_output_current state.
    Używane tylko tymczasowo do diagnozy, czy pliki się nadpisują.
    """
    try:
        json_path = HOVER_OUTPUT_CURRENT_DIR / "hover_output.json"
        png_path = HOVER_OUTPUT_CURRENT_DIR / "hover_output.png"
        json_info = (
            f"exists={json_path.exists()} "
            f"size={(json_path.stat().st_size if json_path.exists() else 0)}"
        )
        png_info = (
            f"exists={png_path.exists()} "
            f"size={(png_path.stat().st_size if png_path.exists() else 0)}"
        )
        debug(f"hover_output_current json {json_info} | png {png_info}")
    except Exception as exc:
        debug(f"hover_output_current debug failed: {exc}")


def _write_current_artifact(src: Path, dest_dir: Path, dest_name: Optional[str] = None) -> Optional[Path]:
    """
    Copy a file into a "current" directory, replacing previous content.
    Returns the destination path or None if the source is missing.
    """
    if src is None or not src.exists():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / (dest_name or src.name)
    if target.exists() and target.is_file():
        with contextlib.suppress(Exception):
            target.unlink()
    shutil.copy2(src, target)
    return target


def _archive_artifact(src: Path, dest_dir: Path, dest_name: Optional[str] = None) -> Optional[Path]:
    """
    Copy a file into an archive/run directory (non-destructive).
    """
    if src is None or not src.exists():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / (dest_name or src.name)
    shutil.copy2(src, target)
    return target


def capture_fullscreen(target: Path) -> Path:
    def _get_mss():
        return _mss_singleton

    def _set_mss(v: object):
        global _mss_singleton
        _mss_singleton = v

    return _capture_fullscreen_mod(
        target,
        mss_singleton_get=_get_mss,
        mss_singleton_set=_set_mss,
        log=log,
    )


def _downscale_for_region(src: Path) -> Path:
    if REGION_RESIZE_MAX_SIDE <= 0:
        return src
    try:
        with Image.open(src) as im:
            w, h = im.size
            side = max(w, h)
            if side <= REGION_RESIZE_MAX_SIDE:
                return src
            scale = REGION_RESIZE_MAX_SIDE / float(side)
            nw, nh = int(w * scale), int(h * scale)
            resized = im.resize((nw, nh), Image.LANCZOS)
            # Keep raw screenshot directory clean: store transient region image in current_run.
            CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
            tmp_path = CURRENT_RUN_DIR / f"{src.stem}_rg_small{src.suffix}"
            resized.save(tmp_path)
            debug(f"region resize: {w}x{h} -> {nw}x{nh} at {tmp_path}")
            return tmp_path
    except Exception as exc:
        debug(f"region resize failed: {exc}")
        return src


def prepare_hover_image(full_image: Path) -> Optional[Path]:
    return _prepare_hover_image_mod(
        full_image,
        raw_current_dir=RAW_CURRENT_DIR,
        hover_top_crop=HOVER_TOP_CROP,
        hover_input_dir=HOVER_INPUT_DIR,
        hover_input_current_dir=HOVER_INPUT_CURRENT_DIR,
        write_current_artifact=_write_current_artifact,
        debug=debug,
        log=log,
    )


def run_region_grow(image_path: Path) -> Optional[Path]:
    return _run_region_grow_mod(
        image_path,
        get_region_grow_module=_get_region_grow_module,
        data_screen_dir=DATA_SCREEN_DIR,
        region_grow_current_dir=REGION_GROW_CURRENT_DIR,
        region_grow_annot_dir=REGION_GROW_ANNOT_DIR,
        region_grow_json_dir=REGION_GROW_JSON_DIR,
        screen_boxes_dir=SCREEN_BOXES_DIR,
        region_grow_script=REGION_GROW_SCRIPT,
        root=ROOT,
        region_grow_timeout=REGION_GROW_TIMEOUT,
        debug_mode=DEBUG_MODE,
        debug=debug,
        log=log,
        write_current_artifact=_write_current_artifact,
        subprocess_kw=SUBPROCESS_KW,
        sys_executable=sys.executable,
    )


def run_arrow_post(json_path: Path) -> None:
    _run_arrow_post_mod(json_path, log=log)


def run_rating(json_path: Path) -> bool:
    return _run_rating_mod(
        json_path,
        get_rating_module=_get_rating_module,
        rating_script=RATING_SCRIPT,
        root=ROOT,
        subprocess_kw=SUBPROCESS_KW,
        sys_executable=sys.executable,
        log=log,
    )


def _legacy_build_hover_from_region_results(json_path: Path) -> Optional[Path]:
    return build_hover_from_region_results(json_path)


def build_hover_from_region_results(json_path: Path) -> Optional[Path]:
    return _hover_runtime.build_hover_from_region_results(
        json_path,
        screenshot_dir=SCREENSHOT_DIR,
        raw_current_dir=RAW_CURRENT_DIR,
        hover_output_dir=HOVER_OUTPUT_DIR,
        hover_output_current_dir=HOVER_OUTPUT_CURRENT_DIR,
        write_current_artifact=_write_current_artifact,
        debug_hover_output_current=_debug_hover_output_current,
        debug=debug,
        log=log,
    )


def run_hover_bot(
    hover_image: Path,
    base_name: str,
    start_time: Optional[float] = None,
) -> Optional[Tuple[subprocess.Popen, Path, Path]]:
    def _cache_get():
        return _hover_reader_cache

    def _cache_set(v: Any):
        global _hover_reader_cache
        _hover_reader_cache = v

    return _hover_runtime.run_hover_bot(
        hover_image,
        base_name,
        start_time=start_time,
        hover_output_dir=HOVER_OUTPUT_DIR,
        hover_output_current_dir=HOVER_OUTPUT_CURRENT_DIR,
        screenshot_dir=SCREENSHOT_DIR,
        root=ROOT,
        subprocess_kw=SUBPROCESS_KW,
        hover_single_script=HOVER_SINGLE_SCRIPT,
        write_current_artifact=_write_current_artifact,
        debug_hover_output_current=_debug_hover_output_current,
        dispatch_hover_to_control_agent=dispatch_hover_to_control_agent,
        update_overlay_status=update_overlay_status,
        debug=debug,
        log=log,
        hover_reader_cache_get=_cache_get,
        hover_reader_cache_set=_cache_set,
    )


def _send_click_from_bbox(
    bbox: List[float],
    image_path: Optional[Path],
    context_label: str,
) -> bool:
    return _send_click_from_bbox_mod(
        bbox,
        image_path,
        context_label,
        send_control_agent=_send_control_agent,
        control_agent_port=CONTROL_AGENT_PORT,
        log=log,
        update_overlay_status=update_overlay_status,
    )


def scroll_on_box(
    bbox: List[float],
    image_path: Optional[Path],
    context_label: str,
    *,
    total_notches: int = 8,
    direction: str = "down",
) -> bool:
    return _scroll_on_box_mod(
        bbox,
        image_path,
        context_label,
        send_control_agent=_send_control_agent,
        control_agent_port=CONTROL_AGENT_PORT,
        log=log,
        total_notches=total_notches,
        direction=direction,
    )


def _hover_rect(box: List[List[float]]) -> Tuple[int, int, int, int]:
    return _hover_runtime.hover_rect(box)


def _inside_any(x: float, y: float, rects: List[Tuple[int, int, int, int]]) -> bool:
    return _hover_runtime.inside_any(x, y, rects)


def _group_hover_lines(seqs: List[Dict[str, Any]]) -> List[List[int]]:
    return _hover_runtime.group_hover_lines(seqs)


def _build_hover_path(
    seqs: List[Dict[str, Any]],
    offset_x: int,
    offset_y: int,
) -> Optional[Dict[str, Any]]:
    return _hover_runtime.build_hover_path(
        seqs,
        offset_x,
        offset_y,
        brain_agent=BRAIN_AGENT,
        hover_speed_defaults=HOVER_SPEED_DEFAULTS,
    )

def _send_udp_payload(payload: Dict[str, Any], port: int) -> bool:
    return _send_udp_payload_mod(payload, port, log=log)


def _control_agent_running() -> bool:
    target = "control_agent.py"
    try:
        import psutil  # type: ignore
    except Exception:
        return False
    for proc in psutil.process_iter(["cmdline"]):
        cmdline = " ".join(proc.info.get("cmdline") or [])
        if target in cmdline:
            return True
    return False


def _save_hover_path_visual(points: List[Dict[str, int]], points_json: Path) -> None:
    _hover_runtime.save_hover_path_visual(
        points,
        points_json,
        screenshot_dir=SCREENSHOT_DIR,
        hover_input_current_dir=HOVER_INPUT_CURRENT_DIR,
        raw_current_dir=RAW_CURRENT_DIR,
        hover_path_dir=HOVER_PATH_DIR,
        hover_path_current_dir=HOVER_PATH_CURRENT_DIR,
        debug=debug,
        log=log,
    )


def _save_hover_overlay_from_json(points_json: Path) -> None:
    _hover_runtime.save_hover_overlay_from_json(
        points_json,
        region_grow_annot_current_dir=REGION_GROW_ANNOT_CURRENT_DIR,
        raw_current_dir=RAW_CURRENT_DIR,
        screenshot_dir=SCREENSHOT_DIR,
        hover_output_dir=HOVER_OUTPUT_DIR,
        hover_output_current_dir=HOVER_OUTPUT_CURRENT_DIR,
        write_current_artifact=_write_current_artifact,
        debug_hover_output_current=_debug_hover_output_current,
        debug=debug,
    )


def ensure_control_agent(port: int) -> Optional[subprocess.Popen]:
    return _ensure_control_agent_mod(
        control_agent_running=_control_agent_running,
        control_agent_script=CONTROL_AGENT_SCRIPT,
        control_agent_port=int(port),
        root=ROOT,
        subprocess_kw=SUBPROCESS_KW,
        log=log,
    )


def _send_control_agent(payload: Dict[str, Any], port: int) -> bool:
    return _send_control_agent_mod(payload, port, log=log)


def dispatch_hover_to_control_agent(points_json: Path) -> None:
    _dispatch_hover_to_control_agent_mod(
        points_json,
        hover_input_current_dir=HOVER_INPUT_CURRENT_DIR,
        raw_current_dir=RAW_CURRENT_DIR,
        hold_left_button=bool(HOLD_LEFT_BUTTON),
        control_agent_port=CONTROL_AGENT_PORT,
        build_hover_path=_build_hover_path,
        save_hover_path_visual=_save_hover_path_visual,
        save_hover_overlay_from_json=_save_hover_overlay_from_json,
        send_control_agent=_send_control_agent,
        start_hover_fallback_timer=start_hover_fallback_timer,
        log=log,
    )



def finalize_hover_bot(task: Tuple[subprocess.Popen, Path, Path]) -> None:
    _hover_runtime.finalize_hover_bot(
        task,
        hover_output_current_dir=HOVER_OUTPUT_CURRENT_DIR,
        write_current_artifact=_write_current_artifact,
        dispatch_hover_to_control_agent=dispatch_hover_to_control_agent,
        debug=debug,
        log=log,
        debug_mode=DEBUG_MODE,
    )


def send_random_click(summary_path: Path, image_path: Path) -> None:
    _send_random_click_mod(
        summary_path,
        image_path,
        send_control_agent=_send_control_agent,
        control_agent_port=CONTROL_AGENT_PORT,
        cancel_hover_fallback_timer=cancel_hover_fallback_timer,
        log=log,
        update_overlay_status=update_overlay_status,
    )


def _latest_file(directory: Path, suffixes: Tuple[str, ...]) -> Optional[Path]:
    return _latest_file_mod(directory, suffixes)


def run_region_grow_latest() -> Optional[Path]:
    latest = _latest_file(SCREENSHOT_DIR, _IMAGE_EXTS)
    if latest is None:
        log("[WARN] No screenshots available for manual region_grow.")
        update_overlay_status("No screenshots for region_grow.")
        return None
    update_overlay_status(f"Manual region_grow on {latest.name}")
    json_path = run_region_grow(latest)
    if json_path:
        run_arrow_post(json_path)
        update_overlay_status(f"region_grow completed ({json_path.name})")
    else:
        update_overlay_status("region_grow failed.")
    return json_path


def run_rating_latest() -> bool:
    latest = _latest_file(SCREEN_BOXES_DIR, _JSON_EXTS)
    if latest is None:
        log("[WARN] No JSON files in screen_boxes for manual rating.")
        update_overlay_status("No JSON for rating.")
        return False
    update_overlay_status(f"Manual rating for {latest.name}")
    run_arrow_post(latest)
    ok = run_rating(latest)
    update_overlay_status("rating completed." if ok else "rating failed.")
    return ok


def _find_screenshot_for_summary(summary_path: Path) -> Optional[Path]:
    return _find_screenshot_for_summary_mod(
        summary_path,
        screenshot_dir=SCREENSHOT_DIR,
        image_exts=_IMAGE_EXTS,
    )


def send_best_click(summary_path: Path, image_path: Optional[Path]) -> None:
    _send_best_click_mod(
        summary_path,
        image_path,
        send_control_agent=_send_control_agent,
        control_agent_port=CONTROL_AGENT_PORT,
        log=log,
        update_overlay_status=update_overlay_status,
    )


def trigger_best_click_from_summary() -> None:
    summary = _latest_file(RATE_SUMMARY_DIR, _JSON_EXTS)
    if summary is None:
        log("[WARN] No summary JSONs available for control agent.")
        update_overlay_status("No summary for control agent.")
        return
    screenshot = _find_screenshot_for_summary(summary)
    if screenshot is None:
        log("[WARN] Matching screenshot not found for summary, using defaults.")
    send_best_click(summary, screenshot)


def _handle_manual_command(
    command: str,
    args: argparse.Namespace,
    recorder_proc: Optional[subprocess.Popen],
) -> Optional[subprocess.Popen]:
    return _handle_manual_command_mod(
        command,
        args=args,
        recorder_proc=recorder_proc,
        run_region_grow_latest=run_region_grow_latest,
        run_rating_latest=run_rating_latest,
        start_ai_recorder=start_ai_recorder,
        trigger_best_click_from_summary=trigger_best_click_from_summary,
        log=log,
        update_overlay_status=update_overlay_status,
    )


def _drain_manual_commands(
    cmd_queue: "queue.Queue[str]",
    args: argparse.Namespace,
    recorder_proc: Optional[subprocess.Popen],
) -> Optional[subprocess.Popen]:
    return _drain_manual_commands_mod(
        cmd_queue,
        args=args,
        recorder_proc=recorder_proc,
        run_region_grow_latest=run_region_grow_latest,
        run_rating_latest=run_rating_latest,
        start_ai_recorder=start_ai_recorder,
        trigger_best_click_from_summary=trigger_best_click_from_summary,
        log=log,
        update_overlay_status=update_overlay_status,
    )


def start_ai_recorder(extra_args: Optional[Iterable[str]] = None) -> Optional[subprocess.Popen]:
    return _start_ai_recorder_mod(
        ai_recorder_script=AI_RECORDER_SCRIPT,
        root=ROOT,
        subprocess_kw=SUBPROCESS_KW,
        log=log,
        extra_args=extra_args,
    )


def stop_process(proc: Optional[subprocess.Popen], timeout: float = 5.0) -> None:
    _stop_process_mod(proc, timeout=timeout, log=log)


def _pipeline_iteration_impl(
    loop_idx: int,
    screenshot_prefix: str = "screen",
    input_image: Optional[Path] = None,
    fast_skip: bool = False,
) -> None:
    _run_iteration_orchestrator(
        loop_idx=loop_idx,
        screenshot_prefix=screenshot_prefix,
        input_image=input_image,
        fast_skip=fast_skip,
        deps={
            "globals_fn": globals,
            "SCREENSHOT_DIR": SCREENSHOT_DIR,
            "RAW_CURRENT_DIR": RAW_CURRENT_DIR,
            "CURRENT_RUN_DIR": CURRENT_RUN_DIR,
            "capture_fullscreen": capture_fullscreen,
            "write_current_artifact": _write_current_artifact,
            "DEBUG_MODE": DEBUG_MODE,
            "debug": debug,
            "log": log,
            "update_overlay_status": update_overlay_status,
            "prepare_hover_image": prepare_hover_image,
            "downscale_for_region": _downscale_for_region,
            "run_region_grow": run_region_grow,
            "build_hover_from_region_results": build_hover_from_region_results,
            "dispatch_hover_to_control_agent": dispatch_hover_to_control_agent,
            "run_arrow_post": run_arrow_post,
            "run_rating": run_rating,
            "RATE_RESULTS_DIR": RATE_RESULTS_DIR,
            "RATE_RESULTS_DEBUG_DIR": RATE_RESULTS_DEBUG_DIR,
            "RATE_RESULTS_CURRENT_DIR": RATE_RESULTS_CURRENT_DIR,
            "RATE_RESULTS_DEBUG_CURRENT_DIR": RATE_RESULTS_DEBUG_CURRENT_DIR,
            "RATE_SUMMARY_DIR": RATE_SUMMARY_DIR,
            "RATE_SUMMARY_CURRENT_DIR": RATE_SUMMARY_CURRENT_DIR,
            "BRAIN_AGENT": BRAIN_AGENT,
            "send_click_from_bbox": _send_click_from_bbox,
            "scroll_on_box": scroll_on_box,
            "find_screenshot_for_summary": _find_screenshot_for_summary,
            "send_best_click": send_best_click,
            "send_random_click": send_random_click,
            "cancel_hover_fallback_timer": cancel_hover_fallback_timer,
        },
    )


def pipeline_iteration(
    loop_idx: int,
    screenshot_prefix: str = "screen",
    input_image: Optional[Path] = None,
    fast_skip: bool = False,
) -> None:
    _pipeline_iteration_external(
        loop_idx=loop_idx,
        screenshot_prefix=screenshot_prefix,
        input_image=input_image,
        fast_skip=fast_skip,
    )


def parse_args() -> argparse.Namespace:
    return _parse_args_mod()


def start_hotkey_listener(
    event: threading.Event, command_queue: "queue.Queue[str]"
) -> Optional["keyboard.Listener"]:
    return _start_hotkey_listener_mod(
        event,
        command_queue,
        log=log,
        debug=debug,
        update_overlay_status=update_overlay_status,
        globals_fn=globals,
        is_debug_mode=lambda: bool(DEBUG_MODE),
    )


def _wait_for_p_in_console() -> None:
    _wait_for_p_in_console_mod()


def main() -> None:
    args = parse_args()
    overlay_status = init_console_overlay(alpha=220)
    turbo_mode = str(os.environ.get("FULLBOT_TURBO_MODE", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
    if turbo_mode:
        log(
            "[INFO] Turbo mode active (GPU-only): "
            f"RATING_MODE={os.environ.get('RATING_MODE', 'off')} "
            f"MAX_SIDE={os.environ.get('REGION_GROW_MAX_SIDE_TURBO', '960')} "
            f"MAX_DET={os.environ.get('REGION_GROW_MAX_DETECTIONS_TURBO', '60')}"
        )

    # Tryb bezpiecznego testu (bez przejmowania myszy).
    if args.safe_test:
        args.autostart_control_agent = False

    # Tryb specjalny: pierwszy agent (hover -> control) i nic więcej.
    # Używamy go do testów opóźnień od screena do pierwszego ruchu agenta.
    if getattr(args, "first_agent", False):
        args.auto = True
        args.loop_count = 1
        args.fast_skip = True  # pomijamy region_grow + rating
        args.disable_recorder = True
        args.autostart_control_agent = True

    # Tryb pomiaru całego pipeline'u bez hotkey'a.
    if args.notime:
        args.auto = True
        if args.loop_count is None:
            args.loop_count = 1
        if args.input_image:
            args.disable_recorder = True
        preload_hover_reader()
    globals()["HOLD_LEFT_BUTTON"] = bool(args.left)
    globals()["DEBUG_MODE"] = bool(args.debug)
    if os.name == "nt":
        if bool(overlay_status.get("transparency_applied")):
            log("[INFO] Console overlay active (green + semi-transparent).")
        else:
            reason = str(overlay_status.get("transparency_reason") or "unknown")
            log(
                f"[WARN] Console overlay transparency not active "
                f"(reason={reason}, console_window_found={bool(overlay_status.get('console_window_found'))})."
            )
            if bool(overlay_status.get("pseudo_overlay_applied")):
                log("[INFO] Pseudo overlay active (VT green background fallback).")
        if not bool(overlay_status.get("vt_enabled")):
            log("[WARN] ANSI VT mode is disabled; WARN/ERROR/TIMER colors may not render.")
    recorder_proc: Optional[subprocess.Popen] = None

    trigger_event = threading.Event()
    command_queue: "queue.Queue[str]" = queue.Queue()
    hotkey_listener = None
    console_p_mode = False
    update_overlay_status("Initializing pipeline...")

    if not args.disable_recorder:
        recorder_proc = start_ai_recorder(args.recorder_args)

    # Upewnij siŽt, ‘•e control_agent jest uruchomiony jeszcze PRZED pierwszym
    # komunikatem "Waiting for hotkey 'P'...", ‘•eby pierwsze wciŽ>niŽtcie P
    # mia‘'o ju‘• gotowego agenta do przyjŽtcia hover path.
    control_agent_proc: Optional[subprocess.Popen] = None
    if not args.safe_test:
        control_agent_proc = ensure_control_agent(CONTROL_AGENT_PORT)

    # Jednorazowy warm-up OCR przed pierwszą iteracją, żeby podczas
    # naciśnięcia 'P' screenshot był robiony od razu na aktualnym ekranie.
    update_overlay_status("Warming OCR models...")
    warm_ocr_once()
    update_overlay_status("OCR ready. Waiting for pipeline start.")
    if DEBUG_MODE:
        debug(f"Args: interval={args.interval} loop_count={args.loop_count} auto={args.auto} disable_recorder={args.disable_recorder} left={args.left} debug={args.debug}")
        debug(f"Paths: raw={SCREENSHOT_DIR} hover_input_current={HOVER_INPUT_CURRENT_DIR} hover_output_current={HOVER_OUTPUT_CURRENT_DIR}")

    if not args.auto:
        hotkey_listener = start_hotkey_listener(trigger_event, command_queue)
        if hotkey_listener is None:
            console_p_mode = True
            update_overlay_status("P mode active (console). Press P for one iteration.")
    else:
        update_overlay_status("Auto mode active.")
                            
    log(
        f"[INFO] Pipeline start (interval={args.interval}s"
        + (f", max_loops={args.loop_count}" if args.loop_count else ", continuous")
        + ")"
    )

    state = {
        "recorder_proc": recorder_proc,
        "control_agent_proc": control_agent_proc,
        "console_p_mode": bool(console_p_mode),
        "debug_mode": bool(DEBUG_MODE),
    }

    try:
        _run_loop_controller(
            args=args,
            trigger_event=trigger_event,
            command_queue=command_queue,
            state=state,
            pipeline_iteration=pipeline_iteration,
            drain_manual_commands=lambda cmd_queue, a, rec: _drain_manual_commands(cmd_queue, a, rec),
            ensure_control_agent=lambda: ensure_control_agent(CONTROL_AGENT_PORT),
            cancel_hover_fallback_timer=cancel_hover_fallback_timer,
            wait_for_p_in_console=_wait_for_p_in_console,
            log=log,
            debug=debug,
            update_overlay_status=update_overlay_status,
        )
    except KeyboardInterrupt:
        log("[INFO] Stopped by user.")
    finally:
        recorder_proc = state.get("recorder_proc")
        control_agent_proc = state.get("control_agent_proc")
        stop_process(recorder_proc)
        if control_agent_proc is not None:
            stop_process(control_agent_proc)
        if hotkey_listener is not None:
            try:
                hotkey_listener.stop()
            except Exception:
                pass
        cancel_hover_fallback_timer()


if __name__ == "__main__":
    with contextlib.suppress(Exception):
        register_main_launch(ROOT)
    main()
