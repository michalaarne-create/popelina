from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image, ImageDraw


def _env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.environ.get(name, default) or default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except Exception:
        return int(default)


def _extract_box_xyxy(item: dict) -> Optional[tuple[int, int, int, int]]:
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


def _write_ocr_boxes_debug_from_json(
    *,
    json_path: Path,
    screenshot_path: Path,
    root: Path,
    log: Callable[[str], None],
) -> Optional[Path]:
    if not _env_flag("FULLBOT_OCR_BOXES_DEBUG", "0"):
        return None
    if not screenshot_path.exists():
        log(f"[WARN] Deferred OCR debug skipped, screenshot missing: {screenshot_path}")
        return None
    if not json_path.exists():
        log(f"[WARN] Deferred OCR debug skipped, JSON missing: {json_path}")
        return None

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        log(f"[WARN] Deferred OCR debug JSON read failed: {exc}")
        return None
    if not isinstance(payload, dict):
        log("[WARN] Deferred OCR debug skipped, JSON payload is not an object.")
        return None

    results = payload.get("results")
    if not isinstance(results, list):
        results = []

    try:
        ocr_dir = root / "data" / "screen" / "OCR boxes png"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        canvas = Image.open(screenshot_path).convert("RGB")
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
        out_path = ocr_dir / f"{screenshot_path.stem}_{ts}_ocr_boxes.png"
        canvas.save(out_path)
        log(f"[INFO] Deferred OCR boxes debug saved: {out_path.name} (boxes={count})")
        return out_path
    except Exception as exc:
        log(f"[WARN] Deferred OCR debug save failed: {exc}")
        return None


@dataclass
class DeferredDebugJob:
    json_path: Path
    screenshot_path: Path
    loop_idx: int
    created_ts: float


class DeferredDebugRenderWorker:
    def __init__(
        self,
        *,
        root: Path,
        run_region_annotation: Callable[[Path, Path], Optional[Path]],
        log: Callable[[str], None],
        enabled: Optional[bool] = None,
    ) -> None:
        self._root = Path(root)
        self._run_region_annotation = run_region_annotation
        self._log = log
        self._enabled = bool(_env_flag("FULLBOT_DEBUG_DEFERRED_RENDER", "1")) if enabled is None else bool(enabled)

        queue_max = _env_int("FULLBOT_DEBUG_DEFERRED_QUEUE_MAX", 0)
        self._queue_max = int(queue_max if queue_max > 0 else 0)
        if self._queue_max > 0:
            self._jobs: "queue.Queue[Optional[DeferredDebugJob]]" = queue.Queue(maxsize=self._queue_max)
        else:
            self._jobs = queue.Queue()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def enabled(self) -> bool:
        return bool(self._enabled)

    def start(self) -> None:
        if not self._enabled:
            self._log("[INFO] Deferred debug worker disabled (FULLBOT_DEBUG_DEFERRED_RENDER=0).")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="DeferredDebugRenderWorker",
            daemon=True,
        )
        self._thread.start()
        q_info = "unbounded" if self._queue_max <= 0 else str(self._queue_max)
        self._log(f"[INFO] Deferred debug worker started (queue_max={q_info}).")

    def schedule(self, json_path: Path, screenshot_path: Path, loop_idx: int) -> bool:
        if not self._enabled:
            return False
        if self._thread is None or not self._thread.is_alive():
            return False
        job = DeferredDebugJob(
            json_path=Path(json_path),
            screenshot_path=Path(screenshot_path),
            loop_idx=int(loop_idx),
            created_ts=time.time(),
        )
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            self._log(
                f"[WARN] Deferred debug queue full (max={self._queue_max}), "
                f"skipping loop={job.loop_idx}."
            )
            return False
        self._log(f"[DEBUG] Deferred debug queued for loop={job.loop_idx}.")
        return True

    def stop(self, *, drain: bool = True, timeout: float = 15.0) -> None:
        if self._thread is None:
            return
        started = time.perf_counter()
        if drain:
            deadline = time.perf_counter() + max(0.1, float(timeout))
            while (not self._jobs.empty()) and time.perf_counter() < deadline:
                time.sleep(0.05)

        self._stop_event.set()
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=max(0.1, float(timeout)))
        if self._thread.is_alive():
            self._log("[WARN] Deferred debug worker did not stop cleanly before timeout.")
        else:
            self._log(f"[INFO] Deferred debug worker stopped in {time.perf_counter() - started:.3f}s.")
        self._thread = None

    def _run(self) -> None:
        while True:
            if self._stop_event.is_set() and self._jobs.empty():
                return
            try:
                item = self._jobs.get(timeout=0.20)
            except queue.Empty:
                continue
            if item is None:
                self._jobs.task_done()
                if self._stop_event.is_set():
                    return
                continue
            try:
                self._run_job(item)
            except Exception as exc:
                self._log(f"[WARN] Deferred debug worker job failed: {exc}")
            finally:
                self._jobs.task_done()

    def _run_job(self, job: DeferredDebugJob) -> None:
        t0 = time.perf_counter()
        queue_delay_ms = max(0.0, (time.time() - float(job.created_ts)) * 1000.0)
        self._log(
            f"[INFO] Deferred debug start loop={job.loop_idx} "
            f"(queue_delay={queue_delay_ms:.1f}ms)."
        )

        try:
            self._run_region_annotation(job.json_path, job.screenshot_path)
        except Exception as exc:
            self._log(f"[WARN] Deferred region annotation failed (loop={job.loop_idx}): {exc}")

        _write_ocr_boxes_debug_from_json(
            json_path=job.json_path,
            screenshot_path=job.screenshot_path,
            root=self._root,
            log=self._log,
        )

        self._log(
            f"[TIMER] deferred_debug.loop_{job.loop_idx} "
            f"{time.perf_counter() - t0:.3f}s"
        )


def create_deferred_debug_worker(
    *,
    root: Path,
    run_region_annotation: Callable[[Path, Path], Optional[Path]],
    log: Callable[[str], None],
) -> DeferredDebugRenderWorker:
    return DeferredDebugRenderWorker(
        root=root,
        run_region_annotation=run_region_annotation,
        log=log,
    )
