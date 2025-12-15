"""Run trained dropdown PPO model on the real desktop (1 FPS capture).

Loads the Phase 2 PPO policy and lets it interact with the actual mouse.
Press ESC at any time to abort safely.

Usage (recommended in venv):
    python scripts/run_live_dropdown_agent.py

Optional flags:
    --model PATH         Override model path.
    --fps N              Capture rate (default 1 FPS).
    --cpu                Force CPU inference.
    --dry-run            Do not move/click the mouse (prints actions only).
    --random             Sample random actions (sanity check pipeline).
    --stochastic         Use stochastic policy instead of deterministic.
    --no-ocr             Disable OCR state extraction.
    --ocr-lang CODE      OCR language (default: en).
    --show-ocr           Print detected OCR boxes (text + coordinates).
"""

import argparse
import sys
import time
import atexit
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Optional

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "envs").exists()),
    Path(__file__).resolve().parent,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import cv2
import torch
from stable_baselines3 import PPO

# Ensure custom extractor is registered for model loading
from models.feature_extractor_onnx_optimized import YOLOv11ONNXExtractorOptimized  # noqa: F401

from pynput import keyboard, mouse

try:
    import mss
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("mss is required for screen capture. Install with `pip install mss`.") from exc

from utils.ocr_features import (
    OCRNotAvailableError,
    STATE_DIM,
    ZERO_STATE,
    AsyncOCRWorker,
    compute_ocr_state,
    create_easyocr_reader,
)


DEFAULT_MODEL_PATH = Path(
    r"C:\Users\user\Desktop\Nowy folder\BOT ANK\bot\moje_AI\yolov8\FULL BOT\models\saved\phase2_transfer\best_model\best_model.zip"
)

def _ensure_capture_context():
    if not hasattr(_ensure_capture_context, "sct"):
        _ensure_capture_context.sct = mss.mss()  # type: ignore[attr-defined]
        _ensure_capture_context.monitor = _ensure_capture_context.sct.monitors[0]  # type: ignore[attr-defined]
        atexit.register(_ensure_capture_context.sct.close)  # type: ignore[attr-defined]
    return _ensure_capture_context.sct, _ensure_capture_context.monitor  # type: ignore[attr-defined]


class SafetyStop:
    """Listens for ESC press to stop the agent."""

    def __init__(self):
        self._stop_event = Event()
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def _on_press(self, key):
        if key == keyboard.Key.esc:
            print("[Safety] ESC pressed - stopping agent.")
            self._stop_event.set()
            return False
        return True

    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def close(self):
        self._listener.stop()


def capture_screen(target_size=(640, 480)) -> np.ndarray:
    sct, monitor = _ensure_capture_context()
    raw = sct.grab(monitor)
    frame = cv2.cvtColor(np.array(raw, dtype=np.uint8), cv2.COLOR_BGRA2BGR)
    if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)


def build_observation(frame: np.ndarray, state: np.ndarray | None = None) -> dict:
    if state is None:
        state = np.zeros(STATE_DIM, dtype=np.float32)
    return {"screen": frame, "state": state.astype(np.float32, copy=False)}


def map_to_screen(action: np.ndarray, screen_size: tuple[int, int]) -> tuple[int, int]:
    x_norm = float(np.clip(action[0], -1.0, 1.0))
    y_norm = float(np.clip(action[1], -1.0, 1.0))
    width, height = screen_size
    x = int((x_norm + 1.0) * 0.5 * (width - 1))
    y = int((y_norm + 1.0) * 0.5 * (height - 1))
    return x, y


def run_agent(args):
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[Agent] Loading model from {model_path}")
    print(f"[Agent] Using device: {device}")

    model = PPO.load(str(model_path), device=device)

    ocr = None
    ocr_worker: Optional[AsyncOCRWorker] = None
    if not args.no_ocr:
        try:
            print(f"[Agent] Initializing EasyOCR (lang={args.ocr_lang})")
            ocr = create_easyocr_reader(lang=args.ocr_lang)
            ocr_worker = AsyncOCRWorker(ocr)
        except OCRNotAvailableError as exc:
            raise ImportError(
                "easyocr is required for OCR state extraction. Install with `pip install easyocr` or pass --no-ocr."
            ) from exc

    stop = SafetyStop()

    mouse_ctl = mouse.Controller()

    _, monitor = _ensure_capture_context()
    screen_w = int(monitor["width"])
    screen_h = int(monitor["height"])
    screen_left = int(monitor.get("left", 0))
    screen_top = int(monitor.get("top", 0))
    target_fps = min(15.0, float(args.fps))
    interval = max(0.01, 1.0 / target_fps)

    ocr_interval = 1.0  # seconds between OCR updates
    last_ocr_time = 0.0
    cached_state = ZERO_STATE.copy()
    cached_boxes: List[Dict[str, Any]] = []
    frame_index = 0

    print("[Agent] Started. Press ESC to stop.")
    try:
        while not stop.should_stop():
            loop_start = time.time()

            frame = capture_screen()

            if ocr_worker and ocr_worker.enabled:
                now = time.time()
                if (now - last_ocr_time) >= ocr_interval:
                    ocr_worker.submit(frame, frame_id=frame_index)
                    last_ocr_time = now
                result = ocr_worker.get_latest()
                if result is not None:
                    cached_state = result.state
                    cached_boxes = result.boxes
                    if args.show_ocr and cached_boxes:
                        preview = cached_boxes[:3]
                        print("[OCR] Boxes:", preview)
            elif ocr is not None:
                now = time.time()
                if (now - last_ocr_time) >= ocr_interval:
                    try:
                        cached_state, boxes = compute_ocr_state(ocr, frame, return_boxes=True)
                        cached_boxes = boxes
                        if args.show_ocr and boxes:
                            print("[OCR] Boxes:", boxes[:3])
                    except Exception as err:
                        print(f"[OCR] Error: {type(err).__name__}: {err}")
                        cached_state = ZERO_STATE.copy()
                        cached_boxes = []
                    last_ocr_time = now

            obs = build_observation(frame, cached_state)

            if args.random:
                action = model.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=not args.stochastic)

            rel_x, rel_y = map_to_screen(action, (screen_w, screen_h))
            x = screen_left + rel_x
            y = screen_top + rel_y

            if args.dry_run:
                print(f"[Dry-Run] action={action} -> move/click ({x}, {y})")
            else:
                try:
                    mouse_ctl.position = (x, y)
                    mouse_ctl.click(mouse.Button.left, 1)
                except Exception as err:
                    print(f"[Mouse] Error: {err}")

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, interval - elapsed))
            frame_index += 1

    finally:
        stop.close()
        if ocr_worker is not None:
            ocr_worker.stop()
        print("[Agent] Stopped.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO dropdown agent on live desktop.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to .zip policy")
    parser.add_argument("--fps", type=float, default=15.0, help="Capture frequency (frames per second, capped at 15 Hz)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--dry-run", action="store_true", help="Do not control the mouse (print actions only)")
    parser.add_argument("--random", action="store_true", help="Ignore model, sample random actions")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy (deterministic off)")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR state features")
    parser.add_argument("--ocr-lang", type=str, default="en", help="OCR language code (default: en)")
    parser.add_argument(
        "--show-ocr",
        action="store_true",
        help="Print detected OCR bounding boxes with text and confidence.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_agent(args)
