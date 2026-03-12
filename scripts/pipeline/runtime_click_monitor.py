from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from PIL import ImageGrab, ImageDraw

try:
    from pynput import mouse  # type: ignore
except Exception:
    mouse = None


class ScreenClickMonitor:
    def __init__(self, *, out_dir: Path, log: Callable[[str], None]) -> None:
        self.out_dir = Path(out_dir)
        self.log = log
        self.listener: Optional["mouse.Listener"] = None
        self._lock = threading.Lock()
        self._enabled = bool(mouse is not None)

    def start(self) -> bool:
        if not self._enabled:
            self.log("[WARN] Click monitor unavailable: pynput.mouse import failed.")
            return False
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.listener = mouse.Listener(on_click=self._on_click)
        self.listener.daemon = True
        self.listener.start()
        self.log(f"[INFO] Click monitor active -> {self.out_dir}")
        return True

    def stop(self) -> None:
        if self.listener is None:
            return
        try:
            self.listener.stop()
        except Exception:
            pass
        self.listener = None

    def _on_click(self, x: int, y: int, button, pressed: bool) -> None:
        if not pressed:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        payload = {
            "ts": ts,
            "epoch_s": round(time.time(), 6),
            "x": int(x),
            "y": int(y),
            "button": str(button),
        }
        with self._lock:
            self._append_jsonl(payload)
            self._save_click_image(payload)
        self.log(f"[INFO] Screen click captured: x={int(x)} y={int(y)} button={button}")

    def _append_jsonl(self, payload: dict) -> None:
        out = self.out_dir / "clicks.jsonl"
        line = json.dumps(payload, ensure_ascii=False)
        with out.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _save_click_image(self, payload: dict) -> None:
        x = int(payload["x"])
        y = int(payload["y"])
        ts = str(payload["ts"])
        # Full-screen snapshot with red point at click position.
        img = ImageGrab.grab(all_screens=True).convert("RGB")
        draw = ImageDraw.Draw(img)
        r = 5
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0), outline=(180, 0, 0))
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 255, 255))

        hist = self.out_dir / f"click_{ts}.png"
        current = self.out_dir / "clicks_on_screen_current.png"
        img.save(hist)
        img.save(current)


def start_screen_click_monitor(*, out_dir: Path, log: Callable[[str], None]) -> Optional[ScreenClickMonitor]:
    mon = ScreenClickMonitor(out_dir=out_dir, log=log)
    if not mon.start():
        return None
    return mon

