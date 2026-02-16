from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

from PIL import Image


def capture_fullscreen(
    target: Path,
    *,
    mss_singleton_get: Callable[[], object],
    mss_singleton_set: Callable[[object], None],
    log,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    errors: List[Exception] = []
    try:
        from mss import mss, tools

        sct = mss_singleton_get()
        if sct is None:
            sct = mss()
            mss_singleton_set(sct)
        monitor = sct.monitors[0]
        raw = sct.grab(monitor)
        tools.to_png(raw.rgb, raw.size, output=str(target))
        return target
    except Exception as exc:
        errors.append(exc)
        log(f"[WARN] mss capture failed: {exc}")
    try:
        from PIL import ImageGrab

        img = ImageGrab.grab(all_screens=True)
        img.save(target)
        return target
    except Exception as exc:
        errors.append(exc)
        log(f"[WARN] ImageGrab capture failed: {exc}")
    last = errors[-1] if errors else RuntimeError("Unknown capture error")
    raise RuntimeError("Unable to capture fullscreen screenshot") from last


def prepare_hover_image(
    full_image: Path,
    *,
    raw_current_dir: Path,
    hover_top_crop: int,
    hover_input_dir: Path,
    hover_input_current_dir: Path,
    write_current_artifact,
    debug,
    log,
) -> Optional[Path]:
    try:
        current_candidate = raw_current_dir / "screenshot.png"
        if current_candidate.exists():
            debug(f"hover input: using current screenshot {current_candidate}")
            full_image = current_candidate
        else:
            debug(f"hover input: current screenshot missing, using {full_image}")
        with Image.open(full_image) as im:
            width, height = im.size
            if height <= hover_top_crop + 10:
                cropped = im.copy()
            else:
                top = min(max(0, hover_top_crop), height - 10)
                cropped = im.crop((0, top, width, height))
        hover_input_dir.mkdir(parents=True, exist_ok=True)
        out_path = hover_input_dir / f"{full_image.stem}_hover.png"
        cropped.save(out_path)
        write_current_artifact(out_path, hover_input_current_dir, "hover_input.png")
        debug(f"hover input saved: {out_path} | current copy -> {hover_input_current_dir / 'hover_input.png'} (size={cropped.size})")
        return out_path
    except Exception as exc:
        log(f"[WARN] Could not prepare hover image: {exc}")
        return None
