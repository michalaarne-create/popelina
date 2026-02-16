from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def hover_rect(box: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def inside_any(x: float, y: float, rects: List[Tuple[int, int, int, int]]) -> bool:
    for (x0, y0, x1, y1) in rects:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def group_hover_lines(seqs: List[Dict[str, Any]]) -> List[List[int]]:
    entries = []
    for idx, seq in enumerate(seqs):
        box = seq.get("box") or []
        if not box:
            continue
        ys = [float(p[1]) for p in box]
        min_y, max_y = min(ys), max(ys)
        height = max(1.0, max_y - min_y)
        dots = seq.get("dots") or []
        if dots:
            line_y = float(sum(d[1] for d in dots) / len(dots))
        else:
            line_y = 0.5 * (min_y + max_y)
        entries.append((idx, min_y, max_y, line_y, height))

    groups: List[List[int]] = []
    ranges: List[Tuple[float, float]] = []
    for idx, min_y, max_y, line_y, height in sorted(entries, key=lambda t: t[3]):
        placed = False
        for gi, (gmin, gmax) in enumerate(ranges):
            gc = 0.5 * (gmin + gmax)
            gh = max(1.0, gmax - gmin)
            if abs(line_y - gc) <= 0.45 * max(height, gh):
                ranges[gi] = (min(gmin, min_y), max(gmax, max_y))
                groups[gi].append(idx)
                placed = True
                break
        if not placed:
            ranges.append((min_y, max_y))
            groups.append([idx])
    return groups


def build_hover_path(
    seqs: List[Dict[str, Any]],
    offset_x: int,
    offset_y: int,
    *,
    brain_agent: Any,
    hover_speed_defaults: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not seqs:
        return None

    usable: List[Dict[str, Any]] = []
    gap_rects: List[Tuple[int, int, int, int]] = []
    for s in seqs:
        try:
            conf = float(s.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        box = s.get("box") or []
        if conf < 0.0:
            if box and isinstance(box, (list, tuple)) and len(box) >= 2:
                try:
                    xs = [float(p[0]) for p in box]
                    ys = [float(p[1]) for p in box]
                    x1, y1 = int(min(xs)), int(min(ys))
                    x2, y2 = int(max(xs)), int(max(ys))
                    gap_rects.append((x1, y1, x2, y2))
                except Exception:
                    pass
            continue
        usable.append(s)

    if not usable:
        return None

    clusters: Dict[int, Dict[str, float]] = {}

    def _box_center(seq: Dict[str, Any]) -> Tuple[float, float]:
        if "center_x" in seq and "center_y" in seq:
            try:
                return float(seq["center_x"]), float(seq["center_y"])
            except Exception:
                pass
        box = seq.get("box") or []
        if box and isinstance(box, (list, tuple)) and len(box) >= 2:
            try:
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                return (min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5
            except Exception:
                pass
        dots = seq.get("dots") or []
        if dots:
            try:
                xs = [float(d[0]) for d in dots]
                ys = [float(d[1]) for d in dots]
                return sum(xs) / len(xs), sum(ys) / len(ys)
            except Exception:
                pass
        return float("inf"), float("inf")

    for s in usable:
        try:
            cid = int(s.get("bg_cluster_id", 0) or 0)
        except Exception:
            cid = 0
        cx, cy = _box_center(s)
        if cid not in clusters:
            clusters[cid] = {"id": cid, "min_x": cx, "max_x": cx, "min_y": cy, "max_y": cy}
        else:
            c = clusters[cid]
            c["min_x"] = min(c["min_x"], cx)
            c["max_x"] = max(c["max_x"], cx)
            c["min_y"] = min(c["min_y"], cy)
            c["max_y"] = max(c["max_y"], cy)

    preferred_cluster_id: Optional[int] = None
    try:
        brain_state = brain_agent.load_state()
        hints = brain_state.get("reading_hints") or {}
        primary = hints.get("primary_bg_cluster_id")
        if primary is not None:
            preferred_cluster_id = int(primary)
    except Exception:
        preferred_cluster_id = None

    def _cluster_sort_key(c: Dict[str, float]) -> Tuple[float, float]:
        return (c["min_x"], c["min_y"])

    cluster_order = sorted(clusters.values(), key=_cluster_sort_key)
    if preferred_cluster_id is not None and preferred_cluster_id in clusters:
        cluster_order = sorted(
            clusters.values(),
            key=lambda c: (0 if int(c["id"]) == preferred_cluster_id else 1, *_cluster_sort_key(c)),
        )

    ordered_seqs: List[Dict[str, Any]] = []
    for c in cluster_order:
        cid = c["id"]
        region_seqs = [s for s in usable if int(s.get("bg_cluster_id", 0) or 0) == cid]
        region_seqs.sort(key=lambda s: (_box_center(s)[1], _box_center(s)[0]))
        ordered_seqs.extend(region_seqs)

    points: List[Dict[str, int]] = []
    line_jump_indices: List[int] = []

    for seq in ordered_seqs:
        dots = seq.get("dots") or []
        if not dots:
            continue
        if points and dots:
            line_jump_indices.append(len(points) - 1)
        for d in dots:
            px = int(round(d[0])) + offset_x
            py = int(round(d[1])) + offset_y
            points.append({"x": px, "y": py})

    if len(points) < 2:
        return None

    payload: Dict[str, Any] = {
        "cmd": "path",
        "points": points,
        "speed": hover_speed_defaults["speed"],
        "min_total_ms": 0.0,
        "speed_factor": hover_speed_defaults["speed_factor"],
        "min_dt": hover_speed_defaults["min_dt"],
        "gap_rects": [list(r) for r in gap_rects],
        "gap_boost": hover_speed_defaults["gap_boost"],
        "line_jump_indices": line_jump_indices,
        "line_jump_boost": hover_speed_defaults["line_jump_boost"],
    }
    if hover_speed_defaults.get("speed_px_per_s", 0) > 0:
        payload["speed_px_per_s"] = hover_speed_defaults["speed_px_per_s"]
    return payload


def send_udp_payload(payload: Dict[str, Any], port: int, *, log: Callable[[str], None]) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = json.dumps(payload).encode("utf-8")
        sock.sendto(data, ("127.0.0.1", port))
        try:
            pts = payload.get("points", [])
            log(f"[DEBUG] UDP sent to {port}: size={len(data)} bytes, points={len(pts)}")
        except Exception:
            log(f"[DEBUG] UDP sent to {port}: size={len(data)} bytes")
        return True
    except Exception as exc:
        log(f"[WARN] Failed to send payload to control agent: {exc}")
        return False


def save_hover_path_visual(
    points: List[Dict[str, int]],
    points_json: Path,
    *,
    screenshot_dir: Path,
    hover_input_current_dir: Path,
    raw_current_dir: Path,
    hover_path_dir: Path,
    hover_path_current_dir: Path,
    debug: Callable[[str], None],
    log: Callable[[str], None],
) -> None:
    stem = points_json.stem
    if stem.endswith("_hover"):
        stem = stem[:-6]
    screen_path = screenshot_dir / f"{stem}.png"

    if screen_path.exists():
        src_img = screen_path
    else:
        candidate = hover_input_current_dir / "hover_input.png"
        if candidate.exists():
            src_img = candidate
        else:
            src_img = raw_current_dir / "screenshot.png"

    if not src_img.exists():
        log(f"[WARN] hover path visualisation skipped (no screenshot found for {points_json})")
        return

    try:
        img = Image.open(src_img).convert("RGB")
    except Exception as exc:
        log(f"[WARN] Could not open screenshot for hover path ({src_img}): {exc}")
        return

    arr = np.array(img, dtype=np.uint8)
    h, w, _ = arr.shape
    counts = np.zeros((h, w), dtype=np.uint8)

    def _clip_xy(x: int, y: int) -> Tuple[int, int]:
        return int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))

    for p0, p1 in zip(points, points[1:]):
        x0, y0 = _clip_xy(p0.get("x", 0), p0.get("y", 0))
        x1, y1 = _clip_xy(p1.get("x", 0), p1.get("y", 0))
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            counts[y0, x0] = np.clip(counts[y0, x0] + 1, 0, 255)
            continue
        xs = np.linspace(x0, x1, steps + 1, dtype=np.int32)
        ys = np.linspace(y0, y1, steps + 1, dtype=np.int32)
        counts[ys, xs] = np.clip(counts[ys, xs] + 1, 0, 255)

    out = arr.copy()
    out[counts == 1] = np.array([0, 255, 0], dtype=np.uint8)
    out[counts == 2] = np.array([0, 0, 255], dtype=np.uint8)
    out[counts == 3] = np.array([255, 255, 0], dtype=np.uint8)
    out[counts == 4] = np.array([255, 165, 0], dtype=np.uint8)
    out[counts >= 5] = np.array([255, 0, 0], dtype=np.uint8)

    hover_path_dir.mkdir(parents=True, exist_ok=True)
    hover_path_current_dir.mkdir(parents=True, exist_ok=True)

    hist_path = hover_path_dir / f"{stem}_hover_path.png"
    Image.fromarray(out).save(hist_path)
    current_path = hover_path_current_dir / "hover_path.png"
    Image.fromarray(out).save(current_path)
    debug(f"hover path visual saved: hist={hist_path} current={current_path}")


def save_hover_overlay_from_json(
    points_json: Path,
    *,
    region_grow_annot_current_dir: Path,
    raw_current_dir: Path,
    screenshot_dir: Path,
    hover_output_dir: Path,
    hover_output_current_dir: Path,
    write_current_artifact: Callable[[Path, Path, Optional[str]], Optional[Path]],
    debug_hover_output_current: Callable[[], None],
    debug: Callable[[str], None],
) -> None:
    image_hint: Optional[str] = None
    try:
        data = json.loads(points_json.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            image_hint = data.get("image") or data.get("background_image")
            seqs = data.get("sequences") or []
        else:
            seqs = data
        if not isinstance(seqs, list) or not seqs:
            return
    except Exception as exc:
        debug(f"hover overlay: could not read {points_json}: {exc}")
        return

    annot_current = region_grow_annot_current_dir / "region_grow_annot_current.png"
    if annot_current.exists():
        screen_path = annot_current
    else:
        if image_hint:
            try:
                p = Path(image_hint)
                if not p.is_absolute():
                    p = points_json.parent / p
                if p.exists():
                    screen_path = p
                else:
                    raise FileNotFoundError
            except Exception:
                screen_path = raw_current_dir / "screenshot.png"
        else:
            stem = points_json.stem
            if stem.endswith("_hover"):
                stem = stem[:-6]
            screen_path = screenshot_dir / f"{stem}.png"
            if not screen_path.exists():
                screen_path = raw_current_dir / "screenshot.png"

    if not screen_path.exists():
        debug(f"hover overlay: no screenshot found for {points_json}")
        return

    try:
        img = Image.open(screen_path).convert("RGB")
    except Exception as exc:
        debug(f"hover overlay: could not open screenshot {screen_path}: {exc}")
        return

    draw = ImageDraw.Draw(img)
    for seq in seqs:
        try:
            conf = float(seq.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        if conf < 0.0:
            continue
        for d in (seq.get("dots") or []):
            if not isinstance(d, (list, tuple)) or len(d) < 2:
                continue
            try:
                x, y = int(d[0]), int(d[1])
            except Exception:
                continue
            r = 3
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=2)

    try:
        hover_output_dir.mkdir(parents=True, exist_ok=True)
        stem = points_json.stem
        if stem.endswith("_hover"):
            stem = stem[:-6]
        out_path = hover_output_dir / f"{stem}_hover.png"
        img.save(out_path)
        current_png = write_current_artifact(out_path, hover_output_current_dir, "hover_output.png")
        debug(f"hover overlay saved: {out_path} | current copy: {current_png}")
        debug_hover_output_current()
    except Exception as exc:
        debug(f"hover overlay save failed: {exc}")


def build_hover_from_region_results(
    json_path: Path,
    *,
    screenshot_dir: Path,
    raw_current_dir: Path,
    hover_output_dir: Path,
    hover_output_current_dir: Path,
    write_current_artifact: Callable[[Path, Path, Optional[str]], Optional[Path]],
    debug_hover_output_current: Callable[[], None],
    debug: Callable[[str], None],
    log: Callable[[str], None],
) -> Optional[Path]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"[WARN] build_hover_from_region_results: could not read {json_path}: {exc}")
        return None

    results = data.get("results") or []
    if not isinstance(results, list) or not results:
        log(f"[WARN] build_hover_from_region_results: no results in {json_path.name}")
        return None

    image_field = data.get("image")
    image_candidates: List[Path] = []
    if isinstance(image_field, str) and image_field:
        try:
            p = Path(image_field)
            if not p.is_absolute():
                p = json_path.parent / p
            image_candidates.append(p)
        except Exception:
            pass
    base_stem = Path(image_field).stem if isinstance(image_field, str) and image_field else json_path.stem
    stems = [base_stem]
    if base_stem.endswith("_rg_small"):
        stems.insert(0, base_stem[: -len("_rg_small")])
    for s in stems:
        image_candidates.append(screenshot_dir / f"{s}.png")
    image_candidates.append(raw_current_dir / "screenshot.png")

    image_path: Optional[Path] = None
    for cand in image_candidates:
        if cand and cand.exists():
            image_path = cand
            break
    if image_path is None:
        log(f"[WARN] build_hover_from_region_results: no image found for {json_path.name}")
        return None

    boxes: List[Tuple[List[Tuple[float, float]], str, float]] = []
    box_clusters: List[int] = []
    box_centers: List[Tuple[float, float]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        try:
            raw_text = r.get("text") or ""
            text = str(raw_text).strip()
            tb = r.get("text_box") or r.get("bbox") or r.get("box")
            conf = float(r.get("conf", 0.0) or 0.0)
        except Exception:
            continue
        if not tb or not isinstance(tb, (list, tuple)) or len(tb) != 4:
            continue
        if not text:
            continue
        if not any(ch.isalpha() for ch in text):
            continue
        if conf < 0.20:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in tb]
        except Exception:
            continue
        box_poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        boxes.append((box_poly, text, conf))
        cid = int(r.get("bg_cluster_id", 0) or 0)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        box_clusters.append(cid)
        box_centers.append((cx, cy))

    if not boxes:
        log(f"[WARN] build_hover_from_region_results: no suitable text boxes in {json_path.name}")
        return None

    try:
        import cv2  # type: ignore
        from dataclasses import asdict
        from scripts.click.hover import hover_bot as hb  # type: ignore

        sequences, annotated, timings = hb.process_from_boxes(image_path, boxes)
        debug(f"build_hover_from_region_results: image={image_path.name} seqs={len(sequences)} timings={timings}")

        payload = []
        for seq in sequences:
            d = asdict(seq)
            try:
                idx = int(seq.index)
            except Exception:
                idx = len(payload)
            if 0 <= idx < len(box_clusters):
                d["bg_cluster_id"] = int(box_clusters[idx])
                cx, cy = box_centers[idx]
                d["center_x"] = float(cx)
                d["center_y"] = float(cy)
            payload.append(d)

        hover_output_dir.mkdir(parents=True, exist_ok=True)
        base = Path(image_field or json_path.stem)
        out_path = hover_output_dir / f"{base.stem}_hover.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        current_json = write_current_artifact(out_path, hover_output_current_dir, "hover_output.json")

        annot_path = hover_output_dir / f"{base.stem}_hover.png"
        try:
            cv2.imwrite(str(annot_path), annotated)
        except Exception as exc:
            debug(f"build_hover_from_region_results: could not save hover overlay: {exc}")
        if annot_path.exists():
            write_current_artifact(annot_path, hover_output_current_dir, "hover_output.png")
        debug_hover_output_current()
        return current_json or out_path
    except Exception as exc:
        log(f"[WARN] build_hover_from_region_results: hover_bot process_from_boxes failed: {exc}")
        return None


def run_hover_bot(
    hover_image: Path,
    base_name: str,
    *,
    start_time: Optional[float],
    hover_output_dir: Path,
    hover_output_current_dir: Path,
    screenshot_dir: Path,
    root: Path,
    subprocess_kw: Dict[str, Any],
    hover_single_script: Path,
    write_current_artifact: Callable[[Path, Path, Optional[str]], Optional[Path]],
    debug_hover_output_current: Callable[[], None],
    dispatch_hover_to_control_agent: Callable[[Path], None],
    update_overlay_status: Callable[[str], None],
    debug: Callable[[str], None],
    log: Callable[[str], None],
    hover_reader_cache_get: Callable[[], Any],
    hover_reader_cache_set: Callable[[Any], None],
) -> Optional[Tuple[subprocess.Popen, Path, Path]]:
    points_json = hover_output_dir / f"{base_name}_hover.json"
    annot_out = hover_output_dir / f"{base_name}_hover.png"
    hover_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2  # type: ignore
        from dataclasses import asdict
        from scripts.click.hover import hover_bot as hb  # type: ignore

        def _get_hover_reader():
            cache = hover_reader_cache_get()
            if cache is None:
                cache = hb.create_paddleocr_reader(lang=hb.OCR_LANG)
                hover_reader_cache_set(cache)
                log("[INFO] hover_bot reader initialized (inline).")
            return cache

        def _worker():
            t0 = start_time or time.perf_counter()
            try:
                reader = _get_hover_reader()
                t_proc_start = time.perf_counter()
                sequences, annotated, timings = hb.process_image(Path(hover_image), reader=reader)
                t_proc_end = time.perf_counter()
                debug(f"hover inline: seqs={len(sequences)} timings={timings}")
                log(f"[TIMER] hover_process_image {t_proc_end - t_proc_start:.3f}s (seqs={len(sequences)})")

                payload = [asdict(seq) for seq in sequences]
                points_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                current_json = write_current_artifact(points_json, hover_output_current_dir, "hover_output.json")
                debug_hover_output_current()
                dispatch_hover_to_control_agent(current_json or points_json)
                log(f"[TIMER] hover_total {time.perf_counter() - t0:.3f}s (screen -> control_agent)")

                try:
                    cv2.imwrite(str(annot_out), annotated)
                except Exception:
                    pass
                if annot_out.exists():
                    write_current_artifact(annot_out, hover_output_current_dir, "hover_output.png")
                else:
                    try:
                        screenshot_candidate = screenshot_dir / f"{base_name}.png"
                        if screenshot_candidate.exists():
                            write_current_artifact(screenshot_candidate, hover_output_current_dir, "hover_output.png")
                    except Exception:
                        pass
            except Exception as exc:
                log(f"[WARN] Inline hover_bot failed: {exc}")

        t = threading.Thread(target=_worker, name="hover_bot_inline", daemon=True)
        t.start()
        log(f"[INFO] hover_bot (inline) started for {hover_image.name}")
        update_overlay_status(f"hover_bot running ({hover_image.name})")
        return None
    except Exception as exc:
        debug(f"Inline hover_bot unavailable, falling back to hover_single.py: {exc}")

    if not hover_single_script.exists():
        log("[WARN] hover_single.py not found; skipping hover bot.")
        return None
    cmd = [
        os.sys.executable,
        str(hover_single_script),
        "--image",
        str(hover_image),
        "--json-out",
        str(points_json),
        "--annot-out",
        str(annot_out),
    ]
    try:
        proc = subprocess.Popen(cmd, cwd=str(root), **subprocess_kw)
        log(f"[INFO] hover_bot started for {hover_image.name}")
        update_overlay_status(f"hover_bot running ({hover_image.name})")
        return proc, points_json, annot_out
    except Exception as exc:
        log(f"[WARN] Failed to launch hover bot: {exc}")
        return None


def finalize_hover_bot(
    task: Tuple[subprocess.Popen, Path, Path],
    *,
    hover_output_current_dir: Path,
    write_current_artifact: Callable[[Path, Path, Optional[str]], Optional[Path]],
    dispatch_hover_to_control_agent: Callable[[Path], None],
    debug: Callable[[str], None],
    log: Callable[[str], None],
    debug_mode: bool,
) -> None:
    proc, json_path, annot_path = task
    try:
        code = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        log("[WARN] hover_bot timed out and was killed.")
        if debug_mode:
            debug(f"hover_bot timeout for {json_path}")
        return
    if code == 0:
        log(f"[INFO] hover_bot completed ({json_path.name})")
        current_json = write_current_artifact(json_path, hover_output_current_dir, "hover_output.json")
        if annot_path.exists():
            write_current_artifact(annot_path, hover_output_current_dir, "hover_output.png")
        dispatch_hover_to_control_agent(current_json or json_path)
    else:
        log(f"[WARN] hover_bot exited with code {code}")

