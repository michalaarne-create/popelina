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


def prepare_hover_image(full_image: Path) -> Optional[Path]:
    """
    Create a cropped version of the screenshot for hover_bot:
    - removes HOVER_TOP_CROP pixels from the top (to skip tab bars)
    - removes 300 px from both left and right sides.
    """
    try:
        with Image.open(full_image) as im:
            width, height = im.size
            if width <= HOVER_SIDE_CROP * 2 or height <= HOVER_TOP_CROP + 10:
                cropped = im.copy()
            else:
                left = HOVER_SIDE_CROP
                right = width - HOVER_SIDE_CROP
                top = min(max(0, HOVER_TOP_CROP), height - 10)
                box = (left, top, right, height)
                cropped = im.crop(box)
        HOVER_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = HOVER_INPUT_DIR / f"{full_image.stem}_hover.png"
        cropped.save(out_path)
        return out_path
    except Exception as exc:
        log(f"[WARN] Could not prepare hover image: {exc}")
        return None


def run_region_grow(image_path: Path) -> Optional[Path]:
    """
    Run utils/region_grow.py for the provided screenshot and return the JSON
    path it generates (if any).
    """
    log(f"[INFO] Running region_grow on {image_path.name}")
    cmd = [sys.executable, str(REGION_GROW_SCRIPT), str(image_path)]
    result = subprocess.run(cmd, cwd=str(ROOT), **SUBPROCESS_KW)
    if result.returncode != 0:
        log(f"[ERROR] region_grow failed with code {result.returncode}")
        return None

    json_path = SCREEN_BOXES_DIR / f"{image_path.stem}.json"
    if not json_path.exists():
        log(f"[ERROR] Expected JSON missing: {json_path}")
        return None

    return json_path


def run_arrow_post(json_path: Path) -> None:
    """
    Opcjonalny krok po region_grow: wykrywa strzałki na obrazie na podstawie
    JSON-a ze screen_boxes i uzupełnia ten JSON o pole `triangles`.
    """
    if not ARROW_POST_SCRIPT.exists():
        return
    log(f"[INFO] Running arrow_post_region on {json_path.name}")
    cmd = [sys.executable, str(ARROW_POST_SCRIPT), str(json_path)]
    result = subprocess.run(cmd, cwd=str(ROOT), **SUBPROCESS_KW)
    if result.returncode != 0:
        log(f"[WARN] arrow_post_region failed with code {result.returncode}")


def run_rating(json_path: Path) -> bool:
    """Invoke scripts/numpy_rate/rating.py for the produced JSON."""
    log(f"[INFO] Running rating on {json_path.name}")
    cmd = [sys.executable, str(RATING_SCRIPT), str(json_path)]
    result = subprocess.run(cmd, cwd=str(ROOT), **SUBPROCESS_KW)
    if result.returncode != 0:
        log(f"[ERROR] rating failed with code {result.returncode}")
        return False
    return True


def run_hover_bot(hover_image: Path, base_name: str) -> Optional[Tuple[subprocess.Popen, Path]]:
    """Launch hover_single.py for the prepared image."""
    if not HOVER_SINGLE_SCRIPT.exists():
        log("[WARN] hover_single.py not found; skipping hover bot.")
        return None
    points_json = HOVER_OUTPUT_DIR / f"{base_name}_hover.json"
    annot_out = HOVER_OUTPUT_DIR / f"{base_name}_hover.png"
    HOVER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(HOVER_SINGLE_SCRIPT),
        "--image",
        str(hover_image),
        "--json-out",
        str(points_json),
        "--annot-out",
        str(annot_out),
    ]
    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), **SUBPROCESS_KW)
        log(f"[INFO] hover_bot started for {hover_image.name}")
        update_overlay_status(f"hover_bot running ({hover_image.name})")
        return proc, points_json
    except Exception as exc:
        log(f"[WARN] Failed to launch hover bot: {exc}")
        return None


def _hover_rect(box: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def _inside_any(x: float, y: float, rects: List[Tuple[int, int, int, int]]) -> bool:
    for (x0, y0, x1, y1) in rects:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def _group_hover_lines(seqs: List[Dict[str, Any]]) -> List[List[int]]:
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


def _build_hover_path(
    seqs: List[Dict[str, Any]],
    offset_x: int,
    offset_y: int,
) -> Optional[Dict[str, Any]]:
    if not seqs:
        return None

    real_indices = [i for i, s in enumerate(seqs) if float(s.get("confidence", 0.0)) >= 0.0]
    ordered_groups = _group_hover_lines([seqs[i] for i in real_indices])
    points: List[Dict[str, int]] = []
    line_jump_indices: List[int] = []
    seg_index = -1
    for group in ordered_groups:
        order = sorted(group, key=lambda i: min(p[0] for p in seqs[real_indices[i]].get("box", [[0, 0]])))
