
    rects_all = [
        _hover_rect([[float(x), float(y)] for x, y in seq.get("box", [])])
        for seq in seqs
        if float(seq.get("confidence", 0.0)) >= 0.0 and seq.get("box")
    ]

    def outside_ratio(p0: Tuple[int, int], p1: Tuple[int, int], samples: int = 9) -> float:
        if samples <= 1:
            return 1.0
        out = 0
        for k in range(samples):
            t = k / (samples - 1)
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])
            if not _inside_any(x, y, rects_all):
                out += 1
        return out / samples

    for i in range(len(points) - 1):
        if outside_ratio((points[i]["x"], points[i]["y"]), (points[i + 1]["x"], points[i + 1]["y"])) >= 0.8:
            line_jump_indices.append(i)

    payload = {
        "cmd": "path",
        "points": points,
        "speed": "normal",
        "min_total_ms": 0.0,
        "speed_factor": 1.0,
        "min_dt": 0.004,
        "gap_rects": [],
        "gap_boost": 3.0,
        "line_jump_indices": line_jump_indices,
        "line_jump_boost": 1.5,
    }
    return payload


def _send_udp_payload(payload: Dict[str, Any], port: int) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(json.dumps(payload).encode("utf-8"), ("127.0.0.1", port))
        return True
    except Exception as exc:
        log(f"[WARN] Failed to send payload to control agent: {exc}")
        return False


def _send_control_agent(payload: Dict[str, Any], port: int) -> bool:
    cmd = payload.get("cmd")
    if cmd == "path":
        points = payload.get("points") or []
        if not points:
            return False
        min_dt = float(payload.get("min_dt", 0.01))
        ok = False
        for pt in points:
            move_payload = {"cmd": "move", "x": int(pt.get("x", 0)), "y": int(pt.get("y", 0))}
            if _send_udp_payload(move_payload, port):
                ok = True
                time.sleep(max(0.0, min_dt))
        return ok
    else:
        return _send_udp_payload(payload, port)


def dispatch_hover_to_control_agent(points_json: Path) -> None:
    try:
        seqs = json.loads(points_json.read_text(encoding="utf-8"))
        if not isinstance(seqs, list):
            raise ValueError("Hover JSON must contain a list")
    except Exception as exc:
        log(f"[WARN] Could not read hover JSON {points_json}: {exc}")
        return

    payload = _build_hover_path(
        seqs,
        offset_x=HOVER_SIDE_CROP,
        offset_y=HOVER_TOP_CROP,
    )
    if not payload:
        log("[WARN] hover_bot produced insufficient points.")
        return

    if _send_control_agent(payload, CONTROL_AGENT_PORT):
        log(f"[INFO] Sent hover path ({len(payload['points'])} pts) to control agent port {CONTROL_AGENT_PORT}")
        start_hover_fallback_timer()


def finalize_hover_bot(task: Tuple[subprocess.Popen, Path]) -> None:
    proc, json_path = task
    try:
        code = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        log("[WARN] hover_bot timed out and was killed.")
        return
    if code == 0:
        log(f"[INFO] hover_bot completed ({json_path.name})")
        dispatch_hover_to_control_agent(json_path)
    else:
        log(f"[WARN] hover_bot exited with code {code}")


def send_random_click(summary_path: Path, image_path: Path) -> None:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"[WARN] Could not read summary JSON {summary_path}: {exc}")
        update_overlay_status("Summary JSON missing or invalid.")
        return
    top = data.get("top_labels") or {}
    candidates = [entry for entry in top.values() if isinstance(entry, dict) and entry.get("bbox")]
    if not candidates:
        log("[WARN] Summary has no candidates for random click.")
        update_overlay_status("No candidates for random click.")
        return
    try:
        with Image.open(image_path) as im:
            screen_w, screen_h = im.size
    except Exception:
        screen_w, screen_h = (1920, 1080)
    chosen = random.choice(candidates)
    bbox = chosen.get("bbox") or []
    if not bbox or len(bbox) != 4:
        log("[WARN] Candidate without bbox for random click.")
        update_overlay_status("Candidate without bbox for random click.")
        return
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(screen_w, int(x2))
    y2 = min(screen_h, int(y2))
    if x2 - x1 <= 10 or y2 - y1 <= 10:
        log("[WARN] Bounding box too small for random click.")
        update_overlay_status("Bounding box too small for random click.")
        return
    rx_min = max(5, x1 + 5)
    rx_max = min(screen_w - 5, x2 - 5)
    ry_min = max(5, y1 + 5)
