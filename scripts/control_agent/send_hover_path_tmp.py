"""
Send hover-bot dots as a polyline path to ControlAgent over UDP.

Usage (manual):
  python scripts/control_agent/send_hover_path.py --json data/screen/hover/hover_output_current/hover_output.json \
      --port 8765 --speed-factor 1.0 --gap-boost 3.0 --line-jump-boost 1.5 \
      [--offset-x 0 --offset-y 0]

Notes:
  - Assumes dot coordinates are already in screen space; use --offset-* if needed.
  - ControlAgent must be running with UDP receiver (see control_agent.py --port).
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Domyślne ustawienia „ludzkiej” trajektorii hovera.
# Używane zarówno przez to CLI, jak i przez main.py (HOVER_SPEED_DEFAULTS).
DEFAULT_SPEED_MODE = "normal"       # tryb prędkości w control_agent (_target_duration_ms)
DEFAULT_SPEED_FACTOR = 1.0          # globalny multiplicator prędkości segmentów
DEFAULT_MIN_DT = 0.008              # minimalny odstęp czasowy między punktami (~125 Hz)
DEFAULT_SPEED_PX_PER_S = 800.0      # docelowa średnia prędkość kursora w px/s
DEFAULT_GAP_BOOST = 2.0             # przyspieszenie na „dziurach” między boxami (sakkady)
DEFAULT_LINE_JUMP_BOOST = 2.5       # dodatkowe przyspieszenie przy skokach między liniami

# Katalog z profilem „prawdziwego” hovera nagranym przez record_hover_profile.py.
PROFILE_PATH = Path(__file__).resolve().parents[2] / "data" / "hover_profile" / "profile.json"


def _apply_profile_overrides() -> None:
    """
    Jeśli istnieje data/hover_profile/profile.json (tworzone przez
    record_hover_profile.py), nadpisz domyślne parametry trajektorii tak,
    aby odpowiadały prawdziwym ruchom użytkownika.
    """
    global DEFAULT_SPEED_MODE, DEFAULT_SPEED_FACTOR, DEFAULT_MIN_DT
    global DEFAULT_SPEED_PX_PER_S, DEFAULT_GAP_BOOST, DEFAULT_LINE_JUMP_BOOST

    try:
        if not PROFILE_PATH.exists():
            return
        data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        rec = data.get("stats", {}).get("recommended") or data.get("recommended") or data

        mode = rec.get("DEFAULT_SPEED_MODE")
        if isinstance(mode, str):
            DEFAULT_SPEED_MODE = mode
        if rec.get("DEFAULT_SPEED_FACTOR") is not None:
            DEFAULT_SPEED_FACTOR = float(rec["DEFAULT_SPEED_FACTOR"])
        if rec.get("DEFAULT_MIN_DT") is not None:
            DEFAULT_MIN_DT = float(rec["DEFAULT_MIN_DT"])
        if rec.get("DEFAULT_SPEED_PX_PER_S") is not None:
            DEFAULT_SPEED_PX_PER_S = float(rec["DEFAULT_SPEED_PX_PER_S"])
        if rec.get("DEFAULT_GAP_BOOST") is not None:
            DEFAULT_GAP_BOOST = float(rec["DEFAULT_GAP_BOOST"])
        if rec.get("DEFAULT_LINE_JUMP_BOOST") is not None:
            DEFAULT_LINE_JUMP_BOOST = float(rec["DEFAULT_LINE_JUMP_BOOST"])
    except Exception as exc:  # pragma: no cover - tylko log, niekrytyczne
        print(f"[send_hover_path] WARN: could not load hover profile ({exc})")


_apply_profile_overrides()


def _load_sequences(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a list of sequences in JSON")
    return data


def _as_rect(box: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))
    return (x0, y0, x1, y1)


def _group_lines(seqs: List[Dict[str, Any]]) -> List[List[int]]:
    """
    Grupuj boxy w wiersze na podstawie średniej wysokości.
    Zwraca listę grup z indeksami sekwencji.
    """
    entries: List[Tuple[int, float, float, float, float]] = []
    for i, s in enumerate(seqs):
        box = s.get("box") or []
        if not box:
            continue
        ys = [float(p[1]) for p in box]
        min_y, max_y = min(ys), max(ys)
        height = max(1.0, max_y - min_y)
        dots = s.get("dots") or []
        if dots:
            line_y = float(sum(d[1] for d in dots) / len(dots))
        else:
            line_y = 0.5 * (min_y + max_y)
        entries.append((i, min_y, max_y, line_y, height))

    groups: List[List[int]] = []
    ranges: List[Tuple[float, float]] = []
    for i, min_y, max_y, line_y, h in sorted(entries, key=lambda t: t[3]):
        placed = False
        for gi, (gmin, gmax) in enumerate(ranges):
            gc = 0.5 * (gmin + gmax)
            gh = max(1.0, gmax - gmin)
            if abs(line_y - gc) <= 0.45 * max(h, gh):
                ranges[gi] = (min(gmin, min_y), max(gmax, max_y))
                groups[gi].append(i)
                placed = True
                break
        if not placed:
            ranges.append((min_y, max_y))
            groups.append([i])
    return groups


def _is_spacing_like(seq: Dict[str, Any]) -> bool:
    """
    Heurystyka: czy box jest separatorem/odstępem a nie realnym tekstem.
    Używana, aby nie generować punktów tam, gdzie są tylko „dziury”.
    """
    try:
        conf = float(seq.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    if conf < 0.0:
        return True
    text = str(seq.get("text", ""))
    if not text or not text.strip():
        return True
    stripped = text.strip()
    if len(stripped) <= 1 and not any(ch.isalnum() for ch in stripped):
        return True
    return False


def _split_columns(real_idx: List[int], seqs: List[Dict[str, Any]]) -> List[List[int]]:
    """
    Podziel sekwencje na pionowe kolumny (np. różne panele/taby) na
    podstawie zakresów X.
    """
    if not real_idx:
        return []
    entries: List[Tuple[int, float, float]] = []
    widths: List[float] = []
    for idx in real_idx:
        box = seqs[idx].get("box") or []
        if not box:
            continue
        xs = [float(p[0]) for p in box]
        x0, x1 = min(xs), max(xs)
        entries.append((idx, x0, x1))
        widths.append(max(1.0, x1 - x0))
    if not entries:
        return [real_idx]

    entries.sort(key=lambda e: e[1])
    widths.sort()
    median_w = widths[len(widths) // 2] if widths else 80.0
    gap_thr = max(80.0, median_w * 1.5)

    columns: List[List[int]] = []
    cur_col: List[int] = []
    _, cur_min, cur_max = entries[0]
    cur_col.append(entries[0][0])
    for idx, x0, x1 in entries[1:]:
        if x0 - cur_max > gap_thr:
            columns.append(cur_col)
            cur_col = [idx]
            cur_min, cur_max = x0, x1
        else:
            cur_col.append(idx)
            cur_min = min(cur_min, x0)
            cur_max = max(cur_max, x1)
    columns.append(cur_col)
    return columns


def _inside_any(x: float, y: float, rects: List[Tuple[int, int, int, int]]) -> bool:
    for (x0, y0, x1, y1) in rects:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def _build_path_and_meta(
    seqs: List[Dict[str, Any]],
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    spacing_x: float = 1.0,
) -> Tuple[List[Dict[str, int]], List[Tuple[int, int, int, int]], List[int]]:
    """
    Zbuduj listę punktów hovera oraz prostokąty „dziur” i indeksy skoków
    między liniami. To jest wolniejszy, ale czytelniejszy wariant do
    użycia z linii komend / debug.
    """
    # 1) Filtr: dziury vs realny tekst.
    gaps: List[Tuple[int, int, int, int]] = []
    real_idx: List[int] = []
    for i, s in enumerate(seqs):
        if _is_spacing_like(s):
            box = s.get("box") or []
            if box:
                rect = _as_rect([[float(x), float(y)] for x, y in box])
                gaps.append(rect)
            continue
        real_idx.append(i)

    # 2) Podział na kolumny (panele).
    column_indices = _split_columns(real_idx, seqs)

    # 3) Wiersze + lewo->prawo.
    points: List[Dict[str, int]] = []
    line_jump_indices: List[int] = []
    seg_index = -1
    for col in column_indices:
        if not col:
            continue
        local_seqs = [seqs[i] for i in col]
        groups = _group_lines(local_seqs)
        for g in groups:
            order = sorted(
                g,
                key=lambda local_i: min(
                    p[0] for p in local_seqs[local_i].get("box", [[0, 0]])
                ),
            )
            first_in_line = True
            for local_i in order:
                s = local_seqs[local_i]
                dots = s.get("dots") or []
                if not dots:
                    continue
                if first_in_line:
                    if points:
                        line_jump_indices.append(max(0, seg_index))
                    first_in_line = False
                for d in dots:
                    x = int(round(d[0])) + offset_x
                    y = int(round(d[1])) + offset_y
                    points.append({"x": x, "y": y})
                    seg_index += 1

    # 4) Skalowanie odległości w poziomie (spacing_x).
    if len(points) >= 2 and spacing_x != 1.0:
        spaced: List[Dict[str, int]] = [points[0]]
        for i in range(1, len(points)):
            dx = (points[i]["x"] - points[i - 1]["x"]) * spacing_x
            spaced_x = int(round(spaced[i - 1]["x"] + dx))
            spaced.append({"x": spaced_x, "y": points[i]["y"]})
        points = spaced

    return points, gaps, line_jump_indices


def send_udp(port: int, payload: Dict[str, Any]) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = json.dumps(payload).encode("utf-8")
    sock.sendto(data, ("127.0.0.1", port))


def build_payload(
    json_path: Path,
    *,
    port: int = 8765,
    speed_factor: float = DEFAULT_SPEED_FACTOR,
    speed_px_per_s: float = DEFAULT_SPEED_PX_PER_S,
    gap_boost: float = DEFAULT_GAP_BOOST,
    line_jump_boost: float = DEFAULT_LINE_JUMP_BOOST,
    min_dt: float = DEFAULT_MIN_DT,
    offset_x: int = 0,
    offset_y: int = 0,
    spacing_x: float = 1.0,
    speed: str = DEFAULT_SPEED_MODE,
    min_total_ms: float = 0.0,
) -> Dict[str, Any]:
    seqs = _load_sequences(json_path)
    points, gaps, line_jumps = _build_path_and_meta(
        seqs, offset_x=offset_x, offset_y=offset_y, spacing_x=spacing_x
    )
    if len(points) < 2:
        raise ValueError("Not enough points to build a path.")

    payload: Dict[str, Any] = {
        "cmd": "path",
        "points": points,
        "speed": speed,
        "min_total_ms": float(min_total_ms),
        "speed_factor": float(speed_factor),
        "min_dt": float(min_dt),
        "gap_rects": [list(r) for r in gaps],
        "gap_boost": float(gap_boost),
        "line_jump_indices": line_jumps,
        "line_jump_boost": float(line_jump_boost),
    }
    if speed_px_per_s > 0:
        payload["speed_px_per_s"] = float(speed_px_per_s)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    default_json = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "screen"
        / "hover"
        / "hover_output_current"
        / "hover_output.json"
    )
    ap.add_argument(
        "--json",
        type=str,
        default=str(default_json),
        help=f"Path to hover_bot JSON with sequences (default: {default_json})",
    )
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--speed-factor", type=float, default=DEFAULT_SPEED_FACTOR)
    ap.add_argument(
        "--speed-px",
        type=float,
        default=DEFAULT_SPEED_PX_PER_S,
        help="Target speed in px/s (0=disabled)",
    )
    ap.add_argument("--gap-boost", type=float, default=DEFAULT_GAP_BOOST)
    ap.add_argument("--line-jump-boost", type=float, default=DEFAULT_LINE_JUMP_BOOST)
    ap.add_argument("--min-dt", type=float, default=DEFAULT_MIN_DT)
    ap.add_argument("--offset-x", type=int, default=0)
    ap.add_argument("--offset-y", type=int, default=0)
    ap.add_argument("--spacing-x", type=float, default=1.0, help="Scale horizontal spacing between dots")
    ap.add_argument("--speed", type=str, default=DEFAULT_SPEED_MODE, choices=["slow", "normal", "fast"])
    ap.add_argument("--min-total-ms", type=float, default=0.0)
    args = ap.parse_args()

    payload = build_payload(
        Path(args.json),
        port=args.port,
        speed_factor=args.speed_factor,
        speed_px_per_s=args.speed_px,
        gap_boost=args.gap_boost,
        line_jump_boost=args.line_jump_boost,
        min_dt=args.min_dt,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        spacing_x=args.spacing_x,
        speed=args.speed,
        min_total_ms=args.min_total_ms,
    )
    send_udp(args.port, payload)
    print(
        f"[send_hover_path] Sent path with {len(payload['points'])} points, "
        f"gaps={len(payload['gap_rects'])}, line_jumps={len(payload['line_jump_indices'])} to port {args.port}"
    )


if __name__ == "__main__":
    main()

