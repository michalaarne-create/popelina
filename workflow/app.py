try:
    import dearpygui.dearpygui as dpg
except ModuleNotFoundError:
    dpg = None  # type: ignore[assignment]
from PIL import Image, ImageDraw
import numpy as np
import os
import sys
import json as pyjson
import math
import time
import threading
from pathlib import Path
import contextlib

WINDOW_TAG = "main_window"
CANVAS_TAG = "canvas"
TEXREG_TAG = "texture_registry"
JSON_VIEWER_TAG = "json_viewer"
JSON_TEXT_TAG = "json_viewer_text"
NOTE_EDITOR_TAG = "note_editor"
NOTE_TEXT_TAG = "note_editor_text"
STATE_FILE = "flowui_state.json"
PIPELINE_STATUS_TAG = "pipeline_status"
BACKUP_WINDOW_TAG = "pipeline_backup_window"
PIPELINE_LIST_TAG = "pipeline_run_list"
PIPELINE_TREE_TAG = "pipeline_tree"
MAX_BACKUP_DAYS = 7
POLISH_DAY_NAMES = ["Poniedzialek", "Wtorek", "Sroda", "Czwartek", "Piatek", "Sobota", "Niedziela"]
ROOT_DIR = Path(__file__).resolve().parents[1]
PIPELINE_RUNS_DIR = ROOT_DIR / "data" / "screen" / "pipeline_runs"
DATA_SCREEN_DIR = ROOT_DIR / "data" / "screen"
DOM_LIVE_DIR = ROOT_DIR / "dom_live"
DOM_LIVE_DEBUG_DIR = DOM_LIVE_DIR / "debug"
BRAIN_STATE_FILE = ROOT_DIR / "data" / "brain_state.json"
DEBUG_SCREEN_DIR = DATA_SCREEN_DIR / "debug"
DEBUG_SCREEN_DIR.mkdir(parents=True, exist_ok=True)
RAW_SCREENS_CURRENT_DIR = DATA_SCREEN_DIR / "raw" / "raw_screens_current"
RAW_SCREENS_DIR = DATA_SCREEN_DIR / "raw" / "raw screen"
SCREEN_BOXES_DIR = DATA_SCREEN_DIR / "numpy_points" / "screen_boxes"
CURRENT_RUN_DIR = DATA_SCREEN_DIR / "current_run"
REGION_GROW_BASE_DIR = DATA_SCREEN_DIR / "region_grow"
REGION_GROW_ANNOT_DIR = REGION_GROW_BASE_DIR / "region_grow_annot_current"
REGION_GROW_JSON_DIR = REGION_GROW_BASE_DIR / "region_grow"
REGION_GROW_ANNOT_HISTORY_DIR = REGION_GROW_BASE_DIR / "region_grow_annot"
REGION_GROW_CURRENT_DIR = REGION_GROW_BASE_DIR / "region_grow_current"
REGION_GROW_REGIONS_CURRENT_DIR = REGION_GROW_BASE_DIR / "regions_current"
REGION_GROW_REGIONS_DIR = REGION_GROW_BASE_DIR / "regions"
REGION_GROW_ANNOT_CURRENT_FILE = REGION_GROW_BASE_DIR / "region_grow_annot_current.png"
RATE_RESULTS_CURRENT_DIR = DATA_SCREEN_DIR / "rate" / "rate_results_current"
RATE_SUMMARY_CURRENT_DIR = DATA_SCREEN_DIR / "rate" / "rate_summary_current"
RATE_RESULTS_DIR = DATA_SCREEN_DIR / "rate" / "rate_results"
RATE_SUMMARY_DIR = DATA_SCREEN_DIR / "rate" / "rate_summary"
HOVER_OUTPUT_DIR = DATA_SCREEN_DIR / "hover" / "hover_output_current"
RATE_RESULTS_DEBUG_CURRENT_DIR = DATA_SCREEN_DIR / "rate" / "rate_results_debug_current"
HOVER_INPUT_CURRENT_DIR = DATA_SCREEN_DIR / "hover" / "hover_input_current"
HOVER_PATH_CURRENT_DIR = DATA_SCREEN_DIR / "hover" / "hover_path_current"
HOVER_SPEED_CURRENT_DIR = DATA_SCREEN_DIR / "hover" / "hover_speed_current"
HOVER_POINTS_ON_PATH_CURRENT_DIR = DATA_SCREEN_DIR / "hover" / "hover_points_on_path_current"
HOVER_POINTS_ON_SPEED_CURRENT_DIR = DATA_SCREEN_DIR / "hover" / "hover_points_on_speed_current"
FILE_DIALOG_TAG = "file_dialog_id"

# Stan
screens = []
json_objects = []
notes = []
screen_counter = 0
texture_counter = 0
undo_stack = []
UNDO_LIMIT = 20

lines = []

pan_x = 0.0
pan_y = 0.0
zoom = 1.0

active_object_id = None
drag_offset = (0.0, 0.0)

is_panning = False
last_mouse_pos = (0.0, 0.0)

# Resize
resize_mode = False
resize_enabled = False
resize_object_id = None
resize_edge_right = False
resize_edge_bottom = False
MIN_SIZE = 50

# Rysowanie linii (D)
draw_line_mode = False
line_start = None

# Gumka do linii (C)
erase_mode = False
ERASE_DISTANCE_PX = 15

# Auto-reload
auto_reload_enabled = True
AUTO_RELOAD_INTERVAL = 2

# Tryb dodawania notatki (N)
add_note_mode = False

# Aktywna notatka do edycji
active_note_id = None

# Flaga - czy trzeba zaktualizowa─ç tekstur─Ö po resize
needs_texture_update = False


# ================== PIPELINE DEFAULT STATE ==================

def _latest_file(dir_path: Path, suffixes) -> str | None:
    try:
        candidates = [p for p in Path(dir_path).iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    except Exception:
        return None
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def _all_files(dir_path: Path, suffixes) -> list[Path]:
    """Zwróć wszystkie pliki z katalogu o podanych rozszerzeniach (posortowane po czasie)."""
    try:
        candidates = [p for p in Path(dir_path).iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    except Exception:
        return []
    return sorted(candidates, key=lambda p: p.stat().st_mtime)


def _default_size_for_image(path: str) -> tuple[float, float]:
    def _scale_to_max(w: float, h: float, max_width: float = 700.0) -> tuple[float, float]:
        """Skaluje proporcjonalnie do max_width (używane tylko do WYŚWIETLANIA)."""
        if w <= max_width:
            return w, h
        scale = max_width / w
        return w * scale, h * scale

    try:
        with Image.open(path) as im:
            w, h = float(im.width), float(im.height)
            return _scale_to_max(w, h)
    except Exception:
        return 1920.0, 1080.0


def _build_pipeline_state_legacy() -> dict:
    """Zbuduj uporządkowany widok CAŁEGO pipeline'u z artefaktów *_current* i debug.

    Główna ścieżka (capture -> region -> hover/rating -> summary) leci w pierwszym rzędzie,
    a wszystkie debugowe pliki wiszą w kolumnach poniżej swoich „rodziców”.
    """
    # Kolumny (x) – osobne dla każdego głównego etapu
    x_capture = 0.0
    x_region_annot = 1400.0
    x_region_json = 2600.0
    x_hover = 3800.0
    x_rating = 5000.0
    x_summary = 6200.0

    # Rzędy (y)
    y_main = 0.0          # główna linia pipeline'u
    row_step = 520.0      # odstęp między kolejnymi „piętrami” debugów

    screens_state: list[dict] = []
    json_state: list[dict] = []
    notes_state: list[dict] = []
    lines_state: list[dict] = []

    added_paths: set[str] = set()

    def add_screen(path: Path, x: float, y: float):
        sp = str(path)
        if sp in added_paths:
            return None
        w, h = _default_size_for_image(sp)
        obj = {"path": sp, "x": x, "y": y, "w": w, "h": h}
        screens_state.append(obj)
        added_paths.add(sp)
        # Niebieski przypis z nazwą pliku tuż pod screenem
        note_text = os.path.basename(sp)
        note_width = 260.0
        note_height = 70.0
        notes_state.append(
            {
                "text": note_text,
                "x": x,
                "y": y + h + 30.0,
                "w": note_width,
                "h": note_height,
            }
        )
        return obj

    def add_json_obj(path: Path, x: float, y: float):
        sp = str(path)
        if sp in added_paths:
            return None
        obj = {"path": sp, "x": x, "y": y, "w": 260.0, "h": 180.0}
        json_state.append(obj)
        added_paths.add(sp)
        return obj

    # ===== CAPTURE + OCR DEBUG =====
    capture_files = _all_files(RAW_SCREENS_CURRENT_DIR, {".png", ".jpg", ".jpeg"})
    if not capture_files:
        capture_files = _all_files(CURRENT_RUN_DIR, {".png", ".jpg", ".jpeg"})

    capture_main = None
    debug_row = 1
    for idx, p in enumerate(capture_files):
        if idx == 0:
            capture_main = add_screen(p, x_capture, y_main)
        else:
            add_screen(p, x_capture, y_main + debug_row * row_step)
            debug_row += 1

    ocr_debug_files = _all_files(DEBUG_SCREEN_DIR, {".png"})
    for p in ocr_debug_files:
        add_screen(p, x_capture, y_main + debug_row * row_step)
        debug_row += 1

    # ===== REGION_GROW =====
    annot_files = _all_files(REGION_GROW_ANNOT_DIR, {".png", ".jpg", ".jpeg"})
    region_annot_main = None
    debug_row_region_annot = 1
    for idx, p in enumerate(annot_files):
        if idx == 0:
            region_annot_main = add_screen(p, x_region_annot, y_main)
        else:
            add_screen(p, x_region_annot, y_main + debug_row_region_annot * row_step)
            debug_row_region_annot += 1

    rg_json_files = _all_files(REGION_GROW_CURRENT_DIR, {".json"})
    region_json_main = None
    debug_row_region_json = 1
    for idx, p in enumerate(rg_json_files):
        if idx == 0:
            region_json_main = add_json_obj(p, x_region_json, y_main)
        else:
            add_json_obj(p, x_region_json, y_main + debug_row_region_json * row_step)
            debug_row_region_json += 1

    # ===== HOVER (wszystkie debugowe obrazki i JSON-y) =====
    hover_main_json = None
    debug_row_hover = 1

    # input / path / speed obrazy
    hover_input_files = _all_files(HOVER_INPUT_CURRENT_DIR, {".png", ".jpg", ".jpeg"})
    for p in hover_input_files:
        add_screen(p, x_hover, y_main + debug_row_hover * row_step)
        debug_row_hover += 1

    hover_path_files = _all_files(HOVER_PATH_CURRENT_DIR, {".png", ".jpg", ".jpeg"})
    for p in hover_path_files:
        add_screen(p, x_hover, y_main + debug_row_hover * row_step)
        debug_row_hover += 1

    hover_speed_files = _all_files(HOVER_SPEED_CURRENT_DIR, {".png", ".jpg", ".jpeg"})
    for p in hover_speed_files:
        add_screen(p, x_hover, y_main + debug_row_hover * row_step)
        debug_row_hover += 1

    # punkty na ścieżce / prędkości – JSON/NPY/TXT
    hover_points_path_files = _all_files(HOVER_POINTS_ON_PATH_CURRENT_DIR, {".json", ".npy", ".txt"})
    for p in hover_points_path_files:
        add_json_obj(p, x_hover, y_main + debug_row_hover * row_step)
        debug_row_hover += 1

    hover_points_speed_files = _all_files(HOVER_POINTS_ON_SPEED_CURRENT_DIR, {".json", ".npy", ".txt"})
    for p in hover_points_speed_files:
        add_json_obj(p, x_hover, y_main + debug_row_hover * row_step)
        debug_row_hover += 1

    # hover_output_current – JSON (ostatni jako główny) + PNG
    hover_output_jsons = _all_files(HOVER_OUTPUT_DIR, {".json"})
    if hover_output_jsons:
        main_hover_json_path = hover_output_jsons[-1]
        for p in hover_output_jsons:
            if p == main_hover_json_path:
                hover_main_json = add_json_obj(p, x_hover, y_main)
            else:
                add_json_obj(p, x_hover, y_main + debug_row_hover * row_step)
                debug_row_hover += 1

    hover_output_imgs = _all_files(HOVER_OUTPUT_DIR, {".png", ".jpg", ".jpeg"})
    for p in hover_output_imgs:
        add_screen(p, x_hover, y_main + debug_row_hover * row_step)
        debug_row_hover += 1

    # ===== RATING (wyniki + debug + summary) =====
    rating_main_json = None
    summary_main_json = None
    debug_row_rating = 1
    debug_row_summary = 1

    rate_results_files = _all_files(RATE_RESULTS_CURRENT_DIR, {".json"})
    for idx, p in enumerate(rate_results_files):
        if idx == 0:
            rating_main_json = add_json_obj(p, x_rating, y_main)
        else:
            add_json_obj(p, x_rating, y_main + debug_row_rating * row_step)
            debug_row_rating += 1

    # Debug ratingu – każdy osobny box pod rating_main_json
    rate_debug_files = _all_files(RATE_RESULTS_DEBUG_CURRENT_DIR, {".json"})
    for p in rate_debug_files:
        add_json_obj(p, x_rating, y_main + debug_row_rating * row_step)
        debug_row_rating += 1

    # Summary (ostatni jako główny w osobnej kolumnie)
    rate_summary_files = _all_files(RATE_SUMMARY_CURRENT_DIR, {".json"})
    if rate_summary_files:
        main_summary_path = rate_summary_files[-1]
        for p in rate_summary_files:
            if p == main_summary_path:
                summary_main_json = add_json_obj(p, x_summary, y_main)
            else:
                add_json_obj(p, x_summary, y_main + debug_row_summary * row_step)
                debug_row_summary += 1

    # ===== Połączenia (główna gałąź + pionowe debugi) =====
    def center_x(item): return item["x"] + item["w"] * 0.5
    def center_y(item): return item["y"] + item["h"] * 0.5
    def left_x(item): return item["x"]
    def right_x(item): return item["x"] + item["w"]
    def top_y(item): return item["y"]
    def bottom_y(item): return item["y"] + item["h"]

    def add_line(src, dst):
        if not src or not dst:
            return

        dx = dst["x"] - src["x"]
        dy = dst["y"] - src["y"]

        # Poziome połączenia między etapami głównej ścieżki:
        # linia idzie od prawej krawędzi źródła do lewej krawędzi celu,
        # dzięki czemu nie przechodzi przez wnętrze boxów.
        if abs(dy) < 1e-3:
            if src["x"] <= dst["x"]:
                x1 = right_x(src)
                y1 = center_y(src)
                x2 = left_x(dst)
                y2 = center_y(dst)
            else:
                x1 = left_x(src)
                y1 = center_y(src)
                x2 = right_x(dst)
                y2 = center_y(dst)
        # Pionowe połączenia w kolumnie debugowej: od dołu elementu głównego
        # do góry dziecka, by linia nie przecinała wnętrza.
        elif abs(dx) < 1e-3:
            if src["y"] <= dst["y"]:
                x1 = center_x(src)
                y1 = bottom_y(src)
                x2 = center_x(dst)
                y2 = top_y(dst)
            else:
                x1 = center_x(src)
                y1 = top_y(src)
                x2 = center_x(dst)
                y2 = bottom_y(dst)
        # Fallback dla nietypowych/ręcznie rysowanych połączeń: środki obu boxów.
        else:
            x1 = center_x(src)
            y1 = center_y(src)
            x2 = center_x(dst)
            y2 = center_y(dst)

        lines_state.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    # Główna, prosta ścieżka w pierwszym rzędzie
    add_line(capture_main, region_annot_main or region_json_main)
    add_line(region_annot_main, region_json_main)
    add_line(region_json_main, hover_main_json)
    add_line(region_json_main, rating_main_json)
    add_line(rating_main_json, summary_main_json)

    # Pionowe gałęzie debugowe
    def connect_column(parent, x_col: float):
        """Połącz element główny z wszystkimi obiektami w tej samej kolumnie poniżej."""
        if not parent:
            return
        for obj in screens_state + json_state:
            if obj is parent:
                continue
            if abs(obj["x"] - x_col) < 1e-3 and obj["y"] > parent["y"]:
                add_line(parent, obj)

    connect_column(capture_main, x_capture)
    connect_column(region_annot_main, x_region_annot)
    connect_column(region_json_main, x_region_json)
    connect_column(hover_main_json, x_hover)
    connect_column(rating_main_json, x_rating)
    connect_column(summary_main_json, x_summary)

    return {
        "pan_x": -800.0,
        "pan_y": -320.0,
        "zoom": 0.23,
        "screens": screens_state,
        "json_objects": json_state,
        "notes": notes_state,
        "lines": lines_state,
    }


def build_pipeline_state() -> dict:
    """Buduje uporzadkowany diagram pipeline'u (pliki + debug) z zachowaniem kolejnosci.

    Zasady:
    - input z lewej, output z prawej,
    - pomiedzy nimi notatka z nazwa skryptu,
    - debugowe artefakty (png/json) wisza ponizej i sa polaczone NIEBIESKIMI liniami,
    - wszystkie linie sa ortogonalne (kat prosty) i wychodza z krawedzi boxow,
    - linie prowadzone sa "po korytarzach" (poza boxami), zeby nie nachodzily na screeny/jsony.

    Kolory:
    - zielony: przeplyw danych (flow),
    - szary: wywolania/relacje "kto co odpala" (call),
    - niebieski: debug.
    """
    # ===== Layout (kolumny) =====
    x_main_note = -12000.0
    x_capture_note = -8000.0
    x_dom_out = -8000.0
    x_capture_out = -5000.0

    x_region_note = -1000.0
    x_region_out = 3000.0

    x_action_note = 7000.0
    x_action_out = 10000.0
    x_summary_out = 13000.0

    x_brain_note = 16000.0
    x_brain_out = 19000.0

    # ===== Layout (rzedy) =====
    row_step = 900.0
    y_screen_main = 0.0
    y_json_main = 520.0
    y_note_main = -750.0

    y_hover_base = 8000.0  # osobny "tor" dla hover (duzo miejsca)

    # ===== Zbieranie obiektow =====
    screens_state: list[dict] = []
    json_state: list[dict] = []
    notes_state: list[dict] = []
    lines_state: list[dict] = []

    added_paths: set[str] = set()

    COLOR_FLOW = [0, 255, 0, 255]
    COLOR_CALL = [160, 160, 160, 255]
    COLOR_DEBUG = [120, 120, 255, 255]

    def _basename(p: str) -> str:
        try:
            return os.path.basename(p)
        except Exception:
            return p

    def add_label(text: str, x: float, y: float, w: float = 260.0, h: float = 140.0):
        notes_state.append({"text": text, "x": x, "y": y, "w": w, "h": h, "role": "label"})

    def add_note_node(text: str, x: float, y: float, w: float = 420.0, h: float = 220.0, role: str = "process"):
        obj = {"text": text, "x": x, "y": y, "w": w, "h": h, "role": role}
        notes_state.append(obj)
        return obj

    def add_screen(path: Path, x: float, y: float):
        sp = str(path)
        if sp in added_paths:
            return None
        w, h = _default_size_for_image(sp)
        obj = {"path": sp, "x": x, "y": y, "w": w, "h": h}
        screens_state.append(obj)
        added_paths.add(sp)
        add_label(_basename(sp), x, y + h + 30.0)
        return obj

    def add_json_obj(path: Path, x: float, y: float):
        sp = str(path)
        if sp in added_paths:
            return None
        obj = {"path": sp, "x": x, "y": y, "w": 260.0, "h": 180.0}
        json_state.append(obj)
        added_paths.add(sp)
        return obj

    # ===== Linie (ortogonalne) =====
    def center_x(item): return float(item["x"]) + float(item["w"]) * 0.5
    def center_y(item): return float(item["y"]) + float(item["h"]) * 0.5
    def left_x(item): return float(item["x"])
    def right_x(item): return float(item["x"]) + float(item["w"])
    def top_y(item): return float(item["y"])
    def bottom_y(item): return float(item["y"]) + float(item["h"])

    def _anchor(item, side: str) -> list[float]:
        if side == "right":
            return [right_x(item), center_y(item)]
        if side == "left":
            return [left_x(item), center_y(item)]
        if side == "top":
            return [center_x(item), top_y(item)]
        if side == "bottom":
            return [center_x(item), bottom_y(item)]
        return [center_x(item), center_y(item)]

    def _orth_points(start: list[float], end: list[float], via_x: float | None = None, via_y: float | None = None) -> list[list[float]]:
        sx, sy = float(start[0]), float(start[1])
        ex, ey = float(end[0]), float(end[1])
        if abs(sx - ex) < 1e-6 or abs(sy - ey) < 1e-6:
            return [[sx, sy], [ex, ey]]
        if via_x is not None and via_y is not None:
            vx = float(via_x)
            vy = float(via_y)
            return [[sx, sy], [vx, sy], [vx, vy], [ex, vy], [ex, ey]]
        if via_x is not None:
            vx = float(via_x)
            return [[sx, sy], [vx, sy], [vx, ey], [ex, ey]]
        if via_y is not None:
            vy = float(via_y)
            return [[sx, sy], [sx, vy], [ex, vy], [ex, ey]]
        mx = (sx + ex) * 0.5
        return [[sx, sy], [mx, sy], [mx, ey], [ex, ey]]

    def add_conn(
        src,
        dst,
        color=None,
        thickness: int = 2,
        from_side: str | None = None,
        to_side: str | None = None,
        via_x: float | None = None,
        via_y: float | None = None,
    ):
        if not src or not dst:
            return
        if color is None:
            color = COLOR_FLOW

        if from_side is None or to_side is None:
            if left_x(dst) >= right_x(src):
                from_side = from_side or "right"
                to_side = to_side or "left"
            elif left_x(src) >= right_x(dst):
                from_side = from_side or "left"
                to_side = to_side or "right"
            elif top_y(dst) >= bottom_y(src):
                from_side = from_side or "bottom"
                to_side = to_side or "top"
            else:
                from_side = from_side or "top"
                to_side = to_side or "bottom"

        start = _anchor(src, from_side)
        end = _anchor(dst, to_side)

        stub = 30.0
        if from_side == "right":
            start2 = [start[0] + stub, start[1]]
        elif from_side == "left":
            start2 = [start[0] - stub, start[1]]
        elif from_side == "bottom":
            start2 = [start[0], start[1] + stub]
        elif from_side == "top":
            start2 = [start[0], start[1] - stub]
        else:
            start2 = [start[0], start[1]]

        if to_side == "left":
            end2 = [end[0] - stub, end[1]]
        elif to_side == "right":
            end2 = [end[0] + stub, end[1]]
        elif to_side == "top":
            end2 = [end[0], end[1] - stub]
        elif to_side == "bottom":
            end2 = [end[0], end[1] + stub]
        else:
            end2 = [end[0], end[1]]

        mid = _orth_points(start2, end2, via_x=via_x, via_y=via_y)
        if not mid:
            return

        pts = [start]
        if abs(pts[-1][0] - start2[0]) > 1e-9 or abs(pts[-1][1] - start2[1]) > 1e-9:
            pts.append(start2)
        for p in mid[1:]:
            if abs(pts[-1][0] - p[0]) > 1e-9 or abs(pts[-1][1] - p[1]) > 1e-9:
                pts.append(p)
        if abs(pts[-1][0] - end[0]) > 1e-9 or abs(pts[-1][1] - end[1]) > 1e-9:
            pts.append(end)

        lines_state.append({"points": pts, "color": list(color), "thickness": int(thickness)})

    def add_debug_fan(parent, children: list, color=None, thickness: int = 2, bus_x: float | None = None):
        if not parent or not children:
            return
        if color is None:
            color = COLOR_DEBUG
        if bus_x is None:
            col_left = min([left_x(parent)] + [left_x(c) for c in children])
            bus_x = col_left - 120.0
        bus_x = float(bus_x)
        for ch in children:
            # Jesli "szyna" jest po prawej stronie dziecka, dopinaj do PRAWEJ krawedzi,
            # w przeciwnym razie do LEWEJ, zeby nie przechodzic przez wnętrze boxa.
            to_side = "right" if bus_x >= right_x(ch) else "left"
            add_conn(parent, ch, color=color, thickness=thickness, from_side="bottom", to_side=to_side, via_x=bus_x)

    # ===== Notatki (pliki skryptow / etapy) =====
    y_header = y_note_main - 1300.0
    main_note = add_note_node("main.py (orchestrator)", x_main_note, y_header, w=520.0, h=260.0, role="process")
    capture_note = add_note_node("dom_renderer/ai_recorder_live.py", x_capture_note, y_note_main, role="process")
    region_note = add_note_node("utils/region_grow.py", x_region_note, y_note_main, role="process")
    rating_note = add_note_node("scripts/numpy_rate/rating.py", x_action_note, y_note_main, role="process")
    brain_note = add_note_node("utils/pipeline_brain_agent.py\\nscripts/pipeline_brain.py", x_brain_note, y_note_main, role="process")
    hover_note = add_note_node(
        "scripts/control_agent/control_agent.py\\n(+ hover_speed_recorder.py)",
        x_action_note,
        y_hover_base + y_note_main,
        role="process",
    )
    arrow_post_note = add_note_node(
        "scripts/arrow_post_region.py (MISSING)",
        x_region_note,
        y_note_main + 260.0,
        w=420.0,
        h=180.0,
        role="process",
    )

    meta_lane_y = y_header + 80.0
    add_conn(main_note, capture_note, color=COLOR_CALL, thickness=2, via_y=meta_lane_y)
    add_conn(main_note, region_note, color=COLOR_CALL, thickness=2, via_y=meta_lane_y)
    add_conn(main_note, rating_note, color=COLOR_CALL, thickness=2, via_y=meta_lane_y)
    add_conn(main_note, brain_note, color=COLOR_CALL, thickness=2, via_y=meta_lane_y)
    add_conn(main_note, hover_note, color=COLOR_CALL, thickness=2, via_y=meta_lane_y)

    # ===== CAPTURE (screenshot) =====
    def add_screen_if_exists(p: Path, x: float, y: float):
        if p and Path(p).is_file():
            return add_screen(Path(p), x, y)
        return None

    def add_json_if_exists(p: Path, x: float, y: float):
        if p and Path(p).is_file():
            return add_json_obj(Path(p), x, y)
        return None

    capture_main = add_screen_if_exists(RAW_SCREENS_CURRENT_DIR / "screenshot.png", x_capture_out, y_screen_main)
    if not capture_main:
        capture_main = add_screen_if_exists(CURRENT_RUN_DIR / "screenshot.png", x_capture_out, y_screen_main)

    add_conn(capture_note, capture_main, color=COLOR_CALL, thickness=2)

    # capture debug: kopia w current_run + OCR stripy (data/screen/debug + dom_live/debug)
    capture_debug_screens: list = []
    # Tylko pliki "current"/auto-updated (bez historycznych dumpow)
    for p in [
        CURRENT_RUN_DIR / "screenshot.png",
        DEBUG_SCREEN_DIR / "ocr_strip1.png",
        DEBUG_SCREEN_DIR / "ocr_strip1_active.png",
        DEBUG_SCREEN_DIR / "ocr_strip2_center.png",
        DOM_LIVE_DEBUG_DIR / "ocr_strip1.png",
        DOM_LIVE_DEBUG_DIR / "ocr_strip1_active.png",
        DOM_LIVE_DEBUG_DIR / "ocr_strip2_center.png",
    ]:
        obj = add_screen_if_exists(p, x_capture_out, y_screen_main + row_step * (len(capture_debug_screens) + 1))
        if obj:
            capture_debug_screens.append(obj)

    # Historia raw_screens (raw screen/*) - najnowszy (opcjonalnie)
    try:
        raw_hist = _all_files(RAW_SCREENS_DIR, {".png", ".jpg", ".jpeg"})
    except Exception:
        raw_hist = []
    if raw_hist:
        obj = add_screen(raw_hist[-1], x_capture_out, y_screen_main + row_step * (len(capture_debug_screens) + 1))
        if obj:
            capture_debug_screens.append(obj)

    add_debug_fan(capture_main, capture_debug_screens)

    # capture debug JSON-y (OCR/timings + pomocnicze)
    capture_debug_jsons: list = []
    y_capture_json = y_screen_main + row_step * (len(capture_debug_screens) + 3)

    for p in [
        DEBUG_SCREEN_DIR / "ocr_debug.json",
        DOM_LIVE_DEBUG_DIR / "ocr_debug.json",
        CURRENT_RUN_DIR / "screenshot_timings.json",
    ]:
        obj = add_json_if_exists(p, x_capture_out, y_capture_json + row_step * len(capture_debug_jsons))
        if obj:
            capture_debug_jsons.append(obj)

    # Screen boxes (debug) - najnowszy JSON
    try:
        sb_files = _all_files(SCREEN_BOXES_DIR, {".json"})
    except Exception:
        sb_files = []
    if sb_files:
        obj = add_json_obj(sb_files[-1], x_capture_out, y_capture_json + row_step * len(capture_debug_jsons))
        if obj:
            capture_debug_jsons.append(obj)

    add_debug_fan(capture_main or capture_note, capture_debug_jsons, bus_x=x_capture_out - 180.0)

    # DOM LIVE / Recorder debug (osobna kolumna, zeby nie mieszać z capture)
    dom_debug_jsons: list = []
    dom_live_objs: dict[str, dict] = {}
    y_dom_json = y_json_main

    cpu_profile = add_json_if_exists(DOM_LIVE_DEBUG_DIR / "cpu_profile.json", x_dom_out, y_dom_json + row_step * len(dom_debug_jsons))
    if cpu_profile:
        dom_debug_jsons.append(cpu_profile)
        dom_live_objs["cpu_profile.json"] = cpu_profile

    dom_live_json_names = [
        "current_snapshot.json",
        "current_page.json",
        "current_tabs.json",
        "current_clickables.json",
        "out.json",
        "stats.json",
    ]
    for name in dom_live_json_names:
        p = DOM_LIVE_DIR / name
        obj = add_json_if_exists(p, x_dom_out, y_dom_json + row_step * len(dom_debug_jsons))
        if obj:
            dom_debug_jsons.append(obj)
            dom_live_objs[name] = obj

    # Dodatkowe JSON-y z dom_live/debug (jeśli są)
    try:
        extra_dom_debug = _all_files(DOM_LIVE_DEBUG_DIR, {".json"})
    except Exception:
        extra_dom_debug = []
    extra_added = 0
    for p in extra_dom_debug:
        if p.name in ("cpu_profile.json", "ocr_debug.json"):
            continue
        obj = add_json_if_exists(p, x_dom_out, y_dom_json + row_step * len(dom_debug_jsons))
        if obj:
            dom_debug_jsons.append(obj)
            dom_live_objs[p.name] = obj
            extra_added += 1
        if extra_added >= 6:
            break

    add_debug_fan(capture_note, dom_debug_jsons, bus_x=x_dom_out - 220.0)

    # current_question.json jako INPUT do ratingu (po prawej od region_grow output)
    current_question = None
    cq_path = DOM_LIVE_DIR / "current_question.json"
    if cq_path.is_file():
        # Osobny "input box" dla ratingu, tak zeby nie nachodzil na region_grow debug PNG.
        current_question = add_json_obj(cq_path, x_action_note - 700.0, y_json_main + 250.0)
        y_mid_lane = y_note_main + 260.0
        add_conn(capture_note, current_question, color=COLOR_FLOW, thickness=2, via_y=y_mid_lane)

    # ===== REGION_GROW =====
    region_json_main = None
    region_json_main = add_json_if_exists(REGION_GROW_CURRENT_DIR / "region_grow.json", x_region_out, y_json_main)

    add_conn(capture_main, region_note)
    add_conn(region_note, region_json_main, color=COLOR_CALL, thickness=2)
    add_conn(region_note, arrow_post_note, color=COLOR_CALL, thickness=2)

    # region debug PNG-y (annot + regions)
    region_debug_screens: list = []
    obj = add_screen_if_exists(REGION_GROW_ANNOT_DIR / "region_grow_annot_current.png", x_region_out, y_screen_main + row_step)
    if obj:
        region_debug_screens.append(obj)

    if REGION_GROW_ANNOT_CURRENT_FILE.is_file():
        obj = add_screen(
            REGION_GROW_ANNOT_CURRENT_FILE,
            x_region_out,
            y_screen_main + row_step * (len(region_debug_screens) + 2),
        )
        if obj:
            region_debug_screens.append(obj)

    regions_current = REGION_GROW_REGIONS_CURRENT_DIR / "regions_current.png"
    if regions_current.is_file():
        obj = add_screen(regions_current, x_region_out, y_screen_main + row_step * (len(region_debug_screens) + 2))
        if obj:
            region_debug_screens.append(obj)

    # Historia: najnowszy annot (region_grow_annot/*)
    try:
        annot_hist = _all_files(REGION_GROW_ANNOT_HISTORY_DIR, {".png", ".jpg", ".jpeg"})
    except Exception:
        annot_hist = []
    if annot_hist:
        obj = add_screen(annot_hist[-1], x_region_out, y_screen_main + row_step * (len(region_debug_screens) + 2))
        if obj:
            region_debug_screens.append(obj)

    # Regions JSON (input dla dalszego pipeline'u/brain) – bez notatki pod JSON-em.
    # Dodajemy zawsze (nawet jeśli plik jeszcze nie istnieje), żeby pipeline był kompletny w UI.
    regions_current_json = add_json_obj(
        REGION_GROW_REGIONS_CURRENT_DIR / "regions_current.json",
        x_region_out,
        y_json_main + 240.0,
    )
    add_conn(region_note, regions_current_json, color=COLOR_CALL, thickness=2)

    add_debug_fan(region_json_main or region_note, region_debug_screens, bus_x=x_region_out + 700.0 + 120.0)

    # region debug JSON-y (np. current_run/region_grow.json)
    region_debug_jsons: list = []
    rg_run = CURRENT_RUN_DIR / "region_grow.json"
    obj = add_json_if_exists(rg_run, x_region_out, y_screen_main + row_step * (len(region_debug_screens) + 4))
    if obj:
        region_debug_jsons.append(obj)

    # Historia RG (region_grow/*.json) - najnowszy
    try:
        rg_hist = _all_files(REGION_GROW_JSON_DIR, {".json"})
    except Exception:
        rg_hist = []
    if rg_hist:
        obj = add_json_obj(
            rg_hist[-1],
            x_region_out,
            y_screen_main + row_step * (len(region_debug_screens) + 4) + row_step * len(region_debug_jsons),
        )
        if obj:
            region_debug_jsons.append(obj)
    add_debug_fan(region_json_main or region_note, region_debug_jsons, bus_x=x_region_out + 700.0 + 120.0)

    # ===== RATING =====
    rating_main_json = None
    rating_main_json = add_json_if_exists(RATE_RESULTS_CURRENT_DIR / "rating.json", x_action_out, y_json_main)

    add_conn(region_json_main, rating_note)
    if current_question:
        add_conn(current_question, rating_note)
    add_conn(rating_note, rating_main_json, color=COLOR_CALL, thickness=2)

    # Debug ratingu (rate_results_debug_current) + current_run/rating.json
    rating_debug_jsons: list = []
    # *_current powinno miec max 1 plik, ale jeśli jest wiecej - bierzemy najnowszy.
    try:
        dbg_files = _all_files(RATE_RESULTS_DEBUG_CURRENT_DIR, {".json"})
    except Exception:
        dbg_files = []
    if dbg_files:
        obj = add_json_obj(dbg_files[-1], x_action_out, y_screen_main + row_step * (len(rating_debug_jsons) + 1))
        if obj:
            rating_debug_jsons.append(obj)

    rating_run = CURRENT_RUN_DIR / "rating.json"
    if rating_run.is_file():
        obj = add_json_obj(rating_run, x_action_out, y_screen_main + row_step * (len(rating_debug_jsons) + 1))
        if obj:
            rating_debug_jsons.append(obj)

    # Historia wyników ratingu (rate_results/*.json) - najnowszy
    try:
        rating_hist = _all_files(RATE_RESULTS_DIR, {".json"})
    except Exception:
        rating_hist = []
    if rating_hist:
        obj = add_json_obj(rating_hist[-1], x_action_out, y_screen_main + row_step * (len(rating_debug_jsons) + 1))
        if obj:
            rating_debug_jsons.append(obj)

    add_debug_fan(rating_main_json or rating_note, rating_debug_jsons)

    # Summary (ostatni jako main po prawej, reszta jako debug ponizej)
    summary_main_json = None
    summary_debug_jsons: list = []
    # Bez historycznych: preferuj summary.json, w razie braku bierz najnowszy.
    summary_path = RATE_SUMMARY_CURRENT_DIR / "summary.json"
    if not summary_path.is_file():
        summaries = _all_files(RATE_SUMMARY_CURRENT_DIR, {".json"})
        summary_path = summaries[-1] if summaries else summary_path
    if summary_path.is_file():
        summary_main_json = add_json_obj(summary_path, x_summary_out, y_json_main)

    add_conn(rating_main_json, summary_main_json)

    summary_run = CURRENT_RUN_DIR / "summary.json"
    if summary_run.is_file() and summary_main_json:
        obj = add_json_obj(summary_run, x_summary_out, y_screen_main + row_step * (len(summary_debug_jsons) + 2))
        if obj:
            summary_debug_jsons.append(obj)

    # Historia summary (rate_summary/*.json) - najnowszy
    try:
        summary_hist = _all_files(RATE_SUMMARY_DIR, {".json"})
    except Exception:
        summary_hist = []
    if summary_hist and summary_main_json:
        obj = add_json_obj(summary_hist[-1], x_summary_out, y_screen_main + row_step * (len(summary_debug_jsons) + 2))
        if obj:
            summary_debug_jsons.append(obj)

    add_debug_fan(summary_main_json, summary_debug_jsons)

    # ===== BRAIN (stan + wejścia z pipeline) =====
    brain_state_json = add_json_if_exists(BRAIN_STATE_FILE, x_brain_out, y_json_main)
    add_conn(summary_main_json, brain_note)
    add_conn(rating_main_json, brain_note)
    if regions_current_json:
        add_conn(regions_current_json, brain_note)
    try:
        stats_obj = dom_live_objs.get("stats.json") if isinstance(dom_live_objs, dict) else None
    except Exception:
        stats_obj = None
    if stats_obj:
        add_conn(stats_obj, brain_note)
    add_conn(brain_note, brain_state_json, color=COLOR_CALL, thickness=2)

    # ===== HOVER (osobny tor) =====
    hover_input = None
    hover_input = add_screen_if_exists(HOVER_INPUT_CURRENT_DIR / "hover_input.png", x_capture_out, y_hover_base + y_screen_main)
    if hover_input:
        add_conn(capture_main, hover_input, from_side="bottom", to_side="top")

    hover_main_json = None
    hover_main_json = add_json_if_exists(HOVER_OUTPUT_DIR / "hover_output.json", x_action_out, y_hover_base + y_json_main)

    # Branch do hover: prowadzenie po korytarzu pomiedzy region_grow obrazkami (do 2700)
    # a kolumna skryptow (od 3000), zeby nie przejsc przez region_grow_annot_current.png.
    hover_lane_x = (x_region_out + 700.0 + x_action_note) * 0.5
    add_conn(region_json_main, hover_note, via_x=hover_lane_x)
    add_conn(hover_input, hover_note)
    add_conn(hover_note, hover_main_json, color=COLOR_CALL, thickness=2)

    hover_debug_nodes: list = []
    y_hover_debug = y_hover_base + y_screen_main + row_step

    for p in [
        HOVER_OUTPUT_DIR / "hover_output.png",
        HOVER_PATH_CURRENT_DIR / "hover_path.png",
        HOVER_SPEED_CURRENT_DIR / "hover_speed.png",
        HOVER_POINTS_ON_PATH_CURRENT_DIR / "hover_points_on_path.png",
        HOVER_POINTS_ON_SPEED_CURRENT_DIR / "hover_points_on_speed.png",
    ]:
        obj = add_screen_if_exists(p, x_action_out, y_hover_debug + row_step * len(hover_debug_nodes))
        if obj:
            hover_debug_nodes.append(obj)

    add_debug_fan(hover_main_json or hover_note, hover_debug_nodes)

    return {
        "pan_x": 420.0,
        "pan_y": 180.0,
        "zoom": 0.06,
        "screens": screens_state,
        "json_objects": json_state,
        "notes": notes_state,
        "lines": lines_state,
    }


# ================== POMOCNICZE ==================

# Pipeline history
pipeline_runs = []
pipeline_run_index = 0
pipeline_runs_meta = []


# ================== POMOCNICZE ==================

def world_from_screen(mx, my):
    """Screen -> world coords."""
    cx, cy = dpg.get_item_rect_min(CANVAS_TAG)
    local_mx = mx - cx
    local_my = my - cy
    wx = (local_mx - pan_x) / zoom
    wy = (local_my - pan_y) / zoom
    return wx, wy

def default_spawn_coords(margin: float = 40.0):
    """Zwraca punkt startowy w centrum bieżącego widoku/ekranu."""
    try:
        vw = dpg.get_viewport_client_width()
        vh = dpg.get_viewport_client_height()
        if vw and vh:
            return world_from_screen(vw * 0.5, vh * 0.5)
    except Exception:
        pass
    if dpg.does_item_exist(CANVAS_TAG):
        try:
            cx, cy = dpg.get_item_rect_min(CANVAS_TAG)
            return world_from_screen(cx + margin, cy + margin)
        except Exception:
            pass
    return 0.0, 0.0


def _mouse_pos_canvas_local() -> tuple[float, float]:
    """Mouse position in canvas-local coordinates (0,0 == drawlist top-left)."""
    with contextlib.suppress(Exception):
        x, y = dpg.get_drawing_mouse_pos()
        return float(x), float(y)

    mx, my = dpg.get_mouse_pos()
    cx, cy = dpg.get_item_rect_min(CANVAS_TAG)
    return float(mx - cx), float(my - cy)


def _canvas_size_local() -> tuple[float, float]:
    if not dpg.does_item_exist(CANVAS_TAG):
        return 0.0, 0.0
    with contextlib.suppress(Exception):
        return float(dpg.get_item_width(CANVAS_TAG)), float(dpg.get_item_height(CANVAS_TAG))
    with contextlib.suppress(Exception):
        min_x, min_y = dpg.get_item_rect_min(CANVAS_TAG)
        max_x, max_y = dpg.get_item_rect_max(CANVAS_TAG)
        return float(max_x - min_x), float(max_y - min_y)
    return 0.0, 0.0


def _world_bounds_from_objects() -> tuple[float, float, float, float] | None:
    bounds: tuple[float, float, float, float] | None = None

    def add_pt(x: float, y: float):
        nonlocal bounds
        if bounds is None:
            bounds = (x, y, x, y)
        else:
            x1, y1, x2, y2 = bounds
            bounds = (min(x1, x), min(y1, y), max(x2, x), max(y2, y))

    def add_rect(x: float, y: float, w: float, h: float):
        add_pt(float(x), float(y))
        add_pt(float(x) + float(w), float(y) + float(h))

    for obj in screens + json_objects + notes:
        with contextlib.suppress(Exception):
            add_rect(obj.get("x", 0.0), obj.get("y", 0.0), obj.get("w", 0.0), obj.get("h", 0.0))

    for ln in lines:
        for p in _line_points_world(ln):
            with contextlib.suppress(Exception):
                add_pt(float(p[0]), float(p[1]))

    return bounds


def fit_view_to_content(padding_px: float = 120.0):
    """Ustaw pan/zoom tak, żeby cała zawartość była widoczna na canvasie."""
    global pan_x, pan_y, zoom
    cw, ch = _canvas_size_local()
    if cw <= 0 or ch <= 0:
        return

    bounds = _world_bounds_from_objects()
    if bounds is None:
        zoom = 1.0
        pan_x = cw * 0.5
        pan_y = ch * 0.5
        return

    min_x, min_y, max_x, max_y = bounds
    bw = max(1e-6, max_x - min_x)
    bh = max(1e-6, max_y - min_y)
    usable_w = max(1.0, cw - 2 * padding_px)
    usable_h = max(1.0, ch - 2 * padding_px)
    zoom = max(0.1, min(5.0, float(min(usable_w / bw, usable_h / bh))))

    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    pan_x = cw * 0.5 - cx * zoom
    pan_y = ch * 0.5 - cy * zoom


def _recover_view_if_broken(reason: str = ""):
    global pan_x, pan_y, zoom
    try:
        zoom = float(zoom)
    except Exception:
        zoom = 1.0
    zoom = max(0.1, min(5.0, zoom))

    bad_pan = False
    with contextlib.suppress(Exception):
        bad_pan = (not math.isfinite(pan_x)) or (not math.isfinite(pan_y))
    if not bad_pan and (abs(pan_x) > 1_000_000 or abs(pan_y) > 1_000_000):
        bad_pan = True

    if bad_pan:
        fit_view_to_content()
        msg = f"View reset (bad pan/zoom) {reason}".strip()
        with contextlib.suppress(Exception):
            dpg.set_value("result_text", msg)
        print(f"[flowui] {msg}")



def distance_point_to_line_segment(px, py, x1, y1, x2, y2):
    """Odleg┼éo┼Ť─ç punktu od odcinka."""
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)


def _orth_points_simple(x1: float, y1: float, x2: float, y2: float) -> list[list[float]]:
    """Zamien prosty odcinek na polyline (kat prosty)."""
    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)
    if abs(x1 - x2) < 1e-6 or abs(y1 - y2) < 1e-6:
        return [[x1, y1], [x2, y2]]
    return [[x1, y1], [x2, y1], [x2, y2]]


def _line_points_world(ln: dict) -> list[list[float]]:
    """Zwraca punkty linii w world coords (wspiera nowy i stary format)."""
    if not isinstance(ln, dict):
        return []

    pts = ln.get("points")
    if isinstance(pts, list) and len(pts) >= 2:
        out: list[list[float]] = []
        for p in pts:
            try:
                if isinstance(p, dict):
                    x = float(p.get("x", 0.0))
                    y = float(p.get("y", 0.0))
                else:
                    x = float(p[0])
                    y = float(p[1])
            except Exception:
                continue
            if not out or abs(out[-1][0] - x) > 1e-9 or abs(out[-1][1] - y) > 1e-9:
                out.append([x, y])
        if len(out) >= 2:
            return out

    if all(k in ln for k in ("x1", "y1", "x2", "y2")):
        return _orth_points_simple(float(ln["x1"]), float(ln["y1"]), float(ln["x2"]), float(ln["y2"]))

    return []


def _line_segments_world(ln: dict) -> list[tuple[list[float], list[float]]]:
    pts = _line_points_world(ln)
    if len(pts) < 2:
        return []
    return list(zip(pts, pts[1:]))


def _min_distance_to_line_world(px: float, py: float, ln: dict) -> float:
    """Minimalna odleglosc punktu od polyline (world coords)."""
    best = float("inf")
    for a, b in _line_segments_world(ln):
        d = distance_point_to_line_segment(px, py, a[0], a[1], b[0], b[1])
        if d < best:
            best = d
    return best


def _normalize_line_dict(ln: dict) -> dict | None:
    """Normalizuje line do formatu: {points: [[x,y]...], color: [r,g,b,a], thickness: int}."""
    if not isinstance(ln, dict):
        return None
    pts = _line_points_world(ln)
    if len(pts) < 2:
        return None

    col = ln.get("color", [0, 255, 0, 255])
    try:
        color = [int(col[0]), int(col[1]), int(col[2]), int(col[3])]
    except Exception:
        color = [0, 255, 0, 255]

    try:
        thickness = int(ln.get("thickness", 2))
    except Exception:
        thickness = 2

    return {"points": pts, "color": color, "thickness": thickness}


def create_json_texture(filename, width=200, height=150):
    """Tworzy tekstur─Ö dla JSON."""
    img = Image.new("RGBA", (width, height), (255, 220, 100, 255))
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([0, 0, width-1, height-1], outline=(200, 150, 0, 255), width=3)
    draw.rectangle([10, 10, width-10, 50], fill=(255, 200, 50, 255))
    
    try:
        draw.text((width//2 - 30, 18), "{ JSON }", fill=(0, 0, 0, 255))
    except:
        pass
    
    short_name = filename[:25] if len(filename) <= 25 else filename[:22] + "..."
    try:
        draw.text((15, 60), short_name, fill=(50, 50, 50, 255))
    except:
        pass
    
    for i in range(4):
        y = 85 + i * 12
        line_width = width - 30 - (i % 2) * 40
        draw.rectangle([15, y, line_width, y+6], fill=(255, 255, 200, 255))
    
    draw.ellipse([width-35, height-35, width-10, height-10], fill=(100, 200, 100, 255))
    
    return img


def create_note_texture(text, width=250, height=200):
    """Tworzy tekstur─Ö dla notatki/bloczku."""
    width = max(100, int(width))
    height = max(80, int(height))
    
    img = Image.new("RGBA", (width, height), (88, 34, 135, 255))
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([0, 0, width-1, height-1], outline=(120, 60, 170, 255), width=3)
    
    header_height = min(35, height // 4)
    draw.rectangle([5, 5, width-5, header_height], fill=(105, 50, 155, 255))
    
    try:
        draw.text((10, 10), "­čôŁ Notatka", fill=(180, 255, 180, 255))
    except:
        draw.text((10, 10), "NOTE", fill=(180, 255, 180, 255))
    
    lines_text = []
    current_line = ""
    words = text.split()
    
    max_chars = max(15, width // 9)
    
    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars:
            current_line += word + " "
        else:
            if current_line:
                lines_text.append(current_line.strip())
            current_line = word + " "
    
    if current_line:
        lines_text.append(current_line.strip())
    
    y_offset = header_height + 10
    line_height = 14
    max_lines = (height - header_height - 30) // line_height
    
    for i, line in enumerate(lines_text[:max_lines]):
        try:
            draw.text((10, y_offset + i * line_height), line, fill=(180, 255, 180, 255))
        except:
            pass
    
    if len(lines_text) > max_lines:
        try:
            draw.text((10, height - 20), "...", fill=(100, 100, 100, 255))
        except:
            pass
    
    if width > 40 and height > 40:
        draw.ellipse([width-30, height-30, width-5, height-5], fill=(100, 200, 255, 255))
        try:
            draw.text((width-25, height-25), "ÔťÄ", fill=(255, 255, 255, 255))
        except:
            pass
    
    return img


# ================== RYSOWANIE ==================

def redraw_canvas():
    """Przerysuj canvas."""
    if not dpg.does_item_exist(CANVAS_TAG):
        return

    dpg.delete_item(CANVAS_TAG, children_only=True)

    def _nice_step(target: float) -> float:
        if not (target > 0.0):
            return 100.0
        exp = 10 ** math.floor(math.log10(target))
        f = target / exp
        if f <= 1.0:
            nf = 1.0
        elif f <= 2.0:
            nf = 2.0
        elif f <= 5.0:
            nf = 5.0
        else:
            nf = 10.0
        return float(nf * exp)

    def _draw_checkerboard():
        if zoom <= 0:
            return
        try:
            cw = float(dpg.get_item_width(CANVAS_TAG))
            ch = float(dpg.get_item_height(CANVAS_TAG))
        except Exception:
            return
        if cw <= 0 or ch <= 0:
            return

        # Widoczny fragment w world coords
        min_wx = (0.0 - pan_x) / zoom
        max_wx = (cw - pan_x) / zoom
        min_wy = (0.0 - pan_y) / zoom
        max_wy = (ch - pan_y) / zoom

        desired_cell_px = 90.0
        step_world = _nice_step(desired_cell_px / max(zoom, 1e-6))
        step_world = max(10.0, min(5000.0, step_world))

        ix0 = int(math.floor(min(min_wx, max_wx) / step_world)) - 2
        ix1 = int(math.ceil(max(min_wx, max_wx) / step_world)) + 2
        iy0 = int(math.floor(min(min_wy, max_wy) / step_world)) - 2
        iy1 = int(math.ceil(max(min_wy, max_wy) / step_world)) + 2

        fill_a = (26, 26, 26, 55)
        fill_b = (34, 34, 34, 55)
        line_minor = (70, 70, 70, 70)
        line_major = (95, 95, 95, 95)
        major_every = 5

        for ix in range(ix0, ix1):
            wx1 = ix * step_world
            wx2 = (ix + 1) * step_world
            x1 = pan_x + wx1 * zoom
            x2 = pan_x + wx2 * zoom
            for iy in range(iy0, iy1):
                wy1 = iy * step_world
                wy2 = (iy + 1) * step_world
                y1 = pan_y + wy1 * zoom
                y2 = pan_y + wy2 * zoom
                fill = fill_a if ((ix + iy) & 1) == 0 else fill_b
                dpg.draw_rectangle((x1, y1), (x2, y2), fill=fill, color=(0, 0, 0, 0), thickness=0, parent=CANVAS_TAG)

        for ix in range(ix0, ix1 + 1):
            wx = ix * step_world
            x = pan_x + wx * zoom
            col = line_major if (ix % major_every == 0) else line_minor
            dpg.draw_line((x, 0.0), (x, ch), color=col, thickness=2 if (ix % major_every == 0) else 1, parent=CANVAS_TAG)
        for iy in range(iy0, iy1 + 1):
            wy = iy * step_world
            y = pan_y + wy * zoom
            col = line_major if (iy % major_every == 0) else line_minor
            dpg.draw_line((0.0, y), (cw, y), color=col, thickness=2 if (iy % major_every == 0) else 1, parent=CANVAS_TAG)

        # Oś world (0,0)
        axis_col = (120, 120, 255, 140)
        x0 = pan_x + 0.0 * zoom
        y0 = pan_y + 0.0 * zoom
        dpg.draw_line((x0, 0.0), (x0, ch), color=axis_col, thickness=2, parent=CANVAS_TAG)
        dpg.draw_line((0.0, y0), (cw, y0), color=axis_col, thickness=2, parent=CANVAS_TAG)

    _draw_checkerboard()

    for s in screens:
        x1 = pan_x + s["x"] * zoom
        y1 = pan_y + s["y"] * zoom
        x2 = x1 + s["w"] * zoom
        y2 = y1 + s["h"] * zoom
        dpg.draw_image(s["tex"], (x1, y1), (x2, y2), parent=CANVAS_TAG)

    for j in json_objects:
        x1 = pan_x + j["x"] * zoom
        y1 = pan_y + j["y"] * zoom
        x2 = x1 + j["w"] * zoom
        y2 = y1 + j["h"] * zoom
        dpg.draw_image(j["tex"], (x1, y1), (x2, y2), parent=CANVAS_TAG)

    for n in notes:
        x1 = pan_x + n["x"] * zoom
        y1 = pan_y + n["y"] * zoom
        x2 = x1 + n["w"] * zoom
        y2 = y1 + n["h"] * zoom
        dpg.draw_image(n["tex"], (x1, y1), (x2, y2), parent=CANVAS_TAG)

    for ln in lines:
        pts = _line_points_world(ln)
        if len(pts) < 2:
            continue

        col = ln.get("color", (0, 255, 0, 255))
        try:
            color = (int(col[0]), int(col[1]), int(col[2]), int(col[3]))
        except Exception:
            color = (0, 255, 0, 255)

        try:
            thickness = int(ln.get("thickness", 2))
        except Exception:
            thickness = 2

        highlight = False
        if erase_mode:
            mx, my = dpg.get_mouse_pos()
            wx, wy = world_from_screen(mx, my)
            dist_world = _min_distance_to_line_world(wx, wy, ln)
            dist_screen = dist_world * zoom
            if dist_screen <= ERASE_DISTANCE_PX:
                highlight = True

        draw_color = (255, 50, 50, 255) if highlight else color
        draw_thickness = 4 if highlight else thickness

        for a, b in _line_segments_world(ln):
            x1 = pan_x + a[0] * zoom
            y1 = pan_y + a[1] * zoom
            x2 = pan_x + b[0] * zoom
            y2 = pan_y + b[1] * zoom
            dpg.draw_line((x1, y1), (x2, y2), color=draw_color, thickness=draw_thickness, parent=CANVAS_TAG)

    if draw_line_mode and line_start is not None:
        mx, my = dpg.get_mouse_pos()
        wx, wy = world_from_screen(mx, my)
        preview_pts = _orth_points_simple(line_start[0], line_start[1], wx, wy)
        for a, b in zip(preview_pts, preview_pts[1:]):
            x1 = pan_x + a[0] * zoom
            y1 = pan_y + a[1] * zoom
            x2 = pan_x + b[0] * zoom
            y2 = pan_y + b[1] * zoom
            dpg.draw_line((x1, y1), (x2, y2), color=(0, 200, 0, 150), thickness=1, parent=CANVAS_TAG)


def add_screen_from_path(path, x=0.0, y=0.0):
    """Dodaje PNG jako screen."""
    global screen_counter, texture_counter

    push_undo_state()

    if not os.path.exists(path):
        msg = f"ÔŁî Plik nie istnieje: {path}"
        print(msg)
        dpg.set_value("result_text", msg)
        return

    image = Image.open(path).convert("RGBA")
    w, h = image.size

    raw = np.frombuffer(image.tobytes(), dtype=np.uint8).astype(np.float32) / 255.0
    data = raw.tolist()

    tex_tag = f"tex_{texture_counter}"
    texture_counter += 1

    dpg.add_static_texture(w, h, data, tag=tex_tag, parent=TEXREG_TAG)

    screen_id = f"screen_{screen_counter}"
    screen_counter += 1

    screens.append({
        "id": screen_id,
        "tex": tex_tag,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "path": path,
        "type": "png",
        "mtime": os.path.getmtime(path)
    })

    info = f"Ôťů PNG: {os.path.basename(path)} ({w}x{h})"
    print(info)
    dpg.set_value("result_text", info)
    _recover_view_if_broken("(load_state)")
    redraw_canvas()


def add_json_to_canvas(path, x=0.0, y=0.0):
    """Dodaje JSON jako graficzny element."""
    global screen_counter, texture_counter

    push_undo_state()

    if not os.path.exists(path):
        msg = f"ÔŁî JSON nie istnieje: {path}"
        print(msg)
        dpg.set_value("result_text", msg)
        return

    filename = os.path.basename(path)
    img = create_json_texture(filename, width=200, height=150)
    w, h = img.size

    raw = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32) / 255.0
    data = raw.tolist()

    tex_tag = f"tex_{texture_counter}"
    texture_counter += 1

    dpg.add_static_texture(w, h, data, tag=tex_tag, parent=TEXREG_TAG)

    json_id = f"json_{screen_counter}"
    screen_counter += 1

    json_objects.append({
        "id": json_id,
        "tex": tex_tag,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "path": path,
        "type": "json",
        "mtime": os.path.getmtime(path)
    })

    info = f"Ôťů JSON: {filename}"
    print(info)
    dpg.set_value("result_text", info)
    redraw_canvas()




def update_screen_texture(obj: dict, path: str) -> None:
    """Podmienia teksture istniejacego screena bez zmiany pozycji."""
    global texture_counter
    if not os.path.exists(path):
        dpg.set_value("result_text", f"Brak pliku PNG: {path}")
        return
    image = Image.open(path).convert("RGBA")
    w, h = image.size
    raw = np.frombuffer(image.tobytes(), dtype=np.uint8).astype(np.float32) / 255.0
    data = raw.tolist()
    old_tex = obj.get("tex")
    if old_tex and dpg.does_item_exist(old_tex):
        dpg.delete_item(old_tex)
    tex_tag = f"tex_reload_{texture_counter}"
    texture_counter += 1
    dpg.add_static_texture(w, h, data, tag=tex_tag, parent=TEXREG_TAG)
    obj.update({"tex": tex_tag, "w": w, "h": h, "path": path, "mtime": os.path.getmtime(path)})


def update_json_texture(obj: dict, path: str) -> None:
    """Podmienia teksture istniejacego JSON-a, zachowujac pozycje/rozmiar."""
    global texture_counter
    filename = os.path.basename(path) if path else "missing"
    width = max(1, int(obj.get("w", 200)))
    height = max(1, int(obj.get("h", 150)))
    img = create_json_texture(filename, width=width, height=height)
    w, h = img.size
    raw = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32) / 255.0
    data = raw.tolist()
    old_tex = obj.get("tex")
    if old_tex and dpg.does_item_exist(old_tex):
        dpg.delete_item(old_tex)
    tex_tag = f"tex_reload_{texture_counter}"
    texture_counter += 1
    dpg.add_static_texture(w, h, data, tag=tex_tag, parent=TEXREG_TAG)
    mtime = os.path.getmtime(path) if path and os.path.exists(path) else 0.0
    obj.update({"tex": tex_tag, "w": w, "h": h, "path": path, "mtime": mtime})

def add_note_to_canvas(x=0.0, y=0.0, text="Nowa notatka"):
    """Dodaje notatk─Ö na canvas."""
    global screen_counter, texture_counter

    push_undo_state()

    img = create_note_texture(text, width=250, height=200)
    w, h = img.size

    raw = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32) / 255.0
    data = raw.tolist()

    tex_tag = f"tex_{texture_counter}"
    texture_counter += 1

    dpg.add_static_texture(w, h, data, tag=tex_tag, parent=TEXREG_TAG)

    note_id = f"note_{screen_counter}"
    screen_counter += 1

    notes.append({
        "id": note_id,
        "tex": tex_tag,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "text": text,
        "type": "note"
    })

    info = f"Ôťů Notatka: {text[:20]}..."
    print(info)
    dpg.set_value("result_text", info)
    redraw_canvas()


def update_note_texture(note):
    """Aktualizuje tekstur─Ö notatki po edycji."""
    global texture_counter
    
    img = create_note_texture(note["text"], width=int(note["w"]), height=int(note["h"]))
    w, h = img.size
    
    raw = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32) / 255.0
    data = raw.tolist()
    
    old_tex = note["tex"]
    if dpg.does_item_exist(old_tex):
        dpg.delete_item(old_tex)
    
    new_tex_tag = f"tex_note_{texture_counter}"
    texture_counter += 1
    
    dpg.add_static_texture(w, h, data, tag=new_tex_tag, parent=TEXREG_TAG)
    
    note["tex"] = new_tex_tag
    redraw_canvas()


def delete_note(note_id):
    """Usu┼ä notatk─Ö."""
    global notes
    
    for i, note in enumerate(notes):
        if note["id"] == note_id:
            if dpg.does_item_exist(note["tex"]):
                dpg.delete_item(note["tex"])
            
            notes.pop(i)
            print(f"­čŚĹ´ŞĆ Usuni─Öto notatk─Ö: {note['text'][:30]}...")
            
            if dpg.does_item_exist(NOTE_EDITOR_TAG):
                dpg.hide_item(NOTE_EDITOR_TAG)
            
            redraw_canvas()
            dpg.set_value("result_text", "­čŚĹ´ŞĆ Notatka usuni─Öta")
            return
    
    print(f"ÔŁî Nie znaleziono notatki: {note_id}")


def open_note_editor(note_id):
    """Otw├│rz edytor notatki."""
    global active_note_id
    
    note = None
    for n in notes:
        if n["id"] == note_id:
            note = n
            break
    
    if not note:
        return
    
    active_note_id = note_id
    
    if not dpg.does_item_exist(NOTE_EDITOR_TAG):
        with dpg.window(label=f"Edycja notatki", tag=NOTE_EDITOR_TAG,
                        width=500, height=450, pos=(100, 100), modal=True, no_collapse=True):
            dpg.add_text("Edytuj tekst notatki:")
            dpg.add_input_text(tag=NOTE_TEXT_TAG,
                               default_value=note["text"],
                               multiline=True,
                               width=-1, height=300)
            with dpg.group(horizontal=True):
                dpg.add_button(label="­čĺż Zapisz", callback=save_note_edit, width=120)
                dpg.add_button(label="­čŚĹ´ŞĆ Usu┼ä notatk─Ö", callback=lambda: delete_note(active_note_id), width=120)
                dpg.add_button(label="ÔŁî Anuluj", callback=lambda: dpg.hide_item(NOTE_EDITOR_TAG), width=120)
    else:
        dpg.configure_item(NOTE_EDITOR_TAG, label=f"Edycja notatki", show=True)
        dpg.set_value(NOTE_TEXT_TAG, note["text"])
    
    print(f"­čôŁ Edycja notatki: {note_id}")


def save_note_edit():
    """Zapisz edycj─Ö notatki."""
    global active_note_id
    
    if not active_note_id:
        return
    
    new_text = dpg.get_value(NOTE_TEXT_TAG)
    
    for n in notes:
        if n["id"] == active_note_id:
            n["text"] = new_text
            update_note_texture(n)
            print(f"­čĺż Zaktualizowano notatk─Ö: {new_text[:30]}...")
            break
    
    dpg.hide_item(NOTE_EDITOR_TAG)
    active_note_id = None


# ================== HOT RELOAD ==================

def check_and_reload_files():
    """Sprawdza czy pliki si─Ö zmieni┼éy i prze┼éadowuje."""
    global texture_counter
    
    if not dpg.does_item_exist(CANVAS_TAG):
        return
    
    reloaded = []
    needs_redraw = False
    
    for s in screens:
        if not os.path.exists(s["path"]):
            continue
        
        try:
            current_mtime = os.path.getmtime(s["path"])
            if current_mtime > s["mtime"]:
                print(f"­čöä Prze┼éadowuj─Ö PNG: {os.path.basename(s['path'])}")
                
                image = Image.open(s["path"]).convert("RGBA")
                tex_w, tex_h = image.size  # pełna rozdzielczość tekstury
                raw = np.frombuffer(image.tobytes(), dtype=np.uint8).astype(np.float32) / 255.0
                data = raw.tolist()
                
                old_tex = s["tex"]
                if dpg.does_item_exist(old_tex):
                    dpg.delete_item(old_tex)
                
                new_tex_tag = f"tex_reload_{texture_counter}"
                texture_counter += 1
                
                dpg.add_static_texture(tex_w, tex_h, data, tag=new_tex_tag, parent=TEXREG_TAG)
                
                s["tex"] = new_tex_tag
                # Rozmiar WYŚWIETLANIA – użyj tej samej logiki skalowania co przy pierwszym razie
                disp_w, disp_h = _default_size_for_image(s["path"])
                s["w"] = disp_w
                s["h"] = disp_h
                s["mtime"] = current_mtime
                
                reloaded.append(os.path.basename(s["path"]))
                needs_redraw = True
                
        except Exception as e:
            print(f"ÔŁî B┼é─ůd prze┼éadowania PNG {s['path']}: {e}")
    
    for j in json_objects:
        if not os.path.exists(j["path"]):
            continue
        
        try:
            current_mtime = os.path.getmtime(j["path"])
            if current_mtime > j["mtime"]:
                print(f"­čöä Prze┼éadowuj─Ö JSON: {os.path.basename(j['path'])}")
                
                filename = os.path.basename(j["path"])
                img = create_json_texture(filename, width=int(j["w"]), height=int(j["h"]))
                w, h = img.size
                raw = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32) / 255.0
                data = raw.tolist()
                
                old_tex = j["tex"]
                if dpg.does_item_exist(old_tex):
                    dpg.delete_item(old_tex)
                
                new_tex_tag = f"tex_reload_{texture_counter}"
                texture_counter += 1
                
                dpg.add_static_texture(w, h, data, tag=new_tex_tag, parent=TEXREG_TAG)
                
                j["tex"] = new_tex_tag
                j["mtime"] = current_mtime
                
                reloaded.append(os.path.basename(j["path"]))
                needs_redraw = True
                
        except Exception as e:
            print(f"ÔŁî B┼é─ůd prze┼éadowania JSON {j['path']}: {e}")
    
    if needs_redraw:
        redraw_canvas()
        msg = f"­čöä Od┼Ťwie┼╝ono: {', '.join(reloaded)}"
        dpg.set_value("result_text", msg)


def auto_reload_thread():
    """W─ůtek sprawdzaj─ůcy zmiany."""
    while True:
        time.sleep(AUTO_RELOAD_INTERVAL)
        if auto_reload_enabled:
            try:
                check_and_reload_files()
            except Exception as e:
                print(f"ÔŁî B┼é─ůd auto-reload: {e}")


def toggle_auto_reload():
    """Prze┼é─ůcz auto-reload."""
    global auto_reload_enabled
    auto_reload_enabled = not auto_reload_enabled
    status = "ON" if auto_reload_enabled else "OFF"
    color = (100, 255, 100) if auto_reload_enabled else (255, 100, 100)
    dpg.configure_item("auto_reload_status", default_value=f"Auto: {status}", color=color)
    print(f"­čöä Auto-reload: {status}")


def open_json_viewer(path):
    """Otw├│rz viewer JSON."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        try:
            obj = pyjson.loads(content)
            pretty = pyjson.dumps(obj, ensure_ascii=False, indent=2)
        except:
            pretty = content
    except Exception as e:
        pretty = f"ÔŁî B┼é─ůd: {e}"

    title = f"JSON: {os.path.basename(path)}"

    if not dpg.does_item_exist(JSON_VIEWER_TAG):
        with dpg.window(label=title, tag=JSON_VIEWER_TAG, width=600, height=700, pos=(50, 50)):
            dpg.add_input_text(tag=JSON_TEXT_TAG, default_value=pretty,
                             multiline=True, readonly=True, width=-1, height=-1)
    else:
        dpg.configure_item(JSON_VIEWER_TAG, label=title, show=True)
        dpg.set_value(JSON_TEXT_TAG, pretty)
    
    print(f"­čôľ Otwarto JSON: {os.path.basename(path)}")


# ================== ZAPIS/ODCZYT ==================

def _clear_all_objects():
    """Usuń istniejące obiekty i powiązane tekstury (przed wczytaniem stanu)."""
    global screens, json_objects, notes, lines, screen_counter, texture_counter
    for obj in screens + json_objects + notes:
        tex = obj.get("tex")
        if tex and dpg.does_item_exist(tex):
            with contextlib.suppress(Exception):
                dpg.delete_item(tex)
    screens = []
    json_objects = []
    notes = []
    lines = []
    screen_counter = 0
    texture_counter = 0


def _delete_texture_for(obj: dict):
    tex = obj.get("tex")
    if tex and dpg.does_item_exist(tex):
        with contextlib.suppress(Exception):
            dpg.delete_item(tex)


def apply_state(state: dict):
    """Załaduj stan (screens/json/notes/lines) na canvas."""
    global pan_x, pan_y, zoom, lines
    _clear_all_objects()

    pan_x = float(state.get("pan_x", 0.0))
    pan_y = float(state.get("pan_y", 0.0))
    zoom = max(0.1, min(5.0, float(state.get("zoom", 1.0))))

    for s in state.get("screens", []):
        path = s.get("path")
        if path and os.path.exists(path):
            add_screen_from_path(path, x=float(s.get("x", 0.0)), y=float(s.get("y", 0.0)))
            screens[-1]["w"] = float(s.get("w", screens[-1]["w"]))
            screens[-1]["h"] = float(s.get("h", screens[-1]["h"]))

    for j in state.get("json_objects", []):
        path = j.get("path")
        if path and os.path.exists(path):
            add_json_to_canvas(path, x=float(j.get("x", 0.0)), y=float(j.get("y", 0.0)))
            json_objects[-1]["w"] = float(j.get("w", json_objects[-1]["w"]))
            json_objects[-1]["h"] = float(j.get("h", json_objects[-1]["h"]))

    for n in state.get("notes", []):
        text = n.get("text", "")
        add_note_to_canvas(x=float(n.get("x", 0.0)), y=float(n.get("y", 0.0)), text=text)
        notes[-1]["w"] = float(n.get("w", notes[-1]["w"]))
        notes[-1]["h"] = float(n.get("h", notes[-1]["h"]))
        if "role" in n:
            notes[-1]["role"] = n.get("role")

    for ln in state.get("lines", []):
        norm = _normalize_line_dict(ln) if isinstance(ln, dict) else None
        if norm:
            lines.append(norm)

    _recover_view_if_broken("(apply_state)")
    redraw_canvas()


def capture_state_snapshot() -> dict:
    """Aktualny stan canvasu (do undo/zapisu)."""
    return {
        "pan_x": pan_x,
        "pan_y": pan_y,
        "zoom": zoom,
        "screens": [{"path": s["path"], "x": s["x"], "y": s["y"], "w": s["w"], "h": s["h"]} for s in screens],
        "json_objects": [{"path": j["path"], "x": j["x"], "y": j["y"], "w": j["w"], "h": j["h"]} for j in json_objects],
        "notes": [{"text": n["text"], "x": n["x"], "y": n["y"], "w": n["w"], "h": n["h"], "role": n.get("role")} for n in notes],
        "lines": [norm for ln in lines if (norm := _normalize_line_dict(ln)) is not None],
    }


def push_undo_state():
    """Zachowaj stan do undo."""
    try:
        snapshot = capture_state_snapshot()
    except Exception:
        return
    undo_stack.append(snapshot)
    if len(undo_stack) > UNDO_LIMIT:
        undo_stack.pop(0)


def restore_state(snapshot: dict):
    """Przywróć stan (undo)."""
    apply_state(snapshot)


def save_state():
    """Zapisz stan."""
    try:
        state = {
            "pan_x": pan_x,
            "pan_y": pan_y,
            "zoom": zoom,
            "screens": [{"path": s["path"], "x": s["x"], "y": s["y"], "w": s["w"], "h": s["h"]} for s in screens],
            "json_objects": [{"path": j["path"], "x": j["x"], "y": j["y"], "w": j["w"], "h": j["h"]} for j in json_objects],
            "notes": [{"text": n["text"], "x": n["x"], "y": n["y"], "w": n["w"], "h": n["h"], "role": n.get("role")} for n in notes],
            "lines": [norm for ln in lines if (norm := _normalize_line_dict(ln)) is not None],
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            pyjson.dump(state, f, ensure_ascii=False, indent=2)
        print(f"­čĺż Stan zapisany")
    except Exception as e:
        print(f"ÔŁî B┼é─ůd zapisu: {e}")


def load_state():
    """Wczytaj stan."""
    global pan_x, pan_y, zoom, lines

    if not os.path.exists(STATE_FILE):
        return False

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = pyjson.load(f)
    except Exception as e:
        print(f"ÔŁî B┼é─ůd odczytu: {e}")
        return False

    pan_x = float(state.get("pan_x", 0.0))
    pan_y = float(state.get("pan_y", 0.0))
    zoom = max(0.1, min(5.0, float(state.get("zoom", 1.0))))

    for s in state.get("screens", []):
        path = s.get("path")
        if path and os.path.exists(path):
            add_screen_from_path(path, x=float(s.get("x", 0.0)), y=float(s.get("y", 0.0)))
            screens[-1]["w"] = float(s.get("w", screens[-1]["w"]))
            screens[-1]["h"] = float(s.get("h", screens[-1]["h"]))

    for j in state.get("json_objects", []):
        path = j.get("path")
        if path and os.path.exists(path):
            add_json_to_canvas(path, x=float(j.get("x", 0.0)), y=float(j.get("y", 0.0)))
            json_objects[-1]["w"] = float(j.get("w", json_objects[-1]["w"]))
            json_objects[-1]["h"] = float(j.get("h", json_objects[-1]["h"]))

    for n in state.get("notes", []):
        text = n.get("text", "")
        add_note_to_canvas(x=float(n.get("x", 0.0)), y=float(n.get("y", 0.0)), text=text)
        notes[-1]["w"] = float(n.get("w", notes[-1]["w"]))
        notes[-1]["h"] = float(n.get("h", notes[-1]["h"]))
        if "role" in n:
            notes[-1]["role"] = n.get("role")

    for ln in state.get("lines", []):
        norm = _normalize_line_dict(ln) if isinstance(ln, dict) else None
        if norm:
            lines.append(norm)

    redraw_canvas()
    print("­čôé Stan wczytany")
    return True


def load_state_pipeline() -> bool:
    """Najpierw spróbuj wczytać stan, jeśli brak – zbuduj pipeline z artefaktów *_current."""
    try:
        if load_state():
            return True
    except Exception:
        pass
    try:
        pipeline_state = build_pipeline_state()
        apply_state(pipeline_state)
        save_state()
        print("╞��c Zainicjalizowano widok pipeline (current*)")
        return True
    except Exception as e:
        print(f"�ʼ� Błąd inicjalizacji pipeline: {e}")
        return False


def on_exit():
    save_state()


# ================== BACKUP / PIPELINE RUNS ==================


def _format_run_label(run: Path) -> str:
    """Zwraca etykietę runu z czasem modyfikacji i wiekiem."""
    ts = run.stat().st_mtime
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    delta = max(0.0, time.time() - ts)
    if delta < 90:
        human = f"{int(delta)}s ago"
    elif delta < 3600:
        human = f"{int(delta // 60)}m ago"
    elif delta < 86400:
        hours = int(delta // 3600)
        minutes = int((delta % 3600) // 60)
        human = f"{hours}h {minutes}m ago"
    else:
        days = int(delta // 86400)
        hours = int((delta % 86400) // 3600)
        human = f"{days}d {hours}h ago"
    return f"{ts_str} ({human}) | {run.name}"


def _update_backup_list():
    """Buduje widok backupow w formie drzewa (dzien -> godzina -> minuta -> sekundy)."""
    if not dpg.does_item_exist(PIPELINE_LIST_TAG):
        return

    # Wyczysc kontener drzewa
    if dpg.does_item_exist(PIPELINE_LIST_TAG):
        dpg.delete_item(PIPELINE_LIST_TAG, children_only=True)

    if not pipeline_runs:
        if dpg.does_item_exist(PIPELINE_STATUS_TAG):
            dpg.set_value(PIPELINE_STATUS_TAG, "Brak zapisanych runow")
        dpg.add_text("Brak zapisanych runow", parent=PIPELINE_LIST_TAG)
        return

    # Metadane z czasem
    runs_meta = pipeline_runs_meta or []
    if not runs_meta:
        for idx, run in enumerate(pipeline_runs):
            try:
                ts = run.stat().st_mtime
            except Exception:
                continue
            runs_meta.append((idx, run, ts))

    # Sort desc
    runs_meta.sort(key=lambda x: x[2], reverse=True)

    # Grupowanie po dacie (max MAX_BACKUP_DAYS)
    day_groups = {}
    day_order = []
    for idx, run, ts in runs_meta:
        tm = time.localtime(ts)
        day_key = time.strftime("%Y-%m-%d", tm)
        if day_key not in day_groups and len(day_groups) >= MAX_BACKUP_DAYS:
            continue
        day_groups.setdefault(day_key, []).append((idx, run, tm))
        if day_key not in day_order:
            day_order.append(day_key)

    tree_root = dpg.add_tree_node(label="Ostatnie backupy", default_open=True, parent=PIPELINE_LIST_TAG, tag=PIPELINE_TREE_TAG)

    for day_key in day_order:
        entries = day_groups.get(day_key, [])
        # sort per day desc
        entries.sort(key=lambda x: time.mktime(x[2]), reverse=True)
        if not entries:
            continue
        tm0 = entries[0][2]
        weekday = POLISH_DAY_NAMES[tm0.tm_wday] if 0 <= tm0.tm_wday < 7 else day_key
        day_label = f"{weekday} ({day_key})"
        day_node = dpg.add_tree_node(label=day_label, default_open=False, parent=tree_root)

        # hour -> minute -> runs
        hour_groups = {}
        for idx, run, tm in entries:
            hour_groups.setdefault(tm.tm_hour, []).append((idx, run, tm))
        for hour, hour_entries in sorted(hour_groups.items(), key=lambda x: x[0], reverse=True):
            hour_label = f"{hour:02d}:00"
            hour_node = dpg.add_tree_node(label=hour_label, default_open=False, parent=day_node)

            minute_groups = {}
            for idx, run, tm in hour_entries:
                minute_groups.setdefault(tm.tm_min, []).append((idx, run, tm))
            for minute, minute_entries in sorted(minute_groups.items(), key=lambda x: x[0], reverse=True):
                minute_label = f"{hour:02d}:{minute:02d}"
                minute_node = dpg.add_tree_node(label=minute_label, default_open=False, parent=hour_node)

                for idx, run, tm in sorted(minute_entries, key=lambda x: x[2].tm_sec, reverse=True):
                    sec_label = f"{hour:02d}:{minute:02d}:{tm.tm_sec:02d} | {run.name}"
                    dpg.add_selectable(label=sec_label, callback=on_backup_run_click, user_data=idx, parent=minute_node)

    if dpg.does_item_exist(PIPELINE_STATUS_TAG):
        dpg.set_value(PIPELINE_STATUS_TAG, f"{len(pipeline_runs)} runs (ostatnie {len(day_order)} dni)")


def refresh_pipeline_runs():
    """Zbiera katalogi z backupami pipeline."""
    global pipeline_runs, pipeline_run_index, pipeline_runs_meta
    runs = []
    meta = []
    if PIPELINE_RUNS_DIR.exists():
        try:
            for entry in os.scandir(PIPELINE_RUNS_DIR):
                if entry.is_dir():
                    p = Path(entry.path)
                    ts = entry.stat().st_mtime
                    runs.append(p)
                    meta.append((len(meta), p, ts))
        except Exception:
            runs = [p for p in PIPELINE_RUNS_DIR.iterdir() if p.is_dir()]
            meta = []
            for idx, p in enumerate(runs):
                try:
                    meta.append((idx, p, p.stat().st_mtime))
                except Exception:
                    pass
    runs_sorted = sorted(meta, key=lambda x: x[2], reverse=True)
    pipeline_runs = [r for _, r, _ in runs_sorted]
    pipeline_runs_meta = [(idx, r, ts) for idx, (_, r, ts) in enumerate(runs_sorted)] if runs_sorted else []
    if pipeline_runs:
        pipeline_run_index = min(pipeline_run_index, len(pipeline_runs) - 1)
    else:
        pipeline_run_index = 0
    _update_backup_list()


def clear_canvas_objects():
    """Czysci wszystkie obiekty i tekstury z canvasa."""
    global screens, json_objects, notes, lines
    global screen_counter, texture_counter, active_object_id, active_note_id
    for obj in screens + json_objects + notes:
        tex = obj.get("tex")
        if tex and dpg.does_item_exist(tex):
            dpg.delete_item(tex)
    screens = []
    json_objects = []
    notes = []
    lines = []
    screen_counter = 0
    texture_counter = 0
    active_object_id = None
    active_note_id = None
    if dpg.does_item_exist(CANVAS_TAG):
        dpg.delete_item(CANVAS_TAG, children_only=True)
    redraw_canvas()


def load_pipeline_run(target_index: int = 0):
    """Laduje PNG + JSON z wybranego runu (0 = najnowszy)."""
    global pipeline_run_index
    refresh_pipeline_runs()
    if not pipeline_runs:
        dpg.set_value("result_text", "Brak zapisanych runow.")
        return
    pipeline_run_index = max(0, min(target_index, len(pipeline_runs) - 1))
    _update_backup_list()
    run_dir = pipeline_runs[pipeline_run_index]

    screenshot = run_dir / "screenshot.png"
    region_json = run_dir / "region_grow.json"
    summary_json = run_dir / "summary.json"
    rating_json = run_dir / "rating.json"

    if screenshot.exists():
        if screens:
            update_screen_texture(screens[0], str(screenshot))
        else:
            add_screen_from_path(str(screenshot), x=0.0, y=0.0)

    target_jsons = []
    if region_json.exists():
        target_jsons.append(region_json)
    target_summary = summary_json if summary_json.exists() else None
    if not target_summary and rating_json.exists():
        target_summary = rating_json
    if target_summary:
        target_jsons.append(target_summary)

    for idx, jpath in enumerate(target_jsons):
        if idx < len(json_objects):
            update_json_texture(json_objects[idx], str(jpath))
        else:
            add_json_to_canvas(str(jpath), x=0.0, y=0.0)

    if len(json_objects) > len(target_jsons):
        for idx in range(len(target_jsons), len(json_objects)):
            update_json_texture(json_objects[idx], "")

    dpg.set_value("result_text", f"Wczytano run: {run_dir.name}")
    redraw_canvas()


def on_backup_select(sender, app_data):
    """Wybor elementu z listy backupow."""
    # zachowane dla zgodnosci (listbox juz nieuzywany)
    if isinstance(app_data, int):
        load_pipeline_run(app_data)


def toggle_backup_window():
    """Pokazuje/ukrywa okno backupów."""
    if not dpg.does_item_exist(BACKUP_WINDOW_TAG):
        return
    visible = dpg.is_item_shown(BACKUP_WINDOW_TAG)
    if visible:
        dpg.hide_item(BACKUP_WINDOW_TAG)
    else:
        refresh_pipeline_runs()
        dpg.configure_item(BACKUP_WINDOW_TAG, show=True, on_top=True)


def on_backup_run_click(sender, app_data, user_data):
    """Klik w konkretny run w drzewie."""
    try:
        idx = int(user_data)
    except Exception:
        return
    load_pipeline_run(idx)


# ================== CALLBACKI ==================
# ================== CALLBACKI ==================

def get_all_objects():
    return screens + json_objects + notes


def on_file_dialog(sender, app_data):
    if not app_data:
        return
    path = app_data.get("file_path_name")
    if not path:
        return

    lower = path.lower()
    if lower.endswith(".png"):
        spawn_x, spawn_y = default_spawn_coords()
        add_screen_from_path(path, x=spawn_x, y=spawn_y)
    elif lower.endswith(".json"):
        spawn_x, spawn_y = default_spawn_coords()
        add_json_to_canvas(path, x=spawn_x, y=spawn_y)
    else:
        dpg.set_value("result_text", f"Nieobs?ugiwany typ: {path}")

def on_file_button():
    # Pokaż dialog plików jako okno modalne i ustaw go na wierzchu.
    try:
        dpg.configure_item(FILE_DIALOG_TAG, show=True, modal=True)
    except Exception:
        dpg.configure_item(FILE_DIALOG_TAG, show=True)
    try:
        dpg.focus_item(FILE_DIALOG_TAG)
    except Exception:
        pass


def reset_pipeline_layout():
    """Wymuś odbudowanie domyślnego workflow (nadpisuje aktualny układ na canvasie)."""
    try:
        state = build_pipeline_state()
        apply_state(state)
        save_state()
        with contextlib.suppress(Exception):
            dpg.set_value("result_text", "Reset layout (pipeline)")
    except Exception as e:
        with contextlib.suppress(Exception):
            dpg.set_value("result_text", f"Reset layout failed: {e}")


def on_key_d(sender, app_data):
    global draw_line_mode, line_start, erase_mode, add_note_mode
    
    if erase_mode:
        erase_mode = False
        dpg.set_value("erase_text", "Gumka (C): OFF")
    
    if add_note_mode:
        add_note_mode = False
        dpg.set_value("note_text", "Notatka (N): OFF")
    
    draw_line_mode = not draw_line_mode
    line_start = None
    mode = "ON" if draw_line_mode else "OFF"
    dpg.set_value("mode_text", f"Tryb linii (D): {mode}")
    print(f"­čľŐ´ŞĆ Rysowanie: {mode}")
    redraw_canvas()


def on_key_c(sender, app_data):
    global erase_mode, draw_line_mode, line_start, add_note_mode
    
    if draw_line_mode:
        draw_line_mode = False
        line_start = None
        dpg.set_value("mode_text", "Tryb linii (D): OFF")
    
    if add_note_mode:
        add_note_mode = False
        dpg.set_value("note_text", "Notatka (N): OFF")
    
    erase_mode = not erase_mode
    mode = "ON" if erase_mode else "OFF"
    dpg.set_value("erase_text", f"Gumka (C): {mode}")
    print(f"­čž╣ Gumka: {mode}")
    redraw_canvas()


def on_key_n(sender, app_data):
    """Tryb dodawania notatki."""
    global add_note_mode, draw_line_mode, erase_mode, line_start
    
    if draw_line_mode:
        draw_line_mode = False
        line_start = None
        dpg.set_value("mode_text", "Tryb linii (D): OFF")
    
    if erase_mode:
        erase_mode = False
        dpg.set_value("erase_text", "Gumka (C): OFF")
    
    add_note_mode = not add_note_mode
    mode = "ON" if add_note_mode else "OFF"
    dpg.set_value("note_text", f"Notatka (N): {mode}")
    print(f"?? Dodawanie notatki: {mode}")


def on_key_z(sender, app_data):
    """Undo (Ctrl+Z)."""
    ctrl_keys = []
    for key_name in ("mvKey_Control", "mvKey_LControl", "mvKey_RControl"):
        ctrl_key = getattr(dpg, key_name, None)
        if ctrl_key is not None:
            ctrl_keys.append(ctrl_key)

    if not ctrl_keys:
        return
    if not any(dpg.is_key_down(k) for k in ctrl_keys):
        return

    if not undo_stack:
        return
    snapshot = undo_stack.pop()
    restore_state(snapshot)
    dpg.set_value("result_text", "Cofnięto (Ctrl+Z)")


def on_key_b(sender, app_data):
    """Toggle okna backup?w (lista runow)."""
    toggle_backup_window()


def on_key_r(sender, app_data):
    """Toggle resize gating (R)."""
    global resize_enabled, resize_mode, resize_object_id, resize_edge_right, resize_edge_bottom, needs_texture_update
    resize_enabled = not resize_enabled
    status = "ON" if resize_enabled else "OFF"
    if dpg.does_item_exist("resize_text"):
        dpg.set_value("resize_text", f"Resize (R): {status}")
    if not resize_enabled:
        resize_mode = False
        resize_object_id = None
        resize_edge_right = False
        resize_edge_bottom = False
        needs_texture_update = False


def on_key_f(sender, app_data):
    """Fit view to content (F)."""
    fit_view_to_content()
    redraw_canvas()
    with contextlib.suppress(Exception):
        dpg.set_value("result_text", "Fit view (F)")

def on_left_click(sender, app_data):
    global active_object_id, drag_offset
    global resize_mode, resize_object_id, resize_edge_right, resize_edge_bottom
    global line_start, add_note_mode, needs_texture_update

    mx, my = dpg.get_mouse_pos()
    wx, wy = world_from_screen(mx, my)
    push_undo_state()

    # PRIORYTET 0: Dodawanie notatki
    if add_note_mode:
        add_note_to_canvas(x=wx, y=wy, text="Nowa notatka\nKliknij dwukrotnie aby edytowa─ç")
        add_note_mode = False
        dpg.set_value("note_text", "Notatka (N): OFF")
        return

    # PRIORYTET 1: Gumka
    if erase_mode:
        removed_count = 0
        i = 0
        while i < len(lines):
            ln = lines[i]
            dist_world = _min_distance_to_line_world(wx, wy, ln)
            dist_screen = dist_world * zoom
            
            if dist_screen <= ERASE_DISTANCE_PX:
                removed = lines.pop(i)
                print(f"­čŚĹ´ŞĆ Usuni─Öto lini─Ö")
                removed_count += 1
            else:
                i += 1

        # Usuń screeny (PNG) pod kursorem
        for idx in range(len(screens) - 1, -1, -1):
            s = screens[idx]
            sx, sy, sw, sh = s["x"], s["y"], s["w"], s["h"]
            inside = sx <= wx <= sx + sw and sy <= wy <= sy + sh
            if inside:
                _delete_texture_for(s)
                screens.pop(idx)
                removed_count += 1
                print("🧽 Usunięto screen (gumka).")
                break

        # Usuń JSON (tekstura podglądu) pod kursorem
        for idx in range(len(json_objects) - 1, -1, -1):
            j = json_objects[idx]
            sx, sy, sw, sh = j["x"], j["y"], j["w"], j["h"]
            inside = sx <= wx <= sx + sw and sy <= wy <= sy + sh
            if inside:
                _delete_texture_for(j)
                json_objects.pop(idx)
                removed_count += 1
                print("🧽 Usunięto JSON (gumka).")
                break
        
        if removed_count > 0:
            dpg.set_value("result_text", f"Usunięto obiekty: {removed_count}")
            redraw_canvas()
        return

    # PRIORYTET 2: Rysowanie linii
    if draw_line_mode:
        if line_start is None:
            line_start = (wx, wy)
        else:
            pts = _orth_points_simple(line_start[0], line_start[1], wx, wy)
            lines.append({"points": pts, "color": [0, 255, 0, 255], "thickness": 2})
            line_start = None
        redraw_canvas()
        return

    # PRIORYTET 3: Resize/Drag - POPRAWIONA WERSJA
    active_object_id = None
    resize_mode = False
    resize_object_id = None
    resize_edge_right = False
    resize_edge_bottom = False
    needs_texture_update = False

    # Wi─Ökszy margines dla lepszego wykrywania
    RESIZE_MARGIN_WORLD = 20 / max(zoom, 0.001)  # 20 pikseli w world units

    all_objs = get_all_objects()
    for obj in reversed(all_objs):
        sx, sy, sw, sh = obj["x"], obj["y"], obj["w"], obj["h"]
        
        # Sprawd┼║ czy jeste┼Ťmy blisko prawej kraw─Ödzi
        near_right = abs(wx - (sx + sw)) <= RESIZE_MARGIN_WORLD
        # Sprawd┼║ czy jeste┼Ťmy blisko dolnej kraw─Ödzi
        near_bottom = abs(wy - (sy + sh)) <= RESIZE_MARGIN_WORLD
        
        # Sprawd┼║ czy jeste┼Ťmy w obszarze obiektu (z marginesem resize)
        in_x_range = (sx - RESIZE_MARGIN_WORLD) <= wx <= (sx + sw + RESIZE_MARGIN_WORLD)
        in_y_range = (sy - RESIZE_MARGIN_WORLD) <= wy <= (sy + sh + RESIZE_MARGIN_WORLD)
        
        # Je┼Ťli jeste┼Ťmy blisko kraw─Ödzi i w zakresie obiektu
        if in_x_range and in_y_range and (near_right or near_bottom):
            resize_mode = True
            resize_object_id = obj["id"]
            resize_edge_right = near_right
            resize_edge_bottom = near_bottom
            
            # Debug
            print(f"­čöž RESIZE MODE: {obj['id']}")
            print(f"   Pozycja myszy: wx={wx:.1f}, wy={wy:.1f}")
            print(f"   Obiekt: x={sx:.1f}, y={sy:.1f}, w={sw:.1f}, h={sh:.1f}")
            print(f"   Prawa kraw─Öd┼║: {sx + sw:.1f}, odleg┼éo┼Ť─ç: {abs(wx - (sx + sw)):.1f}")
            print(f"   Dolna kraw─Öd┼║: {sy + sh:.1f}, odleg┼éo┼Ť─ç: {abs(wy - (sy + sh)):.1f}")
            print(f"   Resize RIGHT: {resize_edge_right}, BOTTOM: {resize_edge_bottom}")
            print(f"   Margin: {RESIZE_MARGIN_WORLD:.1f}")
            
            dpg.set_value("result_text", f"­čöž Resize: {'prawo' if near_right else ''} {'d├│┼é' if near_bottom else ''}")
            return

        # Sprawd┼║ czy klikni─Öto WEWN─äTRZ obiektu (dla drag)
        inside = sx <= wx <= sx + sw and sy <= wy <= sy + sh
        
        if inside:
            active_object_id = obj["id"]
            drag_offset = (wx - sx, wy - sy)
            print(f"Ôťő DRAG MODE: {obj['id']}")
            dpg.set_value("result_text", f"Ôťő Przeci─ůganie: {obj['id']}")
            return


def on_left_double_click(sender, app_data):
    mx, my = dpg.get_mouse_pos()
    wx, wy = world_from_screen(mx, my)

    for obj in reversed(notes):
        sx, sy, sw, sh = obj["x"], obj["y"], obj["w"], obj["h"]
        inside = sx <= wx <= sx + sw and sy <= wy <= sy + sh

        if inside:
            open_note_editor(obj["id"])
            return

    for obj in reversed(json_objects):
        sx, sy, sw, sh = obj["x"], obj["y"], obj["w"], obj["h"]
        inside = sx <= wx <= sx + sw and sy <= wy <= sy + sh

        if inside:
            open_json_viewer(obj["path"])
            return


def on_left_release(sender, app_data):
    global active_object_id, resize_mode, resize_object_id, needs_texture_update
    
    if resize_mode and resize_object_id and needs_texture_update:
        for obj in notes:
            if obj["id"] == resize_object_id:
                update_note_texture(obj)
                break
        needs_texture_update = False
    
    active_object_id = None
    resize_mode = False
    resize_object_id = None


def on_left_drag(sender, app_data):
    global needs_texture_update
    
    mx, my = dpg.get_mouse_pos()
    wx, wy = world_from_screen(mx, my)

    # RESIZE MODE
    if resize_mode and resize_object_id is not None:
        all_objs = get_all_objects()
        for obj in all_objs:
            if obj["id"] == resize_object_id:
                old_w = obj["w"]
                old_h = obj["h"]
                
                # Aktualizuj szeroko┼Ť─ç
                if resize_edge_right:
                    new_w = max(MIN_SIZE, wx - obj["x"])
                    obj["w"] = new_w
                
                # Aktualizuj wysoko┼Ť─ç
                if resize_edge_bottom:
                    new_h = max(MIN_SIZE, wy - obj["y"])
                    obj["h"] = new_h
                
                # Debug - poka┼╝ co si─Ö zmienia
                if obj["w"] != old_w or obj["h"] != old_h:
                    print(f"­čôĆ Resize: {old_w:.0f}x{old_h:.0f} -> {obj['w']:.0f}x{obj['h']:.0f}")
                
                # Oznacz ┼╝e trzeba zaktualizowa─ç tekstur─Ö notatki
                if obj["type"] == "note":
                    needs_texture_update = True
                
                break
        
        redraw_canvas()
        return

    # DRAG MODE
    if active_object_id is None:
        return

    all_objs = get_all_objects()
    for obj in all_objs:
        if obj["id"] == active_object_id:
            obj["x"] = wx - drag_offset[0]
            obj["y"] = wy - drag_offset[1]
            break

    redraw_canvas()


def on_middle_down(sender, app_data):
    global is_panning, last_mouse_pos
    is_panning = True
    last_mouse_pos = _mouse_pos_canvas_local()


def on_middle_drag(sender, app_data):
    global pan_x, pan_y, last_mouse_pos
    if not is_panning:
        return
    
    mx, my = _mouse_pos_canvas_local()
    dx = mx - last_mouse_pos[0]
    dy = my - last_mouse_pos[1]
    pan_x += dx
    pan_y += dy
    last_mouse_pos = (mx, my)
    redraw_canvas()


def on_middle_release(sender, app_data):
    global is_panning
    is_panning = False


def on_mouse_wheel(sender, app_data):
    global zoom, pan_x, pan_y

    # Zoom działa TYLKO gdy kursor jest nad canvasem,
    # i NIE działa nad oknami dialogowymi (backup, file dialog).
    try:
        mx, my = dpg.get_mouse_pos()

        # 1) Jeśli mysz nie jest nad canvasem -> żadnego zoomu.
        if dpg.does_item_exist(CANVAS_TAG):
            try:
                c_min_x, c_min_y = dpg.get_item_rect_min(CANVAS_TAG)
                c_max_x, c_max_y = dpg.get_item_rect_max(CANVAS_TAG)
                if not (c_min_x <= mx <= c_max_x and c_min_y <= my <= c_max_y):
                    return
            except Exception:
                # Gdyby pobranie recta się wywaliło, zachowaj się konserwatywnie.
                return

        # 2) Jeżeli kursor jest na oknie backupów lub file dialogu -> też blokuj zoom.
        for overlay in (BACKUP_WINDOW_TAG, FILE_DIALOG_TAG):
            if not dpg.does_item_exist(overlay):
                continue
            try:
                min_x, min_y = dpg.get_item_rect_min(overlay)
                max_x, max_y = dpg.get_item_rect_max(overlay)
            except Exception:
                continue
            if min_x <= mx <= max_x and min_y <= my <= max_y:
                return
    except Exception:
        pass

    delta = app_data
    if delta == 0:
        return

    old_zoom = zoom
    factor = 1.1 if delta > 0 else 1 / 1.1
    zoom = max(0.1, min(5.0, zoom * factor))

    mx, my = _mouse_pos_canvas_local()
    if zoom != old_zoom:
        pan_x = mx - (mx - pan_x) * (zoom / old_zoom)
        pan_y = my - (my - pan_y) * (zoom / old_zoom)

    redraw_canvas()


def on_mouse_move(sender, app_data):
    if (draw_line_mode and line_start is not None) or erase_mode:
        redraw_canvas()


def on_viewport_resize(sender, app_data):
    """Dopasuj okno i canvas do aktualnego rozmiaru viewportu (fullscreen)."""
    try:
        width, height = app_data
    except Exception:
        # Fallback, gdy DearPyGui zwraca inne dane
        try:
            width = dpg.get_viewport_client_width()
            height = dpg.get_viewport_client_height()
        except Exception:
            width = dpg.get_viewport_width()
            height = dpg.get_viewport_height()

    # Rozmiar głównego okna
    try:
        dpg.set_item_width(WINDOW_TAG, width)
        dpg.set_item_height(WINDOW_TAG, height)
    except Exception:
        pass

    # Zostaw trochę miejsca na toolbar u góry
    canvas_margin_x = 20
    canvas_margin_top = 150
    canvas_width = max(200, width - 2 * canvas_margin_x)
    canvas_height = max(200, height - canvas_margin_top - 20)

    try:
        dpg.set_item_width(CANVAS_TAG, canvas_width)
        dpg.set_item_height(CANVAS_TAG, canvas_height)
    except Exception:
        pass


# ================== UI ==================

if __name__ == "__main__" and "--export-pipeline" in sys.argv:
    try:
        state = build_pipeline_state()
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            pyjson.dump(state, f, ensure_ascii=False, indent=2)
        print(f"[flowui] Exported pipeline state -> {STATE_FILE}")
    except Exception as e:
        print(f"[flowui] Export failed: {e}")
        raise
    raise SystemExit(0)

if dpg is None:
    raise ModuleNotFoundError(
        "dearpygui is required to run the UI. Install it (e.g. `pip install dearpygui`) "
        "or run `python workflow/app.py --export-pipeline` to only generate flowui_state.json."
    )

dpg.create_context()

with dpg.texture_registry(tag=TEXREG_TAG):
    pass

with dpg.window(label="Flow GUI", tag=WINDOW_TAG, width=1200, height=800, pos=(0, 0)):
    dpg.add_text("­čŚ║´ŞĆ Canvas: PNG + JSON + Notatki + Linie + Pan/Zoom + Gumka")

    with dpg.group(horizontal=True):
        dpg.add_button(label="­čôü Dodaj PNG/JSON", callback=on_file_button)
        dpg.add_button(label="­čöä Od┼Ťwie┼╝", callback=lambda: check_and_reload_files())
        dpg.add_button(label="Reset", callback=reset_pipeline_layout)
        dpg.add_button(label="ÔĆ»´ŞĆ", callback=toggle_auto_reload, width=30)
        dpg.add_text("Auto: ON", tag="auto_reload_status", color=(100, 255, 100))
        dpg.add_button(label="R", callback=on_key_r, width=30)
        dpg.add_text("Resize (R): OFF", tag="resize_text", color=(255, 120, 120))
        dpg.add_button(label="F", callback=on_key_f, width=30)
        dpg.add_text("Gotowy", tag="result_text")

    with dpg.group(horizontal=True):
        dpg.add_text("Tryb linii (D): OFF", tag="mode_text")
        dpg.add_text("  |  ", color=(100, 100, 100))
        dpg.add_text("Gumka (C): OFF", tag="erase_text")
        dpg.add_text("  |  ", color=(100, 100, 100))
        dpg.add_text("Notatka (N): OFF", tag="note_text")

    dpg.add_separator()
    dpg.add_text("LPM: drag/resize | DWUKLIK: edytuj notatk─Ö/JSON | PPM: pan | Scroll: zoom")
    dpg.add_text("D: rysuj linie | C: gumka | N: dodaj notatk─Ö (kliknij gdzie ma by─ç)")
    dpg.add_text("F: dopasuj widok (fit) | R: resize ON/OFF | Reset: odbuduj workflow")
    dpg.add_separator()

    # Rozmiar zostanie nadpisany w on_viewport_resize, tu tylko startowe wartości
    dpg.add_drawlist(width=1200, height=650, tag=CANVAS_TAG)

with dpg.file_dialog(
    directory_selector=False,
    show=False,
    callback=on_file_dialog,
    id="file_dialog_id",
    width=700,
    height=400
):
    dpg.add_file_extension(".*", color=(200, 200, 200, 255))
    dpg.add_file_extension(".png", color=(150, 255, 150, 255))
    dpg.add_file_extension(".json", color=(255, 200, 0, 255))

with dpg.window(label="Backups (B)", tag=BACKUP_WINDOW_TAG, width=500, height=320, show=False):
    dpg.add_text("Runs: brak", tag=PIPELINE_STATUS_TAG)
    dpg.add_child_window(tag=PIPELINE_LIST_TAG, width=-1, height=250, border=True)
    dpg.add_button(label="Odswiez liste", callback=refresh_pipeline_runs)


with dpg.handler_registry():
    dpg.add_mouse_click_handler(button=0, callback=on_left_click)
    dpg.add_mouse_double_click_handler(button=0, callback=on_left_double_click)
    dpg.add_mouse_release_handler(button=0, callback=on_left_release)
    dpg.add_mouse_drag_handler(button=0, threshold=2, callback=on_left_drag)

    dpg.add_mouse_click_handler(button=1, callback=on_middle_down)
    dpg.add_mouse_release_handler(button=1, callback=on_middle_release)
    dpg.add_mouse_drag_handler(button=1, threshold=0, callback=on_middle_drag)

    dpg.add_mouse_wheel_handler(callback=on_mouse_wheel)
    dpg.add_mouse_move_handler(callback=on_mouse_move)

    dpg.add_key_press_handler(key=dpg.mvKey_D, callback=on_key_d)
    dpg.add_key_press_handler(key=dpg.mvKey_C, callback=on_key_c)
    dpg.add_key_press_handler(key=dpg.mvKey_N, callback=on_key_n)
    dpg.add_key_press_handler(key=dpg.mvKey_R, callback=on_key_r)
    dpg.add_key_press_handler(key=dpg.mvKey_F, callback=on_key_f)
    dpg.add_key_press_handler(key=dpg.mvKey_Z, callback=on_key_z)
    dpg.add_key_press_handler(key=dpg.mvKey_B, callback=on_key_b)

dpg.create_viewport(title="Flow UI - Canvas", width=1280, height=720)
dpg.setup_dearpygui()
dpg.show_viewport()
with contextlib.suppress(Exception):
    dpg.set_viewport_resize_callback(on_viewport_resize)
with contextlib.suppress(Exception):
    dpg.maximize_viewport()
    try:
        vw = dpg.get_viewport_client_width()
        vh = dpg.get_viewport_client_height()
    except Exception:
        vw = dpg.get_viewport_width()
        vh = dpg.get_viewport_height()
    on_viewport_resize(None, (vw, vh))

dpg.set_exit_callback(on_exit)

if not load_state_pipeline():
    default_image_path = str(DEBUG_SCREEN_DIR / "ocr_strip1_active.png")
    if os.path.exists(default_image_path):
        add_screen_from_path(default_image_path, x=0.0, y=0.0)

refresh_pipeline_runs()

reload_thread = threading.Thread(target=auto_reload_thread, daemon=True)
reload_thread.start()

try:
    dpg.start_dearpygui()
except KeyboardInterrupt:
    print("\nÔÜá´ŞĆ Ctrl+C")
    save_state()
finally:
    dpg.destroy_context()
