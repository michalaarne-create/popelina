# -*- coding: utf-8 -*-
"""
ADVANCED UI ELEMENT SCORER
Ocena kaĹĽdego boxa z przemyĹ›lanym systemem punktacji.
NIE polega na detekcji graficznych trĂłjkÄ…tĂłw (PaddleOCR ich nie widzi).
"""

import os
import sys
import json
import math
import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import shutil

# Optional OCR deps
try:
    from PIL import Image
except Exception:
    Image = None
try:
    import pytesseract
except Exception:
    pytesseract = None

OCR_LANG = os.environ.get("OCR_LANG", "pol+eng")

# ======================== DEBUG MODE ============================
DEBUG_MODE = False  # Domyślnie bez hałaśliwego outputu (Windows console encoding fix)
DEBUG_RESULTS = []  # Konsolowe zestawienie top wyników dropdown (legacy)
# Szczegółowe debug per element (zapisywane do pliku obok wyników)
ELEMENT_DEBUG: Dict[str, Dict[str, dict]] = {}

# ======================== ĹšCIEĹ»KI I/O ===========================
ROOT = Path(__file__).resolve().parents[2]
DATA_SCREEN_DIR = ROOT / "data" / "screen"
REGION_GROW_DIR = DATA_SCREEN_DIR / "region_grow" / "region_grow"
INPUT_DIR = str(REGION_GROW_DIR)
RATE_RESULTS_DIR = str(DATA_SCREEN_DIR / "rate" / "rate_results")
RATE_RESULTS_DEBUG_DIR = str(DATA_SCREEN_DIR / "rate" / "rate_results_debug")
RATE_SUMMARY_DIR = str(DATA_SCREEN_DIR / "rate" / "rate_summary")
RATE_RESULTS_CURRENT_DIR = str(DATA_SCREEN_DIR / "rate" / "rate_results_current")
RATE_RESULTS_DEBUG_CURRENT_DIR = str(DATA_SCREEN_DIR / "rate" / "rate_results_debug_current")
RATE_SUMMARY_CURRENT_DIR = str(DATA_SCREEN_DIR / "rate" / "rate_summary_current")
DOM_LIVE_DIR = ROOT / "dom_live"
CURRENT_QUESTION_PATH = DOM_LIVE_DIR / "current_question.json"

# ======================== PROGI DECYZYJNE =======================
THRESHOLDS = {
    "next_active": 0.45,
    "next_inactive": 0.45,
    "dropdown": 0.50,
    "answer_single": 0.40,
    "answer_multi": 0.40,
    "cookie_accept": 0.65,
    "cookie_reject": 0.65
}

# ======================== UTILITY FUNCTIONS =====================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def norm_text(s: str) -> str:
    if not s: return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def lower_strip_acc(s: str) -> str:
    """Zwraca wersjÄ™ z ogonkami + bez ogonkĂłw (dla lepszego matchowania)"""
    if not s: return ""
    w = s.lower()
    nfkd = unicodedata.normalize('NFKD', w)
    no = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return w + "\n" + no

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _ocr_text_from_image(img, bbox, lang=OCR_LANG) -> str:
    if Image is None or pytesseract is None or img is None or bbox is None:
        return ""
    try:
        x1, y1, x2, y2 = [int(b) for b in bbox]
        W, H = img.size
        pad = max(2, int(0.03 * max(1, x2 - x1, y2 - y1)))
        x1 = _clamp(x1 - pad, 0, max(0, W - 1))
        y1 = _clamp(y1 - pad, 0, max(0, H - 1))
        x2 = _clamp(x2 + pad, 1, W)
        y2 = _clamp(y2 + pad, 1, H)
        if x2 <= x1 or y2 <= y1:
            return ""
        crop = img.crop((x1, y1, x2, y2))
        text = pytesseract.image_to_string(crop, lang=lang, config='--psm 6').strip()
        return norm_text(text)
    except Exception:
        return ""


def combine_scores(scores: List[float]) -> float:
    """
    Kombinuje wiele niezaleĹĽnych scores probabilistycznie.
    Lepsze niĹĽ suma (nie przekracza 1.0)
    """
    if not scores:
        return 0.0
    product = 1.0
    for s in scores:
        product *= (1.0 - max(0.0, min(1.0, s)))
    return min(1.0 - product, 1.0)

def soft_cap(score: float, cap: float = 1.0, softness: float = 0.1) -> float:
    """MiÄ™kkie ograniczenie gĂłrne"""
    if score <= cap - softness:
        return score
    excess = score - (cap - softness)
    return cap - softness + softness * (1 - math.exp(-excess / softness))

def bbox_distance(a: List[int], b: List[int]) -> float:
    """Minimalna odlegĹ‚oĹ›Ä‡ miÄ™dzy dwoma boksami"""
    if not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]):
        return 0.0
    dx = max(0, b[0] - a[2], a[0] - b[2])
    dy = max(0, b[1] - a[3], a[1] - b[3])
    return math.sqrt(dx*dx + dy*dy)

def distance_score(distance_px: float, max_distance: float = 100, steepness: float = 1.5) -> float:
    """Konwertuje odlegĹ‚oĹ›Ä‡ na score 0-1 z nieliniowym spadkiem"""
    if distance_px <= 0:
        return 1.0
    if distance_px >= max_distance:
        return 0.0
    ratio = distance_px / max_distance
    return (1.0 - ratio) ** steepness

def is_below(ref_bbox: List[int], test_bbox: List[int], margin: int = 5) -> bool:
    """Czy test_bbox jest poniĹĽej ref_bbox"""
    return int(test_bbox[1]) >= int(ref_bbox[3]) + margin

def bbox_height(b: List[int]) -> int:
    return max(0, b[3] - b[1])

def bbox_width(b: List[int]) -> int:
    return max(0, b[2] - b[0])

def bbox_center(b: List[int]) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

def bbox_aspect_ratio(b: List[int]) -> float:
    h = bbox_height(b)
    w = bbox_width(b)
    return w / max(1, h)

def point_in_bbox(pt: Tuple[float, float], bbox: List[int], margin: float = 0.0) -> bool:
    if pt is None or bbox is None:
        return False
    x, y = pt
    return (bbox[0] - margin) <= x <= (bbox[2] + margin) and (bbox[1] - margin) <= y <= (bbox[3] + margin)

# --- UTIL ADDITIONS (obok bbox_* funkcji) ---
def rect_area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

def rect_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    return inter / max(1, rect_area(a) + rect_area(b) - inter)

def center_distance(a, b):
    ax, ay = bbox_center(a); bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)

def looks_global_box(b, img_w, img_h, area_ratio=0.40):
    if not b: return False
    w = max(1, b[2]-b[0]); h = max(1, b[3]-b[1])
    if b[0] <= 2 and b[1] <= 2 and (img_w - b[2]) <= 2 and (img_h - b[3]) <= 2:
        return True
    return (w*h) >= area_ratio * max(1, img_w*img_h)

def _point_from_any(value) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return (float(value["x"]), float(value["y"]))
        if "cx" in value and "cy" in value:
            return (float(value["cx"]), float(value["cy"]))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    return None

def _points_from_record(record) -> Optional[List[Tuple[float, float]]]:
    if record is None:
        return None
    candidates = []
    if isinstance(record, dict):
        for key in ("points_original", "points_scaled", "points", "vertices"):
            pts = record.get(key)
            if isinstance(pts, (list, tuple)) and len(pts) >= 3:
                norm = []
                for p in pts:
                    pt = _point_from_any(p)
                    if pt is None:
                        norm = []
                        break
                    norm.append(pt)
                if len(norm) >= 3:
                    return norm
        if all(k in record for k in ("x1", "y1", "x2", "y2", "x3", "y3")):
            pts = [
                (float(record["x1"]), float(record["y1"])),
                (float(record["x2"]), float(record["y2"])),
                (float(record["x3"]), float(record["y3"])),
            ]
            return pts
    if isinstance(record, (list, tuple)):
        if len(record) >= 3 and all(isinstance(p, (list, tuple)) for p in record):
            norm = []
            for p in record:
                pt = _point_from_any(p)
                if pt is None:
                    norm = []
                    break
                norm.append(pt)
            if len(norm) >= 3:
                return norm
        flat = []
        for v in record:
            if isinstance(v, (int, float)):
                flat.append(float(v))
        if len(flat) >= 6 and len(flat) % 2 == 0:
            pts = []
            for i in range(0, len(flat), 2):
                pts.append((flat[i], flat[i+1]))
            if len(pts) >= 3:
                return pts
    return None

def triangle_bbox_from_record(record) -> Optional[List[float]]:
    if record is None:
        return None
    if isinstance(record, dict):
        for key in ("bbox", "box", "rect", "xyxy"):
            val = record.get(key)
            if isinstance(val, (list, tuple)) and len(val) == 4:
                return [float(val[0]), float(val[1]), float(val[2]), float(val[3])]
        if all(k in record for k in ("x", "y", "w", "h")):
            x = float(record["x"]); y = float(record["y"])
            w = float(record["w"]); h = float(record["h"])
            return [x, y, x + w, y + h]
        if all(k in record for k in ("x", "y", "width", "height")):
            x = float(record["x"]); y = float(record["y"])
            w = float(record["width"]); h = float(record["height"])
            return [x, y, x + w, y + h]
    if isinstance(record, (list, tuple)) and len(record) == 4 and all(isinstance(v, (int, float)) for v in record):
        return [float(record[0]), float(record[1]), float(record[2]), float(record[3])]
    pts = _points_from_record(record)
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _mirror_to_current(src_path: str, current_dir: str) -> None:
    """
    Skopiuj plik do katalogu *_current, trzymając tam zawsze tylko jeden plik.
    Jeżeli coś pójdzie nie tak, po prostu pomiń (bez przerywania ratingu).
    """
    try:
        if not src_path or not os.path.isfile(src_path):
            return
        ensure_dir(current_dir)
        # Usuń stare pliki w katalogu *_current
        for name in os.listdir(current_dir):
            full = os.path.join(current_dir, name)
            if os.path.isfile(full):
                try:
                    os.remove(full)
                except Exception:
                    pass
        dst = os.path.join(current_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst)
    except Exception:
        # Brak loga tutaj, żeby nie hałasować w konsoli przy drobnych problemach
        pass


def _load_scrollable_regions_from_dom() -> List[List[int]]:
    """
    Odczytuje z current_question.json listę prostokątów DOM, które są
    oznaczone jako scrollowalne (scrollable==True). Używane do annotacji
    elementów w summary (np. dropdownów) polem 'scrollable'.
    """
    try:
        if not CURRENT_QUESTION_PATH.exists():
            return []
        data = json.loads(CURRENT_QUESTION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    rects: List[List[int]] = []
    elems = data.get("question_elements") or data.get("text_elements") or []
    for el in elems:
        if not isinstance(el, dict):
            continue
        if not (el.get("scrollable") or el.get("scrollableY") or el.get("scrollableX")):
            continue
        bbox = el.get("bbox") or {}
        try:
            x = int(bbox.get("x", 0))
            y = int(bbox.get("y", 0))
            w = int(bbox.get("width", 0))
            h = int(bbox.get("height", 0))
        except Exception:
            continue
        if w <= 4 or h <= 4:
            continue
        rects.append([x, y, x + w, y + h])
    return rects

def triangle_centroid_from_record(record) -> Optional[Tuple[float, float]]:
    if record is None:
        return None
    if isinstance(record, dict):
        for key in ("centroid_original", "centroid_scaled", "centroid", "center"):
            pt = _point_from_any(record.get(key))
            if pt:
                return pt
    pts = _points_from_record(record)
    if pts:
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        n = max(1, len(pts))
        return (sx / n, sy / n)
    bbox = triangle_bbox_from_record(record)
    if bbox:
        return bbox_center([bbox[0], bbox[1], bbox[2], bbox[3]])
    if isinstance(record, (list, tuple)) and len(record) == 2:
        try:
            return (float(record[0]), float(record[1]))
        except Exception:
            return None
    return None

def normalize_triangles(raw_triangles: List[dict]) -> List[dict]:
    normalized = []
    for tri in raw_triangles:
        bbox = triangle_bbox_from_record(tri)
        centroid = triangle_centroid_from_record(tri)
        if bbox is None and centroid is not None:
            cx, cy = centroid
            bbox = [cx - 2.0, cy - 2.0, cx + 2.0, cy + 2.0]
        normalized.append({"bbox": bbox, "centroid": centroid, "raw": tri})
    return normalized

def triangles_hit_bbox(bbox: List[int], triangles: List[dict], margin: float = 3.0, iou_thresh: float = 0.05) -> int:
    if not triangles or not bbox:
        return 0
    hits = 0
    for tri in triangles:
        tri_bbox = tri.get("bbox")
        tri_centroid = tri.get("centroid")
        if tri_centroid and point_in_bbox(tri_centroid, bbox, margin):
            hits += 1
            continue
        if tri_bbox and rect_iou(
            [bbox[0], bbox[1], bbox[2], bbox[3]],
            [tri_bbox[0], tri_bbox[1], tri_bbox[2], tri_bbox[3]],
        ) >= iou_thresh:
            hits += 1
    return hits

# ======================== PATTERN DEFINITIONS ===================

# Unicode symbole (mogÄ… byÄ‡ w tekĹ›cie OCR)
DROPDOWN_UNICODE = "â–ľâ–żâ–ĽâŻ†â®źâ¨â‡âŚ„đź”˝Ë…âŹ·"
DROPDOWN_ASCII = "â†“â¬‡â‡“"
BULLET_GLYPHS = "â€˘Â·â€Łâ—¦âž˘â–şâ–Şâ–«"
CHECK_GLYPHS = "ââ‘â’â–ˇâ– âś“âś”"
RADIO_GLYPHS = "â—‹â—Żâ—Źâ—‰â—ŚâšŞâš«"

# Regex patterns
NEXT_KEYWORDS_PL = [
    r"\bdalej\b", r"\bnast[eÄ™]pny\b", r"\bnast[eÄ™]pna\b", 
    r"\bkontynuuj\b", r"\bprzejd[Ĺşz]\b"
]
NEXT_KEYWORDS_EN = [
    r"\bnext\b", r"\bcontinue\b", r"\bproceed\b", r"\bforward\b"
]
NEXT_RE = re.compile("|".join(NEXT_KEYWORDS_PL + NEXT_KEYWORDS_EN), re.IGNORECASE)

# Dodatkowe frazy „ankietowe” dla przycisków następnego kroku.
NEXT_EXTRA_PL = [
    r"\bnast[eę]pne\s+pytanie\b",
    r"\bnast[eę]pna\s+strona\b",
    r"\bprzejd[źz]\s+dalej\b",
    r"\bprzejd[źz]\s+do\s+(nast[eę]pnego|kolejnego)\b",
    r"\bzako[nń]cz\b",
    r"\bzako[nń]cz\s+ankiet[ęey]\b",
    r"\bwy[sś]lij\b",
    r"\bwy[sś]lij\s+odpowiedzi\b",
]
NEXT_EXTRA_EN = [
    r"\bfinish\b",
    r"\bsubmit\b",
    r"\bdone\b",
    r"\bnext\s+question\b",
]
NEXT_KEYWORDS_PL = NEXT_KEYWORDS_PL + NEXT_EXTRA_PL
NEXT_KEYWORDS_EN = NEXT_KEYWORDS_EN + NEXT_EXTRA_EN
NEXT_RE = re.compile("|".join(NEXT_KEYWORDS_PL + NEXT_KEYWORDS_EN), re.IGNORECASE)

DROPDOWN_STRONG_PL = [
    r"\blista\s+rozwijana\b", r"\bmenu\s+rozwijan[ea]\b",
    r"\brozwi[nĹ„]\s+menu\b", r"\bkliknij\s+aby\s+rozwi[nnÄ…][Ä‡Ä‡]\b"
]
DROPDOWN_STRONG_EN = [
    r"\bdrop\s*down\b", r"\bpull\s*down\s+menu\b", r"\bexpand\s+menu\b"
]
DROPDOWN_MODERATE_PL = [
    r"\bwybierz\s+z\s+listy\b", r"\bwybierz\s+opcj[eÄ™]\b", r"\brozwi[nĹ„]\b"
]
DROPDOWN_MODERATE_EN = [
    r"\bselect\s+from\s+list\b", r"\bselect\s+option\b", r"\bselect\s+an\s+option\b"
]
DROPDOWN_WEAK_PL = [r"\bwybierz\b", r"\bopcje\b"]
DROPDOWN_WEAK_EN = [r"\bselect\b", r"\bchoose\b", r"\bpick\b"]

DROPDOWN_STRONG_RE = re.compile("|".join(DROPDOWN_STRONG_PL + DROPDOWN_STRONG_EN), re.IGNORECASE)
DROPDOWN_MODERATE_RE = re.compile("|".join(DROPDOWN_MODERATE_PL + DROPDOWN_MODERATE_EN), re.IGNORECASE)
DROPDOWN_WEAK_RE = re.compile("|".join(DROPDOWN_WEAK_PL + DROPDOWN_WEAK_EN), re.IGNORECASE)

DROPDOWN_PLACEHOLDER_PATTERNS = [
    r"^--.*--$",
    r"^\-\s+.+",
    r"^Select\.\.\.$",
    r"^Choose.*$",
    r"^\(.+\)$",
    r"^<.+>$",
    r"^\.\.\.$"
]
DROPDOWN_PLACEHOLDER_RE = [re.compile(p) for p in DROPDOWN_PLACEHOLDER_PATTERNS]

QUESTION_WORDS_PL = [
    r"\b(kto|kogo|komu|kim|czyj|czyja|czyje|czyi)\b",
    r"\b(co|czym|gdzie|kiedy|dok[aÄ…]d|sk[aÄ…]d|dlaczego|po\s+co)\b",
    r"\b(jak|jaka|jakie|jaki|ile|kt[oĂł]ry|kt[oĂł]ra|kt[oĂł]re|kt[oĂł]rzy)\b"
]
QUESTION_WORDS_EN = [
    r"\b(who|what|which|when|where|why|how|whose)\b"
]
QUESTION_RE = re.compile("|".join(QUESTION_WORDS_PL + QUESTION_WORDS_EN), re.IGNORECASE)

SINGLE_STRICT_PL = [
    r"\bktor(y|a)\s+z\s+ponizszych\s+odpowiedzi\b",
    r"\bktor(y|a)\s+z\s+ponizszych\s+stwierdzen\b",
    r"\bktor(y|a)\s+z\s+nich\s+jest\s+(prawidlowa|poprawna)\b",
    r"\bjedna\s+z\s+ponizszych\s+odpowiedzi\s+jest\s+(prawidlowa|poprawna)\b",
    r"\btylko\s+jedna\s+odpowiedz\s+jest\s+(prawidlowa|poprawna)\b",
    r"\bwybierz\s+dokladnie\s+jedna\s+odpowiedz\b",
    r"\bzaznacz\s+dokladnie\s+jedna\s+odpowiedz\b",
    r"\bkt[oĂł]ry\s+z\s+(poni[zĹĽ]szych|tych|odpowiedzi|stwierdze[nĹ„])\b",
    r"\bwybierz\s+jedn[aÄ…]\b", r"\bjedna\s+odpowied[zĹş]\b",
    r"\bjednokrotnego\s+wyboru\b", r"\bzaznacz\s+jedn[aÄ…]\b"
]
SINGLE_STRICT_EN = [
    r"\bwhich\s+of\s+the\s+following\b.*\b(is|was)\b",
    r"\bselect\s+one\b", r"\bchoose\s+one\b", r"\bpick\s+one\b",
    r"\bsingle\s+choice\b", r"\bsingle\s+answer\b", r"\bone\s+correct\s+answer\b"
]
SINGLE_STRICT_RE = re.compile("|".join(SINGLE_STRICT_PL + SINGLE_STRICT_EN), re.IGNORECASE)

MULTI_STRICT_PL = [
    r"\bktore\s+z\s+ponizszych\s+odpowiedzi\s+sa\s+(prawidlowe|poprawne)\b",
    r"\bktore\s+z\s+ponizszych\s+stwierdzen\s+sa\s+(prawdziwe|prawidlowe|poprawne)\b",
    r"\bktore\s+z\s+nich\s+sa\s+(prawdziwe|prawidlowe|poprawne)\b",
    r"\bzaznacz\s+wszystkie\s+poprawne\s+odpowiedzi\b",
    r"\bzaznacz\s+wszystkie\s+prawidlowe\s+odpowiedzi\b",
    r"\bwybierz\s+wszystkie\s+poprawne\s+odpowiedzi\b",
    r"\bmozesz\s+zaznaczyc\s+kilka\s+odpowiedzi\b",
    r"\bmozesz\s+zaznaczyc\s+wiecej\s+niz\s+jedna\s+odpowiedz\b",
    r"\bwiecej\s+niz\s+jedna\s+odpowiedz\s+jest\s+(prawidlowa|poprawna)\b",
    r"\bco\s+najmniej\s+jedna\s+odpowiedz\s+jest\s+(prawidlowa|poprawna)\b",
    r"\bkt[oĂł]re\s+z\s+(poni[zĹĽ]szych|tych|odpowiedzi|stwierdze[nĹ„])\b",
    r"\bzaznacz\s+wszystkie\b", r"\bwybierz\s+wszystkie\b",
    r"\bwszystkie\s+(kt[oĂł]re|prawid[Ĺ‚l]owe|poprawne)\b",
    r"\bmo[zĹĽ]esz\s+wybra[cÄ‡]\s+wi[eÄ™]cej\s+ni[zĹĽ]\s+jedn[aÄ…]\b",
    r"\bwiele\s+odpowiedzi\b", r"\bwielokrotnego\s+wyboru\b",
    r"\bco\s+najmniej\s+(jeden|jedn[aÄ…])\b"
]
MULTI_STRICT_EN = [
    r"\bwhich\s+of\s+the\s+following\b.*\b(are|were)\b",
    r"\bselect\s+all\s+that\s+apply\b", r"\bselect\s+all\b", r"\bchoose\s+all\b",
    r"\bcheck\s+all\s+that\s+apply\b", r"\bmultiple\s+choice\b",
    r"\bmultiple\s+answers?\b", r"\bmore\s+than\s+one\b",
    r"\ball\s+that\s+are\s+correct\b"
]
MULTI_STRICT_RE = re.compile("|".join(MULTI_STRICT_PL + MULTI_STRICT_EN), re.IGNORECASE)

MULTI_PARTIAL_PL = [
    r"\(mo[zĹĽ]esz\s+zaznaczy[cÄ‡]\s+(wiele|kilka)\)",
    r"\(wielokrotny\s+wyb[oĂł]r\)"
]
MULTI_PARTIAL_EN = [
    r"\(multiple\s+selections?\s+allowed\)",
    r"\(check\s+all\)",
    r"\(select\s+multiple\)"
]
MULTI_PARTIAL_RE = re.compile("|".join(MULTI_PARTIAL_PL + MULTI_PARTIAL_EN), re.IGNORECASE)

COOKIE_CONTEXT_PL = [
    r"\bcookies?\b", r"\bciasteczk", r"\bpolityka\s+prywatno[sĹ›]ci\b",
    r"\brodo\b", r"\bgdpr\b", r"\bzgod[ya]\s+na\s+przetwarzanie\b"
]
COOKIE_CONTEXT_EN = [
    r"\bcookies?\b", r"\bprivacy\s+policy\b", r"\bdata\s+protection\b",
    r"\bgdpr\b", r"\bprivacy\b"
]
COOKIE_CONTEXT_RE = re.compile("|".join(COOKIE_CONTEXT_PL + COOKIE_CONTEXT_EN), re.IGNORECASE)

COOKIE_ACCEPT_PL = [
    r"\bakceptuj\b", r"\bakceptuj[eÄ™]\b", r"\bzgadzam\s+si[eÄ™]\b",
    r"\bprzyjmuj[eÄ™]\b", r"\bwyra[zĹĽ]am\s+zgod[eÄ™]\b", r"\btak\b"
]
COOKIE_ACCEPT_EN = [
    r"\baccept\b", r"\baccept\s+all\b", r"\bagree\b", r"\ballow\b",
    r"\ballow\s+all\b", r"\bok\b", r"\bgot\s+it\b", r"\bi\s+agree\b", r"\byes\b"
]
COOKIE_ACCEPT_RE = re.compile("|".join(COOKIE_ACCEPT_PL + COOKIE_ACCEPT_EN), re.IGNORECASE)

COOKIE_REJECT_PL = [
    r"\bodrzu[cÄ‡]\b", r"\bodm[oĂł]w\b", r"\bodmawiam\b",
    r"\bnie\s+zgadzam\s+si[eÄ™]\b", r"\bnie\s+ch[cÄ™]\b"
]
COOKIE_REJECT_EN = [
    r"\breject\b", r"\bdecline\b", r"\brefuse\b", r"\bdeny\b",
    r"\bno\s+thanks\b", r"\bdismiss\b", r"\bclose\b", r"\bno\b"
]
COOKIE_REJECT_RE = re.compile("|".join(COOKIE_REJECT_PL + COOKIE_REJECT_EN), re.IGNORECASE)

COOKIE_PARTIAL_PL = [
    r"\btylko\s+niezb[eÄ™]dne\b", r"\btylko\s+konieczne\b",
    r"\bzarz[aÄ…]dzaj\b", r"\bustawienia\b", r"\bpersonalizuj\b"
]
COOKIE_PARTIAL_EN = [
    r"\bonly\s+necessary\b", r"\bonly\s+essential\b", r"\brequired\s+only\b",
    r"\bmanage\b", r"\bsettings\b", r"\bcustomize\b", r"\bconfigure\b"
]
COOKIE_PARTIAL_RE = re.compile("|".join(COOKIE_PARTIAL_PL + COOKIE_PARTIAL_EN), re.IGNORECASE)

# ======================== DETECTION FUNCTIONS ===================

def has_unicode_symbol(text: str, symbols: str) -> bool:
    """Sprawdza czy tekst zawiera ktĂłryĹ› z symboli Unicode"""
    return any(s in text for s in symbols)

def count_unicode_symbol(text: str, symbols: str) -> int:
    """Liczy wystÄ…pienia symboli"""
    return sum(text.count(s) for s in symbols)

def has_list_marker_start(text: str) -> Tuple[bool, str]:
    """
    Sprawdza czy tekst zaczyna siÄ™ od markera listy.
    Zwraca (True/False, typ: 'bullet'/'checkbox'/'radio'/'none')
    """
    stripped = text.lstrip()
    if not stripped:
        return False, 'none'
    
    first_char = stripped[0]
    
    if first_char in CHECK_GLYPHS:
        return True, 'checkbox'
    elif first_char in RADIO_GLYPHS:
        return True, 'radio'
    elif first_char in BULLET_GLYPHS:
        return True, 'bullet'
    elif stripped.startswith("( )"):
        return True, 'radio'
    elif stripped.startswith("[ ]"):
        return True, 'checkbox'
    
    return False, 'none'

def is_questionish(text: str) -> bool:
    """Czy tekst wyglÄ…da na pytanie"""
    if not text:
        return False
    if "?" in text:
        return True
    return bool(QUESTION_RE.search(lower_strip_acc(text)))

def word_count(text: str) -> int:
    """Liczba sĹ‚Ăłw"""
    return len(text.split())

def line_count(text: str) -> int:
    """Liczba linii"""
    return text.count('\n') + 1

def has_arrow_symbols(text: str) -> int:
    """Liczba strzaĹ‚ek w tekĹ›cie"""
    arrows = "â†’âž”â–şÂ»>>"
    return sum(text.count(a) for a in arrows)

def _horiz_overlap_ratio(a: List[int], b: List[int]) -> float:
    """
    Zwraca poziomy overlap w przedziale 0-1 w relacji do w��>szego boxa.
    U�>ywane do wykrywania kolumny odpowiedzi / listy opcji.
    """
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    ov = max(0, min(ax2, bx2) - max(ax1, bx1))
    min_width = max(1, min(ax2 - ax1, bx2 - bx1))
    return ov / min_width

def _answer_candidates_below(question_elem: dict, elements: List[dict]) -> List[dict]:
    """
    Heurystycznie wybiera kandydat�w odpowiedzi znajduj�cych si� pod pytaniem.
    Opiera si� g�'�wnie na geometrii (po�o�enie, odleg�'o�� pionowa, overlap
    poziomy) z minimaln� filtracj� tekstow� (brak '?', niepusty tekst).
    """
    bbox = question_elem["bbox"]
    q_h = bbox_height(bbox)
    max_vertical_gap = max(60, q_h * 10)
    candidates: List[dict] = []

    for e in elements:
        if e is question_elem:
            continue
        eb = e.get("bbox")
        if not eb:
            continue
        if not is_below(bbox, eb):
            continue
        # Odleg�'o�� pionowa od do�>u pytania do g�>ry elementu
        dy = eb[1] - bbox[3]
        if dy > max_vertical_gap:
            continue
        # Wymagaj sensownego pokrycia poziomego z pytaniem
        if _horiz_overlap_ratio(bbox, eb) < 0.4:
            continue
        txt = (e.get("text") or "").strip()
        if not txt:
            continue
        if "?" in txt:
            continue
        candidates.append(e)

    return candidates

# ======================== SCORER FUNCTIONS ======================

def score_next(elem: dict, elements: List[dict], buttons: List[dict], 
               image_width: int, image_height: int) -> Tuple[float, float]:
    """
    Zwraca (next_inactive_score, next_active_score)
    """
    text = elem.get("text", "")
    bbox = elem["bbox"]

    # Pomi? du?e, globalne kontenery obejmuj?ce niemal ca?y ekran
    if looks_global_box(bbox, image_width, image_height):
        return 0.0, 0.0

    scores: List[float] = []

    # Słowa kluczowe
    text_normalized = lower_strip_acc(text)
    if NEXT_RE.search(text_normalized):
        has_pl = any(re.search(p, text_normalized) for p in NEXT_KEYWORDS_PL)
        has_en = any(re.search(p, text_normalized) for p in NEXT_KEYWORDS_EN)
        if has_pl or has_en:
            scores.append(0.35)

    # Proste, krótkie napisy typu „Dalej”, „Next”, „Kontynuuj”
    norm_simple = (text or "").strip().lower()
    SIMPLE_NEXT_WORDS = {
        "dalej",
        "next",
        "continue",
        "kontynuuj",
        "zakończ",
        "zakończ ankietę",
        "finish",
        "submit",
        "done",
    }
    if norm_simple in SIMPLE_NEXT_WORDS:
        scores.append(0.15)

    # Delikatny bonus, jeśli w tekście pada „ankiet...” (kontekst ankiety)
    if "ankiet" in text_normalized:
        scores.append(0.05)

    # Symbole strza?ek
    if has_arrow_symbols(text) > 0:
        scores.append(0.20)

    # Czy jest przyciskiem
    if elem.get("is_button"):
        scores.append(0.30)

    # D?ugo?? tekstu
    text_len = len(text)
    if text_len < 20:
        scores.append(0.15)
    elif text_len < 50:
        scores.append(0.05)
    elif text_len > 50:
        scores.append(-0.30)

    # Pozycja (preferencja dolnej/prawej cz??ci ekranu)
    cx, cy = bbox_center(bbox)
    rel_x = cx / max(1, image_width)
    rel_y = cy / max(1, image_height)
    if rel_y > 0.80:
        scores.append(0.25)
    elif rel_y > 0.60:
        scores.append(0.15)
    if rel_x > 0.70:
        scores.append(0.10)
    elif 0.45 <= rel_x <= 0.65 and rel_y > 0.80:
        scores.append(0.05)

    # Izolacja (brak innych element?w tu? poni?ej)
    max_vertical_gap = int(0.25 * max(1, image_height))
    below = []
    for e in elements:
        if e is elem:
            continue
        eb = e.get("bbox")
        if not eb:
            continue
        if not is_below(bbox, eb):
            continue
        dy = eb[1] - bbox[3]
        if dy > max_vertical_gap:
            continue
        below.append(e)
    if len(below) == 0:
        scores.append(0.10)

    # Wykluczenia
    if re.search(r"(wstecz|back|poprzedni|prev|previous)", text_normalized):
        scores.append(-0.60)
    if "?" in text:
        scores.append(-0.40)
    if text_len > 100:
        scores.append(-0.50)

    # Kombinuj scores
    base_score = combine_scores([s for s in scores if s > 0])
    penalties = sum([s for s in scores if s < 0])
    final_score = max(0.0, min(1.0, base_score + penalties))

    # Okre?l stan (active/inactive)
    btn_state = None
    for b in buttons:
        b_kind = (b.get("kind") or "").lower()
        if b_kind == "next":
            b_bbox = b.get("bbox", [])
            if b_bbox and bbox_distance(bbox, b_bbox) < 30:
                btn_state = bool(b.get("active", b.get("is_enabled", False)))
                final_score = min(1.0, final_score + 0.70)
                break

    if btn_state is not None:
        if btn_state:
            return 0.0, final_score
        else:
            return final_score, 0.0
    else:
        return final_score * 0.5, final_score * 0.5

def score_dropdown(elem: dict, elements: List[dict], triangles: List[dict], all_elements: List[dict]) -> float:
    """
    Zwraca dropdown confidence score (float).
    Debug info zapisywane są w globalnym ELEMENT_DEBUG oraz opcjonalnie w konsoli.
    """
    text = elem.get("text", "")
    bbox = elem["bbox"]

    debug = {
        "text": text,
        "signals": {},
        "bonuses": {},
        "exclusions": {},
        "final_calc": {}
    }

    scores = []

    # MOCNY SYGNAŁ STRUKTURALNY
    if elem.get("kind") == "dropdown":
        scores.append(0.77)
        debug["signals"]["kind_dropdown"] = 0.77
    has_dropdown_box = elem.get("has_dropdown_box", False) or ("dropdown_box" in elem)
    if has_dropdown_box:
        scores.append(0.77)
        debug["signals"]["has_dropdown_box"] = 0.77
    if bool(elem.get("has_frame")) and int(elem.get("frame_hits", 0)) >= 3:
        scores.append(0.25)
        debug["signals"]["ramka_listy"] = 0.25

    # Tekstowe wzorce (silne/umiarkowane/słabe)
    text_normalized = lower_strip_acc(text)
    if text.strip():
        if DROPDOWN_STRONG_RE.search(text_normalized):
            scores.append(0.30)
            debug["signals"]["tekst_strong"] = 0.30
        elif DROPDOWN_MODERATE_RE.search(text_normalized):
            scores.append(0.15)
            debug["bonuses"]["tekst_moderate"] = 0.15
        elif DROPDOWN_WEAK_RE.search(text_normalized):
            scores.append(0.07)
            debug["bonuses"]["tekst_weak"] = 0.07

    # Strzałki w tekście
    if has_arrow_symbols(text) > 0:
        scores.append(0.08)
        debug["bonuses"]["arrow_in_text"] = 0.08

    triangle_hits = triangles_hit_bbox(bbox, triangles)
    if triangle_hits:
        triangle_bonus = min(0.45, 0.28 + 0.08 * (triangle_hits - 1))
        scores.append(triangle_bonus)
        debug["signals"]["triangle_icon"] = triangle_bonus
        debug["signals"]["triangle_hits"] = triangle_hits

    # Układ: kolumna elementów poniżej z dużym pokryciem poziomym
    try:
        bx = bbox
        below = [e for e in all_elements if (e is not elem) and e.get("bbox") and is_below(bx, e["bbox"]) ]
        def horiz_overlap(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ov = max(0, min(ax2, bx2) - max(ax1, bx1))
            return ov / max(1, min(ax2-ax1, bx2-bx1))
        column = [e for e in below if horiz_overlap(bx, e["bbox"]) >= 0.5]
        if len(column) >= 3:
            scores.append(0.25)
            debug["signals"]["pionowa_kolumna_opcji"] = 0.25
        if len(column) >= 5:
            scores.append(0.10)
            debug["bonuses"]["dluga_lista"] = 0.10
    except Exception:
        pass

    # Wykluczenia
    exclusions = []
    text_len = len(text)

    # Puste pola bez OCR – bardzo ostro tniemy score, żeby nie
    # klasyfikować samych ramek / tła jako dropdownów.
    if not text.strip():
        exclusions.append(0.90)
        debug["exclusions"]["no_text"] = -0.90

    if text_len > 100:
        exclusions.append(0.60)
        debug["exclusions"]["long_text"] = -0.60
    elif text_len > 60:
        exclusions.append(0.30)
        debug["exclusions"]["medium_long_text"] = -0.30
    if line_count(text) >= 3:
        exclusions.append(0.50)
        debug["exclusions"]["multiline"] = -0.50
    if re.search(r"\b(zatwierd[zż]|submit|anuluj|cancel|dalej|next|wstecz|back)\b", text_normalized):
        exclusions.append(0.50)
        debug["exclusions"]["action_button"] = -0.50

    # Final
    base_score = combine_scores([s for s in scores if s > 0])
    total_exclusions = sum(exclusions)
    debug["final_calc"]["combined_scores"] = base_score
    debug["final_calc"]["total_exclusions"] = -total_exclusions
    final_score = base_score - total_exclusions
    debug["final_calc"]["before_cap"] = final_score
    final_score = max(0.0, min(0.95, final_score))
    debug["final_calc"]["final"] = final_score

    if DEBUG_MODE:
        DEBUG_RESULTS.append({"text": text[:80], "score": final_score, "signals": debug["signals"], "exclusions": debug["exclusions"], "final": debug["final_calc"]})
    try:
        elid = elem.get("id") or "?"
        if elid not in ELEMENT_DEBUG:
            ELEMENT_DEBUG[elid] = {}
        ELEMENT_DEBUG[elid]["dropdown"] = debug
    except Exception:
        pass

    return final_score  # <-- TYLKO FLOAT!

def score_answer_single(elem: dict, elements: List[dict]) -> float:
    """Zwraca answer_single confidence score"""
    text = elem.get("text", "")
    bbox = elem["bbox"]
    
    scores = []
    
    # === SYGNAĹY BAZOWE - PYTANIE ===
    
    # Znak zapytania
    if "?" in text:
        scores.append(0.25)
    
    # SĹ‚owa pytajÄ…ce
    text_normalized = lower_strip_acc(text)
    if QUESTION_RE.search(text_normalized):
        # SprawdĹş czy forma pojedyncza (ktĂłry, ktĂłra) a nie mnoga (ktĂłre)
        if re.search(r"\bkt[oĂł]r(y|a)\b", text_normalized):
            scores.append(0.30)
        else:
            scores.append(0.20)
    
    # === SYGNAĹY SINGLE-CHOICE ===
    
    # Wzorce strict single
    if SINGLE_STRICT_RE.search(text_normalized):
        scores.append(0.55)
    
    # === SYGNAĹY KONTEKSTOWE - ELEMENTY PONIĹ»EJ ===
    
    below_elements = _answer_candidates_below(elem, elements)
    answer_count = len(below_elements)
    
    # Liczba elementĂłw
    if answer_count == 2:
        scores.append(0.35)
    elif answer_count == 3:
        scores.append(0.40)
    elif answer_count == 4:
        scores.append(0.40)
    elif answer_count in [5, 6]:
        scores.append(0.30)
    elif 7 <= answer_count <= 10:
        scores.append(0.15)
    elif answer_count >= 11:
        scores.append(-0.20)
    elif answer_count == 1:
        scores.append(0.10)
    
    # Radio buttons w elementach
    radio_count = 0
    checkbox_count = 0
    for e in below_elements:
        has_marker, marker_type = has_list_marker_start(e.get("text", ""))
        if marker_type == 'radio':
            radio_count += 1
        elif marker_type == 'checkbox':
            checkbox_count += 1
    
    if radio_count >= 2:
        scores.append(0.70)
        if radio_count == answer_count and answer_count >= 2:
            scores.append(0.15)  # Wszystkie majÄ… radio - bonus
    
    # Numeracja/literowanie
    numbered_count = 0
    for e in below_elements:
        e_text = e.get("text", "").lstrip()
        if re.match(r"^[a-d]\)|^[1-4]\.", e_text):
            numbered_count += 1
    
    if numbered_count >= 2:
        scores.append(0.25)
    
    # === WYKLUCZENIA ===
    exclusions = []
    
    # Wzorce multi
    if MULTI_STRICT_RE.search(text_normalized):
        exclusions.append(0.75)
    
    # Checkboxy
    if checkbox_count >= 2:
        exclusions.append(0.60)
    
    # Brak kontekstu
    if answer_count == 0:
        exclusions.append(0.50)
    if not is_questionish(text):
        exclusions.append(0.40)
    
    # === FINALNY SCORE ===
    base = combine_scores([s for s in scores if s > 0])
    penalties = sum(exclusions) + sum([s for s in scores if s < 0])
    final = base - penalties
    
    # JeĹ›li brak sygnaĹ‚Ăłw pytania i odpowiedzi, zwrĂłÄ‡ 0
    if not is_questionish(text) and answer_count == 0:
        return 0.0
    
    return max(0.0, min(0.95, final))

def score_answer_multi(elem: dict, elements: List[dict]) -> float:
    """Zwraca answer_multi confidence score"""
    text = elem.get("text", "")
    bbox = elem["bbox"]
    
    scores = []
    
    # === SYGNAĹY BAZOWE - PYTANIE ===
    
    # Znak zapytania
    if "?" in text:
        scores.append(0.25)
    
    # SĹ‚owa pytajÄ…ce (forma mnoga!)
    text_normalized = lower_strip_acc(text)
    if re.search(r"\bkt[oĂł]re\b", text_normalized):
        scores.append(0.35)
    elif QUESTION_RE.search(text_normalized):
        scores.append(0.20)
    
    # === SYGNAĹY MULTI-CHOICE ===
    
    # Wzorce strict multi
    if MULTI_STRICT_RE.search(text_normalized):
        scores.append(0.65)
    
    # Instrukcje w nawiasach
    if MULTI_PARTIAL_RE.search(text):
        scores.append(0.40)
    
    # === SYGNAĹY KONTEKSTOWE - ELEMENTY PONIĹ»EJ ===
    
    below_elements = _answer_candidates_below(elem, elements)
    answer_count = len(below_elements)
    
    # Liczba elementĂłw (wiÄ™cej = lepiej dla multi)
    if answer_count == 2:
        scores.append(-0.25)
    elif answer_count == 3:
        scores.append(0.30)
    elif answer_count in [4, 5]:
        scores.append(0.45)
    elif 6 <= answer_count <= 10:
        scores.append(0.55)
    elif answer_count >= 11:
        scores.append(0.60)
    
    # Checkboxy w elementach
    checkbox_count = 0
    radio_count = 0
    bullet_count = 0
    for e in below_elements:
        has_marker, marker_type = has_list_marker_start(e.get("text", ""))
        if marker_type == 'checkbox':
            checkbox_count += 1
        elif marker_type == 'radio':
            radio_count += 1
        elif marker_type == 'bullet':
            bullet_count += 1
    
    if checkbox_count >= 2:
        scores.append(0.75)
        if checkbox_count >= 3:
            scores.append(0.10)
        if checkbox_count == answer_count and answer_count >= 2:
            scores.append(0.15)
    
    # Bullets
    if bullet_count >= 2:
        scores.append(0.20)
    
    # === WYKLUCZENIA ===
    exclusions = []
    
    # Wzorce single
    if SINGLE_STRICT_RE.search(text_normalized):
        exclusions.append(0.75)
    
    # Radio buttons
    if radio_count >= 2:
        exclusions.append(0.65)
    
    # Brak kontekstu
    if answer_count <= 1:
        exclusions.append(0.60)
    if not is_questionish(text):
        exclusions.append(0.40)
    
    # === FINALNY SCORE ===
    base = combine_scores([s for s in scores if s > 0])
    penalties = sum(exclusions) + sum([s for s in scores if s < 0])
    final = base - penalties
    
    # JeĹ›li brak sygnaĹ‚Ăłw pytania i odpowiedzi, zwrĂłÄ‡ 0
    if not is_questionish(text) and answer_count <= 1:
        return 0.0
    
    return max(0.0, min(0.95, final))

def find_cookie_context_nearby(elem: dict, all_elements: List[dict], 
                               max_distance: float = 150) -> bool:
    """Sprawdza czy w pobliĹĽu jest kontekst cookies"""
    bbox = elem["bbox"]
    text = elem.get("text", "")
    
    # SprawdĹş sam element
    if COOKIE_CONTEXT_RE.search(lower_strip_acc(text)):
        return True
    
    # SprawdĹş okoliczne elementy
    for e in all_elements:
        if e == elem:
            continue
        dist = bbox_distance(bbox, e["bbox"])
        if dist < max_distance:
            e_text = e.get("text", "")
            if COOKIE_CONTEXT_RE.search(lower_strip_acc(e_text)):
                return True
    
    return False

def score_cookie_accept(elem: dict, all_elements: List[dict], 
                       buttons: List[dict]) -> float:
    """Zwraca cookie_accept confidence score"""
    text = elem.get("text", "")
    bbox = elem["bbox"]
    
    # WARUNEK WSTÄPNY
    if not find_cookie_context_nearby(elem, all_elements):
        return 0.0
    
    scores = []
    text_normalized = lower_strip_acc(text)
    
    # === SĹOWA AKCEPTACJI ===
    if COOKIE_ACCEPT_RE.search(text_normalized):
        scores.append(0.65)
    
    # Ikony pozytywne
    positive_icons = "âś“âś”âś…đź‘Ť"
    if has_unicode_symbol(text, positive_icons):
        scores.append(0.25)
    
    # === KONTEKST ===
    
    # Przycisk
    if elem.get("is_button"):
        scores.append(0.35)
    
    # KrĂłtki tekst
    if len(text) < 30:
        scores.append(0.20)
    
    # Para z reject
    reject_nearby = False
    for e in all_elements:
        if e == elem:
            continue
        dist = bbox_distance(bbox, e["bbox"])
        if dist < 80:
            e_text = lower_strip_acc(e.get("text", ""))
            if COOKIE_REJECT_RE.search(e_text) or COOKIE_PARTIAL_RE.search(e_text):
                reject_nearby = True
                break
    if reject_nearby:
        scores.append(0.30)
    
    # Banner cookies w pobliĹĽu
    large_cookie_banner = False
    for e in all_elements:
        if e == elem:
            continue
        dist = bbox_distance(bbox, e["bbox"])
        if dist < 120:
            e_text = e.get("text", "")
            e_width = bbox_width(e["bbox"])
            if e_width > 200 and COOKIE_CONTEXT_RE.search(lower_strip_acc(e_text)):
                large_cookie_banner = True
                break
    if large_cookie_banner:
        scores.append(0.35)
    
    # === WYKLUCZENIA ===
    exclusions = []
    
    # SĹ‚owa negatywne
    if re.search(r"\b(nie|not|reject|decline|deny|odm[oĂł]w)\b", text_normalized):
        exclusions.append(0.85)
    
    # CzÄ™Ĺ›ciowa akceptacja
    if COOKIE_PARTIAL_RE.search(text_normalized):
        exclusions.append(0.60)
    
    # DĹ‚ugi tekst
    if len(text) > 100:
        exclusions.append(0.50)
    
    # === FINALNY SCORE ===
    base = combine_scores(scores)
    penalties = sum(exclusions)
    final = base - penalties
    
    return max(0.0, min(0.95, final))

def score_cookie_reject(elem: dict, all_elements: List[dict], 
                       buttons: List[dict]) -> float:
    """Zwraca cookie_reject confidence score"""
    text = elem.get("text", "")
    bbox = elem["bbox"]
    
    # WARUNEK WSTÄPNY
    if not find_cookie_context_nearby(elem, all_elements):
        return 0.0
    
    scores = []
    text_normalized = lower_strip_acc(text)
    
    # === SĹOWA ODRZUCENIA ===
    if COOKIE_REJECT_RE.search(text_normalized):
        scores.append(0.65)
    
    # CzÄ™Ĺ›ciowa akceptacja (zarzÄ…dzaj, tylko niezbÄ™dne)
    if COOKIE_PARTIAL_RE.search(text_normalized):
        scores.append(0.55)
    
    # Ikony negatywne
    negative_icons = "âś—âśâťŚđź‘ŽĂ—"
    if has_unicode_symbol(text, negative_icons):
        scores.append(0.25)
    
    # === KONTEKST ===
    
    # Przycisk
    if elem.get("is_button"):
        scores.append(0.35)
    
    # KrĂłtki tekst
    if len(text) < 40:
        scores.append(0.20)
    
    # Para z accept
    accept_nearby = False
    for e in all_elements:
        if e == elem:
            continue
        dist = bbox_distance(bbox, e["bbox"])
        if dist < 80:
            e_text = lower_strip_acc(e.get("text", ""))
            if COOKIE_ACCEPT_RE.search(e_text):
                accept_nearby = True
                break
    if accept_nearby:
        scores.append(0.30)
    
    # Banner
    large_cookie_banner = False
    for e in all_elements:
        if e == elem:
            continue
        dist = bbox_distance(bbox, e["bbox"])
        if dist < 120:
            e_text = e.get("text", "")
            e_width = bbox_width(e["bbox"])
            if e_width > 200 and COOKIE_CONTEXT_RE.search(lower_strip_acc(e_text)):
                large_cookie_banner = True
                break
    if large_cookie_banner:
        scores.append(0.35)
    
    # === WYKLUCZENIA ===
    exclusions = []
    
    # SĹ‚owa akceptacji
    if COOKIE_ACCEPT_RE.search(text_normalized):
        exclusions.append(0.85)
    
    # "Wszystko" bez kontekstu odrzucenia
    if re.search(r"\b(all|wszystkie)\b", text_normalized):
        if not COOKIE_REJECT_RE.search(text_normalized):
            exclusions.append(0.60)
    
    # === FINALNY SCORE ===
    base = combine_scores(scores)
    penalties = sum(exclusions)
    final = base - penalties
    
    return max(0.0, min(0.95, final))

def resolve_answer_conflict(single_score: float, multi_score: float) -> Tuple[float, float]:
    """RozwiÄ…zuje konflikt gdy oba scores sÄ… wysokie"""
    threshold = 0.40
    
    if single_score > threshold and multi_score > threshold:
        # Konflikt - weĹş wyĹĽszy i drastycznie obniĹĽ oba
        if single_score > multi_score:
            return single_score * 0.30, 0.0
        else:
            return 0.0, multi_score * 0.30
    
    return single_score, multi_score

    out_path = derive_out_path(in_path, result.get("image"))

    debug_out = derive_debug_out_path(in_path, result.get("image"))

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        try:
            debug_payload = {
                "image": result.get("image"),
                "total_elements": result.get("total_elements"),
                "elements": [
                    {
                        "id": el.get("id"),
                        "text": el.get("box"),
                        "bbox": el.get("bbox"),
                        "debug": ELEMENT_DEBUG.get(el.get("id"), {})
                    }
                    for el in result.get("elements", [])
                ]
            }
            ensure_dir(Path(os.path.dirname(debug_out)))
            with open(debug_out, "w", encoding="utf-8") as df:
                json.dump(debug_payload, df, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Nie mogę zapisać debug JSON: {debug_out} :: {e}")
        print(f"[OK] {out_path}")
        print(f"[INFO] Debug zapisany: {debug_out}")
        return out_path
    except Exception as e:
        print(f"[ERROR] Nie mogłem zapisać wyniku: {out_path} :: {e}")
        return None# ======================== MAIN EVALUATOR ========================

def evaluate(data: dict) -> dict:
    """
    GĹ‚Ăłwna funkcja oceniajÄ…ca - ocenia kaĹĽdy element osobno.
    """
    # Wykryj zagnieĹĽdĹĽonÄ… strukturÄ™
    if "detection" in data and isinstance(data["detection"], dict):
        data = data["detection"]
    
    # Wczytaj elementy
    elements = []
    image = data.get("image")
    background_layout = data.get("background_layout") or {}
    
    # Pobierz wymiary obrazu (jeĹ›li sÄ…)
    image_width = 1920  # default
    image_height = 1080  # default
    try:
        iw = data.get("image_width") or data.get("img_w") or data.get("width")
        ih = data.get("image_height") or data.get("img_h") or data.get("height")
        if isinstance(iw, (int, float)) and isinstance(ih, (int, float)) and iw > 0 and ih > 0:
            image_width = int(iw); image_height = int(ih)
    except Exception:
        pass
    # TODO: jeĹ›li masz info o wymiarach w data, uĹĽyj go
    
    source_list = (data.get("results") or data.get("candidates") or data.get("elements") or data.get("boxes") or data.get("detections") or data.get("items") or [])
    
    pil_img = None
    try:
        if Image is not None and isinstance(image, str) and os.path.isfile(image):
            pil_img = Image.open(image).convert('RGB')
    except Exception:
        pil_img = None

    # OCR availability notice
    if pil_img is not None:
        if pytesseract is None:
            print('[WARN] OCR disabled: install pytesseract and Tesseract OCR to extract text')
        else:
            try:
                _ = pytesseract.get_tesseract_version()
            except Exception as _e:
                print(f'[WARN] Tesseract not available: {_e}')

    for i, c in enumerate(source_list):
        bbox = (
            (c.get("text_box") if isinstance(c, dict) else None)
            or (c.get("dropdown_box") if isinstance(c, dict) else None)
            or (c.get("bbox") if isinstance(c, dict) else None)
            or (c.get("box") if isinstance(c, dict) else None)
            or (c.get("rect") if isinstance(c, dict) else None)
            or (c.get("xyxy") if isinstance(c, dict) else None)
            or (c if isinstance(c, (list, tuple)) and len(c) == 4 else None)
        )
        def _normalize_bbox(b, cdict):
            try:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    x1, y1, x2, y2 = [float(x) for x in b]
                    if x2 <= x1 or y2 <= y1:
                        x2 = x1 + x2
                        y2 = y1 + y2
                    return [int(x1), int(y1), int(x2), int(y2)]
                if isinstance(b, dict):
                    if all(k in b for k in ("left", "top", "right", "bottom")):
                        return [int(b["left"]), int(b["top"]), int(b["right"]), int(b["bottom"])]
                    if (all(k in b for k in ("x", "y", "w", "h")) or all(k in b for k in ("x", "y", "width", "height"))):
                        x = b.get("x"); y = b.get("y")
                        w = b.get("w", b.get("width")); h = b.get("h", b.get("height"))
                        return [int(x), int(y), int(x) + int(w), int(y) + int(h)]
                if isinstance(cdict, dict):
                    if all(k in cdict for k in ("left", "top", "right", "bottom")):
                        return [int(cdict["left"]), int(cdict["top"]), int(cdict["right"]), int(cdict["bottom"])]
                    if (all(k in cdict for k in ("x", "y", "w", "h")) or all(k in cdict for k in ("x", "y", "width", "height"))):
                        x = cdict.get("x"); y = cdict.get("y")
                        w = cdict.get("w", cdict.get("width")); h = cdict.get("h", cdict.get("height"))
                        return [int(x), int(y), int(x) + int(w), int(y) + int(h)]
            except Exception:
                return None
            return None
        bbox = _normalize_bbox(bbox, c if isinstance(c, dict) else {})
        if not bbox:
            continue
        txt = ""
        if isinstance(c, dict):
            for key in ["seed_text", "text", "label", "ocr_text", "name", "title", "caption", "value", "placeholder"]:
                val = c.get(key)
                if isinstance(val, str) and val.strip():
                    txt = val
                    break
        txt = norm_text(txt)
        if (not txt) and pil_img is not None and (pytesseract is not None) and (Image is not None):
            pref_raw = (c.get('text_box') if isinstance(c, dict) else None) or (c.get('dropdown_box') if isinstance(c, dict) else None) or bbox
            try:
                pref_bbox = _normalize_bbox(pref_raw, c if isinstance(c, dict) else {})
            except Exception:
                pref_bbox = bbox
            if pref_bbox:
                ocr_txt = _ocr_text_from_image(pil_img, pref_bbox)
                if ocr_txt:
                    txt = ocr_txt
        elements.append({
            "id": (c.get("id") if isinstance(c, dict) else None) or f"cand_{i}",
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            "text": txt,
            "kind": (c.get("kind") if isinstance(c, dict) else None),
            "conf": (c.get("conf", 1.0) if isinstance(c, dict) else 1.0),
            "is_button": bool(c.get("is_button", False)) if isinstance(c, dict) else False,
            "has_triangle": bool(c.get("has_triangle", False)) if isinstance(c, dict) else False,
            "has_dropdown_box": bool(c.get("dropdown_box") is not None) if isinstance(c, dict) else False,
            "has_frame": bool(c.get("has_frame", False)) if isinstance(c, dict) else False,
            "frame_hits": int(c.get("frame_hits", 0)) if isinstance(c, dict) else 0,
            # Przekazanie informacji o tle z region_grow (jeśli istnieje)
            "bg_cluster_id": (int(c.get("bg_cluster_id")) if isinstance(c, dict) and c.get("bg_cluster_id") is not None else None),
            "bg_is_main_like": bool(c.get("bg_is_main_like")) if isinstance(c, dict) and ("bg_is_main_like" in c) else None,
            "bg_mean_rgb": (c.get("bg_mean_rgb") if isinstance(c, dict) else None),
            "bg_dist_to_global": (
                float(c.get("bg_dist_to_global", 0.0))
                if isinstance(c, dict) and ("bg_dist_to_global" in c)
                else None
            ),
        })
    buttons = data.get("buttons", []) or []
    
    raw_triangles = data.get("triangles", []) or []
    triangles = normalize_triangles(raw_triangles)
    print(f"[INFO] Oceniam {len(elements)} elementów (triangles={len(triangles)})")
    
    # WyczyĹ›Ä‡ debug results przed ocenÄ…
    global DEBUG_RESULTS
    DEBUG_RESULTS = []
    
    # OceĹ„ kaĹĽdy element
    results = []
    
    for elem in elements:
        elem_id = elem["id"]
        text = elem["text"]
        bbox = elem["bbox"]
        
        # === OBLICZ SCORES DLA WSZYSTKICH KATEGORII ===
        
        # NEXT
        next_inactive, next_active = score_next(elem, elements, buttons, 
                                                image_width, image_height)
        
        # DROPDOWN
        dropdown_result = score_dropdown(elem, elements, triangles, elements)
        # ObsĹ‚uĹĽ zarĂłwno float jak i tuple (backward compatibility)
        if isinstance(dropdown_result, tuple):
            dropdown = dropdown_result[0]
        else:
            dropdown = dropdown_result
        
        # ANSWERS
        single = score_answer_single(elem, elements)
        multi = score_answer_multi(elem, elements)
        
        # RozwiÄ…ĹĽ konflikty
        single, multi = resolve_answer_conflict(single, multi)
        
        # COOKIES
        cookie_accept = score_cookie_accept(elem, elements, buttons)
        cookie_reject = score_cookie_reject(elem, elements, buttons)
        
        # === ZĹĂ“Ĺ» WYNIK ===
        result = {
            "id": elem_id,
            "box": text,
            "text": text,
            "bbox": bbox,
            "scores": {
                "next_inactive": round(float(next_inactive), 4),
                "next_active": round(float(next_active), 4),
                "dropdown": round(float(dropdown), 4),
                "answer_single": round(float(single), 4),
                "answer_multi": round(float(multi), 4),
                "cookie_accept": round(float(cookie_accept), 4),
                "cookie_reject": round(float(cookie_reject), 4),
            },
            "predictions": {
                "next_inactive": next_inactive > THRESHOLDS["next_inactive"],
                "next_active": next_active > THRESHOLDS["next_active"],
                "dropdown": dropdown > THRESHOLDS["dropdown"],
                "answer_single": single > THRESHOLDS["answer_single"],
                "answer_multi": multi > THRESHOLDS["answer_multi"],
                "cookie_accept": cookie_accept > THRESHOLDS["cookie_accept"],
                "cookie_reject": cookie_reject > THRESHOLDS["cookie_reject"],
            }
        }
        
        results.append(result)
    
    # === WYPISZ DEBUG INFO ===
    if DEBUG_MODE and DEBUG_RESULTS:
        print("\n" + "="*70)
        print("DROPDOWN DEBUG - Top 10 highest scores:")
        print("="*70)
        for i, item in enumerate(sorted(DEBUG_RESULTS, key=lambda x: x["score"], reverse=True)[:10], 1):
            print(f"\n#{i} Score: {item['score']:.3f}")
            print(f"Text: '{item['text']}'")
            if item['signals']:
                print(f"Signals: {item['signals']}")
            if item['exclusions']:
                print(f"Exclusions: {item['exclusions']}")
            print(f"Final: {item['final']}")
        
        print("\n" + "="*70)
        print("DROPDOWN DEBUG - Elements with score > 0.5:")
        print("="*70)
        high_scores = [d for d in DEBUG_RESULTS if d['score'] > 0.5]
        if high_scores:
            for item in high_scores:
                print(f"\nScore: {item['score']:.3f} | Text: '{item['text']}'")
        else:
            print("(brak)")
        
        DEBUG_RESULTS.clear()
    
    # Sortuj wyniki - najwyĹĽsze confidence na gĂłrze
    for r in results:
        max_score = max(r["scores"].values())
        r["_max_score"] = max_score
    
    results.sort(key=lambda x: x["_max_score"], reverse=True)
    
    # UsuĹ„ pomocnicze pole
    for r in results:
        del r["_max_score"]
    
    # ZwrĂłÄ‡ wynik
    output = {
        "image": image,
        "total_elements": len(results),
        "elements": results,
        "thresholds": THRESHOLDS,
        "background_layout": background_layout,
        "summary": {
            "next_detected": sum(1 for r in results if r["predictions"]["next_active"] or r["predictions"]["next_inactive"]),
            "dropdown_detected": sum(1 for r in results if r["predictions"]["dropdown"]),
            "question_detected": sum(1 for r in results if r["predictions"]["answer_single"] or r["predictions"]["answer_multi"]),
            "cookies_detected": sum(1 for r in results if r["predictions"]["cookie_accept"] or r["predictions"]["cookie_reject"]),
        }
    }
    
    return output

# ======================== I/O FUNCTIONS =========================

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_jsons_in_dir(d: str) -> List[str]:
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, fn) for fn in os.listdir(d)
            if fn.lower().endswith(".json") and os.path.isfile(os.path.join(d, fn))]

def derive_out_path(in_json_path: str, image_path: Optional[str]) -> str:
    ensure_dir(RATE_RESULTS_DIR)
    if image_path:
        base = os.path.splitext(os.path.basename(image_path))[0]
    else:
        base = os.path.splitext(os.path.basename(in_json_path))[0]
    return os.path.join(RATE_RESULTS_DIR, f"{base}_rated.json")


def derive_debug_out_path(in_json_path: str, image_path: Optional[str]) -> str:
    ensure_dir(RATE_RESULTS_DEBUG_DIR)
    if image_path:
        base = os.path.splitext(os.path.basename(image_path))[0]
    else:
        base = os.path.splitext(os.path.basename(in_json_path))[0]
    return os.path.join(RATE_RESULTS_DEBUG_DIR, f"{base}_rated_debug.json")


def process_file(in_path: str) -> Optional[str]:
    try:
        data = load_json(in_path)
    except Exception as e:
        print(f"[ERROR] Nie mogłem wczytać JSON: {in_path} :: {e}")
        return None

    # Jeśli w pliku screen_boxes nie ma trójkątów, spróbuj dociągnąć je
    # z osobnego pliku wygenerowanego przez triangle_finder (suffix _triangle.json).
    try:
        if not data.get("triangles") and data.get("image"):
            img_path = Path(data["image"])
            candidates = [
                img_path.with_name(f"{img_path.stem}_triangle.json"),
                DATA_SCREEN_DIR / "numpy_triangles" / f"{img_path.stem}_triangle.json",
            ]
            for tri_json in candidates:
                if tri_json.is_file():
                    with open(tri_json, "r", encoding="utf-8") as tf:
                        tri_data = json.load(tf)
                    tri_list = tri_data.get("triangles") or []
                    if tri_list:
                        data["triangles"] = tri_list
                    break
    except Exception:
        pass

    try:
        result = evaluate(data)
    except Exception as e:
        print(f"[ERROR] Błąd ewaluacji dla {in_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

    out_path = derive_out_path(in_path, result.get("image"))
    debug_out = derive_debug_out_path(in_path, result.get("image"))

    summary_out = derive_summary_out_path(in_path, result.get("image"))
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        try:
            debug_payload = {
                "image": result.get("image"),
                "total_elements": result.get("total_elements"),
                "elements": [
                    {
                        "id": el.get("id"),
                        "text": el.get("box"),
                        "bbox": el.get("bbox"),
                        "debug": ELEMENT_DEBUG.get(el.get("id"), {})
                    }
                    for el in result.get("elements", [])
                ]
            }
            ensure_dir(Path(os.path.dirname(debug_out)))
            with open(debug_out, "w", encoding="utf-8") as df:
                json.dump(debug_payload, df, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Nie mogę zapisać debug JSON: {debug_out} :: {e}")

        try:
            summary_payload = build_summary_payload(result)
            with open(summary_out, "w", encoding="utf-8") as sf:
                json.dump(summary_payload, sf, ensure_ascii=False, indent=2)
            print(f"[INFO] Summary zapisany: {summary_out}")
        except Exception as e:
            print(f"[WARN] Nie mogę zapisać summary JSON: {summary_out} :: {e}")

        print(f"[OK] {out_path}")
        print(f"[INFO] Debug zapisany: {debug_out}")
        return out_path
    except Exception as e:
        print(f"[ERROR] Nie mogłem zapisać wyniku: {out_path} :: {e}")
        return None


def main():
    target_dir = INPUT_DIR
    single_file: Optional[str] = None

    if len(sys.argv) >= 2:
        in_path = sys.argv[1]
        if os.path.isdir(in_path):
            target_dir = in_path
        elif os.path.isfile(in_path):
            single_file = in_path
        else:
            print(f"[ERROR] Brak pliku lub katalogu: {in_path}")
            sys.exit(2)

    if single_file:
        ok = process_file(single_file)
        sys.exit(0 if ok else 2)

    files = list_jsons_in_dir(target_dir)
    if not files:
        print(f"[WARN] Brak plików *.json w: {target_dir}")
        sys.exit(0)

    ok_count = 0
    for p in sorted(files):
        if process_file(p):
            ok_count += 1

    print(f"\n[INFO] Przetworzono {ok_count}/{len(files)} plików")
    print(f"[INFO] Wyniki w: {RATE_RESULTS_DIR}")
def derive_summary_out_path(in_json_path: str, image_path: Optional[str]) -> str:
    ensure_dir(RATE_SUMMARY_DIR)
    if image_path:
        base = os.path.splitext(os.path.basename(image_path))[0]
    else:
        base = os.path.splitext(os.path.basename(in_json_path))[0]
    return os.path.join(RATE_SUMMARY_DIR, f"{base}_summary.json")


def build_summary_payload(result: dict) -> dict:
    elements = result.get("elements", [])
    summary = {
        "image": result.get("image"),
        "total_elements": result.get("total_elements"),
        "background_layout": result.get("background_layout"),
        "top_labels": {},
    }

    scroll_rects = _load_scrollable_regions_from_dom()

    def _pick_best(label: str) -> Optional[dict]:
        best = None
        best_score = -1.0
        for el in elements:
            pred = el.get("predictions", {})
            scores = el.get("scores", {})
            if pred.get(label):
                sc = float(scores.get(label, 0.0))
                if sc > best_score:
                    best_score = sc
                    best = {
                        "id": el.get("id"),
                        "text": el.get("box"),
                        "bbox": el.get("bbox"),
                        "score": round(sc, 4),
                        "label": label,
                    }
                    # Przekaż dalej meta-dane o tle (jeśli są dostępne)
                    if "bg_cluster_id" in el:
                        try:
                            cid = el.get("bg_cluster_id")
                            best["bg_cluster_id"] = int(cid) if cid is not None else None
                        except Exception:
                            best["bg_cluster_id"] = None
                    if "bg_is_main_like" in el:
                        try:
                            best["bg_is_main_like"] = bool(el.get("bg_is_main_like"))
                        except Exception:
                            best["bg_is_main_like"] = None
                    if "bg_mean_rgb" in el and isinstance(el.get("bg_mean_rgb"), (list, tuple)):
                        best["bg_mean_rgb"] = list(el.get("bg_mean_rgb"))
                    if "bg_dist_to_global" in el:
                        try:
                            best["bg_dist_to_global"] = float(el.get("bg_dist_to_global"))
                        except Exception:
                            best["bg_dist_to_global"] = None
                    # Annotacja scrollowalności na podstawie DOM
                    if scroll_rects:
                        bbox = el.get("bbox")
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            try:
                                b = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                                cx, cy = bbox_center(b)
                                for r in scroll_rects:
                                    if point_in_bbox((cx, cy), r, margin=3.0) or rect_iou(b, r) > 0.15:
                                        best["scrollable"] = True
                                        break
                            except Exception:
                                pass
        return best

    for lbl in ("dropdown", "next_active", "next_inactive", "answer_single", "answer_multi", "cookie_accept", "cookie_reject"):
        best = _pick_best(lbl)
        if best:
            summary["top_labels"][lbl] = best

    return summary
