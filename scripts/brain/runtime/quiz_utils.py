from __future__ import annotations

import hashlib
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Iterable, List, Optional, Sequence, Tuple


QUESTION_PREFIX_RE = re.compile(
    r"^(?:\(\s*\d+\s*/\s*\d+\s*\)\s*|"
    r"\(\s*mix(?:\s+\d+)?\s*\)\s*|"
    r"(?:pytanie|question)\s+\d+(?:\s*/\s*\d+)?\s*[:|.-]?\s*)",
    re.IGNORECASE,
)
QUESTION_TRAILER_RE = re.compile(
    r"(?:\|\s*(?:pytanie|question)\s+\d+(?:\s*/\s*\d+)?.*$|"
    r"\s+(?:pytanie|question)\s+\d+(?:\s*/\s*\d+)?.*$)",
    re.IGNORECASE,
)
QUESTION_FIELD_TRAILER_RE = re.compile(
    r"\s*(?:\(|\[)?(?:required|optional|wymagane|opcjonalne)(?:\)|\])?\s*$",
    re.IGNORECASE,
)
SPACE_RE = re.compile(r"\s+")
ARITH_RE = re.compile(r"(-?\d+)\s*([+\-*/])\s*(-?\d+)")
PROMPT_VERBS = (
    "wybierz",
    "zaznacz",
    "wpisz",
    "jakiego",
    "jaki",
    "ktore",
    "które",
    "ile",
    "co ",
    "jaka",
    "jakie",
)
NEXT_TOKENS = ("nast", "next", "dalej", "continue", "kontynu", "wyslij", "wyślij", "submit", "finish", "done")
HEADER_TOKENS = (
    "home",
    "test quiz server",
    "jednokrotna odpowied",
    "wielokrotna odpowied",
    "dropdown",
    "radio +",
    "checkbox",
    "input + next",
    "pytanie ",
)
MOJIBAKE_MARKERS = ("Ă", "Ĺ", "â", "™", "ž", "ź", "ł", "ó")


def md5_text(value: str) -> str:
    return hashlib.md5(value.encode("utf-8", errors="ignore")).hexdigest()


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def normalize_space(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "").strip())


def ascii_fold(value: str) -> str:
    if not value:
        return ""
    norm = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def maybe_fix_mojibake(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    candidates = [text]
    for enc in ("latin1", "cp1250", "cp1252"):
        try:
            candidates.append(text.encode(enc, errors="ignore").decode("utf-8", errors="ignore"))
        except Exception:
            continue
    best = text
    best_score = -10
    for cand in candidates:
        if not cand:
            continue
        score = 0
        score -= sum(cand.count(marker) for marker in MOJIBAKE_MARKERS)
        score += sum(1 for ch in cand if ch in "ąćęłńóśżźĄĆĘŁŃÓŚŻŹ")
        score += 1 if "?" in cand else 0
        if score > best_score:
            best = cand
            best_score = score
    return best


def normalize_ocr_text(value: Any) -> str:
    text = maybe_fix_mojibake(value)
    text = normalize_space(text)
    if not text:
        return ""
    replacements = {
        " i ": " | ",
        " l ": " | ",
        " 1 ": " | ",
    }
    for old, new in replacements.items():
        if "pytanie" in text.lower():
            text = text.replace(old, new)
    return normalize_space(text)


def normalize_question_text(value: Any) -> str:
    text = normalize_ocr_text(value)
    text = QUESTION_PREFIX_RE.sub("", text)
    text = QUESTION_TRAILER_RE.sub("", text)
    text = QUESTION_FIELD_TRAILER_RE.sub("", text)
    return normalize_space(text)


def normalize_match_text(value: Any) -> str:
    text = normalize_question_text(value).lower()
    text = ascii_fold(text)
    text = text.replace("|", " ")
    text = text.replace("„", '"').replace("”", '"').replace("’", "'").replace("`", "'")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def clean_option_text(value: Any) -> str:
    text = normalize_ocr_text(value)
    text = re.sub(r"^[A-Z]\s*[).:-]\s*", "", text)
    return normalize_space(text)


def question_like(text: str) -> bool:
    t = normalize_match_text(text)
    if not t:
        return False
    if "?" in str(text or ""):
        return True
    return any(t.startswith(prefix) for prefix in PROMPT_VERBS)


def next_like(text: str) -> bool:
    t = normalize_match_text(text)
    return bool(t) and any(token in t for token in NEXT_TOKENS)


def header_like(text: str) -> bool:
    t = normalize_match_text(text)
    return bool(t) and any(token in t for token in HEADER_TOKENS)


def text_similarity(a: Any, b: Any) -> float:
    aa = normalize_match_text(a)
    bb = normalize_match_text(b)
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 1.0
    return SequenceMatcher(None, aa, bb).ratio()


def box_iou(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / max(1.0, a_area + b_area - inter)


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_height(box: Sequence[float]) -> float:
    return max(0.0, float(box[3]) - float(box[1]))


def box_width(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0]))


def box_union(boxes: Iterable[Sequence[float]]) -> Optional[List[int]]:
    prepared = [list(map(float, box)) for box in boxes if box and len(box) == 4]
    if not prepared:
        return None
    x1 = min(box[0] for box in prepared)
    y1 = min(box[1] for box in prepared)
    x2 = max(box[2] for box in prepared)
    y2 = max(box[3] for box in prepared)
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def signature_for_question(question_text: str, options_text: Sequence[str], question_type: str) -> str:
    norm_q = normalize_match_text(question_text)
    norm_opts = sorted(clean_option_text(opt) and normalize_match_text(opt) for opt in options_text if clean_option_text(opt))
    payload = f"{question_type}\n{norm_q}\n" + "\n".join(norm_opts)
    return sha1_text(payload)


def extract_arithmetic_answer(question_text: str) -> Optional[str]:
    text = normalize_question_text(question_text)
    match = ARITH_RE.search(text)
    if not match:
        return None
    a = int(match.group(1))
    op = match.group(2)
    b = int(match.group(3))
    try:
        if op == "+":
            return str(a + b)
        if op == "-":
            return str(a - b)
        if op == "*":
            return str(a * b)
        if op == "/" and b != 0:
            value = a / b
            if value.is_integer():
                return str(int(value))
            return str(value)
    except Exception:
        return None
    return None


def quoted_answer(question_text: str) -> Optional[str]:
    text = normalize_question_text(question_text)
    for pattern in (r"'([^']+)'", r'"([^"]+)"'):
        match = re.search(pattern, text)
        if match:
            return normalize_space(match.group(1))
    colon_match = re.search(r":\s*([A-Za-z0-9ąćęłńóśżźĄĆĘŁŃÓŚŻŹ\- ]+)$", text)
    if colon_match:
        return normalize_space(colon_match.group(1))
    return None


def best_text_match(target: str, options: Sequence[str]) -> Tuple[Optional[str], float, int]:
    best_idx = -1
    best_text = None
    best_score = 0.0
    norm_target = normalize_match_text(target)
    for idx, option in enumerate(options):
        score = text_similarity(norm_target, option)
        if score > best_score:
            best_idx = idx
            best_text = option
            best_score = score
    return best_text, best_score, best_idx


def normalized_options_texts(options_map: Any) -> List[str]:
    if isinstance(options_map, dict):
        return [normalize_space(str(v or "")) for _, v in sorted(options_map.items()) if normalize_space(v)]
    if isinstance(options_map, (list, tuple)):
        return [normalize_space(str(v or "")) for v in options_map if normalize_space(v)]
    return []
