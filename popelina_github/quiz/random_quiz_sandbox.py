from __future__ import annotations

import argparse
import json
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]

GLOBAL_TYPES = [
    "single",
    "multi",
    "dropdown",
    "dropdown_scroll",
    "slider",
    "text",
    "triple",
    "mixed",
]

_TYPE_WEIGHTS = {
    "single": 22,
    "multi": 18,
    "dropdown": 16,
    "dropdown_scroll": 10,
    "slider": 12,
    "text": 12,
    "triple": 11,
    "mixed": 9,
}

_PALETTES: List[Dict[str, str]] = [
    {
        "bg_a": "#0b1020",
        "bg_b": "#0f1c30",
        "panel": "#10172a",
        "panel_border": "#253354",
        "text": "#e6edf6",
        "muted": "#9ab0c9",
        "accent": "#3b82f6",
        "accent_text": "#ffffff",
    },
    {
        "bg_a": "#f6f9ff",
        "bg_b": "#edf3ff",
        "panel": "#ffffff",
        "panel_border": "#c8d9ff",
        "text": "#13233f",
        "muted": "#4e6383",
        "accent": "#1e40af",
        "accent_text": "#ffffff",
    },
    {
        "bg_a": "#0b1610",
        "bg_b": "#14291f",
        "panel": "#112319",
        "panel_border": "#2a5641",
        "text": "#dff8eb",
        "muted": "#99c9b0",
        "accent": "#10b981",
        "accent_text": "#072316",
    },
    {
        "bg_a": "#1b1024",
        "bg_b": "#271537",
        "panel": "#211431",
        "panel_border": "#4f3474",
        "text": "#f3eafd",
        "muted": "#cab3e6",
        "accent": "#ec4899",
        "accent_text": "#ffffff",
    },
]

_FONTS = [
    "'Segoe UI', Tahoma, sans-serif",
    "'Trebuchet MS', 'Segoe UI', sans-serif",
    "'Verdana', 'Segoe UI', sans-serif",
    "'Tahoma', 'Segoe UI', sans-serif",
    "'Gill Sans', 'Segoe UI', sans-serif",
    "'Lucida Sans Unicode', 'Segoe UI', sans-serif",
]
_HARD_FONTS = [
    "'Times New Roman', Georgia, serif",
    "'Cambria', 'Times New Roman', serif",
    "'Courier New', monospace",
    "'Consolas', 'Courier New', monospace",
    "'Arial Narrow', Arial, sans-serif",
    "'Franklin Gothic Medium', 'Arial Narrow', sans-serif",
]

_COLORS = ["zielony", "czerwony", "niebieski", "zolty", "pomaranczowy", "fioletowy", "bialy", "czarny"]
_ANIMALS = ["kot", "pies", "kon", "krowa", "zebra", "lis", "sowa", "zaba"]
_FRUITS = ["jablko", "banan", "gruszka", "truskawka", "sliwka", "malina", "arbuz"]
_OBJECTS = ["telefon", "krzeslo", "stol", "kubek", "dlugopis", "zeszyt", "plecak"]
_CITY = ["warszawa", "krakow", "gdansk", "wroclaw", "poznan", "lodz", "szczecin"]


def _pick_many(rng: random.Random, pool: Sequence[str], n: int) -> List[str]:
    if n <= 0:
        return []
    if n >= len(pool):
        out = list(pool)
        rng.shuffle(out)
        return out
    return rng.sample(list(pool), n)


def _rand_prompt(rng: random.Random, qtype: str) -> Tuple[str, List[str], List[str], str]:
    """
    Returns: prompt, options, correct_values, hint_tag
    """
    mode = rng.choice(["pl", "en", "mix"])
    if qtype == "text":
        if rng.random() < 0.5:
            a = rng.randint(1, 40)
            b = rng.randint(1, 40)
            ans = str(a + b)
            if mode == "en":
                prompt = f"Type result: {a}+{b}"
            else:
                prompt = f"Wpisz wynik: {a}+{b}"
            return prompt, [], [ans], "math"
        if rng.random() < 0.5:
            word = rng.choice(_CITY + _ANIMALS + _FRUITS)
            prompt = f"Wpisz slowo: {word}" if mode != "en" else f"Type word: {word}"
            return prompt, [], [word], "word"
        code = f"{rng.randint(100, 999)}-{rng.randint(100, 999)}"
        prompt = f"Przepisz kod: {code}" if mode != "en" else f"Copy code: {code}"
        return prompt, [], [code], "code"

    source = rng.choice(
        [
            ("kolor trawy", "zielony", _COLORS),
            ("zwierze ktore szczeka", "pies", _ANIMALS),
            ("owoc zolta skorka", "banan", _FRUITS),
            ("miasto nad morzem", "gdansk", _CITY),
        ]
    )
    topic, correct, pool = source
    opt_n = rng.randint(4, 7)
    options = _pick_many(rng, pool, max(3, min(opt_n, len(pool))))
    if correct not in options:
        options[rng.randrange(len(options))] = correct
    rng.shuffle(options)
    if qtype == "multi":
        extras = _pick_many(rng, [v for v in pool if v != correct], rng.randint(0, 2))
        correct_values = sorted(set([correct] + extras))
        option_set = list(dict.fromkeys(options))
        for value in correct_values:
            if value not in option_set:
                option_set.append(value)
        target_n = max(len(correct_values), max(3, min(opt_n, len(pool))))
        distractors = [v for v in pool if v not in correct_values]
        rng.shuffle(distractors)
        final_options = list(correct_values)
        for value in distractors:
            if len(final_options) >= target_n:
                break
            final_options.append(value)
        rng.shuffle(final_options)
        if mode == "en":
            prompt = f"Select all that apply: {topic}"
        else:
            prompt = f"Zaznacz wszystkie pasujace: {topic}"
        return prompt, final_options, correct_values, "multi"

    if mode == "en":
        prompt = f"Choose one: {topic}"
    elif mode == "mix":
        prompt = f"Wybierz one option: {topic}"
    else:
        prompt = f"Wybierz jedna odpowiedz: {topic}"
    return prompt, options, [correct], "single"


def _weighted_global_type(rng: random.Random) -> str:
    rows = [(k, int(v)) for k, v in _TYPE_WEIGHTS.items()]
    total = sum(w for _, w in rows)
    x = rng.randint(1, total)
    acc = 0
    for name, w in rows:
        acc += w
        if x <= acc:
            return name
    return "single"


def _style_pack(rng: random.Random, difficulty: str = "normal") -> Dict[str, Any]:
    hard = str(difficulty or "normal").strip().lower() in {"hard", "very_hard", "extreme"}
    palette = dict(rng.choice(_PALETTES))
    if hard and rng.random() < 0.42:
        # Intentionally lower contrast / washed UI for harder OCR+layout cues.
        palette["text"] = rng.choice(["#b8c0cf", "#c7c9d1", "#9fa9ba", "#d0d5de"])
        palette["muted"] = rng.choice(["#8f97a6", "#95a0b3", "#7f8b9e"])
        if rng.random() < 0.5:
            palette["panel_border"] = rng.choice(["#3b4559", "#444f63", "#505d75"])
    marker_variant = rng.choice(["native", "custom"])
    marker_side = rng.choice(["left", "right"])
    font_pool = list(_FONTS) + (list(_HARD_FONTS) if hard else [])
    return {
        **palette,
        "font": rng.choice(font_pool),
        "font_weight": rng.choice([400, 500, 600, 700] if hard else [400, 500, 600]),
        "letter_spacing_em": round(rng.uniform(-0.02, 0.07) if hard else rng.uniform(-0.01, 0.03), 3),
        "word_spacing_em": round(rng.uniform(-0.03, 0.12) if hard else rng.uniform(0.0, 0.05), 3),
        "text_transform_mode": rng.choice(["none", "uppercase", "capitalize"] if hard else ["none", "capitalize"]),
        "radius": rng.randint(6, 24),
        "card_radius": rng.randint(10, 28),
        "spacing": rng.randint(6, 22) if hard else rng.randint(8, 22),
        "row_pad_y": rng.randint(6, 18) if hard else rng.randint(8, 18),
        "row_pad_x": rng.randint(8, 24) if hard else rng.randint(10, 24),
        "marker_size": rng.randint(12, 22),
        "marker_variant": marker_variant,
        "marker_side": marker_side,
        "line_height": round(rng.uniform(1.1, 1.75) if hard else rng.uniform(1.25, 1.6), 2),
        "title_size": rng.randint(24, 42),
        "question_size": rng.randint(17, 34) if hard else rng.randint(19, 33),
        "text_size": rng.randint(12, 20) if hard else rng.randint(14, 20),
        "max_width": rng.choice([920, 980, 1040, 1160, 1280, 1440]),
        "shadow": rng.choice(["none", "soft", "hard"]),
        "stripe_opacity": round(rng.uniform(0.0, 0.28) if hard else rng.uniform(0.0, 0.2), 2),
        "btn_radius": rng.randint(8, 20),
        "btn_pad_y": rng.randint(8, 14),
        "btn_pad_x": rng.randint(14, 28),
        "noise_blocks": rng.randint(2, 7) if hard else rng.randint(0, 3),
        "shell_variant": rng.choice(["plain", "split", "stacked", "dense"]),
        "option_layout": rng.choice(["list", "cards", "pills"]),
        "title_badge": rng.random() < 0.42,
        "show_secondary_cta": rng.random() < 0.35,
        "show_progress_line": rng.random() < 0.48,
        "show_top_nav": rng.random() < 0.38,
        "extra_decoy_buttons": rng.randint(1, 4) if hard else rng.randint(0, 1),
        "dropdown_visual_noise": rng.random() < (0.8 if hard else 0.35),
        "mobile_narrow": rng.random() < (0.24 if hard else 0.08),
        "sticky_header": rng.random() < (0.2 if hard else 0.08),
        "sticky_footer": rng.random() < (0.18 if hard else 0.06),
        "modal_overlay": rng.random() < (0.16 if hard else 0.04),
        "floating_help": rng.random() < (0.22 if hard else 0.08),
        "sidebar_noise": rng.random() < (0.24 if hard else 0.08),
        "loading_stub": rng.random() < (0.18 if hard else 0.05),
        "next_variant": rng.choices(
            ["button", "link", "icon", "top_right", "sticky_bar"],
            weights=[48, 14, 12, 10, 16] if hard else [72, 8, 6, 5, 9],
            k=1,
        )[0],
    }


def _viewport_pack(rng: random.Random, style: Dict[str, Any] | None = None) -> Dict[str, int]:
    style = style or {}
    mobile_narrow = bool(style.get("mobile_narrow"))
    if mobile_narrow:
        base_w = rng.choice([360, 375, 390, 412, 430, 480])
        base_h = rng.choice([700, 740, 780, 820, 860, 920])
        w = max(320, base_w + rng.randint(-8, 12))
        h = max(620, base_h + rng.randint(-20, 40))
    else:
        base_w = rng.choice([1024, 1280, 1366, 1440, 1600, 1920])
        base_h = rng.choice([700, 768, 820, 900, 1000, 1080])
        w = max(960, base_w + rng.randint(-80, 120))
        h = max(620, base_h + rng.randint(-60, 90))
    return {"width": int(w), "height": int(h), "is_mobile_narrow": bool(mobile_narrow)}


def _make_block(rng: random.Random, block_id: str, block_type: str, difficulty: str = "normal") -> Dict[str, Any]:
    hard = str(difficulty or "normal").strip().lower() in {"hard", "very_hard", "extreme"}
    prompt, options, correct, hint = _rand_prompt(rng, "text" if block_type == "text" else block_type)
    if block_type in {"dropdown", "dropdown_scroll"}:
        if not options:
            _, options, correct, hint = _rand_prompt(rng, "single")
        if block_type == "dropdown_scroll":
            long_pool = [f"opcja_{i:02d}" for i in range(1, rng.randint(16, 36))]
            if correct and correct[0] not in long_pool:
                long_pool.insert(rng.randint(4, len(long_pool) - 1), correct[0])
            options = long_pool
        return {
            "block_id": block_id,
            "type": block_type,
            "prompt": prompt,
            "options": options,
            "correct": correct[:1],
            "hint": hint,
            "dropdown_variant": rng.choice(
                ["native", "faux", "search", "segmented", "portal", "virtualized"] if hard else ["native", "faux", "search", "segmented"]
            ),
            "dropdown_open": rng.random() < (0.82 if hard else (0.65 if block_type == "dropdown_scroll" else 0.36)),
        }
    if block_type == "text":
        return {
            "block_id": block_id,
            "type": "text",
            "prompt": prompt,
            "options": [],
            "correct": correct[:1],
            "hint": hint,
        }
    if block_type == "slider":
        slider_mode = rng.choice(["intensity", "rating", "progress", "age"])
        if slider_mode == "intensity":
            left_label, right_label = rng.choice(
                [
                    ("wcale", "bardzo"),
                    ("low", "high"),
                    ("zimno", "goraco"),
                    ("slabo", "mocno"),
                ]
            )
            value = rng.randint(0, 100)
            prompt = rng.choice(
                [
                    "Ustaw poziom intensywnosci",
                    "Jak mocno pasuje ten opis?",
                    "Set the intensity level",
                ]
            )
            unit = "%"
            slider_min, slider_max, slider_step = 0, 100, 5
        elif slider_mode == "rating":
            left_label, right_label = rng.choice(
                [
                    ("1 gwiazdka", "5 gwiazdek"),
                    ("poor", "excellent"),
                    ("slabo", "super"),
                ]
            )
            value = rng.randint(1, 5)
            prompt = rng.choice(
                [
                    "Ocen swoja odpowiedz",
                    "Rate this option",
                    "Jak oceniasz ten wariant?",
                ]
            )
            unit = "/5"
            slider_min, slider_max, slider_step = 1, 5, 1
        elif slider_mode == "age":
            left_label, right_label = "18", "70+"
            value = rng.randint(18, 70)
            prompt = rng.choice(
                [
                    "Wybierz przedzial wieku",
                    "Set your age range",
                    "Przesun suwak do odpowiedniego wieku",
                ]
            )
            unit = " lat"
            slider_min, slider_max, slider_step = 18, 70, 1
        else:
            left_label, right_label = "start", "finish"
            value = rng.randint(0, 10)
            prompt = rng.choice(
                [
                    "Jak daleko jestes w procesie?",
                    "Move slider to your progress",
                    "Okresl postep",
                ]
            )
            unit = "/10"
            slider_min, slider_max, slider_step = 0, 10, 1
        return {
            "block_id": block_id,
            "type": "slider",
            "prompt": prompt,
            "options": [left_label, right_label],
            "correct": [str(value)],
            "hint": "slider",
            "slider_min": slider_min,
            "slider_max": slider_max,
            "slider_step": slider_step,
            "slider_value": value,
            "slider_unit": unit,
        }
    return {
        "block_id": block_id,
        "type": block_type,
        "prompt": prompt,
        "options": options,
        "correct": correct,
        "hint": hint,
        "option_variant": rng.choice(["plain", "shortcut", "badge"]),
    }


def _render_block(block: Dict[str, Any], style: Dict[str, Any], rng: random.Random) -> str:
    b_id = str(block["block_id"])
    b_type = str(block["type"])
    prompt = str(block["prompt"])
    preview_question_only = bool(block.get("preview_question_only"))
    preview_gap_px = int(block.get("preview_gap_px") or 0)
    block_attrs = [f"class='quiz-block{' preview-cutoff' if preview_question_only else ''}'", "data-role='quiz-block'", f"data-block-id='{b_id}'", f"data-block-type='{b_type}'"]
    if preview_question_only:
        block_attrs.append("data-preview-cutoff='1'")
    out: List[str] = [f"<section {' '.join(block_attrs)}>"]
    out.append(f"<h2 class='question' data-role='question' data-block-id='{b_id}'>{prompt}</h2>")
    preview_gap_html = f"<div class='preview-answer-gap' style='height:{preview_gap_px}px' data-role='noise'></div>" if preview_question_only else ""

    marker_side_cls = "marker-right" if style["marker_side"] == "right" else "marker-left"
    marker_variant_cls = "marker-custom" if style["marker_variant"] == "custom" else "marker-native"

    if b_type in {"single", "multi"}:
        input_type = "radio" if b_type == "single" else "checkbox"
        row_variant = rng.choice(["row-soft", "row-solid", "row-outline"])
        option_layout = str(style.get("option_layout") or "list")
        option_variant = str(block.get("option_variant") or "plain")
        if option_layout == "cards":
            out.append("<div class='answers-grid'>")
        elif option_layout == "pills":
            out.append("<div class='answers-pills'>")
        else:
            out.append("<div class='answers-list'>")
        if preview_gap_html:
            out.append(preview_gap_html)
        for idx, text in enumerate(block.get("options") or []):
            marker_shape = "circle" if b_type == "single" else "square"
            tag = ""
            if option_variant == "shortcut":
                tag = f"<span class='answer-tag'>{chr(65 + (idx % 26))}</span>"
            elif option_variant == "badge":
                tag = f"<span class='answer-tag'>{rng.choice(['hot', 'new', 'alt', 'v2'])}</span>"
            out.append(
                "<label class='opt-row "
                + f"{row_variant} {marker_side_cls} {marker_variant_cls}' "
                + f"data-role='answer' data-block-id='{b_id}' data-answer-index='{idx}' data-answer-text='{text}'>"
                + f"<input type='{input_type}' name='g_{b_id}' value='{text}'/>"
                + f"<span class='marker marker-{marker_shape}' aria-hidden='true'></span>"
                + f"<span class='answer-text'>{text}</span>"
                + tag
                + "</label>"
            )
        out.append("</div>")
    elif b_type in {"dropdown", "dropdown_scroll"}:
        dropdown_variant = str(block.get("dropdown_variant") or "native")
        open_state = bool(block.get("dropdown_open"))
        options = block.get("options") or []
        add_visual_noise = bool(style.get("dropdown_visual_noise"))
        trigger_suffix = (
            f"<span class='trigger-meta' data-role='noise'>{rng.choice(['alt+v', 'expand', 'menu'])}</span>"
            if add_visual_noise and rng.random() < 0.55
            else ""
        )
        if dropdown_variant == "native":
            out.append(f"<div class='select-wrap'>")
            if preview_gap_html:
                out.append(preview_gap_html)
            out.append(f"<select data-role='select' data-block-id='{b_id}' class='select-control'>")
            out.append("<option value=''>-- wybierz --</option>")
            for text in options:
                out.append(f"<option value='{text}'>{text}</option>")
            out.append("</select></div>")
        elif dropdown_variant == "segmented":
            out.append(
                f"<div class='faux-select-wrap segmented-wrap' data-role='select' data-block-id='{b_id}'>"
                f"<button type='button' class='faux-select-trigger'>Wybierz opcje</button>{trigger_suffix}"
            )
            if preview_gap_html:
                out.append(preview_gap_html)
            if open_state:
                out.append("<div class='segmented-listbox segmented'>")
                for idx, text in enumerate(options[: min(6, len(options))]):
                    out.append(
                        f"<button type='button' class='seg-btn' data-role='answer' data-block-id='{b_id}' "
                        f"data-answer-index='{idx}' data-answer-text='{text}'>{text}</button>"
                    )
                out.append("</div>")
            out.append("</div>")
        elif dropdown_variant == "portal":
            out.append(
                f"<div class='faux-select-wrap portal-anchor' data-role='select' data-block-id='{b_id}'>"
                f"<button type='button' class='faux-select-trigger'>Wybierz opcje</button>{trigger_suffix}</div>"
            )
            if open_state:
                if preview_gap_html:
                    out.append(preview_gap_html)
                out.append("<div class='portal-popover'>")
                if add_visual_noise and rng.random() < 0.5:
                    out.append("<div class='portal-toolbar' data-role='noise'>Filter | Sort | Recent</div>")
                out.append("<ul class='faux-listbox portal-listbox'>")
                for idx, text in enumerate(options[: min(18, len(options))]):
                    out.append(
                        f"<li class='faux-option' data-role='answer' data-block-id='{b_id}' "
                        f"data-answer-index='{idx}' data-answer-text='{text}'>{text}</li>"
                    )
                out.append("</ul></div>")
        elif dropdown_variant == "virtualized":
            start_idx = rng.randint(0, max(0, len(options) - 18)) if len(options) > 18 else 0
            visible = options[start_idx : start_idx + min(18, len(options))]
            out.append(
                f"<div class='faux-select-wrap virtualized-wrap' data-role='select' data-block-id='{b_id}'>"
                f"<button type='button' class='faux-select-trigger'>Wybierz opcje</button>{trigger_suffix}"
            )
            if open_state:
                if preview_gap_html:
                    out.append(preview_gap_html)
                out.append("<div class='virtualized-shell'>")
                out.append(f"<div class='virtualized-top-spacer' data-role='noise'>+{start_idx} hidden rows</div>")
                out.append("<ul class='faux-listbox virtualized-listbox'>")
                for idx, text in enumerate(visible, start=start_idx):
                    out.append(
                        f"<li class='faux-option' data-role='answer' data-block-id='{b_id}' "
                        f"data-answer-index='{idx}' data-answer-text='{text}'>{text}</li>"
                    )
                out.append("</ul>")
                out.append(
                    f"<div class='virtualized-hint' data-role='noise'>Showing {len(visible)} / {len(options)} options...</div>"
                )
                out.append("</div>")
            out.append("</div>")
        else:
            out.append(
                f"<div class='faux-select-wrap' data-role='select' data-block-id='{b_id}'>"
                f"<button type='button' class='faux-select-trigger'>Wybierz opcje</button>{trigger_suffix}"
            )
            if dropdown_variant == "search":
                out.append("<input class='faux-search' type='text' placeholder='Szukaj...'/>")
            if open_state:
                if preview_gap_html:
                    out.append(preview_gap_html)
                out.append("<ul class='faux-listbox'>")
                for idx, text in enumerate(options[: min(12, len(options))]):
                    out.append(
                        f"<li class='faux-option' data-role='answer' data-block-id='{b_id}' "
                        f"data-answer-index='{idx}' data-answer-text='{text}'>{text}</li>"
                    )
                out.append("</ul>")
            out.append("</div>")
        if b_type == "dropdown_scroll":
            out.append("<p class='hint' data-role='hint'>Lista jest dluga - czesto trzeba przewinac.</p>")
    elif b_type == "text":
        placeholder = rng.choice(["wpisz tutaj", "twoja odpowiedz", "type answer", "uzupelnij pole"])
        out.append(
            "<div class='text-wrap'>"
            + preview_gap_html
            + f"<input data-role='text-input' data-block-id='{b_id}' class='text-control' "
            + f"type='text' autocomplete='off' placeholder='{placeholder}'/>"
            + "</div>"
        )
    elif b_type == "slider":
        slider_min = int(block.get("slider_min") or 0)
        slider_max = int(block.get("slider_max") or 100)
        slider_step = int(block.get("slider_step") or 1)
        slider_value = int(block.get("slider_value") or slider_min)
        slider_unit = str(block.get("slider_unit") or "")
        left_label = str((block.get("options") or ["min", "max"])[0])
        right_label = str((block.get("options") or ["min", "max"])[-1])
        out.append("<div class='slider-wrap'>")
        if preview_gap_html:
            out.append(preview_gap_html)
        out.append(
            f"<div class='slider-value-pill' data-role='noise'>{slider_value}{slider_unit}</div>"
            f"<div class='slider-answer' data-role='answer' data-block-id='{b_id}' data-answer-index='0' data-answer-text='{slider_value}'>"
            f"<input class='slider-control' type='range' min='{slider_min}' max='{slider_max}' step='{slider_step}' value='{slider_value}'/>"
            "</div>"
        )
        out.append(
            f"<div class='slider-scale' data-role='noise'><span>{left_label}</span><span>{right_label}</span></div>"
        )
        out.append("</div>")
    out.append("</section>")
    return "\n".join(out)


def _render_html(sample: Dict[str, Any], rng: random.Random) -> str:
    style = sample["style"]
    blocks = sample["blocks"]
    has_next = bool(sample["has_next"])
    title = sample["title"]
    desc = sample["description"]
    shell_variant = str(style.get("shell_variant") or "plain")
    next_variant = str(style.get("next_variant") or "button")
    sticky_header_enabled = bool(style.get("sticky_header"))
    sticky_footer_enabled = bool(style.get("sticky_footer"))
    modal_overlay_enabled = bool(style.get("modal_overlay"))
    floating_help_enabled = bool(style.get("floating_help"))
    is_mobile_narrow = bool(sample.get("viewport", {}).get("is_mobile_narrow"))

    shadow_css = "none"
    if style["shadow"] == "soft":
        shadow_css = "0 14px 34px rgba(0,0,0,0.22)"
    elif style["shadow"] == "hard":
        shadow_css = "0 20px 44px rgba(0,0,0,0.36)"

    noise_html = []
    for idx in range(int(style["noise_blocks"])):
        tag = rng.choice(["div", "aside", "p"])
        text = rng.choice(
            [
                "Panel info: status online",
                "Navigation: Home / Docs / Contact",
                "Promo: Limited offer",
                "Tip: use keyboard shortcuts",
            ]
        )
        noise_html.append(f"<{tag} class='noise noise-{idx}' data-role='noise'>{text}</{tag}>")
    decoy_buttons = []
    for idx in range(int(style.get("extra_decoy_buttons") or 0)):
        decoy_buttons.append(
            f"<a class='noise-link decoy-link decor-text' href='#' data-role='noise'>"
            f"{rng.choice(['Help', 'About survey', 'Privacy', 'Terms'])} {idx+1}</a>"
        )

    blocks_html = "\n".join(_render_block(block, style, rng) for block in blocks)
    next_html = ""
    sticky_next_html = ""
    hero_next_html = ""
    if has_next:
        next_text = rng.choice(["Nastepne", "Dalej", "Continue", "Nastepny krok"])
        if next_variant == "link":
            next_html = f"<a class='next-link' href='#' data-role='next'>{next_text}</a>"
        elif next_variant == "icon":
            next_html = f"<button class='next-btn next-icon-btn' data-role='next' aria-label='{next_text}'><span class='next-icon-label'>{next_text}</span><span aria-hidden='true'>→</span></button>"
        elif next_variant == "top_right":
            hero_next_html = f"<a class='hero-next-link' href='#' data-role='next'>{next_text} →</a>"
        elif next_variant == "sticky_bar":
            sticky_next_html = f"<button class='next-btn sticky-next-btn' data-role='next'>{next_text}</button>"
        else:
            next_html = f"<button class='next-btn' data-role='next'>{next_text}</button>"

    top_nav_html = ""
    if bool(style.get("show_top_nav")):
        top_nav_html = (
            "<nav class='top-nav decor-text' data-role='nav'>"
            "<a href='#' data-role='nav-item'>Home</a><a href='#' data-role='nav-item'>Catalog</a><a href='#' data-role='nav-item'>Help</a>"
            "</nav>"
        )
    sticky_header_html = ""
    if sticky_header_enabled:
        sticky_header_html = (
            "<div class='sticky-header' data-role='noise'>"
            "<div class='sticky-header-inner'>"
            "<span class='sticky-brand decor-text'>Survey Hub</span>"
            "<nav class='sticky-mini-nav decor-text'><a href='#'>Menu</a><a href='#'>Docs</a><a href='#'>Profile</a></nav>"
            "</div></div>"
        )
    badge_html = f"<span class='hero-badge decor-text' data-role='noise'>{rng.choice(['Beta', 'Pro', 'Live'])}</span>" if bool(style.get("title_badge")) else ""
    progress_html = ""
    if bool(style.get("show_progress_line")):
        progress_html = f"<div class='progress' data-role='noise'><span style='width:{rng.randint(12, 94)}%'></span></div>"
    secondary_cta = ""
    if bool(style.get("show_secondary_cta")):
        secondary_cta = f"<button class='ghost-btn decor-text' type='button' data-role='secondary-cta'>{rng.choice(['Pomin', 'Wroc', 'Zapisz'])}</button>"
    sticky_footer_html = ""
    if sticky_footer_enabled:
        sticky_footer_action = sticky_next_html or (
            f"<a class='sticky-footer-link decor-text' href='#' data-role='noise'>{rng.choice(['Review form', 'Need help?', 'Quick tips'])}</a>"
        )
        sticky_footer_html = (
            "<div class='sticky-footer' data-role='noise'>"
            "<span class='sticky-footer-copy decor-text'>Cookies | Terms | Assistance</span>"
            f"{sticky_footer_action}"
            "</div>"
        )
    modal_shell_open = "<div class='modal-backdrop' data-role='noise'><div class='modal-card-shell'>" if modal_overlay_enabled else ""
    modal_shell_close = "</div></div>" if modal_overlay_enabled else ""
    floating_help_html = ""
    if floating_help_enabled:
        floating_help_html = (
            "<button class='floating-help decor-text' type='button' data-role='noise'>"
            f"{rng.choice(['Help', 'Chat', 'Ask'])}"
            "</button>"
        )

    return f"""<!doctype html>
<html lang='pl'>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>{title}</title>
  <style>
    :root {{
      --bg-a: {style["bg_a"]};
      --bg-b: {style["bg_b"]};
      --panel: {style["panel"]};
      --panel-border: {style["panel_border"]};
      --text: {style["text"]};
      --muted: {style["muted"]};
      --accent: {style["accent"]};
      --accent-text: {style["accent_text"]};
      --radius: {style["radius"]}px;
      --card-radius: {style["card_radius"]}px;
      --space: {style["spacing"]}px;
      --row-py: {style["row_pad_y"]}px;
      --row-px: {style["row_pad_x"]}px;
      --marker-size: {style["marker_size"]}px;
      --max-width: {style["max_width"]}px;
      --title-size: {style["title_size"]}px;
      --question-size: {style["question_size"]}px;
      --text-size: {style["text_size"]}px;
      --line-height: {style["line_height"]};
      --btn-radius: {style["btn_radius"]}px;
      --btn-py: {style["btn_pad_y"]}px;
      --btn-px: {style["btn_pad_x"]}px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: {style["font"]};
      font-weight: {int(style.get("font_weight") or 500)};
      line-height: var(--line-height);
      letter-spacing: {float(style.get("letter_spacing_em") or 0.0)}em;
      word-spacing: {float(style.get("word_spacing_em") or 0.0)}em;
      background:
        radial-gradient(circle at 15% 10%, rgba(255,255,255,{style["stripe_opacity"]}), transparent 38%),
        radial-gradient(circle at 88% 80%, rgba(255,255,255,{style["stripe_opacity"]}), transparent 42%),
        linear-gradient(135deg, var(--bg-a), var(--bg-b));
    }}
    .decor-text {{ text-transform: none; }}
    .wrap {{
      width: min(calc(100vw - 32px), var(--max-width));
      margin: {58 if sticky_header_enabled else 18}px auto {72 if sticky_footer_enabled else 18}px;
      display: grid;
      gap: calc(var(--space) * 1.1);
    }}
    body.mobile-narrow .wrap {{ width: min(calc(100vw - 20px), 520px); }}
    .hero, .quiz-shell {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: var(--card-radius);
      padding: calc(var(--space) * 1.2);
      box-shadow: {shadow_css};
    }}
    .sticky-header {{
      position: fixed; top: 0; left: 0; right: 0; z-index: 30;
      background: color-mix(in srgb, var(--panel), black 10%);
      border-bottom: 1px solid color-mix(in srgb, var(--panel-border), transparent 20%);
      backdrop-filter: blur(8px);
    }}
    .sticky-header-inner {{
      width: min(calc(100vw - 24px), calc(var(--max-width) + 18px));
      margin: 0 auto; padding: 10px 6px; display: flex; justify-content: space-between; gap: 16px; align-items: center;
    }}
    .sticky-mini-nav {{ display: flex; gap: 12px; font-size: calc(var(--text-size) - 2px); }}
    .sticky-mini-nav a {{ color: var(--muted); text-decoration: none; }}
    .sticky-brand {{ font-size: calc(var(--text-size) - 1px); color: var(--text); }}
    .modal-backdrop {{
      position: relative;
      min-height: calc(100vh - 48px);
      padding: {18 if is_mobile_narrow else 36}px 0;
    }}
    .modal-backdrop::before {{
      content: "";
      position: fixed; inset: 0;
      background: rgba(8, 10, 18, 0.52);
      pointer-events: none;
    }}
    .modal-card-shell {{
      position: relative;
      z-index: 2;
      width: min(calc(100vw - 22px), {560 if is_mobile_narrow else 760}px);
      margin: 0 auto;
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: var(--title-size); line-height: 1.1; }}
    .hero-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    .hero-head-side {{ display: flex; align-items: center; gap: 10px; margin-left: auto; }}
    .hero-badge {{ border: 1px solid var(--panel-border); border-radius: 999px; padding: 4px 10px; font-size: 12px; opacity: .88; }}
    .hero p {{ margin: 0; font-size: var(--text-size); color: var(--muted); }}
    .hero-next-link {{ color: var(--accent); text-decoration: none; font-size: calc(var(--text-size) - 1px); white-space: nowrap; }}
    .top-nav {{ display: flex; gap: 14px; margin-bottom: calc(var(--space) * .8); font-size: calc(var(--text-size) - 1px); }}
    .top-nav a {{ color: var(--muted); text-decoration: none; }}
    .noise {{ font-size: calc(var(--text-size) - 2px); opacity: .85; }}
    .noise-link {{ color: var(--muted); text-decoration: none; font-size: calc(var(--text-size) - 2px); opacity: .9; }}
    .quiz-shell.shell-split .quiz-block {{ border-left: 3px solid color-mix(in srgb, var(--accent), transparent 40%); padding-left: calc(var(--space) * .8); }}
    .quiz-shell.shell-dense .quiz-block {{ margin: calc(var(--space) * .6) 0; }}
    .quiz-block {{ margin: calc(var(--space) * 1.2) 0; }}
    .preview-answer-gap {{ width: 100%; pointer-events: none; }}
    .question {{ margin: 0 0 calc(var(--space) * .9); font-size: var(--question-size); line-height: 1.2; }}
    .answers-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 8px; }}
    .answers-pills {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .answers-pills .opt-row {{ width: auto; min-width: 180px; flex: 0 0 auto; }}
    .opt-row {{
      display: flex;
      align-items: center;
      width: 100%;
      gap: 10px;
      cursor: pointer;
      border-radius: var(--radius);
      margin: calc(var(--space) * .55) 0;
      padding: var(--row-py) var(--row-px);
      border: 1px solid var(--panel-border);
      user-select: none;
      font-size: var(--text-size);
    }}
    .row-soft {{ background: color-mix(in srgb, var(--panel), white 3%); }}
    .row-solid {{ background: color-mix(in srgb, var(--panel), black 4%); }}
    .row-outline {{ background: transparent; }}
    .marker-right {{ flex-direction: row-reverse; justify-content: flex-end; }}
    .marker-left {{ flex-direction: row; }}
    .opt-row input {{
      margin: 0;
      width: var(--marker-size);
      height: var(--marker-size);
      accent-color: var(--accent);
      flex: 0 0 auto;
    }}
    .marker-custom input {{ position: absolute; opacity: 0; pointer-events: none; }}
    .marker {{
      display: none;
      width: var(--marker-size);
      height: var(--marker-size);
      border: 2px solid color-mix(in srgb, var(--accent), white 20%);
      background: color-mix(in srgb, var(--panel), white 12%);
      flex: 0 0 auto;
    }}
    .marker-custom .marker {{ display: inline-block; }}
    .marker-circle {{ border-radius: 50%; }}
    .marker-square {{ border-radius: 3px; }}
    .answer-text {{ flex: 1; }}
    .answer-tag {{ font-size: 11px; line-height: 1; border: 1px solid var(--panel-border); border-radius: 999px; padding: 3px 7px; opacity: .85; }}
    .select-wrap, .text-wrap {{ margin: calc(var(--space) * .6) 0; }}
    .slider-wrap {{ margin: calc(var(--space) * .7) 0; display: grid; gap: 10px; }}
    .select-control, .text-control {{
      width: 100%;
      font-size: var(--text-size);
      color: var(--text);
      border-radius: var(--radius);
      border: 1px solid var(--panel-border);
      padding: calc(var(--row-py) - 1px) var(--row-px);
      background: color-mix(in srgb, var(--panel), black 3%);
    }}
    .slider-answer {{ border: 1px solid var(--panel-border); border-radius: var(--radius); padding: 12px 14px; background: color-mix(in srgb, var(--panel), black 4%); }}
    .slider-control {{ width: 100%; accent-color: var(--accent); }}
    .slider-scale {{ display: flex; justify-content: space-between; gap: 12px; color: var(--muted); font-size: calc(var(--text-size) - 3px); }}
    .slider-value-pill {{ justify-self: start; border: 1px solid var(--panel-border); border-radius: 999px; padding: 4px 10px; font-size: calc(var(--text-size) - 2px); color: var(--muted); }}
    .faux-select-wrap {{ border: 1px solid var(--panel-border); border-radius: var(--radius); padding: 8px; background: color-mix(in srgb, var(--panel), black 4%); }}
    .faux-select-trigger {{ width: 100%; text-align: left; border: 1px solid var(--panel-border); border-radius: var(--radius); padding: 8px 12px; background: transparent; color: var(--text); }}
    .trigger-meta {{ display: inline-block; margin-top: 6px; color: var(--muted); font-size: calc(var(--text-size) - 4px); }}
    .faux-search {{ width: 100%; margin-top: 7px; border-radius: var(--radius); border: 1px solid var(--panel-border); padding: 7px 10px; background: transparent; color: var(--text); }}
    .faux-listbox {{ list-style: none; margin: 8px 0 0; padding: 0; max-height: 190px; overflow: auto; border: 1px solid var(--panel-border); border-radius: var(--radius); }}
    .faux-option {{ padding: 8px 10px; border-bottom: 1px solid color-mix(in srgb, var(--panel-border), transparent 35%); font-size: var(--text-size); }}
    .faux-option:last-child {{ border-bottom: 0; }}
    .portal-popover {{ margin-top: 10px; border: 1px solid var(--panel-border); border-radius: var(--radius); background: color-mix(in srgb, var(--panel), white 4%); padding: 8px; box-shadow: 0 10px 20px rgba(0,0,0,.16); }}
    .portal-toolbar {{ color: var(--muted); font-size: calc(var(--text-size) - 4px); margin-bottom: 6px; }}
    .virtualized-shell {{ margin-top: 8px; }}
    .virtualized-top-spacer {{ color: var(--muted); font-size: calc(var(--text-size) - 4px); margin-bottom: 4px; }}
    .segmented {{ display: flex; flex-wrap: wrap; gap: 8px; margin: calc(var(--space) * .4) 0; }}
    .seg-btn {{ border: 1px solid var(--panel-border); border-radius: 999px; background: transparent; color: var(--text); padding: 7px 12px; font-size: calc(var(--text-size) - 1px); }}
    .hint {{ color: var(--muted); font-size: calc(var(--text-size) - 2px); margin: 6px 0 0; }}
    .virtualized-hint {{ color: var(--muted); font-size: calc(var(--text-size) - 3px); margin-top: 6px; opacity: .9; }}
    .progress {{ height: 6px; border-radius: 999px; border: 1px solid var(--panel-border); margin-top: 10px; overflow: hidden; }}
    .progress span {{ display: block; height: 100%; background: var(--accent); }}
    .actions {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    .next-btn {{
      border: 0;
      border-radius: var(--btn-radius);
      padding: var(--btn-py) var(--btn-px);
      font-weight: 700;
      font-size: var(--text-size);
      color: var(--accent-text);
      background: var(--accent);
      cursor: pointer;
      margin-top: calc(var(--space) * .8);
    }}
    .next-link {{
      display: inline-flex;
      align-items: center;
      color: var(--accent);
      font-weight: 700;
      text-decoration: none;
      margin-top: calc(var(--space) * .8);
      padding: 4px 2px;
    }}
    .next-icon-btn {{ min-width: 56px; justify-content: center; gap: 10px; }}
    .next-icon-label {{ font-size: calc(var(--text-size) - 1px); }}
    .sticky-footer {{
      position: fixed; left: 0; right: 0; bottom: 0; z-index: 30;
      display: flex; align-items: center; justify-content: space-between; gap: 14px;
      padding: 10px 16px;
      background: color-mix(in srgb, var(--panel), black 8%);
      border-top: 1px solid color-mix(in srgb, var(--panel-border), transparent 20%);
      backdrop-filter: blur(8px);
    }}
    .sticky-footer-copy {{ color: var(--muted); font-size: calc(var(--text-size) - 3px); }}
    .sticky-footer-link {{ color: var(--muted); text-decoration: none; font-size: calc(var(--text-size) - 2px); }}
    .sticky-next-btn {{ margin-top: 0; }}
    .floating-help {{
      position: fixed; right: 16px; bottom: {92 if sticky_footer_enabled else 18}px; z-index: 35;
      border: 1px solid var(--panel-border); border-radius: 999px; padding: 10px 14px;
      background: color-mix(in srgb, var(--panel), black 6%); color: var(--text);
      box-shadow: 0 10px 22px rgba(0,0,0,.22);
    }}
    .ghost-btn {{
      border: 1px solid var(--panel-border);
      border-radius: var(--btn-radius);
      padding: var(--btn-py) var(--btn-px);
      font-weight: 600;
      font-size: calc(var(--text-size) - 1px);
      color: var(--text);
      background: transparent;
      cursor: pointer;
      margin-top: calc(var(--space) * .8);
    }}
    @media (max-width: 980px) {{
      .answers-grid {{ grid-template-columns: 1fr; }}
      .hero h1 {{ font-size: calc(var(--title-size) - 6px); }}
      .question {{ font-size: calc(var(--question-size) - 3px); }}
      .sticky-header-inner, .sticky-footer {{ padding-left: 12px; padding-right: 12px; }}
      .hero-head {{ align-items: flex-start; }}
    }}
  </style>
</head>
<body class='{"mobile-narrow" if is_mobile_narrow else ""}'>
  {sticky_header_html}
  {modal_shell_open}
  <main class='wrap'>
    {top_nav_html}
    <section class='hero' data-role='hero'>
      <div class='hero-head'><h1 class='decor-text' data-role='hero-title'>{title}</h1><div class='hero-head-side'>{hero_next_html}{badge_html}</div></div>
      <p class='decor-text' data-role='hero-desc'>{desc}</p>
      {progress_html}
      {"".join(noise_html)}
    </section>
    <section class='quiz-shell shell-{shell_variant}' data-role='quiz-shell' data-global-type='{sample["global_type"]}'>
      {blocks_html}
      <div class='actions'>{next_html}{secondary_cta}{"".join(decoy_buttons)}</div>
    </section>
  </main>
  {modal_shell_close}
  {sticky_footer_html}
  {floating_help_html}
</body>
</html>"""


def build_sample(seed: int, index: int, forced_global_type: str | None = None, difficulty: str = "normal") -> Dict[str, Any]:
    rng = random.Random(seed * 1_000_003 + index * 7_919)
    hard = str(difficulty or "normal").strip().lower() in {"hard", "very_hard", "extreme"}
    global_type = str(forced_global_type) if forced_global_type else _weighted_global_type(rng)
    style = _style_pack(rng, difficulty=difficulty)
    if style.get("next_variant") == "sticky_bar":
        style["sticky_footer"] = True
    viewport = _viewport_pack(rng, style)

    if global_type == "triple":
        base_types = [rng.choice(["single", "multi", "dropdown", "slider", "text"]) for _ in range(3)]
        blocks = [_make_block(rng, f"b{i+1}", t, difficulty=difficulty) for i, t in enumerate(base_types)]
    elif global_type == "mixed":
        k = rng.randint(3, 5)
        pool = ["single", "multi", "dropdown", "dropdown_scroll", "slider", "text"]
        rng.shuffle(pool)
        chosen = pool[: max(2, k)]
        while len(chosen) < k:
            chosen.append(rng.choice(pool))
        blocks = [_make_block(rng, f"b{i+1}", t, difficulty=difficulty) for i, t in enumerate(chosen)]
    else:
        blocks = [_make_block(rng, "b1", global_type, difficulty=difficulty)]

    partial_next_question_visible = False
    if len(blocks) >= 2 and global_type in {"triple", "mixed"} and rng.random() < (0.22 if hard else 0.1):
        preview_idx = 1
        blocks[preview_idx]["preview_question_only"] = True
        blocks[preview_idx]["preview_gap_px"] = rng.randint(140, 280) if hard else rng.randint(90, 180)
        partial_next_question_visible = True

    has_next = rng.random() < (0.72 if hard else 0.62)
    auto_next = (not has_next) and (rng.random() < 0.86)
    require_scroll = bool(any(b["type"] == "dropdown_scroll" for b in blocks) or len(blocks) >= 4 or partial_next_question_visible)
    if require_scroll and rng.random() < 0.35:
        viewport["height"] = max(620, viewport["height"] - rng.randint(120, 220))
    if partial_next_question_visible:
        viewport["height"] = min(viewport["height"], rng.randint(620, 760) if hard else rng.randint(680, 820))

    sample_id = f"s_{seed}_{index:06d}"
    title = rng.choice(
        [
            "Quiz Sandbox",
            "Dynamic Survey",
            "Adaptive Form",
            "Knowledge Check",
            "Challenge Board",
        ]
    )
    desc = rng.choice(
        [
            "Losowy wariant interfejsu - styl, spacing i kontrolki zmieniaja sie co probke.",
            "Generowane automatycznie pod dataset: rozne heurystyki i rozne typy pytan.",
            "Renders randomized to avoid overfitting on one visual style.",
        ]
    )
    sample = {
        "sample_id": sample_id,
        "seed": seed,
        "index": index,
        "global_type": global_type,
        "block_types": [str(b["type"]) for b in blocks],
        "blocks": blocks,
        "has_next": has_next,
        "auto_next": auto_next,
        "require_scroll": require_scroll,
        "partial_next_question_visible": partial_next_question_visible,
        "ui_flags": {
            "mobile_narrow": bool(style.get("mobile_narrow")),
            "sticky_header": bool(style.get("sticky_header")),
            "sticky_footer": bool(style.get("sticky_footer")),
            "modal_overlay": bool(style.get("modal_overlay")),
            "floating_help": bool(style.get("floating_help")),
            "next_variant": str(style.get("next_variant") or "button"),
        },
        "style": style,
        "viewport": viewport,
        "title": title,
        "description": desc,
    }
    sample["html"] = _render_html(sample, rng)
    return sample


def build_samples(
    count: int,
    seed: int,
    balanced: bool = False,
    start_index: int = 0,
    difficulty: str = "normal",
) -> List[Dict[str, Any]]:
    n = max(1, int(count))
    if not bool(balanced):
        return [build_sample(seed=seed, index=int(start_index) + i, difficulty=difficulty) for i in range(n)]
    forced_types: List[str] = []
    cls = list(GLOBAL_TYPES)
    for i in range(n):
        forced_types.append(cls[(int(start_index) + i) % len(cls)])
    rng = random.Random(seed * 17 + 91 + int(start_index))
    rng.shuffle(forced_types)
    return [
        build_sample(seed=seed, index=int(start_index) + i, forced_global_type=forced_types[i], difficulty=difficulty)
        for i in range(n)
    ]


def _sample_compact(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "seed": sample["seed"],
        "index": sample["index"],
        "global_type": sample["global_type"],
        "block_types": sample["block_types"],
        "partial_next_question_visible": bool(sample.get("partial_next_question_visible")),
        "ui_flags": sample.get("ui_flags") or {},
        "has_next": sample["has_next"],
        "auto_next": sample["auto_next"],
        "require_scroll": sample["require_scroll"],
        "viewport": sample["viewport"],
        "blocks": [
            {
                "block_id": b["block_id"],
                "type": b["type"],
                "prompt": b["prompt"],
                "options": b["options"],
                "correct": b["correct"],
            }
            for b in sample["blocks"]
        ],
    }


def _serve(samples: List[Dict[str, Any]], host: str, port: int) -> None:
    by_id = {str(s["sample_id"]): s for s in samples}

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str, status: int = 200) -> None:
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path or "/"
            if path == "/":
                links = []
                for s in samples[:400]:
                    sid = s["sample_id"]
                    links.append(f"<li><a href='/s/{sid}'>{sid}</a> [{s['global_type']}]</li>")
                html = (
                    "<!doctype html><html><head><meta charset='utf-8'/>"
                    "<title>Random Quiz Sandbox</title></head><body>"
                    "<h1>Random Quiz Sandbox</h1>"
                    f"<p>samples={len(samples)}</p>"
                    "<p><a href='/api/manifest'>/api/manifest</a></p>"
                    "<ul>"
                    + "".join(links)
                    + "</ul></body></html>"
                )
                self._send_html(html)
                return
            if path.startswith("/s/"):
                sid = path.split("/s/", 1)[1]
                row = by_id.get(sid)
                if not isinstance(row, dict):
                    self._send_html("not found", status=404)
                    return
                self._send_html(str(row["html"]))
                return
            if path == "/api/manifest":
                self._send_json([_sample_compact(s) for s in samples])
                return
            if path.startswith("/api/sample/"):
                sid = path.split("/api/sample/", 1)[1]
                row = by_id.get(sid)
                if not isinstance(row, dict):
                    self._send_json({"error": "not_found"}, status=404)
                    return
                self._send_json(_sample_compact(row))
                return
            self._send_json({"error": "unknown_path", "path": path}, status=404)

    print(f"[sandbox] serving on http://{host}:{port}/ samples={len(samples)}")
    HTTPServer((host, int(port)), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomized quiz sandbox server (screen-only dataset source).")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--balanced", type=int, default=0, help="1=equalize global classes in generated samples.")
    parser.add_argument("--difficulty", type=str, default="hard", help="normal|hard")
    parser.add_argument("--export-manifest", type=str, default="")
    parser.add_argument("--export-only", action="store_true", help="Only export manifest and exit.")
    args = parser.parse_args()

    samples = build_samples(
        count=max(1, int(args.count)),
        seed=int(args.seed),
        balanced=bool(int(args.balanced)),
        start_index=0,
        difficulty=str(args.difficulty or "hard"),
    )
    if args.export_manifest:
        out_path = Path(args.export_manifest).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps([_sample_compact(s) for s in samples], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[sandbox] manifest exported: {out_path}")
    if bool(args.export_only):
        return
    _serve(samples=samples, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
