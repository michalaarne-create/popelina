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
    "text",
    "triple",
    "mixed",
]

_TYPE_WEIGHTS = {
    "single": 22,
    "multi": 18,
    "dropdown": 16,
    "dropdown_scroll": 10,
    "text": 12,
    "triple": 11,
    "mixed": 11,
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
        if mode == "en":
            prompt = f"Select all that apply: {topic}"
        else:
            prompt = f"Zaznacz wszystkie pasujace: {topic}"
        return prompt, options, correct_values, "multi"

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


def _style_pack(rng: random.Random) -> Dict[str, Any]:
    palette = dict(rng.choice(_PALETTES))
    marker_variant = rng.choice(["native", "custom"])
    marker_side = rng.choice(["left", "right"])
    return {
        **palette,
        "font": rng.choice(_FONTS),
        "radius": rng.randint(6, 24),
        "card_radius": rng.randint(10, 28),
        "spacing": rng.randint(8, 22),
        "row_pad_y": rng.randint(8, 18),
        "row_pad_x": rng.randint(10, 24),
        "marker_size": rng.randint(12, 22),
        "marker_variant": marker_variant,
        "marker_side": marker_side,
        "line_height": round(rng.uniform(1.25, 1.6), 2),
        "title_size": rng.randint(24, 42),
        "question_size": rng.randint(19, 33),
        "text_size": rng.randint(14, 20),
        "max_width": rng.choice([920, 980, 1040, 1160, 1280, 1440]),
        "shadow": rng.choice(["none", "soft", "hard"]),
        "stripe_opacity": round(rng.uniform(0.0, 0.2), 2),
        "btn_radius": rng.randint(8, 20),
        "btn_pad_y": rng.randint(8, 14),
        "btn_pad_x": rng.randint(14, 28),
        "noise_blocks": rng.randint(0, 3),
    }


def _viewport_pack(rng: random.Random) -> Dict[str, int]:
    base_w = rng.choice([1024, 1280, 1366, 1440, 1600, 1920])
    base_h = rng.choice([700, 768, 820, 900, 1000, 1080])
    w = max(960, base_w + rng.randint(-80, 120))
    h = max(620, base_h + rng.randint(-60, 90))
    return {"width": int(w), "height": int(h)}


def _make_block(rng: random.Random, block_id: str, block_type: str) -> Dict[str, Any]:
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
    return {
        "block_id": block_id,
        "type": block_type,
        "prompt": prompt,
        "options": options,
        "correct": correct,
        "hint": hint,
    }


def _render_block(block: Dict[str, Any], style: Dict[str, Any], rng: random.Random) -> str:
    b_id = str(block["block_id"])
    b_type = str(block["type"])
    prompt = str(block["prompt"])
    out: List[str] = [f"<section class='quiz-block' data-role='quiz-block' data-block-id='{b_id}' data-block-type='{b_type}'>"]
    out.append(f"<h2 class='question' data-role='question' data-block-id='{b_id}'>{prompt}</h2>")

    marker_side_cls = "marker-right" if style["marker_side"] == "right" else "marker-left"
    marker_variant_cls = "marker-custom" if style["marker_variant"] == "custom" else "marker-native"

    if b_type in {"single", "multi"}:
        input_type = "radio" if b_type == "single" else "checkbox"
        row_variant = rng.choice(["row-soft", "row-solid", "row-outline"])
        for idx, text in enumerate(block.get("options") or []):
            marker_shape = "circle" if b_type == "single" else "square"
            out.append(
                "<label class='opt-row "
                + f"{row_variant} {marker_side_cls} {marker_variant_cls}' "
                + f"data-role='answer' data-block-id='{b_id}' data-answer-index='{idx}' data-answer-text='{text}'>"
                + f"<input type='{input_type}' name='g_{b_id}' value='{text}'/>"
                + f"<span class='marker marker-{marker_shape}' aria-hidden='true'></span>"
                + f"<span class='answer-text'>{text}</span>"
                + "</label>"
            )
    elif b_type in {"dropdown", "dropdown_scroll"}:
        out.append(f"<div class='select-wrap'><select data-role='select' data-block-id='{b_id}' class='select-control'>")
        out.append("<option value=''>-- wybierz --</option>")
        for text in block.get("options") or []:
            out.append(f"<option value='{text}'>{text}</option>")
        out.append("</select></div>")
        if b_type == "dropdown_scroll":
            out.append("<p class='hint'>Lista jest dluga - czesto trzeba przewinac.</p>")
    elif b_type == "text":
        placeholder = rng.choice(["wpisz tutaj", "twoja odpowiedz", "type answer", "uzupelnij pole"])
        out.append(
            "<div class='text-wrap'>"
            + f"<input data-role='text-input' data-block-id='{b_id}' class='text-control' "
            + f"type='text' autocomplete='off' placeholder='{placeholder}'/>"
            + "</div>"
        )
    out.append("</section>")
    return "\n".join(out)


def _render_html(sample: Dict[str, Any], rng: random.Random) -> str:
    style = sample["style"]
    blocks = sample["blocks"]
    has_next = bool(sample["has_next"])
    title = sample["title"]
    desc = sample["description"]

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
        noise_html.append(f"<{tag} class='noise noise-{idx}'>{text}</{tag}>")

    blocks_html = "\n".join(_render_block(block, style, rng) for block in blocks)
    next_html = ""
    if has_next:
        next_text = rng.choice(["Nastepne", "Dalej", "Continue", "Nastepny krok"])
        next_html = f"<button class='next-btn' data-role='next'>{next_text}</button>"

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
      line-height: var(--line-height);
      background:
        radial-gradient(circle at 15% 10%, rgba(255,255,255,{style["stripe_opacity"]}), transparent 38%),
        radial-gradient(circle at 88% 80%, rgba(255,255,255,{style["stripe_opacity"]}), transparent 42%),
        linear-gradient(135deg, var(--bg-a), var(--bg-b));
    }}
    .wrap {{
      width: min(calc(100vw - 32px), var(--max-width));
      margin: 18px auto;
      display: grid;
      gap: calc(var(--space) * 1.1);
    }}
    .hero, .quiz-shell {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: var(--card-radius);
      padding: calc(var(--space) * 1.2);
      box-shadow: {shadow_css};
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: var(--title-size); line-height: 1.1; }}
    .hero p {{ margin: 0; font-size: var(--text-size); color: var(--muted); }}
    .noise {{ font-size: calc(var(--text-size) - 2px); opacity: .85; }}
    .quiz-block {{ margin: calc(var(--space) * 1.2) 0; }}
    .question {{ margin: 0 0 calc(var(--space) * .9); font-size: var(--question-size); line-height: 1.2; }}
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
    .select-wrap, .text-wrap {{ margin: calc(var(--space) * .6) 0; }}
    .select-control, .text-control {{
      width: 100%;
      font-size: var(--text-size);
      color: var(--text);
      border-radius: var(--radius);
      border: 1px solid var(--panel-border);
      padding: calc(var(--row-py) - 1px) var(--row-px);
      background: color-mix(in srgb, var(--panel), black 3%);
    }}
    .hint {{ color: var(--muted); font-size: calc(var(--text-size) - 2px); margin: 6px 0 0; }}
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
  </style>
</head>
<body>
  <main class='wrap'>
    <section class='hero'>
      <h1>{title}</h1>
      <p>{desc}</p>
      {"".join(noise_html)}
    </section>
    <section class='quiz-shell' data-role='quiz-shell' data-global-type='{sample["global_type"]}'>
      {blocks_html}
      {next_html}
    </section>
  </main>
</body>
</html>"""


def build_sample(seed: int, index: int) -> Dict[str, Any]:
    rng = random.Random(seed * 1_000_003 + index * 7_919)
    global_type = _weighted_global_type(rng)
    style = _style_pack(rng)
    viewport = _viewport_pack(rng)

    if global_type == "triple":
        base_types = [rng.choice(["single", "multi", "dropdown", "text"]) for _ in range(3)]
        blocks = [_make_block(rng, f"b{i+1}", t) for i, t in enumerate(base_types)]
    elif global_type == "mixed":
        k = rng.randint(3, 5)
        pool = ["single", "multi", "dropdown", "dropdown_scroll", "text"]
        rng.shuffle(pool)
        chosen = pool[: max(2, k)]
        while len(chosen) < k:
            chosen.append(rng.choice(pool))
        blocks = [_make_block(rng, f"b{i+1}", t) for i, t in enumerate(chosen)]
    else:
        blocks = [_make_block(rng, "b1", global_type)]

    has_next = rng.random() < 0.62
    auto_next = (not has_next) and (rng.random() < 0.86)
    require_scroll = bool(any(b["type"] == "dropdown_scroll" for b in blocks) or len(blocks) >= 4)
    if require_scroll and rng.random() < 0.35:
        viewport["height"] = max(620, viewport["height"] - rng.randint(120, 220))

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
        "style": style,
        "viewport": viewport,
        "title": title,
        "description": desc,
    }
    sample["html"] = _render_html(sample, rng)
    return sample


def build_samples(count: int, seed: int) -> List[Dict[str, Any]]:
    return [build_sample(seed=seed, index=i) for i in range(max(1, int(count)))]


def _sample_compact(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "seed": sample["seed"],
        "index": sample["index"],
        "global_type": sample["global_type"],
        "block_types": sample["block_types"],
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
    parser.add_argument("--export-manifest", type=str, default="")
    parser.add_argument("--export-only", action="store_true", help="Only export manifest and exit.")
    args = parser.parse_args()

    samples = build_samples(count=max(1, int(args.count)), seed=int(args.seed))
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
