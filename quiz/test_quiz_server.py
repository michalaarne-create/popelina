from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "quiz_submissions.log"
QA_CACHE_PATH = BASE_DIR / "data" / "qa_cache.json"


def _json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def _letters(n: int) -> List[str]:
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out: List[str] = []
    for i in range(n):
        out.append(alpha[i] if i < len(alpha) else f"X{i}")
    return out


def _fingerprint(question_text: str, options_text: Sequence[str], question_type: str, key: str) -> str:
    opts = [(o or "").strip().lower() for o in options_text]
    payload = f"{question_type}\n{question_text.strip()}\n" + "\n".join(sorted(opts)) + f"\n{key}"
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


@dataclass(frozen=True)
class QuizQuestion:
    key: str
    prompt: str
    qtype: str  # single | multi | dropdown | dropdown_scroll | text | triple | mixed
    options: List[str]
    correct: List[str]  # option texts (or text answer for qtype=text)
    has_next: bool
    auto_next: bool
    require_scroll: bool = False


@dataclass(frozen=True)
class QuizType:
    idx: int
    title: str
    slug: str
    description: str
    questions: List[QuizQuestion]


def _q(
    *,
    t_idx: int,
    q_idx: int,
    prompt: str,
    qtype: str,
    options: List[str],
    correct: List[str],
    has_next: bool,
    auto_next: bool,
    require_scroll: bool = False,
) -> QuizQuestion:
    return QuizQuestion(
        key=f"type{t_idx:02d}_q{q_idx:02d}",
        prompt=prompt,
        qtype=qtype,
        options=options,
        correct=correct,
        has_next=has_next,
        auto_next=auto_next,
        require_scroll=require_scroll,
    )


def build_bank() -> List[QuizType]:
    colors = ["Czerwony", "Zielony", "Niebieski", "Żółty"]
    animals = ["Kot", "Pies", "Koń", "Krowa"]
    fruits = ["Jabłko", "Banan", "Truskawka", "Gruszka"]
    days = ["Poniedziałek", "Wtorek", "Środa", "Czwartek"]

    long_opts = [f"Opcja {i}" for i in range(1, 26)]
    long_opts[18] = "Poprawna"

    bank: List[QuizType] = []

    # 1. single + next
    t1 = [
        _q(t_idx=1, q_idx=1, prompt="Jakiego koloru jest trawa?", qtype="single", options=colors, correct=["Zielony"], has_next=True, auto_next=False),
        _q(t_idx=1, q_idx=2, prompt="Które zwierzę miauczy?", qtype="single", options=animals, correct=["Kot"], has_next=True, auto_next=False),
        _q(t_idx=1, q_idx=3, prompt="Jaki owoc jest żółty?", qtype="single", options=fruits, correct=["Banan"], has_next=True, auto_next=False),
        _q(t_idx=1, q_idx=4, prompt="Ile jest 2+2?", qtype="single", options=["3", "4", "5", "6"], correct=["4"], has_next=True, auto_next=False),
        _q(t_idx=1, q_idx=5, prompt="Który dzień jest po poniedziałku?", qtype="single", options=["Niedziela", "Wtorek", "Piątek", "Sobota"], correct=["Wtorek"], has_next=True, auto_next=False),
    ]
    bank.append(QuizType(1, "1) Jednokrotna odpowiedź + Next", "single_next", "Radio + przycisk Next.", t1))

    # 2. multi + next
    t2 = [
        _q(t_idx=2, q_idx=1, prompt="Zaznacz owoc:", qtype="multi", options=["Jabłko", "Marchew", "Chleb", "Ser"], correct=["Jabłko"], has_next=True, auto_next=False),
        _q(t_idx=2, q_idx=2, prompt="Zaznacz zwierzę:", qtype="multi", options=["Kot", "Krzesło", "Stół", "Kubek"], correct=["Kot"], has_next=True, auto_next=False),
        _q(t_idx=2, q_idx=3, prompt="Zaznacz kolor:", qtype="multi", options=["Zielony", "Auto", "Dom", "But"], correct=["Zielony"], has_next=True, auto_next=False),
        _q(t_idx=2, q_idx=4, prompt="Zaznacz liczbę parzystą:", qtype="multi", options=["3", "7", "8", "9"], correct=["8"], has_next=True, auto_next=False),
        _q(t_idx=2, q_idx=5, prompt="Zaznacz coś do picia:", qtype="multi", options=["Woda", "But", "Łóżko", "Kamień"], correct=["Woda"], has_next=True, auto_next=False),
    ]
    bank.append(QuizType(2, "2) Wielokrotna odpowiedź + Next", "multi_next", "Checkboxy + Next.", t2))

    # 3. single no next
    t3 = [
        _q(t_idx=3, q_idx=1, prompt="Który owoc jest czerwony?", qtype="single", options=["Truskawka", "Banan", "Gruszka", "Ogórek"], correct=["Truskawka"], has_next=False, auto_next=True),
        _q(t_idx=3, q_idx=2, prompt="Co świeci w dzień na niebie?", qtype="single", options=["Słońce", "Księżyc", "Gwiazdy", "Latarka"], correct=["Słońce"], has_next=False, auto_next=True),
        _q(t_idx=3, q_idx=3, prompt="Ile jest 5-3?", qtype="single", options=["1", "2", "3", "4"], correct=["2"], has_next=False, auto_next=True),
        _q(t_idx=3, q_idx=4, prompt="Które zwierzę szczeka?", qtype="single", options=["Pies", "Kot", "Ryba", "Żółw"], correct=["Pies"], has_next=False, auto_next=True),
        _q(t_idx=3, q_idx=5, prompt="Jaki kształt ma piłka?", qtype="single", options=["Okrągły", "Kwadratowy", "Trójkątny", "Prostokątny"], correct=["Okrągły"], has_next=False, auto_next=True),
    ]
    bank.append(QuizType(3, "3) Jednokrotna odpowiedź bez Next", "single_auto", "Radio -> auto następne pytanie.", t3))

    # 4. multi no next
    t4 = [
        _q(t_idx=4, q_idx=1, prompt="Zaznacz kolor nieba w dzień:", qtype="multi", options=["Niebieski", "Czerwony", "Czarny", "Zielony"], correct=["Niebieski"], has_next=False, auto_next=True),
        _q(t_idx=4, q_idx=2, prompt="Zaznacz zwierzę domowe:", qtype="multi", options=["Pies", "Samochód", "Telefon", "Książka"], correct=["Pies"], has_next=False, auto_next=True),
        _q(t_idx=4, q_idx=3, prompt="Zaznacz coś słodkiego:", qtype="multi", options=["Cukierek", "Sól", "Pieprz", "Cebula"], correct=["Cukierek"], has_next=False, auto_next=True),
        _q(t_idx=4, q_idx=4, prompt="Zaznacz literę:", qtype="multi", options=["A", "7", "!", "@"], correct=["A"], has_next=False, auto_next=True),
        _q(t_idx=4, q_idx=5, prompt="Zaznacz coś do jedzenia:", qtype="multi", options=["Chleb", "Kamień", "But", "Piłka"], correct=["Chleb"], has_next=False, auto_next=True),
    ]
    bank.append(QuizType(4, "4) Wielokrotna odpowiedź bez Next", "multi_auto", "Checkbox -> auto następne.", t4))

    # 5. dropdown + next
    t5 = [
        _q(t_idx=5, q_idx=1, prompt="Wybierz kolor trawy:", qtype="dropdown", options=colors, correct=["Zielony"], has_next=True, auto_next=False),
        _q(t_idx=5, q_idx=2, prompt="Wybierz zwierzę, które miauczy:", qtype="dropdown", options=animals, correct=["Kot"], has_next=True, auto_next=False),
        _q(t_idx=5, q_idx=3, prompt="Wybierz owoc:", qtype="dropdown", options=fruits, correct=["Jabłko"], has_next=True, auto_next=False),
        _q(t_idx=5, q_idx=4, prompt="Wybierz liczbę 10:", qtype="dropdown", options=["8", "9", "10", "11"], correct=["10"], has_next=True, auto_next=False),
        _q(t_idx=5, q_idx=5, prompt="Wybierz porę dnia:", qtype="dropdown", options=["Rano", "Południe", "Wieczór", "Noc"], correct=["Rano"], has_next=True, auto_next=False),
    ]
    bank.append(QuizType(5, "5) Dropdown z Next", "dropdown_next", "Select + Next.", t5))

    # 6. dropdown no next
    t6 = [
        _q(t_idx=6, q_idx=1, prompt="Wybierz kolor nieba:", qtype="dropdown", options=["Niebieski", "Zielony", "Czerwony", "Czarny"], correct=["Niebieski"], has_next=False, auto_next=True),
        _q(t_idx=6, q_idx=2, prompt="Wybierz zwierzę:", qtype="dropdown", options=["Koń", "Stół", "Mleko", "But"], correct=["Koń"], has_next=False, auto_next=True),
        _q(t_idx=6, q_idx=3, prompt="Wybierz owoc czerwony:", qtype="dropdown", options=["Truskawka", "Ogórek", "Chleb", "Woda"], correct=["Truskawka"], has_next=False, auto_next=True),
        _q(t_idx=6, q_idx=4, prompt="Wybierz 3:", qtype="dropdown", options=["1", "2", "3", "4"], correct=["3"], has_next=False, auto_next=True),
        _q(t_idx=6, q_idx=5, prompt="Wybierz dzień weekendu:", qtype="dropdown", options=["Poniedziałek", "Wtorek", "Sobota", "Środa"], correct=["Sobota"], has_next=False, auto_next=True),
    ]
    bank.append(QuizType(6, "6) Dropdown bez Next", "dropdown_auto", "Select -> auto następne.", t6))

    # 7. dropdown scroll + next
    t7 = [
        _q(t_idx=7, q_idx=1, prompt="Wybierz właściwą opcję (scroll):", qtype="dropdown_scroll", options=long_opts, correct=["Poprawna"], has_next=True, auto_next=False, require_scroll=True),
        _q(t_idx=7, q_idx=2, prompt="Wybierz właściwą opcję (scroll):", qtype="dropdown_scroll", options=long_opts, correct=["Poprawna"], has_next=True, auto_next=False, require_scroll=True),
        _q(t_idx=7, q_idx=3, prompt="Wybierz właściwą opcję (scroll):", qtype="dropdown_scroll", options=long_opts, correct=["Poprawna"], has_next=True, auto_next=False, require_scroll=True),
        _q(t_idx=7, q_idx=4, prompt="Wybierz właściwą opcję (scroll):", qtype="dropdown_scroll", options=long_opts, correct=["Poprawna"], has_next=True, auto_next=False, require_scroll=True),
        _q(t_idx=7, q_idx=5, prompt="Wybierz właściwą opcję (scroll):", qtype="dropdown_scroll", options=long_opts, correct=["Poprawna"], has_next=True, auto_next=False, require_scroll=True),
    ]
    bank.append(QuizType(7, "7) Dropdown wymagający scrolla z Next", "dropdown_scroll_next", "Select(dużo opcji) + Next.", t7))

    # 8. dropdown scroll no next
    t8 = [
        _q(t_idx=8, q_idx=1, prompt="Wybierz właściwą opcję (scroll, auto):", qtype="dropdown_scroll", options=long_opts, correct=["Poprawna"], has_next=False, auto_next=True, require_scroll=True),
        _q(t_idx=8, q_idx=2, prompt="Wybierz właściwą opcję (scroll, auto):", qtype="dropdown_scroll", options=long_opts, correct=["Poprawna"], has_next=False, auto_next=True, require_scroll=True),
        _q(t_idx=8, q_idx=3, prompt="Wybierz właściwą opcję (scroll, auto):", qtype="dropdown_scroll", options=long_opts, correct=["Poprawna"], has_next=False, auto_next=True, require_scroll=True),
        _q(t_idx=8, q_idx=4, prompt="Wybierz właściwą opcję (scroll, auto):", qtype="dropdown_scroll", options=long_opts, correct=["Poprawna"], has_next=False, auto_next=True, require_scroll=True),
        _q(t_idx=8, q_idx=5, prompt="Wybierz właściwą opcję (scroll, auto):", qtype="dropdown_scroll", options=long_opts, correct=["Poprawna"], has_next=False, auto_next=True, require_scroll=True),
    ]
    bank.append(QuizType(8, "8) Dropdown wymagający scrolla bez Next", "dropdown_scroll_auto", "Select(dużo opcji) -> auto.", t8))

    # 9. 3 questions in one requiring scroll + next (we implement as a scrolly page with 3 blocks)
    t9 = [
        _q(t_idx=9, q_idx=1, prompt="(1/3) Wybierz kolor trawy:", qtype="triple", options=colors, correct=["Zielony"], has_next=True, auto_next=False, require_scroll=True),
        _q(t_idx=9, q_idx=2, prompt="(2/3) Wybierz zwierzę domowe:", qtype="triple", options=["Pies", "Krowa", "Koń", "Ryba"], correct=["Pies"], has_next=True, auto_next=False, require_scroll=True),
        _q(t_idx=9, q_idx=3, prompt="(3/3) Wybierz owoc:", qtype="triple", options=fruits, correct=["Jabłko"], has_next=True, auto_next=False, require_scroll=True),
        _q(t_idx=9, q_idx=4, prompt="(1/3) Wybierz dzień:", qtype="triple", options=days, correct=["Wtorek"], has_next=True, auto_next=False, require_scroll=True),
        _q(t_idx=9, q_idx=5, prompt="(2/3) Wybierz liczbę 4:", qtype="triple", options=["3", "4", "5", "6"], correct=["4"], has_next=True, auto_next=False, require_scroll=True),
    ]
    bank.append(QuizType(9, "9) 3 pytania w jednym (scroll) + Next", "triple_scroll_next", "Jedna strona: 3 bloki, trzeba scrollować, potem Next.", t9))

    # 10. 3 questions in one requiring scroll no next
    t10 = [
        _q(t_idx=10, q_idx=1, prompt="(1/3) Wybierz kolor nieba:", qtype="triple", options=["Niebieski", "Zielony", "Czerwony", "Czarny"], correct=["Niebieski"], has_next=False, auto_next=True, require_scroll=True),
        _q(t_idx=10, q_idx=2, prompt="(2/3) Wybierz owoc:", qtype="triple", options=fruits, correct=["Banan"], has_next=False, auto_next=True, require_scroll=True),
        _q(t_idx=10, q_idx=3, prompt="(3/3) Wybierz zwierzę:", qtype="triple", options=animals, correct=["Kot"], has_next=False, auto_next=True, require_scroll=True),
        _q(t_idx=10, q_idx=4, prompt="(1/3) Wybierz dzień:", qtype="triple", options=days, correct=["Środa"], has_next=False, auto_next=True, require_scroll=True),
        _q(t_idx=10, q_idx=5, prompt="(2/3) Wybierz kształt piłki:", qtype="triple", options=["Okrągły", "Kwadratowy", "Trójkątny", "Prostokątny"], correct=["Okrągły"], has_next=False, auto_next=True, require_scroll=True),
    ]
    bank.append(QuizType(10, "10) 3 pytania w jednym (scroll) bez Next", "triple_scroll_auto", "Jedna strona: 3 bloki, auto po 3 odpowiedzi.", t10))

    # 11. typing + next
    t11 = [
        _q(t_idx=11, q_idx=1, prompt="Wpisz imię: Ala", qtype="text", options=[], correct=["Ala"], has_next=True, auto_next=False),
        _q(t_idx=11, q_idx=2, prompt="Wpisz wynik 1+1", qtype="text", options=[], correct=["2"], has_next=True, auto_next=False),
        _q(t_idx=11, q_idx=3, prompt="Wpisz kolor nieba: niebieski", qtype="text", options=[], correct=["niebieski"], has_next=True, auto_next=False),
        _q(t_idx=11, q_idx=4, prompt="Wpisz literę A", qtype="text", options=[], correct=["A"], has_next=True, auto_next=False),
        _q(t_idx=11, q_idx=5, prompt="Wpisz 'tak'", qtype="text", options=[], correct=["tak"], has_next=True, auto_next=False),
    ]
    bank.append(QuizType(11, "11) Wpisywanie klawiaturą z Next", "text_next", "Input + Next.", t11))

    # 12. typing no next
    t12 = [
        _q(t_idx=12, q_idx=1, prompt="Wpisz 3", qtype="text", options=[], correct=["3"], has_next=False, auto_next=True),
        _q(t_idx=12, q_idx=2, prompt="Wpisz 'kot'", qtype="text", options=[], correct=["kot"], has_next=False, auto_next=True),
        _q(t_idx=12, q_idx=3, prompt="Wpisz 'woda'", qtype="text", options=[], correct=["woda"], has_next=False, auto_next=True),
        _q(t_idx=12, q_idx=4, prompt="Wpisz 'czerwony'", qtype="text", options=[], correct=["czerwony"], has_next=False, auto_next=True),
        _q(t_idx=12, q_idx=5, prompt="Wpisz 9-4", qtype="text", options=[], correct=["5"], has_next=False, auto_next=True),
    ]
    bank.append(QuizType(12, "12) Wpisywanie klawiaturą bez Next", "text_auto", "Input -> auto.", t12))

    # 13. mixed survey
    t13 = [
        _q(t_idx=13, q_idx=1, prompt="(MIX) Wybierz owoc:", qtype="single", options=fruits, correct=["Jabłko"], has_next=True, auto_next=False, require_scroll=True),
        _q(t_idx=13, q_idx=2, prompt="(MIX) Wybierz kolor:", qtype="dropdown", options=colors, correct=["Niebieski"], has_next=True, auto_next=False, require_scroll=True),
        _q(t_idx=13, q_idx=3, prompt="(MIX) Zaznacz zwierzę:", qtype="multi", options=["Kot", "Stół", "But", "Telefon"], correct=["Kot"], has_next=True, auto_next=False, require_scroll=True),
        _q(t_idx=13, q_idx=4, prompt="(MIX) Wybierz dzień:", qtype="dropdown_scroll", options=long_opts, correct=["Poprawna"], has_next=True, auto_next=False, require_scroll=True),
        _q(t_idx=13, q_idx=5, prompt="(MIX) Wpisz 6-2", qtype="text", options=[], correct=["4"], has_next=True, auto_next=False, require_scroll=True),
    ]
    bank.append(QuizType(13, "13) Mieszana ankieta na jednej stronie", "mixed", "Różne typy na jednej stronie.", t13))

    return bank


BANK = build_bank()


def ensure_qa_cache() -> None:
    QA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if QA_CACHE_PATH.exists():
        try:
            existing = json.loads(QA_CACHE_PATH.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            existing = {}

    items: Dict[str, dict] = existing.get("items", {}) if isinstance(existing.get("items"), dict) else {}

    for qt in BANK:
        for qq in qt.questions:
            if qq.key in items:
                continue
            opts = qq.options or []
            letters = _letters(len(opts))
            options_text = {lab: txt for lab, txt in zip(letters, opts)}
            selected_letters: List[str] = []
            if qq.correct and opts:
                for lab, txt in options_text.items():
                    if txt in set(qq.correct):
                        selected_letters.append(lab)
            rec: Dict[str, Any] = {
                "question_text": qq.prompt,
                "options_text": options_text,
                "question_type": ("text" if qq.qtype == "text" else qq.qtype),
                "selected_options": selected_letters if selected_letters else (["A"] if opts else []),
                "correct_answer": (qq.correct[0] if qq.correct else ""),
                "fingerprint": _fingerprint(qq.prompt, opts, qq.qtype, qq.key),
            }
            if qq.qtype == "text" and qq.correct:
                rec["text_answer"] = qq.correct[0]
            items[qq.key] = rec

    payload = {"version": 1, "items": items}
    QA_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _html(title: str, body: str) -> bytes:
    css = """
    body{font-family:system-ui,Segoe UI,Arial;background:#0b0f14;color:#e6edf3;margin:0}
    a{color:#7dd3fc}
    .wrap{max-width:980px;margin:0 auto;padding:18px}
    .card{background:#111826;border:1px solid #263245;border-radius:12px;padding:16px;margin:12px 0}
    .title{font-size:22px;font-weight:700;margin:0 0 8px}
    .desc{color:#a8b3c2;margin:0 0 12px}
    .btn{display:inline-block;background:#2563eb;color:#fff;border:0;padding:10px 14px;border-radius:10px;cursor:pointer;font-weight:600;text-decoration:none}
    .btn.secondary{background:#334155}
    .q{font-size:20px;font-weight:700;margin:0 0 10px}
    .opt{display:flex;align-items:center;gap:10px;padding:10px 12px;border:1px solid #233044;border-radius:10px;margin:8px 0;background:#0f1623}
    .opt label{cursor:pointer;width:100%}
    select.opt,input.opt{width:100%}
    """
    page = f"<!doctype html><html lang='pl'><head><meta charset='utf-8'/>" \
           f"<meta name='viewport' content='width=device-width, initial-scale=1'/>" \
           f"<title>{title}</title><style>{css}</style></head><body><div class='wrap'>{body}</div></body></html>"
    return page.encode("utf-8")


def _index_page() -> bytes:
    blocks = [
        "<div class='card'><div class='title'>Test Quiz Server</div>"
        "<div class='desc'>Podział na typy pytań (13 kategorii, po 5 pytań). "
        "Każda kategoria w tej samej kolejności jak w wymaganiach.</div></div>"
    ]
    for qt in BANK:
        blocks.append(
            "<div class='card'>"
            f"<div class='title'>{qt.title}</div>"
            f"<div class='desc'>{qt.description}</div>"
            f"<a class='btn' href='/t/{qt.idx}/1'>Start</a> "
            f"<a class='btn secondary' href='/t/{qt.idx}/1?reset=1'>Reset</a>"
            "</div>"
        )
    blocks.append("<div class='card'><div class='desc'>POST /api/submit loguje do logs/quiz_submissions.log</div></div>")
    return _html("Test Quiz Server", "\n".join(blocks))


def _question_page(qt: QuizType, qidx: int, reset: bool) -> bytes:
    qidx = max(1, min(qidx, len(qt.questions)))
    qq = qt.questions[qidx - 1]
    next_url = f"/t/{qt.idx}/{qidx + 1}" if qidx < len(qt.questions) else "/"

    head = (
        "<div class='card'>"
        f"<div class='title'>{qt.title}</div>"
        f"<div class='desc'>{qt.description} | Pytanie {qidx}/5</div>"
        "<a class='btn secondary' href='/'>Home</a>"
        "</div>"
    )

    qblock = [f"<div class='card'><div class='q'>{qq.prompt}</div>"]
    qblock.append(f"<div id='qid' style='display:none'>{qq.key}</div>")

    if qq.qtype in ("single", "multi", "triple", "mixed"):
        itype = "radio" if qq.qtype in ("single", "triple") else ("checkbox" if qq.qtype == "multi" else "radio")
        name = f"q_{qq.key}"
        for i, opt in enumerate(qq.options):
            oid = f"opt_{i}"
            qblock.append(
                f"<div class='opt'><input id='{oid}' type='{itype}' name='{name}' value='{opt}' />"
                f"<label for='{oid}'>{opt}</label></div>"
            )
        if qq.require_scroll:
            qblock.append("<div style='height:700px'></div>")
    elif qq.qtype in ("dropdown", "dropdown_scroll"):
        size_attr = " size='8'" if qq.qtype == "dropdown_scroll" else ""
        opts = "".join([f"<option value='{o}'>{o}</option>" for o in [""] + qq.options])
        qblock.append(f"<select id='sel' class='opt'{size_attr}>{opts}</select>")
    elif qq.qtype == "text":
        qblock.append("<input id='txt' class='opt' placeholder='Wpisz odpowiedź...' />")

    if qq.has_next:
        qblock.append("<div style='margin-top:14px'>")
        qblock.append("<button id='next' class='btn'>Następne ➡️</button>")
        qblock.append("</div>")

    qblock.append("</div>")

    js = f"""
    <script>
    const qid = {json.dumps(qq.key, ensure_ascii=False)};
    const nextUrl = {json.dumps(next_url, ensure_ascii=False)};
    const hasNext = {str(bool(qq.has_next)).lower()};
    const autoNext = {str(bool(qq.auto_next)).lower()};
    const qtype = {json.dumps(qq.qtype, ensure_ascii=False)};

    async function submit(selected) {{
      try {{
        await fetch('/api/submit', {{
          method: 'POST',
          headers: {{'Content-Type':'application/json'}},
          body: JSON.stringify({{qid, qtype, selected, ts: Date.now()}})
        }});
      }} catch (e) {{}}
    }}

    function goNext() {{
      window.location.href = nextUrl;
    }}

    function wireInputs() {{
      const inputs = Array.from(document.querySelectorAll('input[type=radio], input[type=checkbox]'));
      if (!inputs.length) return;
      inputs.forEach(inp => {{
        inp.addEventListener('change', () => {{
          if (inp.type === 'radio') {{
            submit([inp.value]);
            if (autoNext && !hasNext) setTimeout(goNext, 150);
          }} else {{
            const checked = inputs.filter(x=>x.checked).map(x=>x.value);
            submit(checked);
            if (autoNext && !hasNext && checked.length) setTimeout(goNext, 150);
          }}
        }});
      }});
    }}

    function wireSelect() {{
      const sel = document.getElementById('sel');
      if (!sel) return;
      sel.addEventListener('change', () => {{
        if (sel.value) {{
          submit([sel.value]);
          if (autoNext && !hasNext) setTimeout(goNext, 150);
        }}
      }});
    }}

    function wireText() {{
      const el = document.getElementById('txt');
      if (!el) return;
      el.addEventListener('keydown', (ev) => {{
        if (ev.key === 'Enter') {{
          submit([el.value]);
          if (autoNext && !hasNext) setTimeout(goNext, 150);
        }}
      }});
    }}

    function wireNext() {{
      const btn = document.getElementById('next');
      if (!btn) return;
      btn.addEventListener('click', () => {{
        submit(['NEXT_CLICK']);
        goNext();
      }});
    }}

    if ({str(bool(reset)).lower()}) {{
      try {{ localStorage.clear(); }} catch(e){{}}
    }}

    wireInputs();
    wireSelect();
    wireText();
    wireNext();
    </script>
    """

    return _html(f"{qt.title} - {qidx}/5", head + "\n" + "\n".join(qblock) + js)


class QuizRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _send(self, body: bytes, *, status: int = 200, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: Any, status: int = 200) -> None:
        self._send(_json_bytes(data), status=status, content_type="application/json; charset=utf-8")

    def _read_body(self) -> Tuple[bytes, str]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b""
        content_type = self.headers.get("Content-Type", "")
        return body, content_type

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query or "")
        reset = bool(qs.get("reset"))

        if path in ("/", "/index", "/index.html"):
            return self._send(_index_page())

        m = re.match(r"^/t/(\d+)/(\d+)$", path)
        if m:
            t_idx = int(m.group(1))
            q_idx = int(m.group(2))
            qt = next((t for t in BANK if t.idx == t_idx), None)
            if qt is None:
                return self._send(_html("404", "<div class='card'>Not found</div>"), status=404)
            return self._send(_question_page(qt, q_idx, reset=reset))

        return self._send(_html("404", "<div class='card'>Not found</div>"), status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/submit":
            return self._send_json({"error": "Not found"}, status=404)

        body, content_type = self._read_body()
        try:
            if "application/json" in content_type:
                data = json.loads(body.decode("utf-8") or "{}")
            else:
                data = {"raw": body.decode("utf-8", errors="replace")}
        except Exception:
            data = {"raw": body.decode("utf-8", errors="replace")}

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except OSError:
            pass

        return self._send_json({"status": "ok"})


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    ensure_qa_cache()
    server_address = (host, port)
    httpd = HTTPServer(server_address, QuizRequestHandler)
    print(f"Quiz server running at http://{host}:{port}")
    print("Home: http://127.0.0.1:8000/")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run()
