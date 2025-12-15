from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.pipeline_brain_agent import BrainDecision


@dataclass
class StoredAnswer:
    question: str
    answer_key: str
    answer_text: str


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_question_text(question_data: Optional[dict]) -> str:
    if not question_data:
        return ""
    main = question_data.get("main_question") or {}
    text = main.get("text") or question_data.get("full_text") or ""
    return str(text).strip()


def _extract_answer_candidates(summary_data: Optional[dict]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Zwraca listę (key, entry) z top_labels, które mają bbox
    i wyglądają na odpowiedzi.
    """
    if not summary_data:
        return []
    top = summary_data.get("top_labels") or {}
    out: List[Tuple[str, Dict[str, Any]]] = []
    for key, entry in top.items():
        if not isinstance(entry, dict):
            continue
        if not entry.get("bbox"):
            continue
        # prosty filtr: preferuj klucze z 'answer' w nazwie
        out.append((str(key), entry))
    return out


def _load_answers_db(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_answers_db(path: Path, db: Dict[str, Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_stored_answer(
    question: str,
    answers_db: Dict[str, Dict[str, str]],
) -> Optional[StoredAnswer]:
    entry = answers_db.get(question.strip())
    if not isinstance(entry, dict):
        return None
    key = entry.get("answer_key")
    text = entry.get("answer_text", "")
    if not key:
        return None
    return StoredAnswer(question=question, answer_key=str(key), answer_text=str(text))


def _pick_candidate_by_key(
    candidates: List[Tuple[str, Dict[str, Any]]],
    key: str,
) -> Optional[Dict[str, Any]]:
    for k, entry in candidates:
        if k == key:
            return entry
    return None


def answer_from_memory_or_random(
    decision: BrainDecision,
    *,
    answers_path: Path,
) -> Optional[Dict[str, Any]]:
    """
    Główne wejście dla main.py.

    - Jeśli pytanie jest w answers.json -> zwróć zapisany entry (dopasowany po key).
    - Jeśli nie -> wybierz losowy entry z dostępnych odpowiedzi, zapisz go w answers.json
      i zwróć ten entry.

    Zwraca słownik kandydata z summary (ten sam format co w top_labels).
    Jeśli nie ma żadnych kandydatów, zwraca None.
    """
    summary_data = decision.summary_data
    question_data = decision.question_data
    question_text = _extract_question_text(question_data)
    candidates = _extract_answer_candidates(summary_data)
    if not candidates:
        return None

    answers_db = _load_answers_db(answers_path)

    stored = _find_stored_answer(question_text, answers_db)
    if stored is not None:
        entry = _pick_candidate_by_key(candidates, stored.answer_key)
        if entry is not None:
            return entry
        # jeśli key nie pasuje do aktualnego summary, potraktuj jak brak pamięci

    # Brak zapisu albo nie udało się dopasować -> losowo
    import random

    key, entry = random.choice(candidates)
    # spróbuj wyciągnąć przydatny tekst odpowiedzi
    text = ""
    try:
        text = str(entry.get("text") or entry.get("label") or "").strip()
    except Exception:
        text = ""

    if question_text:
        answers_db[question_text] = {
            "answer_key": key,
            "answer_text": text,
        }
        _save_answers_db(answers_path, answers_db)

    return entry


__all__ = ["answer_from_memory_or_random"]

