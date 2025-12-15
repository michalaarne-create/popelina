"""
Profile facts store.

Keeps lightweight key/value facts derived from answered questions so they can
be injected into prompts later.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from .types import SessionMemoryItem

ROOT_PATH = Path(__file__).resolve().parents[1]
PROFILE_FACTS_DIR = ROOT_PATH / "data" / "profile_facts"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class ProfileFactsStore:
    def __init__(self, session_id: str, base_dir: Path = PROFILE_FACTS_DIR):
        self.session_id = session_id
        self.base_dir = Path(base_dir)
        self.path = self.base_dir / f"{session_id}.json"
        self.facts: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self.facts = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.facts = {}

    def save(self) -> None:
        _ensure_dir(self.base_dir)
        self.path.write_text(json.dumps(self.facts, ensure_ascii=False, indent=2), encoding="utf-8")

    def update_from_memory_item(self, item: SessionMemoryItem) -> None:
        if not item.facts:
            return
        self.facts.update(item.facts)
        self.save()

    def update(self, new_facts: Dict[str, Any]) -> None:
        if not new_facts:
            return
        self.facts.update(new_facts)
        self.save()

