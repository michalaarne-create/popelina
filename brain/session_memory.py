"""
Session memory store.

Persists answered questions for the current session:
- append new Q/A entries after every answered cluster
- load/save JSON under data/session_memory/{session_id}.json
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .types import SessionMemoryItem

ROOT_PATH = Path(__file__).resolve().parents[1]
SESSION_MEMORY_DIR = ROOT_PATH / "data" / "session_memory"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class SessionMemoryStore:
    def __init__(self, session_id: str, base_dir: Path = SESSION_MEMORY_DIR):
        self.session_id = session_id
        self.base_dir = Path(base_dir)
        self.path = self.base_dir / f"{session_id}.json"
        self.questions: List[SessionMemoryItem] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for item in data.get("questions", []):
                self.questions.append(SessionMemoryItem(**item))
        except Exception:
            self.questions = []

    def save(self) -> None:
        _ensure_dir(self.base_dir)
        payload = {
            "session_id": self.session_id,
            "questions": [asdict(q) for q in self.questions],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def append(self, item: SessionMemoryItem) -> None:
        if not item.timestamp:
            item.timestamp = datetime.utcnow().isoformat()
        self.questions.append(item)
        self.save()

    def latest_by_canonical_key(self, key: str) -> Optional[SessionMemoryItem]:
        for item in reversed(self.questions):
            if item.canonical_key == key:
                return item
        return None

    def to_history(self) -> List[Dict[str, Any]]:
        return [asdict(q) for q in self.questions]
