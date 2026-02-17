from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def load_json(path: Path, logger: Optional[Callable[[str], None]] = None) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        if callable(logger):
            logger(f"Could not read JSON {path}.")
        return None


def load_state(path: Path, logger: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        if callable(logger):
            logger(f"Could not read brain state {path}.")
        return {}


def save_state(path: Path, state: Dict[str, Any], logger: Optional[Callable[[str], None]] = None) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        if callable(logger):
            logger(f"Failed to save brain state: {exc}")

