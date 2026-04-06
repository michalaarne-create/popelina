from __future__ import annotations

from typing import Any, Dict, Iterable, List


def rotate_test_account(*, accounts: Iterable[Dict[str, Any]], session_index: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = [dict(row) for row in accounts if isinstance(row, dict)]
    if not rows:
        return {}
    return rows[int(session_index) % len(rows)]


def map_environment_profile(*, environment: str, profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    env = str(environment or "").strip().lower()
    if env in profiles:
        return dict(profiles[env])
    return dict(profiles.get("default") or {})
