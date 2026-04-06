from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _persona_key(persona: Mapping[str, Any]) -> str:
    raw = "|".join(
        [
            str(persona.get("persona_id") or persona.get("id") or ""),
            str(persona.get("label") or ""),
            str(persona.get("locale") or ""),
            str(persona.get("site") or ""),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def rotate_persona_profiles(
    *,
    persona_profiles: Sequence[Mapping[str, Any]],
    session_key: str = "",
    prefer_recovery: bool = False,
) -> Dict[str, Any]:
    profiles: List[Dict[str, Any]] = [dict(row) for row in persona_profiles if isinstance(row, Mapping)]
    if not profiles:
        return {
            "selected_persona": {},
            "selected_index": -1,
            "profile_count": 0,
            "reason": "no_personas",
            "rotation_key": "",
        }

    rotation_source = f"{session_key}|{len(profiles)}"
    rotation_key = hashlib.sha1(rotation_source.encode("utf-8", errors="ignore")).hexdigest()
    index = int(rotation_key[:8], 16) % len(profiles)
    if prefer_recovery and len(profiles) > 1:
        index = (index + 1) % len(profiles)

    selected = profiles[index]
    selected.setdefault("persona_key", _persona_key(selected))
    return {
        "selected_persona": selected,
        "selected_index": index,
        "profile_count": len(profiles),
        "reason": "recovery_rotation" if prefer_recovery and len(profiles) > 1 else "deterministic_rotation",
        "rotation_key": rotation_key,
        "selected_persona_key": selected["persona_key"],
    }
