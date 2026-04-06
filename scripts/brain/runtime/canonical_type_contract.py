from __future__ import annotations

from typing import Any, Dict

from scripts.pipeline.contracts import SCHEMA_VERSION


def canonical_operational_type(quiz_type: str, *, fallback: str = "choice") -> str:
    t = str(quiz_type or "").strip().lower()
    if t == "text":
        return "text"
    if t == "autocomplete":
        return "autocomplete"
    if t in {"masked_input", "phone", "date", "postal", "zip", "zipcode", "postal_code"}:
        return "masked_input"
    if t == "dropdown_scroll":
        return "dropdown_scroll"
    if t == "dropdown":
        return "dropdown"
    if t in {"slider", "scale", "rating"}:
        return "slider"
    if t == "matrix":
        return "matrix"
    if t in {"single", "multi", "triple", "mixed", "choice"}:
        return "choice"
    return str(fallback or "choice").strip().lower() or "choice"


def build_canonical_type_contract(
    *,
    detected_quiz_type: str,
    control_kind: str = "",
    active_block_type: str = "",
    source: str = "",
) -> Dict[str, Any]:
    quiz_type = str(detected_quiz_type or "").strip().lower() or str(control_kind or "").strip().lower() or "single"
    operational_type = canonical_operational_type(quiz_type)
    return {
        "schema_version": SCHEMA_VERSION,
        "quiz_type": quiz_type,
        "operational_type": operational_type,
        "control_kind": str(control_kind or "").strip().lower(),
        "active_block_type": str(active_block_type or "").strip().lower(),
        "source": str(source or "").strip() or "runtime",
    }
