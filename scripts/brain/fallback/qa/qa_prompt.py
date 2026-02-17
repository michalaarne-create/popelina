from __future__ import annotations

import hashlib
import string
from typing import Dict, List

from ...shared.types import QuestionCluster


def make_question_fingerprint(question_text: str, options_text: List[str], question_type: str, canonical_key: str = "") -> str:
    normalized_opts = [(opt or "").strip().lower() for opt in options_text]
    payload = f"{question_type}\n{question_text.strip()}\n" + "\n".join(sorted(normalized_opts))
    if canonical_key:
        payload += f"\n{canonical_key}"
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def build_label_mapping(options) -> Dict[str, str]:
    letters = list(string.ascii_uppercase)
    mapping: Dict[str, str] = {}
    for idx, opt in enumerate(options):
        label = letters[idx] if idx < len(letters) else f"X{idx}"
        mapping[label] = opt.id
    return mapping


def build_prompt(cluster: QuestionCluster, labels_map: Dict[str, str], ctx: Dict) -> str:
    lines: List[str] = []
    intro_key = cluster.topic_tags[0] if cluster.topic_tags else "general"
    lines.append(f"[INTRO::{intro_key}] Answer the webform question. Return JSON via `{{\"selected\": [\"A\"]}}`.")

    if cluster.ui_mode == "dropdown":
        lines.append("[UI] Control type: dropdown/select. Choose the option(s) that should be selected, not typed text.")

    if ctx.get("facts_snippet"):
        lines.append("")
        lines.append(ctx["facts_snippet"])

    if ctx.get("topic_history"):
        lines.append("")
        lines.append("Related questions:")
        for i, item in enumerate(ctx["topic_history"]):
            a_text = ", ".join(item.get("selected") or [])
            lines.append(f"Q{i}: {item.get('question')}")
            lines.append(f"A{i}: {a_text}")

    lines.append("")
    q_type_label = "single choice" if cluster.type == "single" else "multi choice"
    lines.append(f"Current question ({q_type_label}):")
    lines.append(cluster.question_text)
    lines.append("Options:")
    for label, opt_id in labels_map.items():
        opt = next((o for o in cluster.options if o.id == opt_id), None)
        if opt is None:
            continue
        lines.append(f"{label}: {opt.text}")
    lines.append("")
    lines.append('Return only JSON: {"selected": ["A"]}')
    return "\n".join(lines)

