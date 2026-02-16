"""
Context selector for QA Gateway.

Chooses the smallest useful subset of session memory and profile facts to feed
into LLM prompts.
"""
from __future__ import annotations

from typing import Dict, List, Any

from .types import QuestionCluster
from .session_memory import SessionMemoryStore
from .profile_facts import ProfileFactsStore


def select_context(
    cluster: QuestionCluster,
    session_memory: SessionMemoryStore,
    profile_facts: ProfileFactsStore,
    max_questions: int = 5,
) -> Dict[str, Any]:
    topic_tags = set(cluster.topic_tags or [])
    history = session_memory.questions
    if topic_tags:
        filtered = [q for q in history if topic_tags.intersection(q.topic_tags or [])]
    else:
        filtered = history

    recent = filtered[-max_questions:]
    topic_history = [
        {"question": item.question_text, "selected": item.selected_options, "facts": item.facts}
        for item in recent
    ]

    facts_snippet = format_facts(profile_facts.facts)
    return {
        "facts_snippet": facts_snippet,
        "topic_history": topic_history,
    }


def format_facts(facts: Dict[str, Any]) -> str:
    if not facts:
        return ""
    lines = ["Known facts:"]
    for key, val in facts.items():
        lines.append(f"- {key}: {val}")
    return "\n".join(lines)
