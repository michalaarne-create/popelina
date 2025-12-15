"""
Brain package bootstrap for the Answer-Bot pipeline.

This package groups the higher-level reasoning layers described in the
architecture draft:
- fusion.py: merge DOM + rated screen elements into unified Element objects
- questions.py: build QuestionCluster structures from fused elements
- session_memory.py / profile_facts.py: lightweight persistence helpers
- brain.py: orchestrator that decides what to do on a given screen
- qa_gateway.py / context_selector.py: prompt/context preparation and caching
- actions.py: translate plans into concrete cursor/keyboard actions

Each module keeps logic small and testable; only shared data shapes live in
brain.types.
"""

from .types import (
    BBox,
    Element,
    QuestionOption,
    QuestionCluster,
    BrainDecision,
    BrainMode,
    QARequest,
    QAResult,
    SessionMemoryItem,
)

__all__ = [
    "BBox",
    "Element",
    "QuestionOption",
    "QuestionCluster",
    "BrainDecision",
    "BrainMode",
    "QARequest",
    "QAResult",
    "SessionMemoryItem",
]
