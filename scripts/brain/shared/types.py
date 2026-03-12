"""
Shared data structures for the brain pipeline.

The goal is to keep the rest of the modules focused on logic while the shapes
of the inputs/outputs are centralized here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal

# Basic geometry
BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

# Brain decisions
BrainMode = Literal["COOKIES", "CLICK_NEXT", "ANSWER_QUESTION", "HIGHLIGHT", "SCROLL", "NOOP"]


@dataclass
class Element:
    """Unified UI element after fusing DOM and rating outputs."""

    id: str
    bbox: BBox
    text_dom: Optional[str] = None
    text_ocr: Optional[str] = None
    dom: Dict[str, Any] = field(default_factory=dict)
    rated: Dict[str, Any] = field(default_factory=dict)
    scores_fused: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bbox": list(self.bbox),
            "text_dom": self.text_dom,
            "text_ocr": self.text_ocr,
            "dom": self.dom,
            "rated": self.rated,
            "scores_fused": self.scores_fused,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Element":
        return cls(
            id=str(data.get("id")),
            bbox=tuple(data.get("bbox", (0, 0, 0, 0))),  # type: ignore[arg-type]
            text_dom=data.get("text_dom"),
            text_ocr=data.get("text_ocr"),
            dom=data.get("dom") or {},
            rated=data.get("rated") or {},
            scores_fused=data.get("scores_fused") or {},
            metadata=data.get("metadata") or {},
        )


@dataclass
class QuestionOption:
    id: str
    bbox: Optional[BBox]
    text: str
    selected: bool = False
    ranked_score: float = 0.0
    dom: Dict[str, Any] = field(default_factory=dict)
    rated: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bbox": list(self.bbox) if self.bbox else None,
            "text": self.text,
            "selected": self.selected,
            "ranked_score": self.ranked_score,
            "dom": self.dom,
            "rated": self.rated,
            "metadata": self.metadata,
        }


@dataclass
class QuestionCluster:
    id: str
    type: Literal["single", "multi"]
    question_bbox: Optional[BBox]
    question_text: str
    ui_mode: Literal["standard", "dropdown", "unknown"] = "standard"
    options: List[QuestionOption] = field(default_factory=list)
    topic_tags: List[str] = field(default_factory=list)
    canonical_key: Optional[str] = None
    source_element_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "ui_mode": self.ui_mode,
            "question_bbox": list(self.question_bbox) if self.question_bbox else None,
            "question_text": self.question_text,
            "options": [opt.to_dict() for opt in self.options],
            "topic_tags": self.topic_tags,
            "canonical_key": self.canonical_key,
            "source_element_id": self.source_element_id,
            "metadata": self.metadata,
        }


@dataclass
class BrainDecision:
    mode: BrainMode
    cluster: Optional[QuestionCluster] = None
    target_element: Optional[Element] = None
    reason: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "cluster": self.cluster.to_dict() if self.cluster else None,
            "target_element": self.target_element.to_dict() if self.target_element else None,
            "reason": self.reason,
            "extras": self.extras,
        }


@dataclass
class QARequest:
    """Payload prepared for LLM/gateway."""

    cache_key: str
    cluster_id: str
    canonical_key: str
    labels_to_option_ids: Dict[str, str]
    prompt: str
    question_type: str = "single"
    options_text: Dict[str, str] = field(default_factory=dict)
    question_text: Optional[str] = None
    facts_snippet: str = ""
    topic_history: List[Dict[str, str]] = field(default_factory=list)
    target_model_hint: Optional[str] = None


@dataclass
class QAResult:
    selected_option_ids: List[str]
    raw_model_output: Optional[str] = None
    cache_hit: bool = False


@dataclass
class SessionMemoryItem:
    cluster_id: str
    canonical_key: str
    question_text: str
    topic_tags: List[str]
    selected_options: List[str]
    facts: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
