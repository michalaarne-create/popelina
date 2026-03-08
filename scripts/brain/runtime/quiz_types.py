from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QuizAction:
    kind: str
    bbox: Optional[List[float]] = None
    text: Optional[str] = None
    combo: Optional[str] = None
    repeat: int = 1
    direction: Optional[str] = None
    amount: Optional[int] = None
    scroll_region_bbox: Optional[List[float]] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResolvedQuizAnswer:
    matched: bool
    question_key: Optional[str]
    question_type: str
    correct_answers: List[str]
    option_indexes: List[int] = field(default_factory=list)
    source: str = "unknown"
    confidence: float = 0.0
    fingerprint: Optional[str] = None
    question_text: str = ""
    normalized_question_text: str = ""
    cache_item: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
