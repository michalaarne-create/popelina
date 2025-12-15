"""
High-level brain stepper.
Decides what to do on the current screen using fused elements, question clusters,
session memory, and profile facts.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .types import (
    BrainDecision,
    BrainMode,
    Element,
    QuestionCluster,
)
from .session_memory import SessionMemoryStore

_SNAPSHOT_ROOT = Path(__file__).resolve().parent
_CURRENT_SNAPSHOT_FILE = _SNAPSHOT_ROOT / "current_question_and_answers.json"
_DEBUG_SNAPSHOT_DIR = _SNAPSHOT_ROOT / "debug"


@dataclass
class BrainConfig:
    min_answer_score: float = 0.35
    scroll_step_px: int = 720
    scroll_on_empty: bool = True
    scroll_on_all_answered: bool = True


class Brain:
    def __init__(self, session_memory: SessionMemoryStore, cfg: Optional[BrainConfig] = None):
        self.session_memory = session_memory
        self.cfg = cfg or BrainConfig()
        self._current_snapshot_file = _CURRENT_SNAPSHOT_FILE
        self._debug_dir = _DEBUG_SNAPSHOT_DIR
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_counter = self._init_snapshot_counter()

    def step(
        self,
        elements: List[Element],
        clusters: List[QuestionCluster],
        specials: Dict[str, Optional[Element]],
    ) -> BrainDecision:
        answered_keys = self._answered_keys()
        screen_type = self._detect_screen_type(clusters, specials)
        if screen_type == "cookies":
            target = specials.get("cookie_reject") or specials.get("cookie_accept")
            return BrainDecision(mode="COOKIES", target_element=target, reason="Cookie banner present")

        if screen_type == "only_next":
            return BrainDecision(mode="CLICK_NEXT", target_element=specials.get("next"), reason="No questions, only Next")

        if screen_type == "question_screen":
            cluster = self._pick_cluster(clusters, answered_keys)
            if cluster:
                src = "memory" if cluster.canonical_key and cluster.canonical_key in answered_keys else "new"
                self._record_question_snapshot(cluster, "Answer top-priority cluster", src)
                return BrainDecision(
                    mode="ANSWER_QUESTION",
                    cluster=cluster,
                    reason="Answer top-priority cluster",
                    extras={"answer_source": src},
                )
            should_scroll, scroll_reason = self._should_scroll(clusters, answered_keys, specials)
            if should_scroll:
                return BrainDecision(
                    mode="SCROLL",
                    reason=scroll_reason,
                    extras={"direction": "down", "amount": self.cfg.scroll_step_px},
                )

        should_scroll, scroll_reason = self._should_scroll(clusters, answered_keys, specials)
        if should_scroll:
            return BrainDecision(
                mode="SCROLL",
                reason=scroll_reason,
                extras={"direction": "down", "amount": self.cfg.scroll_step_px},
            )

        return BrainDecision(mode="NOOP", reason="No actionable items detected")

    def _detect_screen_type(
        self,
        clusters: List[QuestionCluster],
        specials: Dict[str, Optional[Element]],
    ) -> str:
        if specials.get("cookie_accept") or specials.get("cookie_reject"):
            return "cookies"
        if not clusters and specials.get("next"):
            return "only_next"
        if clusters:
            return "question_screen"
        return "other"

    def _pick_cluster(self, clusters: List[QuestionCluster], answered_keys: Set[str]) -> Optional[QuestionCluster]:
        candidates: List[tuple] = []
        for cl in clusters:
            if cl.canonical_key and cl.canonical_key in answered_keys:
                continue
            y_top = cl.question_bbox[1] if cl.question_bbox else 0
            candidates.append((y_top, len(cl.options), cl))
        if not candidates:
            return None
        candidates.sort(key=lambda t: (t[0], t[1]))
        return candidates[0][2]

    def _should_scroll(
        self,
        clusters: List[QuestionCluster],
        answered_keys: Set[str],
        specials: Dict[str, Optional[Element]],
    ) -> Tuple[bool, str]:
        unanswered = [cl for cl in clusters if not (cl.canonical_key and cl.canonical_key in answered_keys)]
        if self.cfg.scroll_on_empty and not clusters and not specials.get("next"):
            return True, "No questions detected; scroll to search for content"
        if self.cfg.scroll_on_all_answered and clusters and not unanswered and not specials.get("next"):
            return True, "All detected clusters already answered; scroll for new quiz content"
        return False, ""

    def _answered_keys(self) -> Set[str]:
        return {item.canonical_key for item in self.session_memory.questions if item.canonical_key}

    def _init_snapshot_counter(self) -> int:
        max_idx = 0
        try:
            for path in self._debug_dir.glob("question_and_answers_*.json"):
                stem = path.stem.split("_")[-1]
                if stem.isdigit():
                    max_idx = max(max_idx, int(stem))
        except Exception:
            pass
        return max_idx + 1

    def _record_question_snapshot(self, cluster: QuestionCluster, reason: str, answer_source: Optional[str]) -> None:
        snapshot: Dict[str, Any] = {
            "recorded_at": datetime.utcnow().isoformat(),
            "reason": reason,
            "answer_source": answer_source,
            "cluster": {
                "id": cluster.id,
                "type": cluster.type,
                "ui_mode": cluster.ui_mode,
                "question_text": cluster.question_text,
                "question_bbox": list(cluster.question_bbox) if cluster.question_bbox else None,
                "canonical_key": cluster.canonical_key,
                "topic_tags": cluster.topic_tags,
                "metadata": cluster.metadata,
            },
            "options": [],
        }
        for opt in cluster.options:
            snapshot["options"].append(
                {
                    "id": opt.id,
                    "text": opt.text,
                    "selected": opt.selected,
                    "ranked_score": opt.ranked_score,
                    "bbox": list(opt.bbox) if opt.bbox else None,
                    "dom": opt.dom,
                    "rated": opt.rated,
                    "metadata": opt.metadata,
                }
            )

        payload = json.dumps(snapshot, ensure_ascii=False, indent=2, default=str)
        try:
            self._current_snapshot_file.write_text(payload, encoding="utf-8")
        except Exception:
            pass

        debug_path = self._debug_dir / f"question_and_answers_{self._snapshot_counter}.json"
        try:
            debug_path.write_text(payload, encoding="utf-8")
            self._snapshot_counter += 1
        except Exception:
            pass
