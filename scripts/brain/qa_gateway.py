"""
QA Gateway 2.0

Responsible for:
- cache lookup by canonical question key
- context selection (session memory + profile facts)
- prompt assembly for the chosen LLM
- recording model answers back into cache + memory + profile facts
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .context_selector import select_context
from .profile_facts import ProfileFactsStore
from .qa_prompt import build_label_mapping, build_prompt, make_question_fingerprint
from .session_memory import SessionMemoryStore
from .types import (
    QARequest,
    QAResult,
    QuestionCluster,
    SessionMemoryItem,
)

ROOT_PATH = Path(__file__).resolve().parents[1]
DEFAULT_QA_CACHE = ROOT_PATH / "data" / "qa_cache.json"


class QACache:
    def __init__(self, path: Path = DEFAULT_QA_CACHE):
        self.path = path
        self.items: Dict[str, Dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.items = data.get("items", {})
        except Exception:
            self.items = {}

    def save(self) -> None:
        payload = {"version": 1, "items": self.items}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, key: str) -> Optional[Dict]:
        return self.items.get(key)

    def find_by_question_text(self, question_text: str) -> Optional[Dict]:
        """Find cached answer by question text (fuzzy match)."""
        if not question_text:
            return None
        q_lower = question_text.strip().lower()
        for key, record in self.items.items():
            cached_q = (record.get("question_text") or "").strip().lower()
            if cached_q and (cached_q == q_lower or q_lower in cached_q or cached_q in q_lower):
                return record
        return None

    def put(self, key: str, record: Dict) -> None:
        self.items[key] = record
        self.save()


class QAGateway:
    def __init__(
        self,
        session_memory: SessionMemoryStore,
        profile_facts: ProfileFactsStore,
        cache: Optional[QACache] = None,
    ):
        self.session_memory = session_memory
        self.profile_facts = profile_facts
        self.cache = cache or QACache()

    def maybe_from_cache(self, cluster: QuestionCluster) -> Optional[QAResult]:
        # First try canonical_key lookup
        cached = None
        if cluster.canonical_key:
            cached = self.cache.get(cluster.canonical_key)
        
        # Fallback: search by question text
        if not cached:
            cached = self.cache.find_by_question_text(cluster.question_text or "")
        
        if not cached:
            return None
        
        # If we have selected_options as letter labels (A, B, C...), map them to element IDs
        selected = cached.get("selected_options", [])
        if selected and cluster.options:
            # Check if selected contains letter labels like ["A", "B", "C"]
            if all(isinstance(s, str) and len(s) <= 2 and s[0].isupper() for s in selected):
                # Map letters to option IDs
                option_ids = []
                for label in selected:
                    idx = ord(label[0]) - ord('A')
                    if 0 <= idx < len(cluster.options):
                        option_ids.append(cluster.options[idx].id)
                selected = option_ids if option_ids else selected
        
        # For text questions, try to match correct_answer with options
        if cached.get("question_type") == "text" and cached.get("text_answer"):
            text_answer = cached.get("text_answer", "")
            # For text inputs, we might need to return the text directly
            return QAResult(
                selected_option_ids=[],
                raw_model_output=text_answer,
                cache_hit=True,
                extras={"text_answer": text_answer, "correct_answer": cached.get("correct_answer")}
            )
        
        return QAResult(
            selected_option_ids=selected,
            raw_model_output=None,
            cache_hit=True,
        )

    def build_request(self, cluster: QuestionCluster) -> QARequest:
        ctx = select_context(cluster, self.session_memory, self.profile_facts)
        labels_map = build_label_mapping(cluster.options)
        options_text = {lbl: opt.text for lbl, opt in zip(labels_map.keys(), cluster.options)}
        prompt = build_prompt(cluster, labels_map, ctx)
        canonical_key = cluster.canonical_key or f"cluster_{cluster.id}"
        return QARequest(
            cache_key=canonical_key,
            cluster_id=cluster.id,
            canonical_key=canonical_key,
            labels_to_option_ids=labels_map,
            prompt=prompt,
            question_type=cluster.type,
            options_text=options_text,
            question_text=cluster.question_text,
            facts_snippet=ctx["facts_snippet"],
            topic_history=ctx["topic_history"],
        )

    def record_result(
        self,
        request: QARequest,
        selected_labels: Sequence[str],
        raw_model_output: Optional[str] = None,
        inferred_facts: Optional[Dict[str, str]] = None,
    ) -> QAResult:
        option_ids = [request.labels_to_option_ids.get(lbl) for lbl in selected_labels]
        option_ids = [oid for oid in option_ids if oid]

        # Update cache
        fingerprint = make_question_fingerprint(
            request.question_text or "",
            list(request.options_text.values()),
            request.question_type,
            request.canonical_key,
        )
        self.cache.put(
            request.canonical_key,
            {
                "selected_options": option_ids,
                "question_text": request.question_text,
                "options_text": request.options_text,
                "question_type": request.question_type,
                "fingerprint": fingerprint,
            },
        )

        # Update session memory
        entry = SessionMemoryItem(
            cluster_id=request.cluster_id,
            canonical_key=request.canonical_key,
            question_text=request.question_text or "",
            topic_tags=[],
            selected_options=option_ids,
            facts=inferred_facts or {},
        )
        self.session_memory.append(entry)

        # Update profile facts
        if inferred_facts:
            self.profile_facts.update(inferred_facts)

        return QAResult(
            selected_option_ids=option_ids,
            raw_model_output=raw_model_output,
            cache_hit=False,
        )
