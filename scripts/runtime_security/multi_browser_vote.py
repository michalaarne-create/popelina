from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _vote_label(row: Mapping[str, Any]) -> str:
    for key in ("vote", "verdict", "result", "outcome", "browser_vote"):
        value = str(row.get(key) or "").strip().lower()
        if value:
            return value
    return "unknown"


def build_multi_browser_vote_state(
    *,
    browser_states: Sequence[Mapping[str, Any]],
    min_agree: int = 2,
) -> Dict[str, Any]:
    rows = [dict(row) for row in browser_states if isinstance(row, Mapping)]
    tally = Counter(_vote_label(row) for row in rows)
    confidence_sum = defaultdict(float)
    confidence_count = defaultdict(int)
    for row in rows:
        label = _vote_label(row)
        confidence_sum[label] += float(row.get("confidence") or row.get("score") or 0.0)
        confidence_count[label] += 1

    winner = "unknown"
    winner_count = 0
    winner_confidence = 0.0
    if tally:
        top_count = max(tally.values())
        tied = sorted([label for label, count in tally.items() if count == top_count])
        winner = tied[0]
        winner_count = int(tally[winner])
        if confidence_count[winner]:
            winner_confidence = confidence_sum[winner] / float(confidence_count[winner])

    is_consensus = bool(winner_count >= int(min_agree) and winner != "unknown")
    reason = "consensus" if is_consensus else ("no_votes" if not rows else "no_consensus")
    if not is_consensus and rows and winner != "unknown":
        reason = "vote_split"

    return {
        "browser_count": len(rows),
        "min_agree": int(min_agree),
        "winner": winner,
        "winner_count": winner_count,
        "winner_confidence": round(float(winner_confidence), 4),
        "is_consensus": is_consensus,
        "reason": reason,
        "vote_counts": dict(tally),
        "votes": rows,
    }
