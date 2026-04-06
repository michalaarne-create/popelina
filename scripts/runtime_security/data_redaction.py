from __future__ import annotations

import re
from typing import Any


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_LONG_DIGIT_RE = re.compile(r"\b\d{6,}\b")


def redact_text(value: Any) -> str:
    text = str(value or "")
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _LONG_DIGIT_RE.sub("[redacted-number]", text)
    return text
