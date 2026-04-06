from __future__ import annotations

import os
from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def build_secret_source_policy(
    *,
    test_data_governance_packet: Optional[Dict[str, Any]] = None,
    allow_review: bool | None = None,
) -> Dict[str, Any]:
    governance = dict(test_data_governance_packet or {})
    environment = str(governance.get("environment") or "default").strip().lower() or "default"
    secret_source = str(governance.get("secret_source") or "").strip().lower()
    requested_secret_names = [
        str(item or "").strip()
        for item in (governance.get("requested_secret_names") or [])
        if str(item or "").strip()
    ]
    loaded_secret_names = [
        str(item or "").strip()
        for item in (governance.get("loaded_secret_names") or [])
        if str(item or "").strip()
    ]

    source_allowed = secret_source in {"environment", "env_file"}
    missing_secret_names = [name for name in requested_secret_names if name not in loaded_secret_names]
    review_allowed = bool(int(str(os.environ.get("FULLBOT_ALLOW_SECRET_SOURCE_REVIEW", "0")))) if allow_review is None else bool(allow_review)
    if not requested_secret_names:
        policy_state = "review"
        reason = "no_requested_secret_names"
    elif not source_allowed:
        policy_state = "block"
        reason = "unknown_secret_source"
    elif environment == "production" and secret_source == "env_file":
        policy_state = "review"
        reason = "production_env_file_source"
    elif missing_secret_names:
        policy_state = "review"
        reason = "missing_requested_secrets"
    else:
        policy_state = "allow"
        reason = "approved_secret_source"

    return {
        "schema_version": SCHEMA_VERSION,
        "environment": environment,
        "decision": policy_state,
        "reason": reason,
        "is_allowed": policy_state == "allow",
        "requires_review": policy_state == "review",
        "review_allowed": review_allowed,
        "is_blocking": policy_state == "block" or (policy_state == "review" and not review_allowed),
        "signals": {
            "secret_source": secret_source or "unknown",
            "requested_secret_names": requested_secret_names,
            "loaded_secret_names": loaded_secret_names,
            "missing_secret_names": missing_secret_names,
        },
    }
