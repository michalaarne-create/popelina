from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION

from .data_redaction import redact_text
from .secrets_loader import load_named_secrets
from .test_data_policy import map_environment_profile, rotate_test_account


_DEFAULT_SECRET_NAMES = ["API_TOKEN", "OPENAI_API_KEY"]
_DEFAULT_ENV_PROFILES: Dict[str, Dict[str, Any]] = {
    "default": {
        "dataset": "safe",
        "account_pool": "default",
    },
    "staging": {
        "dataset": "staging-safe",
        "account_pool": "staging",
    },
    "production": {
        "dataset": "production-safe",
        "account_pool": "production-isolated",
    },
}


def _parse_csv_names(value: Any) -> List[str]:
    items: List[str] = []
    for raw in str(value or "").split(","):
        item = str(raw or "").strip()
        if item and item not in items:
            items.append(item)
    return items


def _parse_json_mapping(value: Any) -> Dict[str, Dict[str, Any]]:
    try:
        parsed = json.loads(str(value or "").strip() or "{}")
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for key, row in parsed.items():
        if isinstance(row, dict):
            result[str(key)] = dict(row)
    return result


def _parse_json_accounts(value: Any) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(str(value or "").strip() or "[]")
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [dict(row) for row in parsed if isinstance(row, dict)]


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _selected_account_summary(account: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(account or {})
    account_id = str(row.get("id") or row.get("alias") or "").strip()
    hint_source = (
        row.get("login")
        or row.get("email")
        or row.get("name")
        or row.get("alias")
        or row.get("id")
        or ""
    )
    return {
        "selected_account_id": account_id,
        "selected_account_hint": redact_text(hint_source),
        "has_selected_account": bool(account_id or hint_source),
    }


def build_test_data_governance_packet(
    *,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    source_env = dict(os.environ if env is None else env)
    environment = str(source_env.get("FULLBOT_ENVIRONMENT") or "default").strip().lower() or "default"
    secret_names = _parse_csv_names(source_env.get("FULLBOT_SECRET_NAMES")) or list(_DEFAULT_SECRET_NAMES)
    env_file = str(source_env.get("FULLBOT_ENV_FILE") or "").strip() or None
    secret_source_override = str(source_env.get("FULLBOT_SECRET_SOURCE") or "").strip().lower()
    profiles = dict(_DEFAULT_ENV_PROFILES)
    profiles.update(_parse_json_mapping(source_env.get("FULLBOT_ENV_PROFILES_JSON")))
    accounts = _parse_json_accounts(source_env.get("FULLBOT_TEST_ACCOUNTS_JSON"))
    session_index = _parse_int(source_env.get("FULLBOT_SESSION_INDEX"), default=0)

    loaded_secrets = load_named_secrets(
        names=secret_names,
        env=source_env,
        env_file=env_file,
    )
    environment_profile = map_environment_profile(
        environment=environment,
        profiles=profiles,
    )
    selected_account = rotate_test_account(
        accounts=accounts,
        session_index=session_index,
    )
    account_summary = _selected_account_summary(selected_account)

    return {
        "schema_version": SCHEMA_VERSION,
        "environment": environment,
        "environment_profile": environment_profile,
        "requested_secret_names": secret_names,
        "loaded_secret_names": sorted(str(name or "").strip() for name in loaded_secrets.keys()),
        "secret_count": int(len(loaded_secrets)),
        "secret_source": secret_source_override or ("env_file" if env_file else "environment"),
        "test_account_rotation": {
            "account_count": int(len(accounts)),
            "session_index": int(session_index),
            **account_summary,
        },
        "is_configured": bool(environment_profile or loaded_secrets or selected_account),
    }


def sanitize_test_data_governance_packet_for_storage(packet: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(packet, dict):
        return {}
    row = dict(packet)
    rotation = dict(row.get("test_account_rotation") or {})
    if rotation:
        if "selected_account_id" in rotation:
            rotation["selected_account_id"] = redact_text(rotation.get("selected_account_id"))
        if "selected_account_hint" in rotation:
            rotation["selected_account_hint"] = redact_text(rotation.get("selected_account_hint"))
        row["test_account_rotation"] = rotation
    return row
