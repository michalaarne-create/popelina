from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def build_environment_profile_separation_contract(
    *,
    test_data_governance_packet: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    governance = dict(test_data_governance_packet or {})
    environment = str(governance.get("environment") or "default").strip().lower() or "default"
    profile = dict(governance.get("environment_profile") or {})
    account_rotation = dict(governance.get("test_account_rotation") or {})

    dataset = str(profile.get("dataset") or "").strip().lower()
    account_pool = str(profile.get("account_pool") or "").strip().lower()
    selected_account_id = str(account_rotation.get("selected_account_id") or "").strip().lower()

    environment_token = "prod" if environment == "production" else environment
    signals = {
        "environment": environment,
        "dataset": dataset,
        "account_pool": account_pool,
        "selected_account_id": selected_account_id,
        "dataset_matches_environment": bool(dataset) and environment_token in dataset,
        "pool_matches_environment": bool(account_pool) and environment_token in account_pool,
        "account_matches_environment": bool(selected_account_id) and environment_token in selected_account_id,
    }

    if environment == "default":
        is_isolated = bool(dataset) and dataset == "safe"
        reason = "default_safe_profile" if is_isolated else "default_profile_not_explicitly_safe"
    elif environment == "staging":
        is_isolated = bool(signals["dataset_matches_environment"] or signals["pool_matches_environment"] or signals["account_matches_environment"])
        reason = "staging_profile_isolated" if is_isolated else "staging_profile_not_isolated"
    elif environment == "production":
        is_isolated = bool(signals["dataset_matches_environment"] and signals["pool_matches_environment"])
        reason = "production_profile_isolated" if is_isolated else "production_profile_not_isolated"
    else:
        is_isolated = False
        reason = "unknown_environment_profile"

    return {
        "schema_version": SCHEMA_VERSION,
        "environment": environment,
        "is_isolated": bool(is_isolated),
        "reason": reason,
        "signals": signals,
    }
