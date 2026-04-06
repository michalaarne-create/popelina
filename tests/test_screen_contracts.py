from __future__ import annotations

import sys
import unittest
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = TESTS_ROOT / "popelina_github"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from popelina_github.scripts.pipeline.screen_contracts import (
    MIGRATION_PLAYBOOK_REGISTRY,
    SCHEMA_REGISTRY,
    build_blank_screen_audit,
    build_broken_screen_shape_guard,
    build_feature_parity_report,
    build_frozen_runtime_snapshot,
    build_model_fingerprint,
    build_payload_semantics_contract,
    build_provenance_payload,
    build_qid_alignment_contract,
    build_screen_to_domain_mapping,
    build_session_language_contract,
    build_source_precedence_contract,
    build_state_freshness_marker,
    build_wrong_context_recovery_policy,
    dedupe_screen_instances,
    detect_blank_screen,
)


class ScreenContractsTests(unittest.TestCase):
    def _screen_state(self) -> dict:
        return {
            "active_question_signature": "sig_q1",
            "question_text": "Wybierz kolor",
            "control_kind": "choice",
            "detected_quiz_type": "single",
            "active_block": {
                "question_signature": "sig_q1",
                "prompt_text": "Wybierz kolor",
                "control_kind": "choice",
                "block_type": "single",
                "answers": [
                    {"text": "Zielony", "bbox": [10, 10, 20, 20]},
                    {"text": "Niebieski", "bbox": [10, 30, 20, 40]},
                ],
                "next_bbox": [100, 100, 150, 130],
            },
        }

    def test_registry_constants_exist(self) -> None:
        self.assertIn("screen_state", SCHEMA_REGISTRY)
        self.assertIn("legacy.unversioned", MIGRATION_PLAYBOOK_REGISTRY)

    def test_build_model_fingerprint_is_stable(self) -> None:
        first = build_model_fingerprint(
            model_parts={"quiz_type_model": "v3", "control_kind_model": "v10"},
            runtime="screen_runtime",
            backend="cpu_fp32",
        )
        second = build_model_fingerprint(
            model_parts={"control_kind_model": "v10", "quiz_type_model": "v3"},
            runtime="screen_runtime",
            backend="cpu_fp32",
        )
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(first["parts"]["runtime"], "screen_runtime")

    def test_build_provenance_payload_prefers_screen_state(self) -> None:
        payload = build_provenance_payload(
            screen_state=self._screen_state(),
            summary_data={"question_candidate": "fallback"},
            page_state={"textboxBbox": [1, 2, 3, 4]},
        )
        self.assertEqual(payload["sources"]["question_text"], "screen_state")
        self.assertIn("screen_state", payload["used_sources"])

    def test_build_frozen_runtime_snapshot_contains_option_texts(self) -> None:
        snapshot = build_frozen_runtime_snapshot(
            screen_state=self._screen_state(),
            labels={"expected_type": "single"},
            artifact_id="iter_001",
        )
        self.assertEqual(snapshot["snapshot"]["artifact_id"], "iter_001")
        self.assertEqual(snapshot["snapshot"]["option_texts"], ["Zielony", "Niebieski"])
        self.assertTrue(snapshot["snapshot_hash"])

    def test_build_feature_parity_report_compares_shared_numeric_keys(self) -> None:
        report = build_feature_parity_report(
            runtime_rows=[{"a": 1.0, "b": 0.5, "only_runtime": 1}, {"a": 3.0, "b": 0.7}],
            label_rows=[{"a": 2.0, "b": 0.2, "only_label": 1}],
        )
        self.assertIn("only_runtime", report["runtime_only_keys"])
        self.assertIn("only_label", report["label_only_keys"])
        self.assertAlmostEqual(report["numeric_drift"]["a"]["delta"], 0.0, places=4)

    def test_dedupe_screen_instances_merges_close_duplicates(self) -> None:
        deduped = dedupe_screen_instances(
            [
                {"ts": 10.0, "screen_signature": "abc", "qid": "q1"},
                {"ts": 10.8, "screen_signature": "abc", "qid": "q1"},
                {"ts": 15.0, "screen_signature": "xyz", "qid": "q1"},
            ]
        )
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["duplicate_count"], 2)

    def test_build_qid_alignment_contract_detects_mismatch(self) -> None:
        contract = build_qid_alignment_contract(
            controls_data={"meta": {"qid": "q1"}},
            screen_state=self._screen_state(),
            page_data={"qid": "q2"},
        )
        self.assertFalse(contract["aligned"])
        self.assertEqual(contract["status"], "mismatch")

    def test_build_source_precedence_contract_prefers_controls_for_selected_values(self) -> None:
        contract = build_source_precedence_contract(
            controls_data={"meta": {"qid": "q1"}},
            screen_state=self._screen_state(),
            page_state={"qid": "q1"},
        )
        self.assertEqual(contract["fields"]["selected_values"]["primary"], "current_controls")
        self.assertEqual(contract["fields"]["question_text"]["primary"], "screen_state")

    def test_build_session_language_contract_detects_polish(self) -> None:
        contract = build_session_language_contract(screen_state={"question_text": "Wybierz odpowiedź"})
        self.assertEqual(contract["language"], "pl")
        self.assertEqual(contract["source"], "heuristic")

    def test_build_payload_semantics_contract_flags_unknown_option(self) -> None:
        contract = build_payload_semantics_contract(
            answer_payload={"selected_values": ["Czerwony", "Fioletowy"]},
            screen_state=self._screen_state(),
            resolved_answer={"source": "cache"},
        )
        self.assertFalse(contract["ok"])
        self.assertIn("unknown_option:Fioletowy", contract["violations"])

    def test_build_state_freshness_marker_marks_stale_screen(self) -> None:
        marker = build_state_freshness_marker(screen_ts=100.0, network_ts=109.0, checkpoint_ts=105.0, now_ts=112.5, max_age_s=8.0)
        self.assertFalse(marker["is_fresh"])
        self.assertIn("screen_age_s", marker["stale_reasons"])

    def test_build_wrong_context_recovery_policy_prefers_refocus_for_wrong_tab(self) -> None:
        policy = build_wrong_context_recovery_policy(
            stop_reason="screen_wrong_tab",
            site_consistency_score=0.55,
            active_url="https://other.example/path",
            expected_host="quiz.example",
            wrong_tab_detected=True,
        )
        self.assertEqual(policy["recommended_action"], "refocus_tab")
        self.assertEqual(policy["severity"], "warning")

    def test_build_screen_to_domain_mapping_exposes_navigation(self) -> None:
        mapping = build_screen_to_domain_mapping(
            screen_state=self._screen_state(),
            page_state={"qid": "q1", "textboxBbox": [1, 2, 3, 4], "backBbox": [5, 6, 7, 8], "draftBbox": [9, 10, 11, 12]},
            controls_data={"meta": {"qid": "q1"}},
        )
        self.assertEqual(mapping["question"]["qid"], "q1")
        self.assertEqual(mapping["navigation"]["next_bbox"], [100, 100, 150, 130])
        self.assertEqual(mapping["navigation"]["back_bbox"], [5, 6, 7, 8])
        self.assertEqual(mapping["navigation"]["draft_bbox"], [9, 10, 11, 12])
        self.assertEqual(len(mapping["answers"]), 2)

    def test_detect_blank_screen_catches_black_and_loader_cases(self) -> None:
        black = detect_blank_screen(image_stats={"mean": 0.5, "std": 0.8})
        loader = detect_blank_screen(page_state={"pageText": "Loading..."})
        self.assertTrue(black["is_blank"])
        self.assertEqual(black["kind"], "black_screen")
        self.assertTrue(loader["is_blank"])
        self.assertEqual(loader["kind"], "loader_only")

    def test_build_broken_screen_shape_guard_flags_answers_without_prompt(self) -> None:
        guard = build_broken_screen_shape_guard(
            {
                "control_kind": "choice",
                "active_block": {
                    "answers": [{"text": "A"}],
                },
            }
        )
        self.assertTrue(guard["is_broken"])
        self.assertIn("answers_without_prompt", guard["issues"])

    def test_build_broken_screen_shape_guard_flags_prompt_without_controls(self) -> None:
        guard = build_broken_screen_shape_guard(
            {
                "control_kind": "choice",
                "active_block": {
                    "prompt_text": "Wybierz kolor",
                },
            }
        )
        self.assertTrue(guard["is_broken"])
        self.assertIn("prompt_without_controls", guard["issues"])

    def test_build_broken_screen_shape_guard_flags_dropdown_without_select_bbox(self) -> None:
        guard = build_broken_screen_shape_guard(
            {
                "control_kind": "dropdown",
                "active_block": {
                    "prompt_text": "Wybierz miasto",
                    "next_bbox": [100, 120, 180, 150],
                },
            }
        )
        self.assertTrue(guard["is_broken"])
        self.assertIn("dropdown_without_select_bbox", guard["issues"])

    def test_build_blank_screen_audit_counts_kinds(self) -> None:
        audit = build_blank_screen_audit(
            [
                {"id": "a", "image_stats": {"mean": 0.0, "std": 0.0}},
                {"id": "b", "page_state": {"pageText": "Loading..."}},
                {"id": "c", "screen_state": self._screen_state()},
            ]
        )
        self.assertEqual(audit["counts"]["black_screen"], 1)
        self.assertEqual(audit["counts"]["loader_only"], 1)
        self.assertEqual(audit["counts"]["not_blank"], 1)


if __name__ == "__main__":
    unittest.main()
