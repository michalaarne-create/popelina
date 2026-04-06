from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import unittest
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = TESTS_ROOT / "popelina_github"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _load_module(module_name: str, relative_path: str):
    module_path = PACKAGE_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


replay_lib = _load_module("replay_lib_test", "Replay and simulation/replay_lib.py")


class ReplayAndSimulationTests(unittest.TestCase):
    def test_diff_decisions_reports_action_and_answer_changes(self) -> None:
        diff = replay_lib.diff_decisions(
            {
                "actions": [{"kind": "screen_click", "reason": "click_answer:A"}],
                "resolved_answer": {"correct_answers": ["A"]},
                "trace": {"stage": "answer"},
                "screen_state": {"question_text": "Q1"},
            },
            {
                "actions": [{"kind": "dom_select_option", "reason": "dom_select_answer"}],
                "resolved_answer": {"correct_answers": ["B"]},
                "trace": {"stage": "next"},
                "screen_state": {"question_text": "Q1"},
            },
        )
        self.assertEqual(diff["action_kind_left"], "screen_click")
        self.assertEqual(diff["action_kind_right"], "dom_select_option")
        self.assertFalse(diff["actions_equal"])
        self.assertFalse(diff["resolved_equal"])

    def test_simulate_counterfactuals_ranks_expected_action_first(self) -> None:
        report = replay_lib.simulate_counterfactuals(
            {
                "options": [{"text": "Kot"}, {"text": "Pies"}],
                "next_bbox": [1, 2, 3, 4],
                "input_bbox": None,
                "select_bbox": None,
            },
            {"post_action_expectation": {"expected_first_action_kind": "screen_click", "expected_values": ["Pies"]}},
            [
                {"name": "wrong", "actions": [{"kind": "noop", "reason": "manual_noop"}]},
                {"name": "right", "actions": [{"kind": "screen_click", "reason": "click_answer:Pies"}]},
            ],
        )
        self.assertEqual(report["ranked_candidates"][0]["candidate_name"], "right")
        self.assertGreater(report["ranked_candidates"][0]["score"], report["ranked_candidates"][1]["score"])

    def test_build_session_bundle_copies_manifest_and_iterations(self) -> None:
        tmp_dir = Path("tmp_test_replay_bundle")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        run_dir = tmp_dir / "run_001"
        iter_dir = run_dir / "iter_001"
        output_root = tmp_dir / "bundles"
        iter_dir.mkdir(parents=True, exist_ok=True)
        try:
            (run_dir / "summary.json").write_text(json.dumps({"completed": False}), encoding="utf-8")
            (run_dir / "trace.jsonl").write_text('{"iteration":1}\n', encoding="utf-8")
            for name in ("screen_state.json", "decision.json", "state.json", "page.html"):
                (iter_dir / name).write_text("{}", encoding="utf-8")
            manifest = replay_lib.build_session_bundle(run_dir, output_root)
            bundle_dir = Path(manifest["bundle_dir"])
            self.assertTrue((bundle_dir / "manifest.json").exists())
            self.assertTrue((bundle_dir / "summary.json").exists())
            self.assertTrue((bundle_dir / "iter_001" / "decision.json").exists())
            self.assertEqual(manifest["iterations"], 1)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_replay_artifact_run_recomputes_actions(self) -> None:
        tmp_dir = Path("tmp_test_replay_artifact")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        run_dir = tmp_dir / "run_001"
        iter_dir = run_dir / "iter_001"
        iter_dir.mkdir(parents=True, exist_ok=True)
        cache_path = tmp_dir / "qa_cache.json"
        try:
            cache_path.write_text(
                json.dumps(
                    {
                        "items": {
                            "q1": {
                                "question_text": "Które zwierzę miauczy?",
                                "options_text": {"A": "Kot", "B": "Pies"},
                                "question_type": "single",
                                "selected_options": ["A"],
                                "correct_answer": "Kot",
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            screen_state = {
                "question_text": "Które zwierzę miauczy?",
                "options": [
                    {"text": "Kot", "bbox": [10, 10, 20, 20]},
                    {"text": "Pies", "bbox": [30, 30, 40, 40]},
                ],
                "screen_signature": "sig-1",
                "active_question_signature": "q-sig",
                "control_kind": "choice",
                "next_bbox": None,
            }
            decision = {
                "screen_state": screen_state,
                "resolved_answer": {"question_type": "single", "correct_answers": ["Kot"]},
                "actions": [{"kind": "screen_click", "reason": "click_answer:Kot", "bbox": [10, 10, 20, 20]}],
                "trace": {"stage": "answer", "post_action_expectation": {"expected_first_action_kind": "screen_click"}},
                "region_payload": {},
            }
            (iter_dir / "screen_state.json").write_text(json.dumps(screen_state, ensure_ascii=False), encoding="utf-8")
            (iter_dir / "decision.json").write_text(json.dumps(decision, ensure_ascii=False), encoding="utf-8")
            report = replay_lib.replay_artifact_run(run_dir, cache_path=cache_path)
            self.assertEqual(report["iterations_total"], 1)
            self.assertEqual(report["records"][0]["replay_plan_kind"], "screen_click")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
