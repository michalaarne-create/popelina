from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import unittest
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[2]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


quality_lib = _load_module("quality_check_lib_test", TESTS_ROOT / "quality_check" / "quality_lib.py")


class QualityCheckTests(unittest.TestCase):
    def test_decision_audit_and_confidence_tracker_use_summary_steps(self) -> None:
        tmp_dir = Path("tmp_test_quality_check_audit")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        run_dir = tmp_dir / "run_001"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            summary = {
                "completed": False,
                "steps": [
                    {
                        "iteration": 1,
                        "question": "Q1",
                        "plan_source": "screen_only",
                        "plan_kind": "screen_click",
                        "resolved_answer": {"confidence": 0.98, "matched": True},
                        "screen_state": {"screen_parse_quality": "actionable", "confidence": {"merged": 0.82}},
                        "trace": {"fused_action_confidence": 0.88},
                        "runtime_state": {"progressed": True},
                        "validation_state": {"category": "", "is_blocking": False},
                    },
                    {
                        "iteration": 2,
                        "question": "Q2",
                        "plan_source": "dom_fallback",
                        "plan_kind": "noop",
                        "resolved_answer": {"confidence": 0.2, "matched": False},
                        "screen_state": {"screen_parse_quality": "question_only", "confidence": {"merged": 0.3}},
                        "trace": {"fused_action_confidence": 0.25},
                        "runtime_state": {"progressed": False},
                        "validation_state": {"category": "required_error", "is_blocking": True},
                    },
                ],
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
            audit = quality_lib.build_decision_audit_table(run_dir)
            tracker = quality_lib.build_confidence_to_outcome_tracker(run_dir)
            self.assertEqual(audit["row_count"], 2)
            self.assertEqual(audit["rows"][0]["parser_confidence"], 0.82)
            self.assertTrue(any(bucket["parser_bucket"] == "high" for bucket in tracker["buckets"]))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_fallback_cost_analyzer_reads_action_meta(self) -> None:
        tmp_dir = Path("tmp_test_quality_check_fallback")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        run_dir = tmp_dir / "run_001"
        meta_dir = run_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        try:
            meta_a = meta_dir / "a.json"
            meta_b = meta_dir / "b.json"
            meta_a.write_text(json.dumps({"ts_ms": 1000}), encoding="utf-8")
            meta_b.write_text(json.dumps({"ts_ms": 1300}), encoding="utf-8")
            summary = {
                "steps": [
                    {
                        "iteration": 1,
                        "plan_source": "dom_fallback",
                        "runtime_state": {"progressed": True},
                        "actions": [
                            {"debug": {"meta": str(meta_a)}},
                            {"debug": {"meta": str(meta_b)}},
                        ],
                    }
                ]
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
            report = quality_lib.build_fallback_cost_analyzer(run_dir)
            self.assertEqual(report["sources"][0]["plan_source"], "dom_fallback")
            self.assertEqual(report["sources"][0]["avg_execute_proxy_ms"], 300.0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_iteration_bottleneck_profiler_uses_artifact_mtimes(self) -> None:
        tmp_dir = Path("tmp_test_quality_check_bottleneck")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        run_dir = tmp_dir / "run_001"
        art_dir = run_dir / "iter_001"
        meta_dir = run_dir / "meta"
        art_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        try:
            screenshot = art_dir / "screen.png"
            decision = art_dir / "decision.json"
            meta = meta_dir / "action.json"
            screenshot.write_text("a", encoding="utf-8")
            decision.write_text("{}", encoding="utf-8")
            meta.write_text(json.dumps({"ts_ms": 2500}), encoding="utf-8")
            # Force deterministic mtimes.
            import os
            os.utime(screenshot, (1.0, 1.0))
            os.utime(decision, (2.0, 2.0))
            summary = {
                "steps": [
                    {
                        "iteration": 1,
                        "actions": [{"debug": {"meta": str(meta)}}],
                        "artifacts": {"screenshot": str(screenshot), "decision": str(decision)},
                    }
                ]
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
            report = quality_lib.build_iteration_bottleneck_profiler(run_dir)
            self.assertEqual(report["rows"][0]["total_proxy_ms"], 1500)
            self.assertGreater(report["rows"][0]["parse_plan_share"], 0.0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_automatic_gap_miner_emits_missing_contract_suggestions(self) -> None:
        tmp_dir = Path("tmp_test_quality_check_gap")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        run_dir = tmp_dir / "run_001"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            summary = {
                "completed": False,
                "steps": [
                    {
                        "iteration": 1,
                        "plan_source": "dom_fallback",
                        "plan_kind": "noop",
                        "resolved_answer": {"matched": False},
                        "screen_state": {"screen_parse_quality": "question_only"},
                        "trace": {"fused_action_confidence": 0.91},
                        "runtime_state": {"progressed": False},
                        "validation_state": {"is_blocking": True},
                    }
                ],
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
            report = quality_lib.build_automatic_gap_miner(run_dir)
            gaps = {item["gap"] for item in report["suggestions"]}
            self.assertIn("answer_resolver_contract", gaps)
            self.assertIn("planner_actionability_contract", gaps)
            self.assertIn("validation_recovery_parser", gaps)
            self.assertIn("screen_executor_targeting", gaps)
            self.assertIn("confidence_calibration", gaps)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
