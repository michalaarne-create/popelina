from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]


def _run_json(cmd: list[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    whole = proc.stdout.strip()
    if whole:
        try:
            parsed = json.loads(whole)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    for raw in reversed(lines):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {"ok": False, "returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run combined quality report for dataset + element role + quiz type.")
    parser.add_argument("--dataset-dir", type=str, default="")
    parser.add_argument("--element-role-manifest", type=str, default="")
    parser.add_argument("--element-role-model", type=str, default="")
    parser.add_argument("--quiz-type-manifest", type=str, default="")
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    py = sys.executable
    report: Dict[str, Any] = {"python": py}

    if args.dataset_dir:
        report["dataset_status"] = _run_json([py, str(ROOT / "quiz" / "dataset_status.py"), "--out-dir", str(Path(args.dataset_dir).resolve())])
        report["dataset_validate"] = _run_json([py, str(ROOT / "quiz" / "dataset_validate.py"), "--out-dir", str(Path(args.dataset_dir).resolve())])
    if args.element_role_manifest and args.element_role_model:
        report["element_role_eval"] = _run_json(
            [
                py,
                str(ROOT / "quiz" / "eval_element_role_model.py"),
                "--manifest",
                str(Path(args.element_role_manifest).resolve()),
                "--model",
                str(Path(args.element_role_model).resolve()),
            ]
        )
    if args.quiz_type_manifest:
        report["quiz_type_benchmark"] = _run_json(
            [
                py,
                str(ROOT / "scripts" / "brain" / "runtime" / "quiz_type_benchmark.py"),
                "--manifest",
                str(Path(args.quiz_type_manifest).resolve()),
            ]
        )

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out_json:
        out_path = Path(args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
