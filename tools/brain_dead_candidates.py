from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Set


ROOT = Path(__file__).resolve().parents[1]
BRAIN_DIR = ROOT / "scripts" / "brain"


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)


def _iter_py_files() -> List[Path]:
    out: List[Path] = []
    for p in ROOT.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        out.append(p)
    return out


def _extract_refs(src: str) -> Set[str]:
    refs: Set[str] = set()
    try:
        tree = ast.parse(src)
    except Exception:
        return refs
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                refs.add(str(n.name))
        elif isinstance(node, ast.ImportFrom):
            base = str(node.module or "")
            if base:
                refs.add(base)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if "scripts.brain" in s:
                refs.add(s.strip())
    return refs


def main() -> None:
    brain_files = sorted([p for p in BRAIN_DIR.rglob("*.py") if "__pycache__" not in p.parts])
    all_py = _iter_py_files()

    refs_by_file: Dict[str, List[str]] = {}
    for p in all_py:
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        refs = sorted(_extract_refs(src))
        refs_by_file[str(p.relative_to(ROOT))] = refs

    brain_modules = {_module_name(p): p for p in brain_files}
    inbound: Dict[str, Set[str]] = {m: set() for m in brain_modules}
    for src_file, refs in refs_by_file.items():
        for ref in refs:
            for mod in brain_modules:
                if ref == mod or ref.startswith(mod + ".") or mod.startswith(ref + "."):
                    inbound[mod].add(src_file)

    report_rows = []
    dead_candidates = []
    for mod, path in sorted(brain_modules.items()):
        refs = sorted(inbound.get(mod, set()) - {str(path.relative_to(ROOT))})
        row = {
            "module": mod,
            "path": str(path.relative_to(ROOT)),
            "references": refs,
            "reference_count": len(refs),
            "dead_candidate": len(refs) == 0,
        }
        report_rows.append(row)
        if row["dead_candidate"]:
            dead_candidates.append(row)

    out_json = BRAIN_DIR / "dead_candidates_report.json"
    out_txt = BRAIN_DIR / "dead_candidates_report.txt"
    out_json.write_text(json.dumps({"files": report_rows, "dead_candidates": dead_candidates}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["Brain dead-candidate report", ""]
    for row in report_rows:
        status = "DEAD_CANDIDATE" if row["dead_candidate"] else "REFERENCED"
        lines.append(f"[{status}] {row['path']} refs={row['reference_count']}")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {out_json}")
    print(f"written: {out_txt}")


if __name__ == "__main__":
    main()
