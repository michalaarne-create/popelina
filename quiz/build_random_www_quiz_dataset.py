from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from random_quiz_sandbox import build_samples


ROOT = Path(__file__).resolve().parents[1]


def _collect_dom_annotations(page: Any) -> Dict[str, Any]:
    script = """
() => {
  const rect = (el) => {
    const r = el.getBoundingClientRect();
    return [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)];
  };
  const rows = (sel, kind) => Array.from(document.querySelectorAll(sel)).map((el) => ({
    kind,
    text: (el.textContent || '').trim(),
    bbox: rect(el),
    attrs: {
      block_id: el.getAttribute('data-block-id') || '',
      block_type: el.getAttribute('data-block-type') || '',
      answer_index: el.getAttribute('data-answer-index') || '',
      answer_text: el.getAttribute('data-answer-text') || '',
    },
  }));
  return {
    viewport: {
      width: window.innerWidth,
      height: window.innerHeight,
      scroll_x: Math.round(window.scrollX || 0),
      scroll_y: Math.round(window.scrollY || 0),
      doc_w: Math.max(document.documentElement.scrollWidth, document.body ? document.body.scrollWidth : 0),
      doc_h: Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0),
    },
    blocks: rows('[data-role="quiz-block"]', 'block'),
    questions: rows('[data-role="question"]', 'question'),
    answers: rows('[data-role="answer"]', 'answer'),
    selects: rows('[data-role="select"]', 'select'),
    text_inputs: rows('[data-role="text-input"]', 'text_input'),
    next_buttons: rows('[data-role="next"]', 'next'),
  };
}
"""
    return page.evaluate(script)


def _max_scroll_y(page: Any) -> int:
    return int(
        page.evaluate(
            "() => Math.max(0, Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0) - window.innerHeight)"
        )
        or 0
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve())


def _try_import_playwright() -> Optional[Any]:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore

        return sync_playwright
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_dataset(
    *,
    count: int,
    seed: int,
    out_dir: Path,
    headless: bool,
    timeout_ms: int,
    html_only: bool,
    cdp_endpoint: str,
) -> Dict[str, Any]:
    samples = build_samples(count=max(1, int(count)), seed=int(seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    html_dir = out_dir / "html"
    manifests_dir = out_dir / "manifests"
    for p in (images_dir, labels_dir, html_dir, manifests_dir):
        p.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    per_type: Dict[str, int] = {}
    total_views = 0

    for row in samples:
        gt = str(row.get("global_type") or "unknown")
        per_type[gt] = int(per_type.get(gt) or 0) + 1

    sync_playwright = _try_import_playwright()
    if (not html_only) and sync_playwright is None:
        raise RuntimeError(
            "Brak playwright w interpreterze. Uzyj .venv312 albo uruchom z --html-only. "
            "Instalacja: python -m pip install playwright. "
            "Jesli masz juz otwarta przegladarke CDP, mozesz podac --cdp-endpoint."
        )

    if html_only:
        for sample in samples:
            sid = str(sample["sample_id"])
            html_path = html_dir / f"{sid}.html"
            html_path.write_text(str(sample["html"]), encoding="utf-8")
            label = {
                "sample_id": sid,
                "global_type": sample["global_type"],
                "block_types": sample["block_types"],
                "has_next": sample["has_next"],
                "auto_next": sample["auto_next"],
                "require_scroll": sample["require_scroll"],
                "viewport": sample["viewport"],
                "blocks": sample["blocks"],
                "style": sample["style"],
                "render": {"mode": "html_only"},
            }
            label_path = labels_dir / f"{sid}.json"
            _write_json(label_path, label)
            manifest_rows.append(
                {
                    "sample_id": sid,
                    "view_id": f"{sid}__html",
                    "expected_global_type": sample["global_type"],
                    "expected_block_types": sample["block_types"],
                    "image_path": "",
                    "html_path": _relative(html_path, out_dir),
                    "label_path": _relative(label_path, out_dir),
                }
            )
            total_views += 1
    else:
        with sync_playwright() as pw:
            cdp = str(cdp_endpoint or "").strip()
            shared_context = None
            if cdp:
                browser = pw.chromium.connect_over_cdp(cdp, timeout=int(timeout_ms))
                shared_context = browser.contexts[0] if browser.contexts else browser.new_context()
                close_browser = False
            else:
                browser = pw.chromium.launch(headless=bool(headless))
                close_browser = True
            try:
                for sample in samples:
                    sid = str(sample["sample_id"])
                    rng = random.Random(int(sample["seed"]) * 97 + int(sample["index"]) * 37)
                    html = str(sample["html"])
                    html_path = html_dir / f"{sid}.html"
                    html_path.write_text(html, encoding="utf-8")
                    viewport = sample.get("viewport") if isinstance(sample.get("viewport"), dict) else {}
                    vw = int(viewport.get("width") or 1366)
                    vh = int(viewport.get("height") or 900)

                    if shared_context is None:
                        context = browser.new_context(
                            viewport={"width": vw, "height": vh},
                            color_scheme="dark" if str(sample["style"].get("bg_a", "")).startswith("#0") else "light",
                            device_scale_factor=1.0,
                        )
                        page = context.new_page()
                    else:
                        context = shared_context
                        page = context.new_page()
                        page.set_viewport_size({"width": vw, "height": vh})
                    try:
                        page.set_content(html, wait_until="domcontentloaded", timeout=int(timeout_ms))
                        page.wait_for_timeout(90)
                        max_scroll = _max_scroll_y(page)
                        states: List[Tuple[str, int]] = [("top", 0)]
                        if bool(sample.get("require_scroll")) and max_scroll > 120:
                            states.append(("mid", int(max_scroll * rng.uniform(0.35, 0.6))))
                            states.append(("bottom", int(max_scroll * rng.uniform(0.78, 0.98))))
                        for state_name, y in states:
                            y_clamped = max(0, min(max_scroll, int(y)))
                            page.evaluate(f"window.scrollTo(0, {y_clamped});")
                            page.wait_for_timeout(60)

                            view_id = f"{sid}__{state_name}"
                            image_path = images_dir / f"{view_id}.png"
                            page.screenshot(path=str(image_path), full_page=False)
                            dom_info = _collect_dom_annotations(page)

                            label_payload = {
                                "sample_id": sid,
                                "view_id": view_id,
                                "global_type": sample["global_type"],
                                "block_types": sample["block_types"],
                                "has_next": sample["has_next"],
                                "auto_next": sample["auto_next"],
                                "require_scroll": sample["require_scroll"],
                                "blocks": sample["blocks"],
                                "style": sample["style"],
                                "viewport": sample["viewport"],
                                "render": {
                                    "state": state_name,
                                    "scroll_y": int(dom_info.get("viewport", {}).get("scroll_y", y_clamped)),
                                    "max_scroll": max_scroll,
                                },
                                "dom": dom_info,
                            }
                            label_path = labels_dir / f"{view_id}.json"
                            _write_json(label_path, label_payload)

                            manifest_rows.append(
                                {
                                    "sample_id": sid,
                                    "view_id": view_id,
                                    "expected_global_type": sample["global_type"],
                                    "expected_block_types": sample["block_types"],
                                    "image_path": _relative(image_path, out_dir),
                                    "html_path": _relative(html_path, out_dir),
                                    "label_path": _relative(label_path, out_dir),
                                }
                            )
                            total_views += 1
                    except Exception as exc:
                        errors.append({"sample_id": sid, "error": str(exc)})
                    finally:
                        if shared_context is None:
                            context.close()
                        else:
                            page.close()
            finally:
                if close_browser:
                    browser.close()

    manifest_json = manifests_dir / "dataset_manifest.json"
    _write_json(manifest_json, {"rows": manifest_rows})
    manifest_jsonl = manifests_dir / "dataset_manifest.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "created_at_unix": int(time.time()),
        "count_requested": int(count),
        "seed": int(seed),
        "total_samples": len(samples),
        "total_views": int(total_views),
        "per_global_type": per_type,
        "errors": errors,
        "out_dir": str(out_dir.resolve()),
        "manifests": {
            "json": str(manifest_json.resolve()),
            "jsonl": str(manifest_jsonl.resolve()),
        },
    }
    _write_json(manifests_dir / "dataset_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fully automatic randomized WWW-like quiz dataset.")
    parser.add_argument("--count", type=int, default=1000, help="How many base samples to generate.")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic random seed.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "data" / "benchmarks" / "random_www_quiz"),
        help="Output directory for images, labels and manifests.",
    )
    parser.add_argument("--headless", type=int, default=1, help="1=headless browser screenshots, 0=visible browser.")
    parser.add_argument("--timeout-ms", type=int, default=8000, help="Per-page render timeout.")
    parser.add_argument("--html-only", action="store_true", help="Generate html+labels only (no screenshots).")
    parser.add_argument(
        "--cdp-endpoint",
        type=str,
        default="",
        help="Optional existing Chromium CDP endpoint, e.g. http://127.0.0.1:9222",
    )
    args = parser.parse_args()

    summary = build_dataset(
        count=int(args.count),
        seed=int(args.seed),
        out_dir=Path(args.out_dir).resolve(),
        headless=bool(int(args.headless)),
        timeout_ms=int(args.timeout_ms),
        html_only=bool(args.html_only),
        cdp_endpoint=str(args.cdp_endpoint or ""),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
