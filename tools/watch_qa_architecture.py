from __future__ import annotations

import argparse
import html
import hashlib
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "qa_ankiety_architektura_status.html"
STATE_DIR = ROOT / ".runtime"
LOG_FILE = STATE_DIR / "qa_architektura_watchdog.log"
STATE_FILE = STATE_DIR / "qa_architektura_watchdog_state.json"
EVENTS_FILE = STATE_DIR / "qa_architektura_watchdog_events.json"
REPORT_FILE = STATE_DIR / "qa_architektura_watchdog_report.html"
MAX_BLOCK_LINES = 25
MAX_EVENTS = 80
MAX_HTML_ITEMS = 40

NODE_PATTERN_N = re.compile(
    r'N\(\s*"((?:[^"\\]|\\.)*)"\s*,\s*"([a-z]+)"',
    re.DOTALL,
)
NODE_PATTERN_OBJ = re.compile(
    r'\{\s*t:\s*"((?:[^"\\]|\\.)*)"\s*,\s*s:\s*"([a-z]+)"',
    re.DOTALL,
)
PALETTE_PATTERN = re.compile(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", re.IGNORECASE)

ANSI = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "gray": "\033[90m",
}

STATUS_LABELS = {
    "red": "czerwony",
    "orange": "pomaranczowy",
    "yellow": "zolty",
    "green": "zielony",
    "gray": "szary",
}

STATUS_COLORS = {
    "red": "red",
    "orange": "yellow",
    "yellow": "yellow",
    "green": "green",
    "gray": "gray",
}


@dataclass(frozen=True)
class NodeRecord:
    key: str
    text: str
    status: str
    line: int


def decode_js_string(value: str) -> str:
    try:
        return bytes(value, "utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return value


def colorize(text: str, color: str | None = None, *, bold: bool = False, dim: bool = False) -> str:
    codes: list[str] = []
    if bold:
        codes.append(ANSI["bold"])
    if dim:
        codes.append(ANSI["dim"])
    if color:
        codes.append(ANSI[color])
    if not codes:
        return text
    return "".join(codes) + text + ANSI["reset"]


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def build_snapshot(content: str) -> dict[str, object]:
    matches: list[tuple[int, str, str]] = []
    for pattern in (NODE_PATTERN_OBJ, NODE_PATTERN_N):
        for match in pattern.finditer(content):
            matches.append((match.start(), decode_js_string(match.group(1)).strip(), match.group(2).strip()))
    matches.sort(key=lambda item: item[0])

    occurrences: Counter[str] = Counter()
    nodes: dict[str, NodeRecord] = {}
    for offset, text, status in matches:
        occurrences[text] += 1
        key = f"{text}##{occurrences[text]}"
        nodes[key] = NodeRecord(
            key=key,
            text=text,
            status=status,
            line=line_for_offset(content, offset),
        )

    palette = {name: value.strip() for name, value in PALETTE_PATTERN.findall(content)}
    digest = hashlib.sha1(content.encode("utf-8", errors="replace")).hexdigest()

    return {
        "digest": digest,
        "nodes": nodes,
        "palette": palette,
        "node_count": len(nodes),
    }


def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status)


def summarize_counts(nodes: Iterable[NodeRecord]) -> str:
    counter = Counter(node.status for node in nodes)
    order = ["red", "orange", "yellow", "green", "gray"]
    parts = []
    for status in order:
        if counter.get(status):
            parts.append(f"{status_label(status)}={counter[status]}")
    return ", ".join(parts) if parts else "brak wezlow"


def render_node(record: NodeRecord) -> str:
    color = STATUS_COLORS.get(record.status)
    return f"L{record.line}: {colorize(record.text, color)} {colorize(f'[{status_label(record.status)}]', 'gray')}"


def diff_snapshots(old: dict[str, object], new: dict[str, object]) -> dict[str, list]:
    old_nodes: dict[str, NodeRecord] = old["nodes"]  # type: ignore[assignment]
    new_nodes: dict[str, NodeRecord] = new["nodes"]  # type: ignore[assignment]
    old_palette: dict[str, str] = old["palette"]  # type: ignore[assignment]
    new_palette: dict[str, str] = new["palette"]  # type: ignore[assignment]

    added = [new_nodes[key] for key in new_nodes.keys() - old_nodes.keys()]
    removed = [old_nodes[key] for key in old_nodes.keys() - new_nodes.keys()]
    changed = []
    for key in new_nodes.keys() & old_nodes.keys():
        before = old_nodes[key]
        after = new_nodes[key]
        if before.status != after.status:
            changed.append((before, after))

    palette_changed = []
    for key in sorted(set(old_palette) | set(new_palette)):
        before = old_palette.get(key)
        after = new_palette.get(key)
        if before != after:
            palette_changed.append((key, before, after))

    added.sort(key=lambda item: item.line)
    removed.sort(key=lambda item: item.line)
    changed.sort(key=lambda item: item[1].line)
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "palette_changed": palette_changed,
    }


def print_block(title: str, lines: list[str], color: str) -> None:
    if not lines:
        return
    print(colorize(title, color, bold=True))
    visible = lines[:MAX_BLOCK_LINES]
    for line in visible:
        print(f"  {line}")
    hidden = len(lines) - len(visible)
    if hidden > 0:
        print(f"  {colorize(f'... i jeszcze {hidden} kolejnych zmian', 'gray')}")


def append_log(message: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def load_events() -> list[dict[str, object]]:
    if not EVENTS_FILE.exists():
        return []
    try:
        data = json.loads(EVENTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def render_html_report(events: list[dict[str, object]]) -> None:
    def badge(status: str, count: int) -> str:
        return (
            f'<span class="badge badge-{html.escape(status)}">'
            f"{html.escape(status_label(status))}: {count}"
            "</span>"
        )

    cards: list[str] = []
    for event in events[:MAX_EVENTS]:
        summary = event.get("summary", {})
        counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
        badges = "".join(
            badge(status, int(counts.get(status, 0)))
            for status in ("red", "orange", "yellow", "green", "gray")
            if int(counts.get(status, 0))
        )

        sections_html: list[str] = []
        for section in event.get("sections", []):
            if not isinstance(section, dict):
                continue
            items = section.get("items", [])
            if not isinstance(items, list) or not items:
                continue
            visible = items[:MAX_HTML_ITEMS]
            rows = "".join(f"<li>{html.escape(str(item))}</li>" for item in visible)
            hidden = len(items) - len(visible)
            if hidden > 0:
                rows += f'<li class="more">... i jeszcze {hidden} kolejnych zmian</li>'
            sections_html.append(
                f"""
                <section class="section">
                  <h4>{html.escape(str(section.get("title", "Zmiany")))}</h4>
                  <ul>{rows}</ul>
                </section>
                """
            )

        kind = str(event.get("kind", "info"))
        cards.append(
            f"""
            <article class="event event-{html.escape(kind)}">
              <div class="event-top">
                <div>
                  <div class="event-kind">{html.escape(str(event.get("kind_label", kind.upper())))}</div>
                  <h3>{html.escape(str(event.get("title", "Aktualizacja")))}</h3>
                </div>
                <div class="event-time">{html.escape(str(event.get("timestamp", "")))}</div>
              </div>
              <div class="meta">
                <span>{html.escape(str(event.get("subtitle", "")))}</span>
                <span>{html.escape(str(event.get("file_mtime", "")))}</span>
              </div>
              <div class="badges">{badges or '<span class="badge badge-neutral">brak statusow</span>'}</div>
              {''.join(sections_html) or '<p class="empty">Brak szczegolowych zmian do pokazania.</p>'}
            </article>
            """
        )

    document = f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Watchdog QA Architektura</title>
  <style>
    :root {{
      --bg: #09111f;
      --panel: #0f1b31;
      --panel-2: #132443;
      --line: #29406d;
      --text: #e7efff;
      --muted: #9eb2d8;
      --red: #ff6f6f;
      --orange: #ffb057;
      --yellow: #ffe06e;
      --green: #51d88a;
      --gray: #95a0b8;
      --shadow: rgba(0, 0, 0, 0.28);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Consolas, "Cascadia Code", monospace;
      color: var(--text);
      background:
        radial-gradient(circle at top, rgba(68, 120, 255, 0.18), transparent 34%),
        linear-gradient(180deg, #050b16 0%, var(--bg) 100%);
    }}
    .wrap {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      padding: 22px 24px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(18, 33, 61, 0.96), rgba(10, 18, 34, 0.98));
      box-shadow: 0 20px 50px var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .hero p {{
      margin: 6px 0;
      color: var(--muted);
    }}
    .events {{
      margin-top: 20px;
      display: grid;
      gap: 16px;
    }}
    .event {{
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(15, 27, 49, 0.97), rgba(9, 17, 31, 0.98));
      box-shadow: 0 18px 42px var(--shadow);
      padding: 18px 20px;
    }}
    .event-start {{ border-left: 4px solid #6ab7ff; }}
    .event-change {{ border-left: 4px solid var(--green); }}
    .event-missing {{ border-left: 4px solid var(--red); }}
    .event-stop {{ border-left: 4px solid var(--gray); }}
    .event-top {{
      display: flex;
      gap: 16px;
      justify-content: space-between;
      align-items: start;
    }}
    .event-kind {{
      color: var(--muted);
      letter-spacing: 0.14em;
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .event h3 {{
      margin: 0;
      font-size: 20px;
    }}
    .event-time, .meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    .meta {{
      margin-top: 10px;
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
    }}
    .badges {{
      margin-top: 14px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .badge {{
      border: 1px solid var(--line);
      padding: 6px 10px;
      font-size: 12px;
      background: rgba(255, 255, 255, 0.04);
    }}
    .badge-red {{ color: var(--red); }}
    .badge-orange {{ color: var(--orange); }}
    .badge-yellow {{ color: var(--yellow); }}
    .badge-green {{ color: var(--green); }}
    .badge-gray, .badge-neutral {{ color: var(--gray); }}
    .section {{
      margin-top: 16px;
      padding-top: 14px;
      border-top: 1px solid rgba(41, 64, 109, 0.65);
    }}
    .section h4 {{
      margin: 0 0 10px;
      font-size: 14px;
      letter-spacing: 0.08em;
      color: #dce7ff;
    }}
    .section ul {{
      margin: 0;
      padding-left: 18px;
      color: #d7e3ff;
      line-height: 1.55;
    }}
    .section li + li {{
      margin-top: 6px;
    }}
    .more, .empty {{
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      .wrap {{ padding: 12px; }}
      .event-top {{ display: block; }}
      .event-time {{ margin-top: 10px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Watchdog QA Architektura</h1>
      <p>Raport odswieza sie przy kazdej zmianie pliku monitorowanego przez watchdoga.</p>
      <p>Plik raportu: {html.escape(str(REPORT_FILE))}</p>
      <p>Liczba zapisanych zdarzen: {len(events[:MAX_EVENTS])}</p>
    </section>
    <section class="events">
      {''.join(cards) or '<article class="event"><p class="empty">Brak zdarzen.</p></article>'}
    </section>
  </div>
</body>
</html>
"""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(document, encoding="utf-8")


def record_event(event: dict[str, object]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events()
    events.insert(0, event)
    events = events[:MAX_EVENTS]
    EVENTS_FILE.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    render_html_report(events)


def save_state(snapshot: dict[str, object], target: Path) -> None:
    nodes: dict[str, NodeRecord] = snapshot["nodes"]  # type: ignore[assignment]
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    counter = Counter(node.status for node in nodes.values())
    payload = {
        "target": str(target),
        "saved_at": now_stamp(),
        "digest": snapshot["digest"],
        "node_count": snapshot["node_count"],
        "summary": summarize_counts(nodes.values()),
        "counts": dict(counter),
        "report_file": str(REPORT_FILE),
    }
    STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def report_initial(snapshot: dict[str, object], target: Path) -> None:
    nodes: dict[str, NodeRecord] = snapshot["nodes"]  # type: ignore[assignment]
    counter = Counter(node.status for node in nodes.values())
    print(colorize("WATCHDOG STARTED", "cyan", bold=True))
    print(colorize(f"Plik: {target}", "gray"))
    print(f"Snapshot: {snapshot['node_count']} wezlow, {summarize_counts(nodes.values())}")
    print(colorize(f"Raport HTML: {REPORT_FILE}", "gray"))
    print(colorize("WATCHDOG READY", "green", bold=True))
    append_log(f"[{now_stamp()}] START {target} :: {snapshot['node_count']} wezlow :: {summarize_counts(nodes.values())}")
    record_event(
        {
            "kind": "start",
            "kind_label": "START",
            "title": f"Watchdog uruchomiony dla {target.name}",
            "subtitle": str(target),
            "timestamp": now_stamp(),
            "file_mtime": "",
            "summary": {
                "node_count": snapshot["node_count"],
                "counts": dict(counter),
            },
            "sections": [
                {
                    "title": "Stan poczatkowy",
                    "items": [
                        f"Wezly: {snapshot['node_count']}",
                        summarize_counts(nodes.values()),
                        f"Raport HTML: {REPORT_FILE}",
                    ],
                }
            ],
        }
    )
    save_state(snapshot, target)


def report_missing(target: Path) -> None:
    message = f"[{now_stamp()}] Brak pliku: {target}"
    print(colorize(message, "red", bold=True))
    append_log(message)
    record_event(
        {
            "kind": "missing",
            "kind_label": "BRAK PLIKU",
            "title": f"Nie znaleziono {target.name}",
            "subtitle": str(target),
            "timestamp": now_stamp(),
            "file_mtime": "",
            "summary": {"node_count": 0, "counts": {}},
            "sections": [{"title": "Szczegoly", "items": [message]}],
        }
    )


def report_change(target: Path, diff: dict[str, list], snapshot: dict[str, object]) -> None:
    changed: list[tuple[NodeRecord, NodeRecord]] = diff["changed"]  # type: ignore[assignment]
    added: list[NodeRecord] = diff["added"]  # type: ignore[assignment]
    removed: list[NodeRecord] = diff["removed"]  # type: ignore[assignment]
    palette_changed: list[tuple[str, str | None, str | None]] = diff["palette_changed"]  # type: ignore[assignment]

    stat = target.stat()
    nodes: dict[str, NodeRecord] = snapshot["nodes"]  # type: ignore[assignment]
    counter = Counter(node.status for node in nodes.values())
    header = f"[{now_stamp()}] Zmiana w {target.name} | mtime={datetime.fromtimestamp(stat.st_mtime).strftime('%H:%M:%S')}"
    print()
    print(colorize("=" * 100, "gray"))
    print(colorize(header, "cyan", bold=True))
    print(colorize(f"Wezly: {snapshot['node_count']} | {summarize_counts(nodes.values())}", "gray"))

    print_block("DODANE", [render_node(item) for item in added], "green")
    print_block("USUNIETE", [render_node(item) for item in removed], "red")

    if changed:
        lines = []
        for before, after in changed:
            after_color = STATUS_COLORS.get(after.status)
            lines.append(
                f"L{after.line}: {after.text} "
                f"{colorize(f'[{status_label(before.status)} -> {status_label(after.status)}]', after_color, bold=True)}"
            )
        print_block("ZMIENIONY STATUS", lines, "yellow")

    if palette_changed:
        palette_lines = []
        for name, before, after in palette_changed:
            palette_lines.append(f"{name}: {before or '<brak>'} -> {after or '<brak>'}")
        print_block("ZMIENIONA PALETA CSS", palette_lines, "magenta")

    if not any((added, removed, changed, palette_changed)):
        print(colorize("Zmiana techniczna bez roznicy w wezłach/statusach.", "gray"))

    log_parts = [
        f"[{now_stamp()}] CHANGE {target.name}",
        f"+{len(added)}",
        f"-{len(removed)}",
        f"status={len(changed)}",
        f"palette={len(palette_changed)}",
    ]
    append_log(" | ".join(log_parts))
    record_event(
        {
            "kind": "change",
            "kind_label": "ZMIANA",
            "title": f"Zmiana w {target.name}",
            "subtitle": str(target),
            "timestamp": now_stamp(),
            "file_mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "node_count": snapshot["node_count"],
                "counts": dict(counter),
            },
            "sections": [
                {
                    "title": "Dodane",
                    "items": [f"L{item.line}: {item.text} [{status_label(item.status)}]" for item in added],
                },
                {
                    "title": "Usuniete",
                    "items": [f"L{item.line}: {item.text} [{status_label(item.status)}]" for item in removed],
                },
                {
                    "title": "Zmieniony status",
                    "items": [
                        f"L{after.line}: {after.text} [{status_label(before.status)} -> {status_label(after.status)}]"
                        for before, after in changed
                    ],
                },
                {
                    "title": "Zmieniona paleta CSS",
                    "items": [f"{name}: {before or '<brak>'} -> {after or '<brak>'}" for name, before, after in palette_changed],
                },
            ],
        }
    )
    save_state(snapshot, target)


def watch(target: Path, poll_interval: float, once: bool) -> int:
    last_snapshot: dict[str, object] | None = None
    last_digest: str | None = None
    missing_reported = False

    while True:
        if not target.exists():
            if not missing_reported:
                report_missing(target)
                missing_reported = True
            if once:
                return 1
            time.sleep(poll_interval)
            continue

        missing_reported = False
        content = target.read_text(encoding="utf-8", errors="replace")
        snapshot = build_snapshot(content)
        digest = snapshot["digest"]  # type: ignore[index]

        if last_snapshot is None:
            report_initial(snapshot, target)
            last_snapshot = snapshot
            last_digest = digest  # type: ignore[assignment]
            if once:
                return 0
        elif digest != last_digest:
            diff = diff_snapshots(last_snapshot, snapshot)
            report_change(target, diff, snapshot)
            last_snapshot = snapshot
            last_digest = digest  # type: ignore[assignment]
            if once:
                return 0

        time.sleep(poll_interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Watchdog zmian dla qa_ankiety_architektura_status.html")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    try:
        return watch(args.target.resolve(), args.poll_interval, args.once)
    except KeyboardInterrupt:
        print()
        print(colorize("WATCHDOG STOPPED", "gray", bold=True))
        append_log(f"[{now_stamp()}] STOP {args.target}")
        record_event(
            {
                "kind": "stop",
                "kind_label": "STOP",
                "title": f"Watchdog zatrzymany dla {args.target.name}",
                "subtitle": str(args.target),
                "timestamp": now_stamp(),
                "file_mtime": "",
                "summary": {"node_count": 0, "counts": {}},
                "sections": [],
            }
        )
        return 130


if __name__ == "__main__":
    sys.exit(main())
