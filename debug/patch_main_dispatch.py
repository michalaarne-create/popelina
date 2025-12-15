from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    main_path = root / "main.py"

    text = main_path.read_text(encoding="utf-8")

    marker = 'debug(f"hover path visual failed: {exc}")'
    idx = text.find(marker)
    if idx == -1:
        raise SystemExit("marker not found")

    # Szukamy poczŽtku bloku try tu‘• przed logiem.
    try_start = text.rfind("    try:", 0, idx)
    if try_start == -1:
        raise SystemExit("try block not found")

    # Koniec bloku to koniec linii z debug(...).
    line_end = text.find("\n", idx)
    if line_end == -1:
        line_end = len(text)
    try_block = text[try_start : line_end + 1]

    new_block = (
        '    try:\n'
        '        pts = payload.get("points") or []\n'
        '        if pts:\n'
        '            _save_hover_path_visual(pts, points_json)\n'
        '        # Równolegle budujemy overlay: screenshot + boxy + kropki jako\n'
        '        # hover_output_current/hover_output.png (dla Flow UI).\n'
        '        _save_hover_overlay_from_json(points_json)\n'
        '    except Exception as exc:\n'
        '        debug(f"hover path/overlay visual failed: {exc}")\n'
    )

    if try_block.strip() == new_block.strip():
        print("try block already patched")
        return

    text = text[:try_start] + new_block + text[line_end + 1 :]
    main_path.write_text(text, encoding="utf-8")
    print("try block patched")


if __name__ == "__main__":
    main()

