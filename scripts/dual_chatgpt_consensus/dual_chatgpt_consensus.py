# dual_chatgpt_consensus.py — orchestrator z dwoma agentami (mouse/click vs kbd/scroll)
import argparse
import json
import os
import random
import socket
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------- UDP helpers ----------
def send_udp(port: int, payload: Dict[str, Any]):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.sendto(json.dumps(payload).encode("utf-8"), ("127.0.0.1", port))
    finally:
        s.close()

def send_mouse_move_click(port_mouse: int, x: int, y: int, speed="normal", min_total_ms=80.0):
    send_udp(port_mouse, {
        "cmd": "move",
        "x": int(x),
        "y": int(y),
        "press": "mouse",
        "speed": speed,
        "min_total_ms": float(min_total_ms)
    })

def send_kbd_keys(port_kbd: int, combo: str):
    send_udp(port_kbd, {"cmd": "keys", "combo": combo})

def send_kbd_type(port_kbd: int, text: str):
    send_udp(port_kbd, {"cmd": "type", "text": text})

def send_kbd_paste(port_kbd: int, text: str):
    send_udp(port_kbd, {"cmd": "paste", "text": text})

def send_kbd_scroll(port_kbd: int, direction="down", amount=3, duration=0.4):
    send_udp(port_kbd, {
        "cmd": "scroll",
        "direction": direction,
        "amount": amount,
        "duration": duration
    })

def send_mouse_scroll(port_mouse: int, direction="down", amount=2, duration=0.35):
    send_udp(port_mouse, {
        "cmd": "scroll",
        "direction": direction,
        "amount": amount,
        "duration": duration
    })

# ---------- File helpers ----------
def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def wait_for_file(path: str, timeout: float = 60.0, poll: float = 0.2):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path):
            return
        time.sleep(poll)
    raise RuntimeError(f"Timeout waiting for {path}")

def load_clickables(rec_dir: str) -> List[Dict[str, Any]]:
    return load_json(os.path.join(rec_dir, "current_clickables.json")) or []

# ---------- Matching by CSS "hints" ----------
def _css_tokens(hint: str) -> List[str]:
    parts = [p.strip().lower() for p in hint.split(">") if p.strip()]
    tail = parts[-2:] if len(parts) >= 2 else parts
    atoms: List[str] = []
    for p in tail:
        atoms.extend([a for a in p.split() if a])
    return [a for a in atoms if a]

def match_clickable_by_hint(clickables: List[Dict[str, Any]], hint: str) -> Optional[Dict[str, Any]]:
    if not hint:
        return None
    atoms = _css_tokens(hint)
    best = None
    for c in clickables:
        attrs = (c.get("attributes") or {})
        aria  = (attrs.get("aria") or {})
        hay = " | ".join([
            (attrs.get("selector") or ""),
            (aria.get("label") or ""),
            (c.get("role") or ""),
            (c.get("type") or ""),
            (c.get("text") or "")
        ]).lower()
        if all(a in hay for a in atoms):
            if best is None or c["bbox"].get("center_y", 0) > best["bbox"].get("center_y", 0):
                best = c
    return best

def random_point_in_bbox(bbox: Dict[str, Any], jitter_ratio: float = 0.22) -> Tuple[int, int]:
    cx, cy = bbox.get("center_x", 0), bbox.get("center_y", 0)
    w, h = bbox.get("width", 0), bbox.get("height", 0)
    jx = (random.uniform(-jitter_ratio, jitter_ratio) * max(6, w * jitter_ratio))
    jy = (random.uniform(-jitter_ratio, jitter_ratio) * max(6, h * jitter_ratio))
    return int(cx + jx), int(cy + jy)

# ---------- Waiting helpers ----------
def wait_css_absent(rec_dir: str, css_hint: str, timeout_s: float = 180.0, poll_s: float = 0.6):
    if not css_hint:
        return
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        clickables = load_clickables(rec_dir)
        if match_clickable_by_hint(clickables, css_hint) is None:
            return
        time.sleep(poll_s)
    raise RuntimeError(f"Timeout waiting for element to disappear: {css_hint}")

def wait_last_present(rec_dir: str, css_hint: str, timeout_s: float = 120.0, poll_s: float = 0.6) -> Dict[str, Any]:
    if not css_hint:
        return {}
    atoms = _css_tokens(css_hint)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        clickables = load_clickables(rec_dir)
        matches = []
        for c in clickables:
            attrs = (c.get("attributes") or {})
            aria  = (attrs.get("aria") or {})
            hay = " | ".join([
                (attrs.get("selector") or ""),
                (aria.get("label") or ""),
                (c.get("role") or ""),
                (c.get("type") or ""),
                (c.get("text") or "")
            ]).lower()
            if all(a in hay for a in atoms):
                matches.append(c)
        if matches:
            # ostatnia (najniżej)
            return max(matches, key=lambda c: c["bbox"].get("center_y", 0))
        time.sleep(poll_s)
    raise RuntimeError(f"Timeout waiting for element present: {css_hint}")

# ---------- Human-like helpers ----------
def human_scroll_mix(rec_dir: str, port_mouse: int, port_kbd: int, down=True, total_notches=6):
    """
    Dla naturalności: trochę scrolla przez kbd_agent, trochę przez control_agent.
    """
    dir_name = "down" if down else "up"
    # porcja #1 kbd
    a1 = max(1, int(total_notches * random.uniform(0.3, 0.6)))
    send_kbd_scroll(port_kbd, direction=dir_name, amount=a1, duration=random.uniform(0.25, 0.5))
    time.sleep(random.uniform(0.08, 0.2))
    # porcja #2 mouse
    a2 = max(1, total_notches - a1)
    send_mouse_scroll(port_mouse, direction=dir_name, amount=a2, duration=random.uniform(0.25, 0.45))
    time.sleep(random.uniform(0.05, 0.15))

# ---------- Main actions ----------

def _wait_for_composer(rec_dir: str, input_hint: str, timeout_s: float = 120.0, poll_s: float = 0.5) -> Dict[str, Any]:
    """Waits for composer (hint or textbox fallback)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        clickables = load_clickables(rec_dir)
        hit = match_clickable_by_hint(clickables, input_hint) if input_hint else None
        if not hit:
            candidates = []
            for c in clickables:
                role = (c.get("role") or "").lower()
                tag = (c.get("tag") or "").lower()
                if role in ("textbox",) or tag in ("textarea", "input"):
                    candidates.append(c)
            if candidates:
                hit = max(candidates, key=lambda c: (c["bbox"].get("width", 0), c["bbox"].get("center_y", 0)))
        if hit:
            return hit
        time.sleep(poll_s)
    raise RuntimeError("Cannot locate ChatGPT input box (timeout)")
def paste_into_composer(rec_dir: str, port_mouse: int, port_kbd: int, input_hint: str, text: str):
    hit = _wait_for_composer(rec_dir, input_hint)
    x, y = random_point_in_bbox(hit["bbox"], jitter_ratio=0.22)
    send_mouse_move_click(port_mouse, x, y, speed="normal", min_total_ms=90.0)
    time.sleep(random.uniform(0.20, 0.35))

    # Naturalnie: czasem zrobimy micro-scroll w dół, żeby "dociągnąć" composer
    if random.random() < 0.35:
        human_scroll_mix(rec_dir, port_mouse, port_kbd, down=True, total_notches=random.randint(1, 3))

    # Preferujemy paste (minimalizuje literówki)
    send_kbd_paste(port_kbd, text)
    time.sleep(random.uniform(0.08, 0.18))
    send_kbd_keys(port_kbd, "enter")
    time.sleep(random.uniform(0.35, 0.6))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec-dir", type=str, required=True)
    ap.add_argument("--port-mouse", type=int, default=8765, help="UDP port control_agent (move/click/scroll)")
    ap.add_argument("--port-kbd", type=int, default=8766, help="UDP port kbd_scroll_agent (type/keys/scroll)")
    ap.add_argument("--max-rounds", type=int, default=1)
    ap.add_argument("--reply-wait", type=int, default=12)
    # HINTy CSS:
    ap.add_argument("--input-css", type=str, default="")
    ap.add_argument("--spinner-css", type=str, default="")
    ap.add_argument("--assistant-ready-css", type=str, default="")
    args = ap.parse_args()

    click_path = os.path.join(args.rec_dir, "current_clickables.json")
    wait_for_file(click_path, timeout=60.0)

    print("Initial prompt for Chat A:", end=" ")
    sys.stdout.flush()
    try:
        initial_prompt = input()
    except EOFError:
        initial_prompt = ""

    # lekki pre-scroll na dół, by na pewno widzieć composer (naturalnie, mieszanka)
    if random.random() < 0.6:
        human_scroll_mix(args.rec_dir, args.port_mouse, args.port_kbd, down=True, total_notches=random.randint(2, 6))

    print("[*] Clicking composer (mouse agent) and sending text (kbd agent)...")
    paste_into_composer(args.rec_dir, args.port_mouse, args.port_kbd, args.input_css, initial_prompt)

    if args.spinner_css:
        print("[*] Waiting for 'thinking' spinner to disappear...")
        wait_css_absent(args.rec_dir, args.spinner_css, timeout_s=180.0, poll_s=0.6)

    if args.assistant_ready_css:
        print("[*] Waiting for assistant last-reply control to appear...")
        last = wait_last_present(args.rec_dir, args.assistant_ready_css, timeout_s=120.0, poll_s=0.6)
        if last:
            print(f"[OK] Last reply control detected (y={last['bbox'].get('center_y', 0)})")

    print("\n[DONE] Etap Chat A zakończony.\n")

if __name__ == "__main__":
    main()
