import argparse
import json
import socket
import sys
import time
from typing import Any, Dict, Tuple, Optional

import ctypes

user32 = ctypes.windll.user32

WM_CHAR = 0x0102
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_MOUSEWHEEL = 0x020A

VK_RETURN = 0x0D
WHEEL_DELTA = 120

_target_hwnd: Optional[int] = None


def debug(msg: str):
    print(f"[kbd_agent] {msg}")
    sys.stdout.flush()


def find_chatgpt_window() -> Optional[int]:
    debug("Szukam okna ChatGPT dla kbd_agent...")

    candidates = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
    def enum_proc(hwnd, lParam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.lower()
        if "chatgpt" in title or "chatgpt.com" in title:
            candidates.append(hwnd)
        return True

    user32.EnumWindows(enum_proc, 0)

    if not candidates:
        debug("Nie znalazlem okna ChatGPT.")
        return None

    hwnd = candidates[0]
    debug(f"Znalazlem hwnd={hwnd} (kbd_agent).")
    return hwnd


def get_target_hwnd() -> Optional[int]:
    global _target_hwnd
    if _target_hwnd is not None and user32.IsWindow(_target_hwnd):
        return _target_hwnd
    _target_hwnd = find_chatgpt_window()
    return _target_hwnd


def send_char(c: str):
    hwnd = get_target_hwnd()
    if not hwnd:
        debug("Brak hwnd – nie moge wyslac WM_CHAR.")
        return
    code = ord(c)
    debug(f"WM_CHAR '{repr(c)}' ({code}) -> hwnd={hwnd}")
    user32.PostMessageW(hwnd, WM_CHAR, code, 0)


def send_text(text: str, delay: float = 0.01):
    for c in text:
        send_char(c)
        time.sleep(delay)


def press_enter():
    hwnd = get_target_hwnd()
    if not hwnd:
        debug("Brak hwnd – nie moge wyslac ENTER.")
        return
    debug(f"ENTER -> hwnd={hwnd}")
    user32.PostMessageW(hwnd, WM_KEYDOWN, VK_RETURN, 0)
    time.sleep(0.02)
    user32.PostMessageW(hwnd, WM_KEYUP, VK_RETURN, 0)


def scroll(direction: str = "down", amount: int = 2, duration: float = 0.35):
    hwnd = get_target_hwnd()
    if not hwnd:
        debug("Brak hwnd – scroll (kbd_agent) ignoruje.")
        return

    delta = -WHEEL_DELTA if direction.lower() == "down" else WHEEL_DELTA
    steps = max(1, int(amount))
    delay = max(0.01, duration / steps if duration > 0 else 0.01)

    debug(f"Scroll (kbd_agent) dir={direction} amount={amount} duration={duration} hwnd={hwnd}")

    for _ in range(steps):
        user32.PostMessageW(hwnd, WM_MOUSEWHEEL, ctypes.c_int(delta << 16).value, 0)
        time.sleep(delay)


def handle_packet(data: bytes, addr: Tuple[str, int]):
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as e:
        debug(f"Nie moge zdekodowac JSON: {e}")
        return

    cmd = payload.get("cmd")
    if cmd == "paste":
        text = payload.get("text", "")
        debug(f"CMD paste, dlugosc={len(text)}")
        send_text(text, delay=0.005)

    elif cmd == "type":
        text = payload.get("text", "")
        debug(f"CMD type, dlugosc={len(text)}")
        send_text(text, delay=0.03)

    elif cmd == "keys":
        combo = (payload.get("combo") or "").lower()
        debug(f"CMD keys combo={combo}")
        # Na razie tylko ENTER
        if combo in ("enter", "return"):
            press_enter()
        else:
            debug(f"Nieobslugiwany combo={combo}")

    elif cmd == "scroll":
        direction = payload.get("direction", "down")
        amount = payload.get("amount", 2)
        duration = payload.get("duration", 0.35)
        try:
            amount = int(amount)
        except Exception:
            amount = 2
        try:
            duration = float(duration)
        except Exception:
            duration = 0.35
        scroll(direction=direction, amount=amount, duration=duration)

    else:
        debug(f"Nieznane cmd={cmd} payload={payload}")


def run_server(port: int):
    debug(f"Startuje kbd_agent, UDP 127.0.0.1:{port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))

    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except Exception as e:
            debug(f"Blad recvfrom: {e}")
            time.sleep(0.2)
            continue
        handle_packet(data, addr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766, help="UDP port kbd_agent")
    args = ap.parse_args()

    debug(f"Uruchamiam kbd_scroll_agent na porcie {args.port}")
    try:
        run_server(args.port)
    except KeyboardInterrupt:
        debug("Przerwane przez uzytkownika (Ctrl+C)")

if __name__ == "__main__":
    main()
