import argparse
import json
import socket
import sys
import threading
import time
from typing import Any, Dict, Tuple, Optional

import ctypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# --- Windows constants ---
WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_MOUSEWHEEL = 0x020A

MK_LBUTTON = 0x0001
WHEEL_DELTA = 120

# Prosty cache hwnd
_target_hwnd: Optional[int] = None


def debug(msg: str):
    print(f"[control_agent] {msg}")
    sys.stdout.flush()


def find_chatgpt_window() -> Optional[int]:
    """
    Szuka okna, którego tytuł zawiera 'chatgpt' lub 'chatgpt.com'.
    To jest okno, do którego będziemy wysyłać kliknięcia / scroll.
    """
    debug("Szukam okna ChatGPT...")

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
        debug("Nie znalazlem zadnego okna z 'ChatGPT' w tytule.")
        return None

    hwnd = candidates[0]
    debug(f"Znalazlem hwnd={hwnd} jako okno ChatGPT.")
    return hwnd


def get_target_hwnd() -> Optional[int]:
    global _target_hwnd
    if _target_hwnd is not None and user32.IsWindow(_target_hwnd):
        return _target_hwnd
    _target_hwnd = find_chatgpt_window()
    return _target_hwnd


def _make_lparam_xy(x: int, y: int) -> int:
    # lParam = (y << 16) | (x & 0xFFFF)
    return (int(y) << 16) | (int(x) & 0xFFFF)


def click_at(x: int, y: int):
    """
    Klikniecie w oknie ChatGPT, BEZ ruszania fizycznej myszy.
    Zakladamy, ze x,y to wspolrzedne w przestrzeni 'client' (jak z DOM-u, viewport).
    """
    hwnd = get_target_hwnd()
    if not hwnd:
        debug("Brak hwnd ChatGPT – nie moge kliknac.")
        return

    lparam = _make_lparam_xy(x, y)
    debug(f"Klikam w oknie hwnd={hwnd} w punkt (x={x}, y={y})")

    # MOUSEMOVE (czasem pomaga zaktualizowac hover)
    user32.PostMessageW(hwnd, WM_MOUSEMOVE, 0, lparam)
    time.sleep(0.01)

    # LBUTTONDOWN / UP
    user32.PostMessageW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
    time.sleep(0.01)
    user32.PostMessageW(hwnd, WM_LBUTTONUP, 0, lparam)


def scroll(direction: str = "down", amount: int = 2, duration: float = 0.35):
    """
    Scroll w oknie ChatGPT przez WM_MOUSEWHEEL (bez ruszania kursora).
    amount – liczba 'notches', duration – laczny czas w sekundach (przyblizony).
    """
    hwnd = get_target_hwnd()
    if not hwnd:
        debug("Brak hwnd ChatGPT – nie moge scrollowac.")
        return

    # Delta: w dół = ujemny, w górę = dodatni
    delta = -WHEEL_DELTA if direction.lower() == "down" else WHEEL_DELTA
    steps = max(1, int(amount))
    delay = max(0.01, duration / steps if duration > 0 else 0.01)

    debug(f"Scroll {direction}, amount={amount}, duration={duration}, hwnd={hwnd}")

    # lParam: czesto ignorowany przez przegladarke, wiec dajemy (0,0)
    lparam = _make_lparam_xy(0, 0)
    for _ in range(steps):
        user32.PostMessageW(hwnd, WM_MOUSEWHEEL, ctypes.c_int(delta << 16).value, lparam)
        time.sleep(delay)


def handle_packet(data: bytes, addr: Tuple[str, int]):
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as e:
        debug(f"Nie moge zdekodowac JSON: {e}")
        return

    cmd = payload.get("cmd")
    if cmd == "move":
        # Interpretujemy "move" jako klikniecie w dane koordy z DOM
        x = int(payload.get("x", 0))
        y = int(payload.get("y", 0))
        debug(f"CMD move -> click_at({x}, {y})")
        click_at(x, y)

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
        debug(f"CMD scroll dir={direction} amount={amount} duration={duration}")
        scroll(direction=direction, amount=amount, duration=duration)

    else:
        debug(f"Otrzymalem nieznane cmd={cmd} z payload={payload}")


def run_server(port: int):
    debug(f"Startuje UDP server na 127.0.0.1:{port}")
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
    ap.add_argument("--port", type=int, default=8765, help="UDP port do nasluchu (move/scroll)")
    ap.add_argument(
        "--config",
        type=str,
        default="",
        help="Nieuzywane (dla kompatybilnosci z istniejacym .bat)",
    )
    args = ap.parse_args()

    debug(f"Uruchamiam control_agent na porcie {args.port}")
    debug(f"Ignoruje ewentualny plik config: {args.config}")

    try:
        run_server(args.port)
    except KeyboardInterrupt:
        debug("Przerwane przez uzytkownika (Ctrl+C)")

if __name__ == "__main__":
    main()
