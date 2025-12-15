# control_agent.py - POPRAWIONA WERSJA ZE SCROLLOWANIEM
import argparse
import json
import math
import os
import platform
import queue
import socket
import threading
import time
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random
from pynput.keyboard import Key

# Project root (for screenshots / hover path visuals)
AGENT_ROOT = Path(__file__).resolve().parents[2]
DATA_SCREEN_DIR = AGENT_ROOT / "data" / "screen"
HOVER_DIR = DATA_SCREEN_DIR / "hover"
RAW_DIR = DATA_SCREEN_DIR / "raw"
SCREENSHOT_DIR = RAW_DIR / "raw screen"
HOVER_INPUT_CURRENT_DIR = HOVER_DIR / "hover_input_current"
RAW_CURRENT_DIR = RAW_DIR / "raw_screens_current"
HOVER_PATH_DIR = HOVER_DIR / "hover_path"
HOVER_PATH_CURRENT_DIR = HOVER_DIR / "hover_path_current"
HOVER_SPEED_DIR = HOVER_DIR / "hover_speed"
HOVER_SPEED_CURRENT_DIR = HOVER_DIR / "hover_speed_current"
HOVER_SPEED_RECORDER = AGENT_ROOT / "scripts" / "control_agent" / "hover_speed_recorder.py"


def _align_path(points: List[Tuple[float, float]], start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
    if not points:
        return []
    src_dx = points[-1][0] - points[0][0]
    src_dy = points[-1][1] - points[0][1]
    dst_dx = end[0] - start[0]
    dst_dy = end[1] - start[1]
    src_len = math.hypot(src_dx, src_dy)
    dst_len = math.hypot(dst_dx, dst_dy)
    if src_len < 1e-6 or dst_len < 1e-6:
        return [start, end]
    scale = dst_len / src_len
    src_angle = math.atan2(src_dy, src_dx)
    dst_angle = math.atan2(dst_dy, dst_dx)
    rot = dst_angle - src_angle
    cos_r = math.cos(rot)
    sin_r = math.sin(rot)
    transformed = []
    for x, y in points:
        rel_x = x - points[0][0]
        rel_y = y - points[0][1]
        scaled_x = rel_x * scale
        scaled_y = rel_y * scale
        tx = scaled_x * cos_r - scaled_y * sin_r
        ty = scaled_x * sin_r + scaled_y * cos_r
        transformed.append((start[0] + tx, start[1] + ty))
    return transformed

# Opcjonalnie NumPy
try:
    import numpy as np
    _NP_OK = True
except Exception:
    _NP_OK = False

# Platform I/O
if platform.system() == "Windows":
    import ctypes
    class POINT(ctypes.Structure): _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    _user32 = ctypes.windll.user32
    try: _winmm = ctypes.windll.winmm
    except Exception: _winmm = None
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 76, 77, 78, 79
    def time_begin_period(ms=1):
        if _winmm:
            try: _winmm.timeBeginPeriod(ms)
            except Exception: pass
    def time_end_period(ms=1):
        if _winmm:
            try: _winmm.timeEndPeriod(ms)
            except Exception: pass
    def get_cursor_pos() -> Tuple[int, int]:
        pt = POINT(); _user32.GetCursorPos(ctypes.byref(pt)); return int(pt.x), int(pt.y)
    def set_cursor_pos_raw(x: int, y: int): _user32.SetCursorPos(int(x), int(y))
    def get_virtual_bounds() -> Tuple[int, int, int, int]:
        left, top, w, h = (_user32.GetSystemMetrics(m) for m in [SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN])
        if w <= 0 or h <= 0: left, top, w, h = 0, 0, _user32.GetSystemMetrics(0), _user32.GetSystemMetrics(1)
        return left, top, left + w - 1, top + h - 1
    def precise_sleep(seconds: float):
        if seconds <= 0: return
        if seconds < 0.002:
            target = time.perf_counter() + seconds
            while time.perf_counter() < target: pass
        else: time.sleep(seconds)
else:
    from pynput import mouse
    _mc = mouse.Controller()
    def time_begin_period(ms=1): pass
    def time_end_period(ms=1): pass
    def get_cursor_pos() -> Tuple[int, int]: x, y = _mc.position; return int(x), int(y)
    def set_cursor_pos_raw(x: int, y: int): _mc.position = (int(x), int(y))
    def get_virtual_bounds() -> Optional[Tuple[int, int, int, int]]: return None
    def precise_sleep(seconds: float): time.sleep(seconds)

SCREEN_BOUNDS = get_virtual_bounds()

def clamp_to_screen(x: int, y: int, margin: int = 1) -> Tuple[int, int]:
    if SCREEN_BOUNDS is None: return x, y
    l, t, r, b = SCREEN_BOUNDS
    return max(l + margin, min(r - margin, x)), max(t + margin, min(b - margin, y))

def clamp_path(points: List[Tuple[int, int]], margin: int = 1) -> List[Tuple[int, int]]:
    return [clamp_to_screen(x, y, margin) for (x, y) in points]

def _path_length(xy: List[Tuple[float, float]]) -> float:
    if len(xy) < 2: return 0.0
    s = 0.0
    for i in range(len(xy) - 1): s += math.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1])
    return s

def _curvature_sign_flips(points: List[Tuple[float, float]]) -> int:
    n = len(points)
    if n < 3: return 0
    signs = []
    for i in range(1, n-1):
        x0,y0 = points[i-1]; x1,y1 = points[i]; x2,y2 = points[i+1]
        cross = (x1-x0)*(y2-y1) - (y1-y0)*(x2-x1)
        s = 1 if cross > 1e-6 else (-1 if cross < -1e-6 else 0)
        if s != 0: signs.append(s)
    flips = 0
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]: flips += 1
    return flips

def apply_micro_jitter(points: List[Tuple[int, int]], max_jitter_px: float = 1.5, step_std: float = 0.4) -> List[Tuple[int, int]]:
    """
    Delikatny, gładki jitter boczny wzdłuż trajektorii.
    - brak teleportów (offset to mały random-walk),
    - pierwszy i ostatni punkt zostają bez zmian.
    """
    if len(points) < 3 or max_jitter_px <= 0.0:
        return points
    out: List[Tuple[int, int]] = [points[0]]
    lat_prev = 0.0
    for i in range(1, len(points) - 1):
        x_prev, y_prev = points[i - 1]
        x_next, y_next = points[i + 1]
        vx, vy = x_next - x_prev, y_next - y_prev
        L = math.hypot(vx, vy) or 1.0
        # jednostkowa normalna do kierunku lokalnego
        nx, ny = -vy / L, vx / L
        lat = lat_prev + random.gauss(0.0, step_std)
        lat = max(-max_jitter_px, min(max_jitter_px, lat))
        x_j = points[i][0] + nx * lat
        y_j = points[i][1] + ny * lat
        out.append((int(round(x_j)), int(round(y_j))))
        lat_prev = lat
    out.append(points[-1])
    return out

def moving_average(points: List[Tuple[int,int]], win: int) -> List[Tuple[int,int]]:
    if win <= 1 or len(points) <= 2: return points
    win = max(3, win | 1)
    half = win // 2
    out = []
    for i in range(len(points)):
        xsum, ysum, cnt = 0, 0, 0
        for j in range(max(0, i-half), min(len(points), i+half+1)):
            xsum += points[j][0]; ysum += points[j][1]; cnt += 1
        out.append((int(round(xsum/cnt)), int(round(ysum/cnt))))
    return out

def limit_lateral(points: List[Tuple[int,int]], start: Tuple[int,int], end: Tuple[int,int], max_lat: float) -> List[Tuple[int,int]]:
    sx, sy = start; tx, ty = end
    vx, vy = tx - sx, ty - sy
    L = math.hypot(vx, vy) or 1.0
    ux, uy = vx / L, vy / L
    px, py = -uy, ux
    out = []
    for x, y in points:
        rx, ry = x - sx, y - sy
        lat = rx*px + ry*py
        lat = max(-max_lat, min(max_lat, lat))
        longi = rx*ux + ry*uy
        nx = sx + longi*ux + lat*px
        ny = sy + longi*uy + lat*py
        out.append((int(round(nx)), int(round(ny))))
    out[-1] = (tx, ty)
    return out

class ExactLibrary:
    def __init__(self, data_path: str, slider_threshold: float = 0.05, max_path_length: float = 1900.0):
        self.samples: List[Dict[str, Any]] = []
        self.max_path_length = max_path_length
        self._load(data_path, slider_threshold)
        self._debug_statistics()
    
    def _load(self, data_path: str, slider_threshold: float):
        try:
            with open(data_path, "r", encoding="utf-8") as f: data = json.load(f)
        except Exception as e:
            print(f"[ExactLib] load error: {e}"); return
        
        total_read = 0
        rejected_length = 0
        rejected_norm = 0
        rejected_time = 0
        
        for ex in data:
            total_read += 1
            traj = ex.get("trajectory")
            if not traj or len(traj) < 3: continue
            
            pts = [(float(p[0]), float(p[1])) for p in traj]
            times = [float(p[2]) for p in traj]
            times = [t - times[0] for t in times]
            
            if times[-1] <= 0:
                rejected_time += 1
                continue
            
            start, end = pts[0], pts[-1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            norm = math.hypot(dx, dy)
            
            if norm < 2.0:
                rejected_norm += 1
                continue
            
            path_len = _path_length(pts)
            
            if path_len > self.max_path_length:
                rejected_length += 1
                continue
            
            is_slider_from_json = ex.get("is_slider", None)
            if is_slider_from_json is not None:
                is_slider = bool(is_slider_from_json)
            else:
                is_slider = times[-1] >= slider_threshold
            
            self.samples.append({
                "pts": pts,
                "times": times,
                "theta": math.atan2(dy, dx),
                "norm": norm,
                "path_length": path_len,
                "is_slider": is_slider,
                "straightness": float(norm / max(1e-6, path_len)),
                "flips": int(_curvature_sign_flips(pts)),
            })
        
        print(f"\n[ExactLib] Loaded {len(self.samples)} trajectories")
    
    def _debug_statistics(self):
        if not self.samples:
            print("[DEBUG] No trajectories loaded!")
            return
        
        total_count = len(self.samples)
        slider_count = sum(1 for s in self.samples if s["is_slider"])
        avg_straight_dist = sum(s["norm"] for s in self.samples) / total_count
        avg_path_length = sum(s["path_length"] for s in self.samples) / total_count
        
        print(f"[Stats] Total: {total_count}, Sliders: {slider_count}")
        print(f"[Stats] Avg straight: {avg_straight_dist:.0f}px, Avg path: {avg_path_length:.0f}px")
    
    @staticmethod
    def _angle_diff(a: float, b: float) -> float: 
        return abs((a - b + math.pi) % (2 * math.pi) - math.pi)
    
    def pick_best(self, start: Tuple[int, int], target: Tuple[int, int], want_slider: Optional[bool], curvy: str = "auto", curvy_min: Optional[float] = None) -> Optional[Dict[str, Any]]:
        sx, sy = start; tx, ty = target
        vx, vy = tx - sx, ty - sy
        vnorm = math.hypot(vx, vy)
        if vnorm < 1.0: return None
        vtheta = math.atan2(vy, vx)
        thr = {"less": 0.995, "auto": 0.985, "more": 0.970}.get(curvy, 0.985)
        if curvy_min is not None: thr = float(curvy_min)
        pool = [s for s in self.samples if (want_slider is None or s["is_slider"] == want_slider) and s["straightness"] >= thr]
        if not pool:
            pool = sorted([s for s in self.samples if (want_slider is None or s["is_slider"] == want_slider)], key=lambda s: s["straightness"], reverse=True)
            pool = pool[:min(50, len(pool))]
        
        scale_candidates = []
        for s in pool:
            scale = vnorm / max(1e-6, s["norm"])
            if s.get("flips", 0) <= (3 if scale > 1.5 else 5):
                scale_candidates.append(s)
        if not scale_candidates: scale_candidates = pool

        best, best_score = None, 1e9
        for s in scale_candidates:
            angle_pen = self._angle_diff(vtheta, s["theta"])
            scale_pen = abs(math.log(vnorm / max(1e-6, s["norm"])))
            flips_pen = (0.15 if (vnorm / max(1e-6, s["norm"])) > 1.5 else 0.08) * max(0, s.get("flips", 0) - 1)
            straight_bonus = 1.2 - s["straightness"]
            score = angle_pen + 0.25 * scale_pen + flips_pen + straight_bonus
            if score < best_score: best, best_score = s, score
        return best if best else (max(self.samples, key=lambda s: s["straightness"]) if self.samples else None)
    
    def pick_random(self, start: Tuple[int, int], target: Tuple[int, int], want_slider: Optional[bool], curvy: str = "auto", top_k: int = 30) -> Optional[Dict[str, Any]]:
        sx, sy = start; tx, ty = target
        vx, vy = tx - sx, ty - sy
        vnorm = math.hypot(vx, vy)
        if vnorm < 1.0: return None
        vtheta = math.atan2(vy, vx)
        thr = {"less": 0.995, "auto": 0.985, "more": 0.970}.get(curvy, 0.985)
        pool = [s for s in self.samples if (want_slider is None or s["is_slider"] == want_slider) and s["straightness"] >= thr]
        if not pool:
            pool = sorted([s for s in self.samples if (want_slider is None or s["is_slider"] == want_slider)], key=lambda s: s["straightness"], reverse=True)
            pool = pool[:min(80, len(pool))]
        if not pool: return None

        scored = []
        for s in pool:
            angle_pen = self._angle_diff(vtheta, s["theta"])
            scale_pen = abs(math.log(vnorm / max(1e-6, s["norm"])))
            flips_pen = (0.15 if (vnorm / max(1e-6, s["norm"])) > 1.5 else 0.08) * max(0, s.get("flips", 0) - 1)
            straight_bonus = 1.2 - s["straightness"]
            score = angle_pen + 0.25 * scale_pen + flips_pen + straight_bonus
            scored.append((score, s))

        scored.sort(key=lambda t: t[0])
        k = min(top_k, len(scored))
        choice = random.choice(scored[:k]) if scored else None
        return choice[1] if choice else None
    
    @staticmethod
    def retarget(sample: Dict[str, Any], start_xy: Tuple[int, int], end_xy: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], List[float]]:
        sx, sy = start_xy; tx, ty = end_xy
        pts, times = sample["pts"], sample["times"]
        ex0, ey0 = pts[0]; ex1, ey1 = pts[-1]
        evx, evy = ex1 - ex0, ey1 - ey0
        vtx, vty = tx - sx, ty - sy
        en, tn = math.hypot(evx, evy), math.hypot(vtx, vty)
        if en < 1e-6 or tn < 1e-6: return [(sx, sy), (tx, ty)], [0.0, 0.016]
        scale = tn / en
        phi = math.atan2(vty, vtx) - math.atan2(evy, evx)
        c, s = math.cos(phi), math.sin(phi)
        out_pts: List[Tuple[int, int]] = []
        for (x, y) in pts:
            xr, yr = x - ex0, y - ey0
            xr2, yr2 = scale * (xr * c - yr * s), scale * (xr * s + yr * c)
            out_pts.append((int(round(sx + xr2)), int(round(sy + yr2))))
        out_pts[-1] = (tx, ty)
        t_src, t_tgt = times[-1], max(0.016, times[-1] * scale)
        return out_pts, [(t / t_src) * t_tgt for t in times]

def _cum_arclen(pts: List[Tuple[int,int]]) -> List[float]:
    if len(pts) < 2: return [0.0]*len(pts)
    out=[0.0]
    for i in range(1,len(pts)): out.append(out[-1] + math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]))
    return out

def _ease_min_jerk(u: float) -> float: u = max(0.0, min(1.0, u)); return u*u*u*(10 - 15*u + 6*u*u)

def _blend_lead_in(pts: List[Tuple[int,int]], times: List[float], lead_px: float = 24.0, lead_ms: float = 80.0) -> Tuple[List[Tuple[int,int]], List[float]]:
    n = len(pts)
    if n < 3 or times[-1] <= 0: return pts, times
    cum = _cum_arclen(pts)
    anchor = max(2, min(next((i for i, d in enumerate(cum) if d >= max(lead_px, 0.03 * cum[-1])), n-1), n-1))
    total_T, lead_T = times[-1], min(lead_ms/1000.0, 0.45 * times[-1])
    P0, P3 = pts[0], pts[anchor]
    v0, v1 = (pts[1][0]-pts[0][0], pts[1][1]-pts[0][1]), (pts[anchor][0]-pts[anchor-1][0], pts[anchor][1]-pts[anchor-1][1])
    def unit(v): L = math.hypot(v[0], v[1]) or 1.0; return (v[0]/L, v[1]/L)
    t0, t1 = unit(v0), unit(v1)
    ctrl_len = max(8.0, min(max(lead_px, 0.03 * cum[-1])*0.6, 0.5*math.hypot(P3[0]-P0[0], P3[1]-P0[1])))
    P1, P2 = (P0[0] + t0[0]*ctrl_len, P0[1] + t0[1]*ctrl_len), (P3[0] - t1[0]*ctrl_len, P3[1] - t1[1]*ctrl_len)
    m = max(8, anchor*3)
    lead_pts, lead_times = [], []
    for i in range(m):
        u = _ease_min_jerk(i/(m-1)); a = 1-u
        x = a**3*P0[0] + 3*a**2*u*P1[0] + 3*a*u**2*P2[0] + u**3*P3[0]
        y = a**3*P0[1] + 3*a**2*u*P1[1] + 3*a*u**2*P2[1] + u**3*P3[1]
        lead_pts.append((int(round(x)), int(round(y)))); lead_times.append(lead_T * (i/(m-1)))
    lead_pts[-1] = P3
    rest_pts, rest_times = pts[anchor+1:], [lead_T + (t - times[anchor])* (times[-1] - lead_T) / max(1e-6, times[-1] - times[anchor]) for t in times[anchor+1:]]
    return lead_pts + rest_pts, lead_times + rest_times

def _ease_slow_last_third(times: List[float], strength: float = 0.9) -> List[float]:
    if len(times) < 2 or times[-1] <= 0: return times
    T = times[-1]
    dt = [times[i+1] - times[i] for i in range(len(times)-1)]
    out_dt = [(d * (1.0 + strength * (((times[i]+times[i+1])*0.5/T - 2/3)*3)**1.2)) if (times[i]+times[i+1])*0.5/T > 2/3 else d for i, d in enumerate(dt)]
    scale = T / sum(out_dt) if sum(out_dt) > 0 else 1.0
    new_times = [0.0]
    for d in out_dt: new_times.append(new_times[-1] + d * scale)
    new_times[-1] = T
    return new_times

class UdpReceiver(threading.Thread):
    def __init__(self, port: int, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self.port, self.out_queue = port, out_queue
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", self.port))
        self.stop_event = threading.Event()
    def run(self):
        print(f"[UDP] Listening on 127.0.0.1:{self.port}")
        self.sock.settimeout(0.5)
        while not self.stop_event.is_set():
            try: data, _ = self.sock.recvfrom(65535)
            except socket.timeout: continue
            try:
                msg = json.loads(data.decode("utf-8"))
                if isinstance(msg, dict): self.out_queue.put(msg)
            except Exception: pass

class InputController:
    def __init__(self):
        from pynput import keyboard, mouse
        self.kb_ctrl, self.ms_ctrl = keyboard.Controller(), mouse.Controller()
        
    def press(self, key_or_button: str):
        from pynput import mouse
        if key_or_button == "mouse": self.ms_ctrl.press(mouse.Button.left)
        else: self.kb_ctrl.press(key_or_button)
        
    def release(self, key_or_button: str):
        from pynput import mouse
        if key_or_button == "mouse": self.ms_ctrl.release(mouse.Button.left)
        else: self.kb_ctrl.release(key_or_button)
    
    def scroll(self, notches: int):
        """Scrolluj o określoną liczbę 'ząbków' (dodatnie = w dół, ujemne = w górę)"""
        if notches == 0:
            return
        # W pynput ujemne = w górę, dodatnie = w dół
        self.ms_ctrl.scroll(0, -notches)

    # MAPOWANIE STRING -> KEY / ZNAK
    def _to_key(self, name: str):
        name = name.lower()
        special = {
            "ctrl": Key.ctrl,
            "control": Key.ctrl,
            "shift": Key.shift,
            "alt": Key.alt,
            "cmd": Key.cmd,
            "win": Key.cmd,
            "enter": Key.enter,
            "tab": Key.tab,
            "esc": Key.esc,
            "escape": Key.esc,
            "space": Key.space,
            "backspace": Key.backspace,
            "delete": Key.delete,
        }
        return special.get(name, name)

    def hotkey(self, *names: str, delay: float = 0.02):
        keys = [self._to_key(n) for n in names]
        for k in keys:
            self.kb_ctrl.press(k)
        time.sleep(delay)
        for k in reversed(keys):
            self.kb_ctrl.release(k)

    def type_text(self, text: str, delay: float = 0.0):
        for ch in text:
            self.kb_ctrl.press(ch)
            self.kb_ctrl.release(ch)
            if delay > 0:
                time.sleep(delay)

    def paste(self):
        self.hotkey("ctrl", "v")

    def copy(self):
        self.hotkey("ctrl", "c")

class ControlAgent:
    def __init__(self, cfg_path: str, udp_port: int = 8765, verbose: bool = False):
        self.cfg = self._load_cfg(cfg_path)
        data_rel = self.cfg.get("dataset", {}).get("source", "trajectory.json")
        data_path = data_rel if os.path.isabs(data_rel) else os.path.join(os.path.dirname(os.path.abspath(cfg_path)), data_rel)
        self.lib = ExactLibrary(data_path=data_path, slider_threshold=0.05, max_path_length=1900.0)
        self.cmd_queue: queue.Queue = queue.Queue()
        self.receiver = UdpReceiver(udp_port, self.cmd_queue)
        self.input_ctrl = InputController()
        self.verbose = bool(verbose or os.environ.get("CONTROL_AGENT_VERBOSE") == "1")
        
        print("[Agent] Using OSU trajectories. Movements will be accelerated.")
        if SCREEN_BOUNDS: 
            print(f"[Agent] Screen bounds detected: {SCREEN_BOUNDS}")

    def _load_cfg(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f: return json.load(f)
        except Exception: return {}

    def start(self):
        self.receiver.start()
        print(f"[Agent] Ready. Waiting for UDP commands on port {self.receiver.port}...")
        try:
            while True:
                try: cmd = self.cmd_queue.get(timeout=0.1)
                except queue.Empty: continue
                self.handle_command(cmd)
        except KeyboardInterrupt: pass
        finally:
            self.receiver.stop_event.set()
            self.receiver.join(timeout=1.0)
            print("[Agent] Stopped.")

    def handle_command(self, cmd: Dict[str, Any]):
        c = (cmd.get("cmd") or "move").lower()
        if c == "move":
            self._cmd_move(cmd)
        elif c == "scroll":
            self._cmd_scroll(cmd)
        elif c == "path":
            self._cmd_path(cmd)
        elif c == "keys":
            self._cmd_keys(cmd)
        elif c == "type":
            self._cmd_type(cmd)
        elif c == "paste":
            self.input_ctrl.paste()
        else:
            print(f"[Agent] Unknown command: {c}")

    def _render_hover_path(self, trace_stem: str, pts: List[Tuple[int,int]]) -> None:
        """
        Render the true executed hover path over a screenshot.

        Historical PNGs -> DATA_SCREEN_DIR/hover/hover_path/<stem>_hover_path.png
        Current PNG     -> DATA_SCREEN_DIR/hover/hover_path_current/hover_path.png

        Colour encoding (per pixel visit count):
            1x  -> green
            2x  -> blue
            3x  -> yellow
            4x  -> orange
            5+x -> red
        """
        if not pts or len(pts) < 2:
            return
        stem = trace_stem.strip()
        if not stem:
            return
        screen_path = SCREENSHOT_DIR / f"{stem}.png"
        if screen_path.exists():
            src_img = screen_path
        else:
            candidate = HOVER_INPUT_CURRENT_DIR / "hover_input.png"
            if candidate.exists():
                src_img = candidate
            else:
                src_img = RAW_CURRENT_DIR / "screenshot.png"
        if not src_img.exists():
            return
        try:
            from PIL import Image  # type: ignore
            import numpy as _np  # local alias
        except Exception:
            return
        try:
            img = Image.open(src_img).convert("RGB")
        except Exception:
            return
        arr = _np.array(img, dtype=_np.uint8)
        h, w, _ = arr.shape
        counts = _np.zeros((h, w), dtype=_np.uint8)

        def _clip_xy(x: int, y: int) -> Tuple[int, int]:
            return int(_np.clip(x, 0, w - 1)), int(_np.clip(y, 0, h - 1))

        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            x0, y0 = _clip_xy(x0, y0)
            x1, y1 = _clip_xy(x1, y1)
            dx = x1 - x0
            dy = y1 - y0
            steps = max(abs(dx), abs(dy))
            if steps == 0:
                counts[y0, x0] = _np.clip(counts[y0, x0] + 1, 0, 255)
                continue
            xs = _np.linspace(x0, x1, steps + 1, dtype=_np.int32)
            ys = _np.linspace(y0, y1, steps + 1, dtype=_np.int32)
            counts[ys, xs] = _np.clip(counts[ys, xs] + 1, 0, 255)

        out = arr.copy()
        mask1 = counts == 1
        out[mask1] = _np.array([0, 255, 0], dtype=_np.uint8)
        mask2 = counts == 2
        out[mask2] = _np.array([0, 0, 255], dtype=_np.uint8)
        mask3 = counts == 3
        out[mask3] = _np.array([255, 255, 0], dtype=_np.uint8)
        mask4 = counts == 4
        out[mask4] = _np.array([255, 165, 0], dtype=_np.uint8)
        mask5 = counts >= 5
        out[mask5] = _np.array([255, 0, 0], dtype=_np.uint8)

        try:
            HOVER_PATH_DIR.mkdir(parents=True, exist_ok=True)
            HOVER_PATH_CURRENT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        hist_path = HOVER_PATH_DIR / f"{stem}_hover_path.png"
        current_path = HOVER_PATH_CURRENT_DIR / "hover_path.png"
        try:
            Image.fromarray(out).save(hist_path)
            Image.fromarray(out).save(current_path)
        except Exception:
            return

    def _render_hover_speed(self, trace_stem: str, pts: List[Tuple[int, int]], times: List[float]) -> None:
        """
        Render heatmap prędkości kursora:
        - kolory kodują szybkość w px/s wzdłuż ścieżki,
        - prędkość liczona na odcinkach między punktami.

        Historical PNGs -> DATA_SCREEN_DIR/hover/hover_speed/<stem>_hover_speed.png
        Current PNG     -> DATA_SCREEN_DIR/hover/hover_speed_current/hover_speed.png
        """
        if not pts or len(pts) < 2 or len(times) < 2:
            return
        stem = str(trace_stem or "").strip()
        if not stem:
            return
        screen_path = SCREENSHOT_DIR / f"{stem}.png"
        if screen_path.exists():
            src_img = screen_path
        else:
            candidate = HOVER_INPUT_CURRENT_DIR / "hover_input.png"
            if candidate.exists():
                src_img = candidate
            else:
                src_img = RAW_CURRENT_DIR / "screenshot.png"
        if not src_img.exists():
            return
        try:
            from PIL import Image  # type: ignore
            import numpy as _np  # type: ignore
        except Exception:
            return
        try:
            img = Image.open(src_img).convert("RGB")
        except Exception:
            return
        arr = _np.array(img, dtype=_np.uint8)
        h, w, _ = arr.shape

        segments: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []
        speeds: List[float] = []
        for (x0, y0), (x1, y1), t0, t1 in zip(pts, pts[1:], times, times[1:]):
            dt = float(t1 - t0)
            if dt <= 1e-4:
                continue
            v = math.hypot(float(x1 - x0), float(y1 - y0)) / dt
            speeds.append(v)
            segments.append(((int(x0), int(y0)), (int(x1), int(y1)), v))
        if not speeds:
            return

        v_min = min(speeds)
        v_max = max(speeds)
        if v_max <= v_min + 1e-6:
            v_max = v_min + 1.0

        def _clip_xy(x: int, y: int) -> Tuple[int, int]:
            return int(_np.clip(x, 0, w - 1)), int(_np.clip(y, 0, h - 1))

        out = arr.copy()
        for (x0, y0), (x1, y1), v in segments:
            x0, y0 = _clip_xy(x0, y0)
            x1, y1 = _clip_xy(x1, y1)
            dx = x1 - x0
            dy = y1 - y0
            steps = max(abs(dx), abs(dy))
            if steps <= 0:
                steps = 1
            xs = _np.linspace(x0, x1, steps + 1, dtype=_np.int32)
            ys = _np.linspace(y0, y1, steps + 1, dtype=_np.int32)

            # Normalizuj prędkość do [0,1] i zmapuj: wolno=niebieski -> średnio=zielony -> szybko=czerwony.
            t = (v - v_min) / (v_max - v_min)
            t = max(0.0, min(1.0, t))
            if t < 0.5:
                # niebieski -> zielony
                k = t / 0.5
                r, g, b = 0, int(255 * k), int(255 * (1.0 - k))
            else:
                # zielony -> czerwony
                k = (t - 0.5) / 0.5
                r, g, b = int(255 * k), int(255 * (1.0 - k)), 0
            colour = _np.array([r, g, b], dtype=_np.uint8)
            out[ys, xs] = colour

        try:
            HOVER_SPEED_DIR.mkdir(parents=True, exist_ok=True)
            HOVER_SPEED_CURRENT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        hist_path = HOVER_SPEED_DIR / f"{stem}_hover_speed.png"
        current_path = HOVER_SPEED_CURRENT_DIR / "hover_speed.png"
        try:
            Image.fromarray(out).save(hist_path)
            Image.fromarray(out).save(current_path)
        except Exception:
            return
        except Exception:
            return

    def _cmd_scroll(self, cmd: Dict[str, Any]):
        """Obsługa komendy scrollowania"""
        direction = cmd.get("direction", "down").lower()
        amount = int(cmd.get("amount", 3))  # liczba notches
        duration = float(cmd.get("duration", 0.5))
        
        print(f"[Scroll] {direction} by {amount} notches")
        
        # Określ kierunek
        if direction == "up":
            amount = -abs(amount)
        else:
            amount = abs(amount)
        
        # Scrolluj
        self._smooth_scroll_notches(amount, duration)

    def _cmd_keys(self, cmd: Dict[str, Any]):
        combo = (cmd.get("combo") or "").lower()
        if not combo:
            return
        parts = [p.strip() for p in combo.split("+") if p.strip()]
        if not parts:
            return
        if self.verbose:
            try:
                print(f"[Keys] combo={'+'.join(parts)}")
            except Exception:
                pass
        if len(parts) == 1:
            key = self.input_ctrl._to_key(parts[0])
            self.input_ctrl.kb_ctrl.press(key)
            time.sleep(0.02)
            self.input_ctrl.kb_ctrl.release(key)
        else:
            self.input_ctrl.hotkey(*parts)

    def _cmd_type(self, cmd: Dict[str, Any]):
        text = cmd.get("text") or ""
        if not text:
            return
        delay = float(cmd.get("delay", 0.0))
        if self.verbose:
            preview = text.replace("\n", " ")
            if len(preview) > 60:
                preview = preview[:57] + "..."
            try:
                print(f"[Type] len={len(text)} delay={delay}s text=\"{preview}\"")
            except Exception:
                pass
        self.input_ctrl.type_text(text, delay=delay)

    def _smooth_scroll_notches(self, total_notches: int, duration: float = 0.5):
        """Scrolluj płynnie o określoną liczbę notches"""
        if total_notches == 0:
            return
            
        abs_notches = abs(total_notches)
        
        if abs_notches <= 3:
            # Mało - scrolluj na raz
            self.input_ctrl.scroll(total_notches)
            time.sleep(0.1)
        else:
            # Dużo - podziel na kroki
            steps = min(abs_notches, 10)  # Max 10 kroków
            notches_per_step = total_notches / steps
            delay_per_step = duration / steps
            
            for i in range(steps):
                # Losowa wariacja dla bardziej ludzkiego ruchu
                this_step = int(notches_per_step + random.uniform(-0.3, 0.3))
                if this_step == 0:
                    this_step = 1 if total_notches > 0 else -1
                    
                self.input_ctrl.scroll(this_step)
                
                # Losowe opóźnienie
                delay = delay_per_step * random.uniform(0.8, 1.2)
                time.sleep(delay)

    def _target_duration_ms(self, dist_px: float, speed_str: str, duration_ms: Optional[float], min_total_ms: float) -> float:
        if duration_ms is not None:
            base = float(duration_ms)
        else:
            d = float(max(0.0, dist_px))
            # Wartości podzielone przez ~2, żeby przyspieszyć ruch
            if d <= 150: base = 375.0 + (d / 150.0) * 125.0    # 375-500ms
            elif d <= 400: base = 500.0 + ((d-150.0)/250.0)*200.0 # 500-700ms
            else: base = 700.0 + min(250.0, (d-400.0)*0.4)     # 700-950ms
        
        base *= random.uniform(0.97, 1.03)
        speed_str = (speed_str or "normal").lower()
        if speed_str == "fast": base *= 0.8
        elif speed_str == "slow": base *= 1.2
        
        return max(max(100.0, float(min_total_ms or 0.0)), base)

    def _densify_linear(self, pts: List[Tuple[int,int]], times: List[float], fps: int = 144) -> Tuple[List[Tuple[int,int]], List[float]]:
        if len(pts) < 2 or times[-1] <= 0:
            return pts, times
        T = times[-1]
        N = max(60, int(T * fps))
        new_times: List[float] = [i * (T / (N - 1)) for i in range(N)]
        new_pts: List[Tuple[int, int]] = []
        j = 0
        for tn in new_times:
            while j + 1 < len(times) and times[j + 1] < tn:
                j += 1
            if j + 1 >= len(times):
                new_pts.append(pts[-1])
                continue
            t0, t1 = times[j], times[j + 1]
            if t1 <= t0:
                new_pts.append(pts[j])
                continue
            a = (tn - t0) / (t1 - t0)
            x = int(round(pts[j][0] + a * (pts[j + 1][0] - pts[j][0])))
            y = int(round(pts[j][1] + a * (pts[j + 1][1] - pts[j][1])))
            new_pts.append((x, y))
        new_pts[-1] = pts[-1]
        return new_pts, new_times

    def _densify_by_pixel(self, pts: List[Tuple[int, int]], times: List[float], max_step_px: float = 1.0) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Densyfikacja œcie¿ki tak, aby kroki kursora by³y maksymalnie
        ~1 px (w metryce euklidesowej). Czas interpolujemy liniowo,
        ¿eby zachowaæ œredni¹ prêdkoœæ.
        """
        if len(pts) < 2 or len(pts) != len(times):
            return pts, times
        new_pts: List[Tuple[int, int]] = [pts[0]]
        new_times: List[float] = [times[0]]
        max_step = max(0.5, float(max_step_px))
        for (x0, y0, t0), (x1, y1, t1) in zip(
            [(px, py, pt) for (px, py), pt in zip(pts, times)],
            [(px, py, pt) for (px, py), pt in zip(pts[1:], times[1:])],
        ):
            dx = x1 - x0
            dy = y1 - y0
            dist = math.hypot(dx, dy)
            if dist <= max_step:
                new_pts.append((x1, y1))
                new_times.append(t1)
                continue
            steps = max(1, int(math.ceil(dist / max_step)))
            for s in range(1, steps + 1):
                a = s / steps
                x = int(round(x0 + dx * a))
                y = int(round(y0 + dy * a))
                t = t0 + (t1 - t0) * a
                new_pts.append((x, y))
                new_times.append(t)
        return new_pts, new_times

    def _postprocess_path(self, pts: List[Tuple[int,int]], times: List[float], start: Tuple[int,int], end: Tuple[int,int]) -> Tuple[List[Tuple[int,int]], List[float]]:
        dist = math.hypot(end[0]-start[0], end[1]-start[1])
        win = 5 if dist < 250 else (7 if dist < 600 else 9)
        pts = moving_average(pts, win)
        max_lat = min(40.0, 0.08 * dist)
        pts = limit_lateral(pts, start, end, max_lat)
        pts, times = _blend_lead_in(pts, times, lead_px=24.0, lead_ms=80.0)
        times = _ease_slow_last_third(times, strength=0.95)
        pts, times = self._densify_linear(pts, times, fps=144)
        # delikatny jitter boczny, proporcjonalny do długości ruchu
        jitter_amp = min(2.0, 0.02 * max(dist, 1.0))
        pts = apply_micro_jitter(pts, max_jitter_px=jitter_amp, step_std=0.3)
        pts = clamp_path(pts, margin=1)
        pts[-1] = end
        return pts, times

    def _run_timed_points(self, points: List[Tuple[int, int]], times_sec: List[float]):
        if not points or len(points) < 2:
            return
        time_begin_period(1)
        try:
            t0 = time.perf_counter()
            first_move_logged = False
            for i in range(len(points)):
                set_cursor_pos_raw(points[i][0], points[i][1])
                if not first_move_logged and i >= 1:
                    first_move_logged = True
                    try:
                        print(f"[Agent TIMER] first_move {time.perf_counter() - t0:.3f}s after path_start")
                    except Exception:
                        pass
                if i < len(points) - 1:
                    delay = (t0 + times_sec[i+1]) - time.perf_counter()
                    if delay > 0:
                        precise_sleep(delay)
        finally:
            time_end_period(1)

    # --- PATH (polyline) execution helpers ---
    @staticmethod
    def _seg_intersects_rect(p0: Tuple[int,int], p1: Tuple[int,int], rect: Tuple[int,int,int,int]) -> bool:
        x0,y0 = p0; x1,y1 = p1
        rx0,ry0,rx1,ry1 = rect
        if rx0 > rx1: rx0,rx1 = rx1,rx0
        if ry0 > ry1: ry0,ry1 = ry1,ry0
        if max(x0,x1) < rx0 or min(x0,x1) > rx1 or max(y0,y1) < ry0 or min(y0,y1) > ry1:
            return False
        if rx0 <= x0 <= rx1 and ry0 <= y0 <= ry1: return True
        if rx0 <= x1 <= rx1 and ry0 <= y1 <= ry1: return True
        def _ccw(ax,ay,bx,by,cx,cy): return (cy-ay)*(bx-ax) > (by-ay)*(cx-ax)
        def _inter(a,b,c,d):
            (ax,ay),(bx,by),(cx,cy),(dx,dy) = a,b,c,d
            return _ccw(ax,ay,cx,cy,dx,dy) != _ccw(bx,by,cx,cy,dx,dy) and _ccw(ax,ay,bx,by,cx,cy) != _ccw(ax,ay,bx,by,dx,dy)
        A=(x0,y0); B=(x1,y1)
        edges=[((rx0,ry0),(rx1,ry0)),((rx1,ry0),(rx1,ry1)),((rx1,ry1),(rx0,ry1)),((rx0,ry1),(rx0,ry0))]
        return any(_inter(A,B,e0,e1) for (e0,e1) in edges)

    def _build_times_for_path(
        self,
        pts: List[Tuple[int,int]],
        *,
        speed: str = "normal",
        duration_ms: Optional[float] = None,
        min_total_ms: float = 0.0,
        global_speed_factor: float = 1.0,
        min_dt: float = 0.004,
        gap_rects: Optional[List[Tuple[int,int,int,int]]] = None,
        gap_boost: float = 1.0,
        line_jump_indices: Optional[List[int]] = None,
        line_jump_boost: float = 1.0,
    ) -> List[float]:
        t_start = time.perf_counter()
        if len(pts) < 2:
            return [0.0]
        total_dist = 0.0
        seg_len: List[float] = []
        for i in range(len(pts)-1):
            d = math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
            seg_len.append(d); total_dist += d
        if total_dist <= 0.0:
            return [0.0, 0.016]
        base_T = self._target_duration_ms(total_dist, speed, duration_ms, min_total_ms) / 1000.0
        base_T = max(0.03, base_T)
        # global speed applied per-segment as seg_speed
        times = [0.0]
        gap_rects = gap_rects or []
        line_jump_set = set(line_jump_indices or [])
        t_acc = 0.0
        seg_speed = max(0.05, float(global_speed_factor))
        for i, d in enumerate(seg_len):
            share = d / total_dist
            dt = base_T * share
            # Przyspieszenia traktujemy jako zmianę „częstotliwości” (czasu),
            # nie jako skok pikseli – więc modyfikujemy tylko dt.
            if any(self._seg_intersects_rect(pts[i], pts[i+1], r) for r in gap_rects):
                eff = 1.0 + max(0.0, gap_boost - 1.0) * 0.6  # zmiękcz efekt
                dt /= max(1.0, eff)
            if i in line_jump_set:
                eff = 1.0 + max(0.0, line_jump_boost - 1.0) * 0.6
                dt /= max(1.0, eff)
            # Per-segment relative speed randomization (very smooth: tiny random walk).
            x = random.uniform(-2.0, 2.0)
            seg_speed = max(0.2, seg_speed + x / 100.0)
            dt /= seg_speed
            dt = max(min_dt, dt)
            t_acc += dt
            times.append(t_acc)
        if self.verbose:
            try:
                print(f"[Agent TIMER] build_times_for_path total={times[-1]:.3f}s compute={time.perf_counter()-t_start:.3f}s")
            except Exception:
                pass
        return times

    def _build_times_for_path_smooth(
        self,
        pts: List[Tuple[int,int]],
        *,
        speed: str = "normal",
        duration_ms: Optional[float] = None,
        min_total_ms: float = 0.0,
        global_speed_factor: float = 1.0,
        min_dt: float = 0.004,
        gap_rects: Optional[List[Tuple[int,int,int,int]]] = None,
        gap_boost: float = 1.0,
        line_jump_indices: Optional[List[int]] = None,
        line_jump_boost: float = 1.0,
    ) -> List[float]:
        t_start = time.perf_counter()
        if len(pts) < 2:
            return [0.0]
        total_dist = 0.0
        seg_len: List[float] = []
        for i in range(len(pts) - 1):
            d = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            seg_len.append(d)
            total_dist += d
        if total_dist <= 0.0:
            return [0.0, 0.016]
        base_T = self._target_duration_ms(total_dist, speed, duration_ms, min_total_ms) / 1000.0
        base_T = max(0.03, base_T)
        times = [0.0]
        gap_rects = gap_rects or []
        line_jump_set = set(line_jump_indices or [])
        t_acc = 0.0
        seg_speed = max(0.05, float(global_speed_factor))
        # Płynny, delikatny akcelerator dla gapów – zamiast
        # skokowego mnożnika utrzymujemy stan, który powoli
        # zbliża się do docelowego podczas wchodzenia/wychodzenia z dziur.
        gap_state = 1.0
        gap_smooth = 0.8  # 80% poprzedniego stanu, 20% nowego celu
        for i, d in enumerate(seg_len):
            share = d / total_dist
            dt = base_T * share
            # Przyspieszenia traktujemy jako zmianę częstotliwości (czasu),
            # nie jako skok pikseli – więc modyfikujemy tylko dt.
            in_gap = any(self._seg_intersects_rect(pts[i], pts[i + 1], r) for r in gap_rects)
            if in_gap and gap_boost > 1.0:
                # Maksymalny efekt gap_boost jest dodatkowo zmiękczony,
                # tak żeby przyspieszenie było wyczuwalne, ale bardzo ludzko.
                target = 1.0 + max(0.0, gap_boost - 1.0) * 0.25
            else:
                target = 1.0
            gap_state = gap_state * gap_smooth + target * (1.0 - gap_smooth)
            dt /= max(1.0, gap_state)
            if i in line_jump_set:
                eff = 1.0 + max(0.0, line_jump_boost - 1.0) * 0.6
                dt /= max(1.0, eff)
            # Per-segment relative speed randomization (very smooth: tiny random walk).
            x = random.uniform(-2.0, 2.0)
            seg_speed = max(0.2, seg_speed + x / 100.0)
            dt /= seg_speed
            dt = max(min_dt, dt)
            t_acc += dt
            times.append(t_acc)
        if self.verbose:
            try:
                print(f"[Agent TIMER] build_times_for_path_smooth total={times[-1]:.3f}s compute={time.perf_counter()-t_start:.3f}s")
            except Exception:
                pass
        return times

    def _cmd_path(self, cmd: Dict[str, Any]):
        # Execute polyline path with optional boosts and global speed factor
        t_cmd_start = time.perf_counter()
        pts_in = cmd.get("points") or []
        if not pts_in or len(pts_in) < 2:
            print("[Path] ignored: too few points")
            return
        pts_raw: List[Tuple[int, int]] = [(int(p["x"]), int(p["y"])) for p in pts_in]
        pts_raw = clamp_path(pts_raw, margin=1)
        speed = str(cmd.get("speed", "normal"))
        duration_ms = cmd.get("duration_ms")
        min_total_ms = float(cmd.get("min_total_ms", 0.0))
        speed_factor = float(cmd.get("speed_factor", 1.0))
        min_dt = float(cmd.get("min_dt", 0.004))
        gap_rects = [tuple(map(int, r)) for r in (cmd.get("gap_rects") or [])]
        gap_boost = max(0.05, float(cmd.get("gap_boost", 1.0)))
        line_jump_indices = [int(i) for i in (cmd.get("line_jump_indices") or [])]
        line_jump_boost = max(0.05, float(cmd.get("line_jump_boost", 1.0)))
        press = str(cmd.get("press", "none")).lower()
        should_hold = press == "mouse"

        # Utrzymaj dokładnie tę samą trajektorię co hover JSON (bez retargetu biblioteki).
        path_pts: List[Tuple[int, int]] = pts_raw[:]
        if len(path_pts) < 2:
            print("[Path] ignored: path too short")
            return

        line_jump_indices_expanded: List[int] = []
        for idx in line_jump_indices:
            if 0 <= idx < len(path_pts):
                line_jump_indices_expanded.append(idx)

        spacing_x = float(cmd.get("spacing_x", 1.0) or 1.0)
        if spacing_x != 1.0 and len(path_pts) >= 2:
            spaced_pts: List[Tuple[int, int]] = [path_pts[0]]
            for i in range(1, len(path_pts)):
                dx = (path_pts[i][0] - path_pts[i - 1][0]) * spacing_x
                spaced_x = int(round(spaced_pts[i - 1][0] + dx))
                spaced_pts.append((spaced_x, path_pts[i][1]))
            path_pts = spaced_pts

        times = self._build_times_for_path_smooth(
            path_pts,
            speed=speed,
            duration_ms=duration_ms,
            min_total_ms=min_total_ms,
            global_speed_factor=speed_factor,
            min_dt=min_dt,
            gap_rects=gap_rects,
            gap_boost=gap_boost,
            line_jump_indices=line_jump_indices_expanded,
            line_jump_boost=line_jump_boost,
        )
        # Densyfikacja po pikselu – maksymalnie ~1 px na krok.
        pts_d, times_d = self._densify_by_pixel(path_pts, times, max_step_px=1.0)
        # delikatny jitter także na ścieżkach hoverowych – zawsze, ale w małej amplitudzie
        path_dist = _path_length([(float(x), float(y)) for x, y in path_pts])
        jitter_amp = min(2.0, 0.015 * max(path_dist, 1.0))
        pts_d = apply_micro_jitter(pts_d, max_jitter_px=jitter_amp, step_std=0.25)
        pts_d = clamp_path(pts_d, margin=1)

        speed_px_per_s = float(cmd.get("speed_px_per_s", 0.0) or 0.0)
        if speed_px_per_s > 0.0 and times_d:
            # Ogranicz do sensownego zakresu „ludzkich” prędkości.
            speed_px_per_s = min(max(speed_px_per_s, 80.0), 2500.0)
            total_len = _path_length([(float(x), float(y)) for x, y in pts_d])
            if total_len > 0.0 and times_d[-1] > 1e-6:
                # Docelowy czas = długość ścieżki / px_per_s, z dolnym limitem,
                # żeby bardzo krótkie ruchy nie były „teleportem”.
                target_T = max(0.15, total_len / speed_px_per_s)
                scale_t = target_T / max(1e-6, times_d[-1])
                times_d = [t * scale_t for t in times_d]

        if self.verbose:
            try:
                print(
                    f"[Path] points_in={len(pts_in)} expanded={len(path_pts)} densified={len(pts_d)} "
                    f"duration={times_d[-1]:.3f}s press={press}"
                )
            except Exception:
                pass

        # Po skalowaniu: dbamy o monotoniczność czasów.
        # Dla klasycznych ruchów trzymamy min_dt, dla trybu px/s pozwalamy na mniejsze kroki,
        # żeby ruch nie był „tępy” i zbyt poszatkowany.
        if times_d:
            adj_times = [times_d[0]]
            for i in range(1, len(times_d)):
                dt = times_d[i] - times_d[i - 1]
                if speed_px_per_s <= 0.0:
                    dt = max(min_dt, dt)
                else:
                    dt = max(1e-4, dt)
                adj_times.append(adj_times[-1] + dt)
            times_d = adj_times

        # Optional real-path visualisation: render on matching screenshot.
        trace_stem = str(cmd.get("trace_stem") or "").strip()
        if trace_stem:
            def _safe_run(fn, *args):
                try:
                    fn(*args)
                except Exception as e:  # pragma: no cover - tylko log
                    try:
                        print(f"[PathTrace] render failed: {e}")
                    except Exception:
                        pass
            # Render mapę odwiedzin oraz mapę prędkości równolegle, bez blokowania ruchu.
            threading.Thread(target=_safe_run, args=(self._render_hover_path, trace_stem, pts_d), daemon=True).start()
            threading.Thread(target=_safe_run, args=(self._render_hover_speed, trace_stem, pts_d, times_d), daemon=True).start()

        try:
            if should_hold:
                self.input_ctrl.press("mouse")
            self._run_timed_points(pts_d, times_d)
        finally:
            if should_hold:
                self.input_ctrl.release("mouse")

    def _cmd_move(self, cmd: Dict[str, Any]):
        """Obsługa komendy ruchu - UPROSZCZONE BEZ AUTO-SCROLLOWANIA"""
        tx, ty = int(cmd.get("x", 0)), int(cmd.get("y", 0))
        press = str(cmd.get("press", "none")).lower()
        speed = str(cmd.get("speed", "normal")).lower()
        duration_ms = cmd.get("duration_ms", None)
        curvy = str(cmd.get("curvy", "auto")).lower()
        min_total_ms = float(cmd.get("min_total_ms", 0.0))
        
        # NOWE: Explicite scrollowanie tylko jeśli bot tego zażąda
        needs_scroll = cmd.get("needs_scroll", False)
        scroll_direction = cmd.get("scroll_direction", "down")
        scroll_amount = int(cmd.get("scroll_amount", 3))
        
        sx, sy = get_cursor_pos()
        if self.verbose:
            desc = cmd.get("desc") or cmd.get("label") or cmd.get("text") or cmd.get("role")
            target_info = f" '{desc}'" if desc else ""
            print(f"[Move] to({tx},{ty}) from({sx},{sy}) press={press} speed={speed} curvy={curvy}{target_info}")
        
        # Scrolluj TYLKO jeśli bot explicite tego zażąda
        if needs_scroll:
            print(f"[Agent] Bot requested scroll {scroll_direction} by {scroll_amount} notches")
            
            # Przesuń mysz w bezpieczne miejsce do scrollowania
            if SCREEN_BOUNDS:
                _, _, right, bottom = SCREEN_BOUNDS
                scroll_x = right // 2
                scroll_y = bottom // 2
            else:
                scroll_x = 960
                scroll_y = 540
                
            # Szybki ruch do miejsca scrollowania
            set_cursor_pos_raw(scroll_x, scroll_y)
            time.sleep(0.1)
            
            # Scrolluj
            if scroll_direction == "up":
                scroll_amount = -abs(scroll_amount)
            else:
                scroll_amount = abs(scroll_amount)
                
            self._smooth_scroll_notches(scroll_amount, duration=0.5)
            
            # Poczekaj aż strona się ustabilizuje
            time.sleep(0.3)
            
            # Pobierz nową pozycję kursora
            sx, sy = get_cursor_pos()
        
        # Kontynuuj normalny ruch do celu
        tx, ty = clamp_to_screen(tx, ty)
        dist = math.hypot(tx - sx, ty - sy)

        want_slider: Optional[bool] = True if str(cmd.get("action", "auto")).lower() == "slider" else False
        sample = self.lib.pick_best((sx, sy), (tx, ty), want_slider, curvy=curvy)
        
        if sample is None:
            Tms = self._target_duration_ms(dist, speed, duration_ms, min_total_ms)
            pts = [(sx, sy), (tx, ty)]
            times = [0.0, Tms / 1000.0]
            pts, times = self._densify_linear(pts, times)
            if self.verbose:
                try:
                    print(f"[Path] linear steps={len(pts)} duration={times[-1]:.3f}s from=({pts[0][0]},{pts[0][1]}) to=({pts[-1][0]},{pts[-1][1]})")
                except Exception:
                    pass
            self._run_timed_points(pts, times)
        else:
            pts_raw, times_raw = ExactLibrary.retarget(sample, (sx, sy), (tx, ty))
            T_target_ms = self._target_duration_ms(dist, speed, duration_ms, min_total_ms)
            scale_t = (T_target_ms / 1000.0) / max(1e-6, times_raw[-1])
            times_raw = [t * scale_t for t in times_raw]
            pts_final, times_final = self._postprocess_path(pts_raw, times_raw, (sx, sy), (tx, ty))
            if self.verbose:
                try:
                    print(
                        "[Pick] slider=" + str(sample.get("is_slider")) +
                        f" straight={sample.get('straightness', 0.0):.3f} flips={int(sample.get('flips', 0))}"
                    )
                    print(f"[Path] steps={len(pts_final)} duration={times_final[-1]:.3f}s from=({pts_final[0][0]},{pts_final[0][1]}) to=({pts_final[-1][0]},{pts_final[-1][1]})")
                except Exception:
                    pass
            self._run_timed_points(pts_final, times_final)

        if press == "mouse":
            time.sleep(random.uniform(0.01, 0.03))
            if self.verbose:
                cx, cy = get_cursor_pos()
                print(f"[Click] left at ({cx},{cy})")
            self.input_ctrl.press("mouse")
            time.sleep(random.uniform(0.02, 0.05))
            self.input_ctrl.release("mouse")

def main():
    parser = argparse.ArgumentParser()
    default_cfg = Path(__file__).resolve().parents[1] / "control_agent" / "train.json"
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_cfg),
        help=f"Config file (e.g. train.json) (default: {default_cfg})",
    )
    parser.add_argument("--port", type=int, default=8765, help="UDP port to listen on")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging of actions and click coordinates")
    parser.add_argument(
        "--coords",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional manual path (x1 y1 x2 y2) to send once after agent start",
    )
    parser.add_argument(
        "--left",
        action="store_true",
        help="Hold left mouse button for manual path.",
    )
    args = parser.parse_args()

    try:
        from pynput import keyboard, mouse
        kb_ctrl = keyboard.Controller()
        ms_ctrl = mouse.Controller()
    except Exception as e:
        print("\n[ERROR] Cannot initialize Pynput.")
        print("Make sure the script has proper permissions (Accessibility/Input Monitoring on macOS).")
        print(f"Details: {e}\n")
        return

    agent = ControlAgent(cfg_path=args.config, udp_port=args.port, verbose=args.verbose)

    if getattr(args, "coords", None) and agent.lib.samples:
        x1, y1, x2, y2 = args.coords
        sample = random.choice(agent.lib.samples)
        pts = _align_path(sample["pts"], (x1, y1), (x2, y2))
        command = {
            "cmd": "path",
            "points": [{"x": int(round(px)), "y": int(round(py))} for px, py in pts],
            "min_dt": 0.01,
        }
        if args.left:
            command["press"] = "mouse"
            command["speed"] = "slow"
        agent.cmd_queue.put(command)

    if getattr(args, "coords", None):
        x1, y1, x2, y2 = args.coords
        command = {
            "cmd": "path",
            "points": [{"x": x1, "y": y1}, {"x": x2, "y": y2}],
            "min_dt": 0.01,
        }
        if args.left:
            command["press"] = "mouse"
        agent.cmd_queue.put(command)

    # Hard-exit listener on "}" key (and ] as fallback)
    stop_evt = threading.Event()
    def _on_press(key):
        try:
            ch = getattr(key, "char", None)
            vk = getattr(key, "vk", None)
            if ch and ch == "}":
                print("[Agent] Hard exit requested via '}'")
                stop_evt.set()
                return False
            if vk in (221,):  # ] / } on many layouts
                print("[Agent] Hard exit requested via vk=221")
                stop_evt.set()
                return False
        except Exception:
            return False
        return True

    listener = keyboard.Listener(on_press=_on_press)
    listener.daemon = True
    listener.start()

    def _run_agent():
        agent.start()
        stop_evt.set()

    t = threading.Thread(target=_run_agent, name="ControlAgentMain", daemon=True)
    t.start()

    try:
        while not stop_evt.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            agent.receiver.stop_event.set()
        except Exception:
            pass
        listener.stop()
        t.join(timeout=1.0)
        print("[Agent] Stopped (hard exit).")

if __name__ == "__main__":
    main()
