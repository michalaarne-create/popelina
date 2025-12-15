from __future__ import annotations

"""
Generator losowych, ale gładkich trajektorii myszy w stylu "ludzkich wygibasów".

Cel:
  - generować sekwencje punktów bez teleportów i szarpnięć,
  - dobrać czas trwania ruchu do rozsądnej ludzkiej prędkości,
  - opcjonalnie od razu złożyć payload w formacie `cmd="path"` dla control_agent.

Użycie (przykład):
    from scripts.control_agent.random_mouse_trajectory import (
        generate_wiggle_traj,
        build_control_agent_payload,
    )

    traj = generate_wiggle_traj(960, 540)
    payload = build_control_agent_payload(traj)
    # payload można wysłać przez UDP tak jak inne komendy do control_agent.
"""

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

PointT = Tuple[float, float, float]  # (x, y, t_sec)
TrajectoryT = List[PointT]


def _path_length(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def _ease_min_jerk(u: float) -> float:
    """
    Klasyczna krzywa minimum-jerk (gładkie przyspieszanie i hamowanie).
    Zwraca u' w [0,1] dla wejścia u z [0,1].
    """
    u = max(0.0, min(1.0, u))
    return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)


@dataclass
class WiggleConfig:
    # Zasięg ruchu wokół punktu startowego (px).
    radius_px: float = 60.0
    # Typowa długość pojedynczego kroku (px).
    step_px: float = 4.5
    # Minimalna / maksymalna długość pojedynczego kroku (px).
    min_step_px: float = 1.5
    max_step_px: float = 12.0
    # Odchylenie standardowe zmiany kąta między kolejnymi krokami (stopnie).
    turn_std_deg: float = 15.0
    # Maksymalny pojedynczy skręt (stopnie), żeby nie było ostrych zakrętów.
    max_turn_deg: float = 40.0
    # Liczba punktów w wygibasie.
    min_points: int = 40
    max_points: int = 100
    # Docelowy czas trwania całej trajektorii (sekundy).
    min_duration_s: float = 0.35
    max_duration_s: float = 1.2
    # Docelowy zakres średniej prędkości kursora (px/s).
    min_speed_px_s: float = 500.0
    max_speed_px_s: float = 1400.0
    # Częstotliwość próbkowania przy generowaniu finalnej trajektorii (Hz).
    sampling_hz: int = 144


def _generate_spatial_wiggle(
    start_x: float,
    start_y: float,
    cfg: WiggleConfig,
) -> List[Tuple[float, float]]:
    """
    Generuje "szkielet" ruchu: listę kolejnych punktów XY bez czasu.
    To jest stonowany random-walk z ograniczonymi skrętami oraz promieniem.
    """
    n_points = random.randint(cfg.min_points, cfg.max_points)
    cx, cy = float(start_x), float(start_y)
    origin_x, origin_y = cx, cy

    theta = random.uniform(0.0, 2.0 * math.pi)
    points: List[Tuple[float, float]] = [(cx, cy)]

    for _ in range(n_points - 1):
        # Losowy, ale ograniczony skręt
        d_theta_deg = random.gauss(0.0, cfg.turn_std_deg)
        d_theta_deg = max(-cfg.max_turn_deg, min(cfg.max_turn_deg, d_theta_deg))
        theta += math.radians(d_theta_deg)

        # Losowa długość kroku z ograniczeniem
        step = random.gauss(cfg.step_px, cfg.step_px * 0.4)
        step = max(cfg.min_step_px, min(cfg.max_step_px, step))

        nx = cx + step * math.cos(theta)
        ny = cy + step * math.sin(theta)

        # Ograniczenie do koła o promieniu radius_px wokół punktu startowego
        dx = nx - origin_x
        dy = ny - origin_y
        r = math.hypot(dx, dy)
        if r > cfg.radius_px:
            # Projekcja z powrotem na okrąg + lekkie "przyciągnięcie" do środka
            scale = cfg.radius_px / max(r, 1e-6)
            dx *= scale
            dy *= scale
            nx = origin_x + dx * 0.9
            ny = origin_y + dy * 0.9

        cx, cy = nx, ny
        points.append((cx, cy))

    # Prosty filtr wygładzający (3-punktowa średnia ruchoma)
    if len(points) >= 3:
        smooth: List[Tuple[float, float]] = [points[0]]
        for i in range(1, len(points) - 1):
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            smooth.append(((x0 + x1 + x2) / 3.0, (y0 + y1 + y2) / 3.0))
        smooth.append(points[-1])
        points = smooth

    return points


def generate_wiggle_traj(
    start_x: int,
    start_y: int,
    cfg: Optional[WiggleConfig] = None,
) -> TrajectoryT:
    """
    Główna funkcja: z punktu startowego generuje trajektorię (x, y, t).

    - Pozycje są gładnie połączone (brak teleportów).
    - Czas narasta zgodnie z krzywą minimum-jerk (wolniej na początku i końcu).
    - Średnia prędkość mieści się w rozsądnym, ludzkim zakresie.
    """
    cfg = cfg or WiggleConfig()

    spatial = _generate_spatial_wiggle(start_x, start_y, cfg)
    if len(spatial) < 2:
        return [(float(start_x), float(start_y), 0.0)]

    total_len = _path_length(spatial)
    if total_len <= 1.0:
        return [(float(start_x), float(start_y), 0.0)]

    # Wybór docelowej prędkości i czasu trwania
    target_speed = random.uniform(cfg.min_speed_px_s, cfg.max_speed_px_s)
    target_T = total_len / max(target_speed, 1.0)
    target_T = max(cfg.min_duration_s, min(cfg.max_duration_s, target_T))

    # Ustalenie liczby próbek w czasie
    hz = max(30, int(cfg.sampling_hz))
    n_samples = max(2, int(target_T * hz))

    # Cumulative arc length do parametryzacji po drodze
    cum: List[float] = [0.0]
    for i in range(1, len(spatial)):
        x0, y0 = spatial[i - 1]
        x1, y1 = spatial[i]
        cum.append(cum[-1] + math.hypot(x1 - x0, y1 - y0))

    L = cum[-1] or 1.0

    traj: TrajectoryT = []
    for i in range(n_samples):
        # u_time – równomierny w czasie; u – po min-jerk
        u_time = i / max(n_samples - 1, 1)
        u = _ease_min_jerk(u_time)
        s = u * L

        # znajdź segment odpowiadający długości s
        j = 0
        while j + 1 < len(cum) and cum[j + 1] < s:
            j += 1
        if j + 1 >= len(cum):
            x, y = spatial[-1]
        else:
            s0, s1 = cum[j], cum[j + 1]
            x0, y0 = spatial[j]
            x1, y1 = spatial[j + 1]
            if s1 <= s0:
                x, y = x0, y0
            else:
                alpha = (s - s0) / (s1 - s0)
                x = x0 + alpha * (x1 - x0)
                y = y0 + alpha * (y1 - y0)

        t = u_time * target_T
        traj.append((x, y, t))

    # Korekta dokładnego czasu końca
    if traj:
        x_last, y_last, _ = traj[-1]
        traj[-1] = (x_last, y_last, target_T)

    return traj


def build_control_agent_payload(
    traj: TrajectoryT,
    *,
    speed: str = "normal",
    speed_factor: float = 1.0,
    min_dt: float = 0.008,
    speed_px_per_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Konwertuje trajektorię (x, y, t) na payload `cmd="path"` dla control_agent.

    Uwaga: control_agent sam buduje rozkład czasów po drodze na podstawie
    `min_dt`, `speed_factor` i opcjonalnie `speed_px_per_s`, więc tutaj
    czasy są użyte tylko do estymacji średniej prędkości.
    """
    if len(traj) < 2:
        return {
            "cmd": "path",
            "points": [{"x": int(round(traj[0][0])), "y": int(round(traj[0][1]))}] if traj else [],
            "speed": speed,
            "speed_factor": float(speed_factor),
            "min_dt": float(min_dt),
            "gap_rects": [],
            "gap_boost": 1.0,
            "line_jump_indices": [],
            "line_jump_boost": 1.0,
        }

    xy = [(p[0], p[1]) for p in traj]
    total_len = _path_length(xy)
    total_T = max(traj[-1][2] - traj[0][2], 1e-3)

    if speed_px_per_s is None:
        avg_speed = total_len / total_T
        # Bezpieczne ograniczenie prędkości
        speed_px_per_s = max(200.0, min(avg_speed, 2000.0))

    payload: Dict[str, Any] = {
        "cmd": "path",
        "points": [{"x": int(round(x)), "y": int(round(y))} for (x, y, _) in traj],
        "speed": speed,
        "speed_factor": float(speed_factor),
        "min_dt": float(min_dt),
        "gap_rects": [],
        "gap_boost": 1.0,
        "line_jump_indices": [],
        "line_jump_boost": 1.0,
        "min_total_ms": float(total_T * 1000.0),
        "speed_px_per_s": float(speed_px_per_s),
    }
    return payload


__all__ = [
    "WiggleConfig",
    "generate_wiggle_traj",
    "build_control_agent_payload",
]

