import math
from typing import List, Tuple, Dict

Point = List[float]  # [x, y, t]
Traj = List[Point]

def translate_to_origin(traj: Traj) -> Traj:
    if not traj:
        return traj
    x0, y0, t0 = traj[0]
    return [[x - x0, y - y0, t - t0] for (x, y, t) in traj]

def _mean_xy(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    n = float(len(points))
    return (sx / n, sy / n)

def _direction_vector(traj: Traj, early_frac: float, late_frac: float) -> Tuple[float, float]:
    n = len(traj)
    if n < 2:
        return (1.0, 0.0)
    k1 = max(1, int(n * early_frac))
    k2 = max(1, int(n * late_frac))
    early = [(x, y) for x, y, _ in traj[:k1]]
    late = [(x, y) for x, y, _ in traj[-k2:]]
    ex, ey = _mean_xy(early)
    lx, ly = _mean_xy(late)
    vx, vy = (lx - ex, ly - ey)
    if abs(vx) + abs(vy) < 1e-9:
        x0, y0, _ = traj[0]
        x1, y1, _ = traj[-1]
        vx, vy = (x1 - x0, y1 - y0)
        if abs(vx) + abs(vy) < 1e-9:
            return (1.0, 0.0)
    return (vx, vy)

def rotate_traj(traj: Traj, angle_deg: float) -> Traj:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    out: Traj = []
    for x, y, t in traj:
        xr = x * c - y * s
        yr = x * s + y * c
        out.append([xr, yr, t])
    return out

def canonicalize_traj(
    traj: Traj,
    orient_cfg: Dict,
) -> Traj:
    if not traj:
        return traj
    # 1) translate to origin
    translated = translate_to_origin(traj)
    # 2) orientation vector
    early_frac = orient_cfg.get("early_frac", 0.1)
    late_frac = orient_cfg.get("late_frac", 0.1)
    ensure_right = orient_cfg.get("ensure_right", True)
    vx, vy = _direction_vector(translated, early_frac, late_frac)
    theta = math.atan2(vy, vx)  # angle to +X
    c, s = math.cos(-theta), math.sin(-theta)
    rotated: Traj = []
    for x, y, t in translated:
        xr = x * c - y * s
        yr = x * s + y * c
        rotated.append([xr, yr, t])
    if ensure_right and rotated[-1][0] < rotated[0][0]:
        for p in rotated:
            p[0] = -p[0]  # mirror X
    return rotated

def resample_uniform(traj: Traj, target_hz: int) -> Traj:
    # linear interpolation to uniform dt
    if not traj:
        return traj
    if len(traj) == 1:
        x, y, _ = traj[0]
        return [[x, y, 0.0]]
    dt = 1.0 / float(target_hz)
    t_end = traj[-1][2]
    if t_end <= 0:
        return [[traj[0][0], traj[0][1], 0.0]]
    # original times and points
    src = traj
    # build target times
    num = max(1, int(t_end / dt) + 1)
    t_new = [i * dt for i in range(num)]
    out: Traj = []
    j = 0
    for tn in t_new:
        while j + 1 < len(src) and src[j + 1][2] < tn:
            j += 1
        if j + 1 >= len(src):
            x, y, _ = src[-1]
            out.append([x, y, tn])
        else:
            t0 = src[j][2]
            t1 = src[j + 1][2]
            if t1 == t0:
                x, y, _ = src[j]
                out.append([x, y, tn])
            else:
                a = (tn - t0) / (t1 - t0)
                x = src[j][0] + a * (src[j + 1][0] - src[j][0])
                y = src[j][1] + a * (src[j + 1][1] - src[j][1])
                out.append([x, y, tn])
    return out

def pad_or_trim(traj: Traj, max_len: int, pad_strategy: str = "repeat_end") -> Tuple[Traj, int]:
    # returns (traj_fixed_len, orig_len)
    n = len(traj)
    if n == 0:
        # create one zero point
        traj = [[0.0, 0.0, 0.0]]
        n = 1
    if n == max_len:
        return traj, n
    if n > max_len:
        return traj[:max_len], max_len
    # pad
    if pad_strategy == "repeat_end":
        last = traj[-1]
        pad = [last[:] for _ in range(max_len - n)]
        return traj + pad, n
    # default zeros
    pad = [[0.0, 0.0, traj[-1][2]] for _ in range(max_len - n)]
    return traj + pad, n