"""
Hover Environment - FIXED VERSION

Sequential + Natural variance + STABILIZED REWARDS
Fixes:
- max_steps: 1000 -> 300
- Reward scaling: /10
- Early termination on success
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HoverEnvMultiLineV2(gym.Env):
    """
    Sequential placement with natural variances and stabilised rewards.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        lines_file: str = "data/text_lines.json",
        *,
        line_randomization: bool = False,
        line_jitter_y: float = 20.0,
        line_jitter_x: float = 40.0,
        line_scale_jitter: float = 0.1,
        line_shuffle_prob: float = 0.0,
    ):
        super().__init__()

        lines_path = Path(lines_file)
        if not lines_path.exists():
            raise FileNotFoundError(f"Lines configuration file not found: {lines_file}")

        with open(lines_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            self.lines: List[Dict[str, float]] = data["lines"]

        self.screen_w = 1920
        self.screen_h = 1080
        self._base_lines: List[Dict[str, float]] = [dict(line) for line in self.lines]
        self.line_randomization = bool(line_randomization)
        self.line_jitter_y = float(max(0.0, line_jitter_y))
        self.line_jitter_x = float(max(0.0, line_jitter_x))
        self.line_scale_jitter = float(max(0.0, line_scale_jitter))
        self.line_shuffle_prob = float(np.clip(line_shuffle_prob, 0.0, 1.0))

        self._recompute_line_cache()
        self._blank_canvas = np.ones((self.screen_h, self.screen_w, 3), dtype=np.uint8) * 240

        # ACTION: [dx_forward, dy_offset]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(12,),
            dtype=np.float32,
        )

        self.current_line_idx = 0
        self.current_pos: Optional[np.ndarray] = None
        self.line_dots: Dict[str, List[Tuple[float, float]]] = {}
        self.step_count = 0

        # Reduced max steps (1000 -> 300)
        self.max_steps = 300

        # Parameters
        self.ideal_dx = 40
        self.dx_min = 25
        self.dx_max = 55
        self.dy_min = -80
        self.dy_max = 15

        self.natural_dx_range = (30, 50)
        self.natural_dy_range = (-15, 10)

        # Target sampling parameters for per-dot guidance
        self.target_dx_range = (25.0, 50.0)
        self.target_y_range = (-60.0, 15.0)
        self.target_dx_tolerance = 6.0
        self.target_y_tolerance = 12.0
        self._line_targets: Dict[str, Dict[str, float]] = {}
        self._last_target_hit = False

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._maybe_randomize_lines()

        self._line_targets = {}
        self.current_line_idx = 0
        self.line_dots = {line["id"]: [] for line in self.lines}
        self.step_count = 0
        self.total_dots = 0
        self.completed_line_flags = {line["id"]: False for line in self.lines}
        self.completed_lines_count = 0

        first_line = self.lines[0]
        first_x = first_line["x1"] + self.np_random.uniform(-5, 15)
        first_y = first_line["y1"] + self.np_random.uniform(-20, 5)

        self.current_pos = np.array([first_x, first_y], dtype=np.float32)
        self._record_dot(first_line["id"], first_x, first_y)
        self._set_next_target(first_line["id"], first_x)

        obs = self._get_obs()
        info = {
            "current_line": 0,
            "total_dots": self.total_dots,
            "lines_completed": self.completed_lines_count,
        }
        return obs, info

    def step(self, action: np.ndarray):
        current_line = self._get_current_line()
        line_id = current_line["id"]

        prev_x = float(self.current_pos[0])
        prev_y = float(self.current_pos[1])

        # dx: [-1, 1] -> [25, 55]px
        dx = 40 + float(action[0]) * 15
        dx = float(np.clip(dx, self.dx_min, self.dx_max))

        # dy: [-1, 1] -> [-80, +15]px
        dy_offset_range = self.dy_max - self.dy_min
        dy_offset = self.dy_min + (float(action[1]) + 1.0) / 2.0 * dy_offset_range
        dy_offset = float(np.clip(dy_offset, self.dy_min, self.dy_max))

        proposed_x = float(prev_x + dx)
        proposed_y = float(current_line["y1"] + dy_offset)

        will_complete = proposed_x >= current_line["x2"]
        effective_x = float(np.clip(proposed_x, current_line["x1"], current_line["x2"]))
        effective_y = proposed_y

        target = self._get_target(line_id, prev_x)
        reward, target_hit, dx_error, dy_error = self._calculate_reward(
            target=target,
            prev_x=prev_x,
            prev_y=prev_y,
            new_x=effective_x,
            new_y=effective_y,
            line=current_line,
            line_completed=will_complete,
        )

        self._record_dot(line_id, effective_x, effective_y)
        self._mark_line_complete(line_id)

        if will_complete:
            reward += 120.0
            self._line_targets.pop(line_id, None)
            if self.current_line_idx < self.total_lines - 1:
                self.current_line_idx += 1
                next_line = self._get_current_line()
                start_x = next_line["x1"] + self.np_random.uniform(-5, 15)
                start_y = next_line["y1"] + self.np_random.uniform(-20, 5)
                self.current_pos = np.array([start_x, start_y], dtype=np.float32)
                self._record_dot(next_line["id"], start_x, start_y)
                self._set_next_target(next_line["id"], start_x)
            else:
                self.current_pos = np.array([current_line["x2"], current_line["y1"]], dtype=np.float32)
        else:
            self.current_pos = np.array([effective_x, effective_y], dtype=np.float32)
            self._set_next_target(line_id, effective_x)

        self.step_count += 1

        all_complete = self.completed_lines_count == self.total_lines
        terminated = all_complete
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()

        info = {
            "position": self.current_pos.tolist(),
            "dx": dx,
            "dy_offset": dy_offset,
            "current_line": self.current_line_idx,
            "total_dots": self.total_dots,
            "lines_completed": self.completed_lines_count,
            "total_lines": self.total_lines,
            "line_completed": will_complete,
            "all_complete": all_complete,
            "target_dx": target["dx"],
            "target_y_offset": target["y_offset"],
            "target_hit": target_hit,
            "dx_error": dx_error,
            "dy_error": dy_error,
        }

        current_target = self._line_targets.get(self._get_current_line()["id"])
        if current_target is not None:
            info["next_target_dx"] = current_target["dx"]
            info["next_target_y_offset"] = current_target["y_offset"]

        return obs, reward, terminated, truncated, info

    def render(self):
        """No on-screen rendering is required for this environment."""
        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _recompute_line_cache(self) -> None:
        self.total_lines = len(self.lines)
        self._line_id_to_index = {line["id"]: idx for idx, line in enumerate(self.lines)}
        self._line_norm_cache = [
            {
                "y": line["y1"] / self.screen_h * 2 - 1,
                "x1": line["x1"] / self.screen_w * 2 - 1,
                "x2": line["x2"] / self.screen_w * 2 - 1,
            }
            for line in self.lines
        ]

    def _maybe_randomize_lines(self) -> None:
        if not self.line_randomization:
            self.lines = [dict(line) for line in self._base_lines]
            self._recompute_line_cache()
            return

        rng = self.np_random
        randomized: List[Dict[str, float]] = []

        for template in self._base_lines:
            line = dict(template)

            if self.line_jitter_y > 0.0:
                y_jitter = rng.uniform(-self.line_jitter_y, self.line_jitter_y)
                line["y1"] = float(np.clip(line["y1"] + y_jitter, 0.0, self.screen_h - 1))

            width = max(line["x2"] - line["x1"], 1.0)

            if self.line_scale_jitter > 0.0:
                scale = rng.uniform(1.0 - self.line_scale_jitter, 1.0 + self.line_scale_jitter)
                width = np.clip(width * scale, 20.0, float(self.screen_w))

            x_shift = rng.uniform(-self.line_jitter_x, self.line_jitter_x) if self.line_jitter_x > 0.0 else 0.0
            center = (line["x1"] + line["x2"]) / 2.0 + x_shift
            half = width / 2.0

            x1 = float(np.clip(center - half, 0.0, self.screen_w - 1))
            x2 = float(np.clip(center + half, x1 + 1.0, self.screen_w))

            line["x1"] = x1
            line["x2"] = x2

            randomized.append(line)

        randomized.sort(key=lambda l: l["y1"])

        if self.line_shuffle_prob > 0.0 and rng.random() < self.line_shuffle_prob:
            rng.shuffle(randomized)
            randomized.sort(key=lambda l: l["y1"])

        self.lines = randomized
        self._recompute_line_cache()

    def _sample_target(self, line: Dict[str, float], reference_x: float) -> Dict[str, float]:
        remaining = max(line["x2"] - reference_x, 0.0)
        dx_min, dx_max = self.target_dx_range
        max_dx = max(5.0, min(dx_max, remaining if remaining > 0 else dx_max))
        dx_low = min(dx_min, max_dx)
        dx_target = float(self.np_random.uniform(dx_low, max_dx)) if max_dx > 0 else 0.0

        y_min, y_max = self.target_y_range
        y_offset = float(self.np_random.uniform(y_min, y_max))
        if abs(y_offset) < 5.0:
            sign = np.sign(y_offset)
            if sign == 0:
                sign = self.np_random.choice([-1.0, 1.0])
            y_offset = sign * max(abs(y_offset), 5.0)

        return {
            "dx": dx_target,
            "y_offset": y_offset,
            "dx_tol": self.target_dx_tolerance,
            "y_tol": self.target_y_tolerance,
        }

    def _set_next_target(self, line_id: str, reference_x: float) -> None:
        line = self.lines[self._line_id_to_index[line_id]]
        self._line_targets[line_id] = self._sample_target(line, reference_x)

    def _get_target(self, line_id: str, reference_x: float) -> Dict[str, float]:
        target = self._line_targets.get(line_id)
        if target is None:
            self._set_next_target(line_id, reference_x)
            target = self._line_targets[line_id]
        return target

    def _get_current_line(self) -> Dict[str, float]:
        return self.lines[self.current_line_idx]

    def _is_line_complete(self, line_idx: int) -> bool:
        line = self.lines[line_idx]
        line_id = line["id"]
        if self.completed_line_flags[line_id]:
            return True

        dots = self.line_dots[line_id]
        if not dots:
            return False

        last_dot_x = dots[-1][0]
        remaining = line["x2"] - last_dot_x
        if remaining < 50:
            self.completed_line_flags[line_id] = True
            self.completed_lines_count += 1
            return True
        return False

    def _get_last_dot_on_line(self, line_id: str) -> Optional[Tuple[float, float]]:
        dots = self.line_dots[line_id]
        if not dots:
            return None
        return dots[-1]

    def _get_obs(self) -> np.ndarray:
        current_line = self._get_current_line()
        line_id = current_line["id"]
        target = self._get_target(line_id, self.current_pos[0])

        line_length = max(current_line["x2"] - current_line["x1"], 1.0)
        progress = np.clip((self.current_pos[0] - current_line["x1"]) / line_length, 0.0, 1.0)
        remaining = np.clip((current_line["x2"] - self.current_pos[0]) / line_length, 0.0, 1.0)
        current_y_offset = self.current_pos[1] - current_line["y1"]

        obs = np.zeros(12, dtype=np.float32)
        obs[0] = progress * 2.0 - 1.0
        obs[1] = remaining * 2.0 - 1.0
        obs[2] = np.clip(target["dx"] / max(self.target_dx_range[1], 1.0), 0.0, 1.0) * 2.0 - 1.0
        obs[3] = np.clip(target["y_offset"] / 60.0, -1.0, 1.0)
        obs[4] = np.clip(current_y_offset / 80.0, -1.0, 1.0)
        obs[5] = (
            (self.current_line_idx / max(self.total_lines - 1, 1))
            if self.total_lines > 1
            else 0.0
        ) * 2.0 - 1.0
        obs[6] = np.clip(self.completed_lines_count / max(self.total_lines, 1), 0.0, 1.0) * 2.0 - 1.0
        obs[7] = np.clip((self.total_dots % 20) / 20.0, 0.0, 1.0) * 2.0 - 1.0
        obs[8] = np.clip((line_length) / self.screen_w, 0.0, 1.0) * 2.0 - 1.0
        obs[9] = np.clip(current_line["y1"] / self.screen_h, 0.0, 1.0) * 2.0 - 1.0
        obs[10] = np.clip((target["dx"] - (self.current_pos[0] - current_line["x1"])) / max(line_length, 1.0), -1.0, 1.0)
        obs[11] = np.clip((self.step_count % 200) / 200.0, 0.0, 1.0) * 2.0 - 1.0

        return obs

    def _calculate_reward(
        self,
        *,
        target: Dict[str, float],
        prev_x: float,
        prev_y: float,
        new_x: float,
        new_y: float,
        line: Dict[str, float],
        line_completed: bool,
    ) -> Tuple[float, bool, float, float]:
        dx_actual = new_x - prev_x
        dy_actual = new_y - line["y1"]

        dx_error = dx_actual - target["dx"]
        dy_error = dy_actual - target["y_offset"]

        reward = 0.0
        target_hit = False

        if dx_actual <= 0:
            reward -= 50.0
        else:
            reward += 25.0 * np.clip(dx_actual / max(self.target_dx_range[1], 1.0), 0.0, 1.0)

        reward += 60.0 * np.exp(-0.5 * (dx_error / max(target["dx_tol"], 1.0)) ** 2)
        reward += 45.0 * np.exp(-0.5 * (dy_error / max(target["y_tol"], 1.0)) ** 2)

        if abs(dx_error) <= target["dx_tol"] and abs(dy_error) <= target["y_tol"]:
            reward += 90.0
            target_hit = True
        else:
            reward -= 12.0 * np.clip(abs(dx_error) / 30.0, 0.0, 2.0)
            reward -= 8.0 * np.clip(abs(dy_error) / 40.0, 0.0, 2.0)

        line_length = max(line["x2"] - line["x1"], 1.0)
        progress = np.clip((new_x - line["x1"]) / line_length, 0.0, 1.0)
        reward += 35.0 * progress

        if line_completed:
            reward += 120.0

        reward -= 1.0

        return reward, target_hit, dx_error, dy_error

    def _record_dot(self, line_id: str, x: float, y: float) -> None:
        self.line_dots[line_id].append((float(x), float(y)))
        self.total_dots += 1

    def _mark_line_complete(self, line_id: str) -> None:
        if self.completed_line_flags[line_id]:
            return
        dots = self.line_dots[line_id]
        if not dots:
            return
        line_idx = self._line_id_to_index[line_id]
        line = self.lines[line_idx]
        last_dot_x = dots[-1][0]
        remaining = line["x2"] - last_dot_x
        if remaining < 50:
            self.completed_line_flags[line_id] = True
            self.completed_lines_count += 1

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #
    def get_stats(self) -> Dict[str, float]:
        """
        Get episode statistics.
        """
        total_dots = self.total_dots
        completed_lines = self.completed_lines_count

        spacings: List[float] = []
        y_offsets: List[float] = []

        for line in self.lines:
            dots = self.line_dots[line["id"]]
            if len(dots) > 1:
                for idx in range(len(dots) - 1):
                    dx = dots[idx + 1][0] - dots[idx][0]
                    spacings.append(dx)

                for dot_x, dot_y in dots:
                    y_offsets.append(dot_y - line["y1"])

        return {
            "total_dots": total_dots,
            "lines_completed": completed_lines,
            "total_lines": self.total_lines,
            "success": completed_lines == self.total_lines,
            "avg_spacing": float(np.mean(spacings)) if spacings else 0.0,
            "spacing_std": float(np.std(spacings)) if spacings else 0.0,
            "avg_y_offset": float(np.mean(y_offsets)) if y_offsets else 0.0,
            "y_offset_std": float(np.std(y_offsets)) if y_offsets else 0.0,
            "dots_per_line": {line_id: len(dots) for line_id, dots in self.line_dots.items()},
        }


if __name__ == "__main__":
    env = HoverEnvMultiLineV2()

    print("Testing HoverEnvMultiLineV2 (FIXED)")
    print()

    obs, info = env.reset()
    total_reward = 0.0

    for step_idx in range(250):
        dx_action = np.random.normal(0, 0.3)
        dy_action = np.random.normal(-0.2, 0.4)
        action = np.clip(np.array([dx_action, dy_action]), -1, 1)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if info.get("line_completed"):
            print(
                f"Line {info['current_line'] - 1} completed (step {step_idx + 1}, dots: {info['total_dots']})"
            )

        if terminated or truncated:
            print(f"\nEpisode finished after {step_idx + 1} steps")
            if terminated:
                print("   Reason: all lines completed!")
            else:
                print("   Reason: max steps reached")
            break

    stats = env.get_stats()
    print("\nEpisode statistics:")
    print(f"   Dots: {stats['total_dots']}, Lines: {stats['lines_completed']}/{stats['total_lines']}")
    print(f"   Spacing: {stats['avg_spacing']:.1f}px ± {stats['spacing_std']:.1f}")
    print(f"   Y offset: {stats['avg_y_offset']:.1f}px ± {stats['y_offset_std']:.1f}")
    print(f"   Total reward: {total_reward:+.1f}")
    print()
    print("Rewards are scaled by 0.1 for stability.")
