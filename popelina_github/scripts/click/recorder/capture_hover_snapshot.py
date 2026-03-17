"""
Capture a single observation snapshot from HoverEnvWithScreenshots.

The script reproduces the training environment configuration (including
optional line randomisation) and saves a PNG preview showing the grid along
with recorded dots.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "envs").exists()),
    Path(__file__).resolve().parent,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import cv2
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scripts.train.hover.train_hover_phase2 import HoverEnvWithScreenshots, TinyHoverCNNExtractor


def _make_env(randomize_layout: bool, training_mode: bool) -> HoverEnvWithScreenshots:
    env = HoverEnvWithScreenshots(
        lines_file="data/text_lines.json",
        training=training_mode,
        state_dropout_prob=0.0,
        state_block_dropout_prob=0.0,
        env_hint_weight=0.0,
        line_randomization=randomize_layout,
        line_jitter_y=32.0 if randomize_layout else 0.0,
        line_jitter_x=60.0 if randomize_layout else 0.0,
        line_scale_jitter=0.15 if randomize_layout else 0.0,
        line_shuffle_prob=0.1 if randomize_layout else 0.0,
    )
    return env


def capture_snapshot(
    *,
    output_path: Path,
    model_path: Path,
    vecnorm_path: Path | None = None,
    deterministic: bool = True,
    steps: int = 10,
    training_mode: bool = True,
    randomize_layout: bool = True,
) -> Dict[str, List[Tuple[float, float]]]:
    base_env = _make_env(randomize_layout=randomize_layout, training_mode=training_mode)

    vec_env = DummyVecEnv([lambda: base_env])
    if vecnorm_path and vecnorm_path.exists():
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        print(f"[INFO] Loaded VecNormalize stats from {vecnorm_path}")
    else:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=50.0)
        print("[WARN] VecNormalize stats missing; using fresh stats (results may differ).")
    vec_env.training = False
    vec_env.norm_obs = False
    vec_env.norm_reward = False
            
    model = PPO.load(model_path)
    model.policy.set_training_mode(False)

    obs = vec_env.reset()
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, dones, _ = vec_env.step(action)
        if bool(dones[0]):
            break

    frame = base_env._render_screenshot().copy()
    full_frame = np.ones((base_env.screen_h, base_env.screen_w, 3), dtype=np.uint8) * 240

    for line in base_env.lines:
        y = int(round(line["y1"]))
        cv2.rectangle(full_frame, (int(line["x1"]), y - 2), (int(line["x2"]), y + 2), (80, 80, 80), thickness=-1)

    cell_w = base_env._grid_cell_w
    cell_h = base_env._grid_cell_h

    line_dots = {line_id: list(dots) for line_id, dots in base_env.line_dots.items()}

    for dots in line_dots.values():
        for x, y in dots:
            col = int(np.clip(np.floor(x / cell_w), 0, frame.shape[1] - 1))
            row = int(np.clip(np.floor(y / cell_h), 0, frame.shape[0] - 1))
            frame[row, col] = (0, 255, 255)
            cv2.circle(full_frame, (int(x), int(y)), 12, (0, 255, 255), -1)

    upscale_size = 1024
    frame = cv2.resize(frame, (upscale_size, upscale_size), interpolation=cv2.INTER_NEAREST)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    cv2.imwrite(str(output_path.with_name("hover_snapshot_full.png")), full_frame)

    vec_env.close()
    return line_dots


def main() -> None:
    output_dir = PROJECT_ROOT / "debug"
    output_path = output_dir / "hover_snapshot.png"
    model_path = PROJECT_ROOT / "models/saved/phase2_transfer/final_model.zip"
    vecnorm_path = PROJECT_ROOT / "models/saved/phase2_transfer/vecnormalize.pkl"

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    line_dots = capture_snapshot(
        output_path=output_path,
        model_path=model_path,
        vecnorm_path=vecnorm_path if vecnorm_path.exists() else None,
        deterministic=True,
        steps=10,
        training_mode=True,
        randomize_layout=True,
    )

    print(f"[INFO] Snapshot saved to: {output_path}")
    print()
    print(f"[INFO] Line dots: {line_dots}")
    print()


if __name__ == "__main__":
    main()
