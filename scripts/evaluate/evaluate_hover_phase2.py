"""
Hover Phase 2 evaluation script for the lightweight CNN policy.

Loads a saved PPO model (and optional VecNormalize statistics) and runs a
configurable number of evaluation episodes in the Hover environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "envs").exists()),
    Path(__file__).resolve().parent,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan

from scripts.train.hover.train_hover_phase2 import HoverEnvWithScreenshots


def _make_env_factory(
    *,
    randomize_layout: bool,
    max_steps: int = 300,
) -> Callable[[], HoverEnvWithScreenshots]:
    def _factory() -> HoverEnvWithScreenshots:
        env = HoverEnvWithScreenshots(
            lines_file="data/text_lines.json",
            training=randomize_layout,
            state_dropout_prob=0.0,
            state_block_dropout_prob=0.0,
            env_hint_weight=0.0,
            line_randomization=randomize_layout,
            line_jitter_y=32.0 if randomize_layout else 0.0,
            line_jitter_x=60.0 if randomize_layout else 0.0,
            line_scale_jitter=0.15 if randomize_layout else 0.0,
            line_shuffle_prob=0.1 if randomize_layout else 0.0,
        )
        env.max_steps = max_steps
        return env

    return _factory


def _build_vec_env(vecnorm_path: Path | None, randomize_layout: bool) -> VecNormalize:
    env = DummyVecEnv([_make_env_factory(randomize_layout=randomize_layout)])

    if vecnorm_path and vecnorm_path.exists():
        vec_env = VecNormalize.load(vecnorm_path, env)
        print(f"[INFO] VecNormalize stats loaded from: {vecnorm_path}")
    else:
        vec_env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=50.0)
        print("[WARN] VecNormalize stats not found; using fresh normalisation.")

    vec_env.training = False
    vec_env.norm_obs = False
    vec_env.norm_reward = False

    return VecCheckNan(vec_env, raise_exception=False, warn_once=True)


def evaluate_model(
    model_path: Path,
    vecnorm_path: Path | None,
    episodes: int,
    deterministic: bool,
    randomize_layout: bool,
) -> Dict[str, float]:
    print("=" * 70)
    print("HOVER PHASE 2 - EVALUATION")
    print("=" * 70)
    print(f"[INFO] Model path:      {model_path}")
    if vecnorm_path:
        print(f"[INFO] VecNormalize:    {vecnorm_path}")
    print(f"[INFO] Episodes:        {episodes}")
    print(f"[INFO] Deterministic:   {deterministic}")
    print(f"[INFO] Layout random:   {randomize_layout}")
    print()

    vec_env = _build_vec_env(vecnorm_path, randomize_layout=randomize_layout)

    model = PPO.load(model_path)
    model.policy.set_training_mode(False)

    episode_rewards: List[float] = []
    episode_steps: List[int] = []
    successes: List[bool] = []

    obs = vec_env.reset()

    for episode_idx in range(episodes):
        done = False
        ep_reward = 0.0
        steps = 0
        info: Dict[str, float] = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, infos = vec_env.step(action)

            reward_value = float(reward[0])
            done = bool(dones[0])
            info = infos[0]

            ep_reward += reward_value
            steps += 1

        episode_rewards.append(ep_reward)
        episode_steps.append(steps)
        successes.append(bool(info.get("all_complete", False)))

        print(
            f"Episode {episode_idx + 1:02d}: reward={ep_reward:+10.2f}  "
            f"steps={steps:3d}  success={'YES' if successes[-1] else 'NO'}"
        )

        obs = vec_env.reset()

    vec_env.close()

    rewards_arr = np.asarray(episode_rewards, dtype=np.float32)
    steps_arr = np.asarray(episode_steps, dtype=np.int32)
    success_rate = float(np.mean(successes) * 100.0)

    summary = {
        "reward_mean": float(np.mean(rewards_arr)),
        "reward_std": float(np.std(rewards_arr)),
        "reward_min": float(np.min(rewards_arr)),
        "reward_max": float(np.max(rewards_arr)),
        "steps_mean": float(np.mean(steps_arr)),
        "success_rate": success_rate,
    }

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Mean reward:   {summary['reward_mean']:+10.2f} ± {summary['reward_std']:.2f}")
    print(f"Reward range:  {summary['reward_min']:+10.2f} .. {summary['reward_max']:+10.2f}")
    print(f"Mean steps:    {summary['steps_mean']:.1f}")
    print(f"Success rate:  {summary['success_rate']:.1f}%")
    print("=" * 70)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Hover Phase 2 PPO agent.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/saved/phase2_transfer/best_model/best_model.zip"),
        help="Path to the saved PPO model.",
    )
    parser.add_argument(
        "--vecnorm",
        type=Path,
        default=Path("models/saved/phase2_transfer/vecnormalize.pkl"),
        help="Path to the saved VecNormalize statistics.",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy.")
    parser.add_argument(
        "--randomize-layout",
        action="store_true",
        help="Evaluate with randomized line layouts (harder, matching training augmentation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path: Path = args.model
    vecnorm_path: Path | None = args.vecnorm if args.vecnorm else None

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    if vecnorm_path is not None and not vecnorm_path.exists():
        print(f"[WARN] VecNormalize stats not found: {vecnorm_path}")
        vecnorm_path = None

    evaluate_model(
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        episodes=max(1, args.episodes),
        deterministic=args.deterministic,
        randomize_layout=args.randomize_layout,
    )


if __name__ == "__main__":
    main()
