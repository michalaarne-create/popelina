"""
Test różnicy między rollout a eval.
"""

import sys
from pathlib import Path

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "envs").exists()),
    Path(__file__).resolve().parent,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.hover_env_multiline import HoverEnvMultiLineV2
import numpy as np

def make_env(rank):
    def _init():
        env = HoverEnvMultiLineV2(lines_file='data/text_lines.json')
        env = Monitor(env)
        return env
    return _init

# Load model
model_path = "models/saved/phase1/checkpoints/ppo_hover_v2_950000_steps.zip"
model = PPO.load(model_path)

print("="*70)
print("TEST 1: Eval env (single)")
print("="*70)

eval_env = Monitor(HoverEnvMultiLineV2(lines_file='data/text_lines.json'))

total_rewards = []
total_lengths = []

for ep in range(5):
    obs, _ = eval_env.reset()
    done = False
    ep_reward = 0
    steps = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        ep_reward += reward
        steps += 1
        
        if steps >= 1000:  # Safety
            break
    
    total_rewards.append(ep_reward)
    total_lengths.append(steps)
    
    print(f"Episode {ep+1}: {steps} steps, reward={ep_reward:.1f}")

print(f"\nEval summary:")
print(f"  Mean reward: {np.mean(total_rewards):.1f}")
print(f"  Mean length: {np.mean(total_lengths):.1f}")

eval_env.close()

print("\n" + "="*70)
print("TEST 2: Vectorized env (like training)")
print("="*70)

vec_env = SubprocVecEnv([make_env(i) for i in range(4)])

obs = vec_env.reset()

episode_rewards = [0] * 4
episode_lengths = [0] * 4
completed_episodes = []

for step in range(500):  # Max 500 steps
    actions, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = vec_env.step(actions)
    
    for i in range(4):
        episode_rewards[i] += rewards[i]
        episode_lengths[i] += 1
        
        if dones[i]:
            print(f"  Env {i}: {episode_lengths[i]} steps, reward={episode_rewards[i]:.1f}")
            completed_episodes.append({
                'reward': episode_rewards[i],
                'length': episode_lengths[i]
            })
            
            episode_rewards[i] = 0
            episode_lengths[i] = 0
    
    if len(completed_episodes) >= 5:
        break

print(f"\nVectorized summary:")
print(f"  Mean reward: {np.mean([e['reward'] for e in completed_episodes]):.1f}")
print(f"  Mean length: {np.mean([e['length'] for e in completed_episodes]):.1f}")

vec_env.close()

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"Eval:       {np.mean(total_rewards):.1f} reward")
print(f"Vectorized: {np.mean([e['reward'] for e in completed_episodes]):.1f} reward")
