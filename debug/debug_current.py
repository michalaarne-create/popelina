"""
Debug obecnego modelu - zobacz CO DOKŁADNIE robi.
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
from envs.hover_env_multiline import HoverEnvMultiLineV2
import numpy as np

def debug_agent():
    # Load latest model
    model_path = "models/saved/phase1/best_model/best_model.zip"
    
    if not Path(model_path).exists():
        checkpoints = sorted(Path("models/saved/phase1/checkpoints").glob("*.zip"))
        if checkpoints:
            model_path = str(checkpoints[-1])
        else:
            print("❌ No model found!")
            return
    
    print(f"📂 Loading: {model_path}")
    model = PPO.load(model_path)
    
    env = HoverEnvMultiLineV2(lines_file='data/text_lines.json')
    
    print("\n" + "="*70)
    print("🔍 DEBUGGING AGENT BEHAVIOR")
    print("="*70 + "\n")
    
    for ep in range(2):
        obs, info = env.reset()
        done = False
        
        print(f"\n{'='*70}")
        print(f"Episode {ep+1}")
        print(f"{'='*70}")
        print(f"Start pos: {env.current_pos}")
        print(f"Start line: {env.lines[0]['text'][:30]}...\n")
        
        step = 0
        total_reward = 0
        last_x = env.current_pos[0]
        
        for _ in range(50):  # Max 50 kroków do debugowania
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Oblicz faktyczne dx
            current_x = info['position'][0]
            actual_dx = current_x - last_x
            last_x = current_x
            
            total_reward += reward
            step += 1
            
            # Print co krok
            print(f"Step {step:3d}: "
                  f"Action=[{action[0]:+.2f}, {action[1]:+.2f}] → "
                  f"dx={info['dx']:.1f}px, dy_off={info['dy_offset']:.1f}px | "
                  f"Pos=[{info['position'][0]:.0f}, {info['position'][1]:.0f}] | "
                  f"Line {info['current_line']}, Dots: {info['total_dots']}, "
                  f"Reward: {reward:+.1f}")
            
            if info.get('line_completed'):
                print(f"    ✅ LINE {info['current_line']-1} COMPLETED!")
            
            if done:
                reason = "all_complete" if info.get('all_complete') else "max_steps"
                print(f"\n    🏁 Done after {step} steps ({reason})")
                break
        
        print(f"\n📊 Episode summary:")
        print(f"   Steps: {step}")
        print(f"   Total reward: {total_reward:+.1f}")
        print(f"   Lines completed: {info['lines_completed']}/{info['total_lines']}")
        print(f"   Total dots: {info['total_dots']}")
    
    # Check env type
    print(f"\n{'='*70}")
    print("🔍 ENVIRONMENT CHECK")
    print(f"{'='*70}")
    print(f"Env class: {env.__class__.__name__}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space shape: {env.observation_space.shape}")
    
    # Check if sequential
    if hasattr(env, 'ideal_dx'):
        print(f"\n✅ SEQUENTIAL ENV detected!")
        print(f"   Ideal dx: {env.ideal_dx}px")
        print(f"   Natural dx range: {env.natural_dx_range}")
        print(f"   Natural dy range: {env.natural_dy_range}")
    else:
        print(f"\n❌ OLD ENV (non-sequential)!")
        print(f"   This explains poor performance!")

if __name__ == "__main__":
    debug_agent()
