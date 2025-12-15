"""
Profilowanie środowiska - gdzie jest bottleneck?
"""

import sys
from pathlib import Path

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "envs").exists()),
    Path(__file__).resolve().parent,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import time
import numpy as np
from envs.dropdown_env import DropdownEnv


def profile_env():
    """Profile env performance"""
    
    env = DropdownEnv(questions_file='data/quiz_dropdowns.json')
    
    print("="*70)
    print("🔬 PROFILING ENVIRONMENT")
    print("="*70)
    print()
    
    # Test 1: Reset performance
    print("Test 1: Reset speed")
    times = []
    for i in range(100):
        start = time.time()
        obs, info = env.reset(seed=i)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_reset = np.mean(times) * 1000
    print(f"   Average reset: {avg_reset:.2f} ms")
    print()
    
    # Test 2: Step performance
    print("Test 2: Step speed")
    env.reset()
    
    step_times = []
    render_times = []
    
    for _ in range(100):
        action = env.action_space.sample()
        
        start = time.time()
        obs, reward, term, trunc, info = env.step(action)
        step_time = time.time() - start
        step_times.append(step_time)
        
        # Measure rendering separately
        start = time.time()
        screen = env._render_screen()
        render_time = time.time() - start
        render_times.append(render_time)
        
        if term or trunc:
            env.reset()
    
    avg_step = np.mean(step_times) * 1000
    avg_render = np.mean(render_times) * 1000
    
    print(f"   Average step: {avg_step:.2f} ms")
    print(f"   Average render: {avg_render:.2f} ms")
    print(f"   Render overhead: {avg_render/avg_step*100:.1f}%")
    print()
    
    # Test 3: FPS estimation
    print("Test 3: Estimated training FPS")
    
    # Step time + network forward pass (estimate ~5ms)
    network_time = 5  # ms
    total_time_per_step = avg_step + network_time
    
    fps = 1000 / total_time_per_step
    
    print(f"   Step: {avg_step:.1f}ms + Network: {network_time:.1f}ms = {total_time_per_step:.1f}ms")
    print(f"   Estimated FPS: {fps:.0f} steps/sec")
    print()
    
    # Diagnosis
    print("="*70)
    print("📊 DIAGNOSIS")
    print("="*70)
    
    if avg_render > 10:
        print("❌ PROBLEM: Rendering is VERY SLOW (>10ms)")
        print("   Solutions:")
        print("   1. Cache rendered screens (pre-render)")
        print("   2. Simplify rendering (remove text, use rectangles only)")
        print("   3. Use state vector only (no vision)")
    elif avg_render > 5:
        print("⚠️  WARNING: Rendering is slow (>5ms)")
        print("   Consider caching or simplifying")
    else:
        print("✅ Rendering is OK (<5ms)")
    
    print()
    
    if fps < 100:
        print("❌ PROBLEM: Estimated FPS too low (<100)")
        print(f"   Current: {fps:.0f} FPS")
        print(f"   Target: 200+ FPS")
    elif fps < 200:
        print("⚠️  WARNING: FPS could be better")
        print(f"   Current: {fps:.0f} FPS")
    else:
        print(f"✅ FPS is good ({fps:.0f} FPS)")
    
    print()
    env.close()


if __name__ == "__main__":
    profile_env()
