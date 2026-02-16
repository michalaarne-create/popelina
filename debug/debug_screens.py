"""
Debug - czy środowiska mają różne pytania?
"""

import sys
from pathlib import Path

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "envs").exists()),
    Path(__file__).resolve().parent,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from envs.dropdown_env import DropdownEnv
import cv2
import numpy as np

def test_diversity():
    """Test czy każdy reset daje inne pytanie"""
    
    env = DropdownEnv(questions_file='data/quiz_dropdowns.json')
    
    print("🧪 Testing environment diversity...")
    print()
    
    questions_seen = []
    screens = []
    
    for i in range(5):
        obs, info = env.reset(seed=i)
        
        question = info['question']
        screen = obs['screen']
        
        questions_seen.append(question)
        screens.append(screen)
        
        # Save
        cv2.imwrite(f'debug_reset_{i}.png', screen)
        
        print(f"Reset {i}:")
        print(f"  Question: {question[:50]}...")
        print(f"  Screen unique pixels: {len(np.unique(screen)):,}")
        print(f"  Dropdown open: {info['dropdown_open']}")
        print()
    
    # Check diversity
    unique_questions = len(set(questions_seen))
    
    print(f"📊 Results:")
    print(f"  Resets: 5")
    print(f"  Unique questions: {unique_questions}")
    
    if unique_questions < 5:
        print(f"  ⚠️  PROBLEM: Same questions appearing!")
    else:
        print(f"  ✅ All different questions!")
    
    # Screen diversity
    print(f"\n🎨 Screen diversity:")
    for i in range(5):
        unique = len(np.unique(screens[i]))
        print(f"  Screen {i}: {unique:,} unique values")
    
    # Are they identical?
    all_same = all(np.array_equal(screens[0], s) for s in screens[1:])
    
    if all_same:
        print(f"\n  ❌ ERROR: All screens IDENTICAL!")
    else:
        print(f"\n  ✅ Screens are different!")
    
    print(f"\n💾 Saved: debug_reset_0.png to debug_reset_4.png")
    print(f"   Open them - should show DIFFERENT dropdowns!")
    
    env.close()


if __name__ == "__main__":
    test_diversity()
