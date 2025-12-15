"""
Ewaluacja wytrenowanego agenta Phase 1.
Pokazuje jak agent stawia kropki i wypełnia linie.
"""

import sys
from pathlib import Path

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "envs").exists()),
    Path(__file__).resolve().parent,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from stable_baselines3 import PPO

# ✅ POPRAWIONY IMPORT:
from envs.hover_env_multiline import HoverEnvMultiLineV2

import json


def evaluate_model(model_path, num_episodes=3):
    """
    Ewaluuj model i pokaż wyniki.
    """
    
    print("=" * 70)
    print("📊 EVALUATION - MULTI-LINE HOVER V2")
    print("=" * 70)
    print()
    
    # Load model
    print(f"📂 Loading model: {model_path}")
    model = PPO.load(model_path)
    print("✅ Model loaded")
    print()
    
    # ✅ POPRAWIONY ENV:
    env = HoverEnvMultiLineV2(lines_file='data/text_lines.json')
    
    # Evaluate
    print(f"🎯 Running {num_episodes} episodes...")
    print()
    
    all_results = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        
        episode_reward = 0
        steps = 0
        trajectory = []
        
        print(f"{'='*70}")
        print(f"Episode {ep+1}/{num_episodes}")
        print(f"{'='*70}")
        
        last_completed = 0
        last_dots = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            trajectory.append(info['position'])
            episode_reward += reward
            steps += 1
            
            # Print progress
            if info['lines_completed'] > last_completed:
                print(f"🎉 Line {info['lines_completed']} completed! (step {steps}, total dots: {info['total_dots']})")
                last_completed = info['lines_completed']
            
            # Print dot placement
            if info['total_dots'] > last_dots:
                if steps % 10 == 0:  # Print every 10th dot
                    print(f"   ✓ Dot {info['total_dots']} placed (step {steps})")
                last_dots = info['total_dots']
            
            if steps % 100 == 0:
                print(f"   Progress - Step {steps}: {info['total_dots']} dots, {info['lines_completed']} lines")
        
        stats = env.get_stats()
        
        print(f"\n📊 Episode {ep+1} Summary:")
        print(f"   Steps: {steps}")
        print(f"   Total reward: {episode_reward:+.1f}")
        print(f"   Reward per step: {episode_reward/steps:+.2f}")
        print(f"   Dots placed: {stats['total_dots']}")
        print(f"   Lines completed: {stats['lines_completed']}/{stats['total_lines']}")
        print(f"   Completion: {stats['lines_completed']/stats['total_lines']*100:.1f}%")
        print(f"   Termination reason: {'Stuck' if info.get('stuck') else 'All lines full' if terminated else 'Max steps'}")
        print()
        
        all_results.append({
            'reward': episode_reward,
            'steps': steps,
            'dots': stats['total_dots'],
            'lines_completed': stats['lines_completed'],
            'trajectory': trajectory,
            'dots_per_line': stats['dots_per_line'],
            'line_dots': {k: list(v) for k, v in env.line_dots.items()},  # Convert to serializable
        })
    
    # Aggregate stats
    print(f"{'='*70}")
    print("📈 AGGREGATE STATISTICS")
    print(f"{'='*70}")
    
    avg_reward = np.mean([r['reward'] for r in all_results])
    std_reward = np.std([r['reward'] for r in all_results])
    avg_dots = np.mean([r['dots'] for r in all_results])
    avg_lines = np.mean([r['lines_completed'] for r in all_results])
    avg_steps = np.mean([r['steps'] for r in all_results])
    
    print(f"Average reward: {avg_reward:+.1f} ± {std_reward:.1f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average dots: {avg_dots:.1f}")
    print(f"Average lines completed: {avg_lines:.1f} / {stats['total_lines']}")
    print(f"Average completion: {avg_lines/stats['total_lines']*100:.1f}%")
    print()
    
    # Benchmarking
    print(f"{'='*70}")
    print("🎯 PERFORMANCE RATING")
    print(f"{'='*70}")
    
    if avg_reward > 2000:
        rating = "🌟 EKSTREMALNIE DOBRY - Gotowy do Phase 2!"
    elif avg_reward > 1500:
        rating = "🥇 BARDZO DOBRY - Gotowy do Phase 2!"
    elif avg_reward > 1000:
        rating = "🥈 DOBRY - Można iść do Phase 2 lub trenować dłużej"
    elif avg_reward > 500:
        rating = "🥉 OK - Rozważ trening do 800k steps"
    else:
        rating = "⚠️  SŁABY - Trenuj dłużej lub debug"
    
    print(f"Rating: {rating}")
    print(f"Dots coverage: {avg_dots/204*100:.1f}% (max ~204 dots possible)")
    print(f"Lines coverage: {avg_lines/stats['total_lines']*100:.1f}%")
    print()
    
    # Visualize
    try:
        visualize_results(all_results, env.lines)
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")
    
    env.close()
    
    return all_results


def visualize_results(results, lines):
    """Wizualizuj trajektorie i kropki"""
    
    # Create figure
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 10))
    
    if len(results) == 1:
        axes = [axes]
    
    for idx, (result, ax) in enumerate(zip(results, axes)):
        # Rysuj linie tekstu
        for line in lines:
            ax.plot([line['x1'], line['x2']], [line['y1'], line['y2']], 
                    'k-', linewidth=2, alpha=0.3)
            
            # Valid area (Y: -80 to +15)
            rect = Rectangle(
                (line['x1'], line['y1'] - 80),
                line['x2'] - line['x1'], 95,
                fill=True, alpha=0.05, color='green', edgecolor='green', linestyle='--', linewidth=0.5
            )
            ax.add_patch(rect)
            
            # Label
            ax.text(line['x1'] - 60, line['y1'], f"{line['id']}", 
                   fontsize=7, ha='right', va='center', alpha=0.6)
        
        # Rysuj kropki
        for line_id, dots in result['line_dots'].items():
            for dot_x, dot_y in dots:
                circle = Circle((dot_x, dot_y), radius=12, 
                              color='blue', alpha=0.6, zorder=10, edgecolor='darkblue', linewidth=0.5)
                ax.add_patch(circle)
        
        # Rysuj trajektorię (subsampled dla czytelności)
        trajectory = np.array(result['trajectory'])
        step_size = max(1, len(trajectory) // 200)  # Max 200 punktów
        trajectory_sub = trajectory[::step_size]
        
        ax.plot(trajectory_sub[:, 0], trajectory_sub[:, 1], 
               'r-', linewidth=0.3, alpha=0.2, label='Trajectory', zorder=1)
        
        # Start/end
        ax.plot(trajectory[0, 0], trajectory[0, 1], 
               'go', markersize=8, label='Start', zorder=11)
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 
               'ro', markersize=8, label='End', zorder=11)
        
        # Styling
        ax.set_xlim(0, 1920)
        ax.set_ylim(0, 1200)
        ax.invert_yaxis()
        ax.set_xlabel('X position (px)', fontsize=10)
        ax.set_ylabel('Y position (px)', fontsize=10)
        ax.set_title(f"Episode {idx+1}\n"
                    f"Dots: {result['dots']}, Lines: {result['lines_completed']}/17\n"
                    f"Reward: {result['reward']:+.0f}",
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('models/saved/phase1/eval')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'evaluation_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Evaluate best model
    model_path = "models/saved/phase1/best_model/best_model.zip"
    
    if not Path(model_path).exists():
        # Try final model
        model_path = "models/saved/phase1/final_model.zip"
        
        if not Path(model_path).exists():
            # Try checkpoints
            checkpoint_dir = Path("models/saved/phase1/checkpoints")
            if checkpoint_dir.exists():
                checkpoints = sorted(checkpoint_dir.glob("*.zip"))
                if checkpoints:
                    model_path = str(checkpoints[-1])  # Latest checkpoint
                    print(f"ℹ️  Using latest checkpoint: {model_path}")
            
            if not Path(model_path).exists():
                print(f"❌ No model found!")
                print("   Train first with: python scripts/04_train_ppo_phase1.py")
                sys.exit(1)
    
    results = evaluate_model(
        model_path=model_path,
        num_episodes=3,
    )
    
    print("\n✅ Evaluation complete!")
