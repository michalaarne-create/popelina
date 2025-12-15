"""
Quick test - sprawdź co się wywala
"""

print("Starting...")

try:
    print("1. Importing libraries...")
    import sys
    from pathlib import Path
    PROJECT_ROOT = next(
        (p for p in Path(__file__).resolve().parents if (p / "envs").exists()),
        Path(__file__).resolve().parent,
    )
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    print("   ✓ Path setup OK")
    
    import torch
    print(f"   ✓ PyTorch OK (CUDA: {torch.cuda.is_available()})")
    
    from envs.dropdown_env import DropdownEnv
    print("   ✓ DropdownEnv import OK")
    
    from models.feature_extractor import DropdownFeaturesExtractor
    print("   ✓ FeatureExtractor import OK")
    
    print("\n2. Creating environment...")
    env = DropdownEnv(questions_file='data/quiz_dropdowns.json')
    print("   ✓ Env created")
    
    print("\n3. Testing env reset...")
    obs, info = env.reset()
    print(f"   ✓ Reset OK")
    print(f"   Screen shape: {obs['screen'].shape}")
    print(f"   State shape: {obs['state'].shape}")
    
    print("\n4. Testing env step...")
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(f"   ✓ Step OK")
    print(f"   Action taken: {info.get('action_taken', 'N/A')}")
    print(f"   Reward: {reward}")
    
    print("\n5. Creating feature extractor...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    from gymnasium import spaces
    obs_space = spaces.Dict({
        'screen': spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype='uint8'),
        'state': spaces.Box(low=-1, high=1, shape=(8,), dtype='float32')
    })
    
    extractor = DropdownFeaturesExtractor(obs_space, features_dim=128)
    extractor = extractor.to(device)
    print(f"   ✓ Extractor created on {device}")
    
    print("\n6. Testing feature extraction...")
    obs_tensor = {
        'screen': torch.FloatTensor(obs['screen']).unsqueeze(0).to(device),
        'state': torch.FloatTensor(obs['state']).unsqueeze(0).to(device)
    }
    
    with torch.no_grad():
        features = extractor(obs_tensor)
    
    print(f"   ✓ Features extracted")
    print(f"   Features shape: {features.shape}")
    print(f"   Features dtype: {features.dtype}")
    print(f"   Features range: [{features.min():.2f}, {features.max():.2f}]")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    
    env.close()
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
