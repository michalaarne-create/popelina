"""
Deep diagnostics for ONNX Runtime CUDA
"""
import sys
import os

print("="*70)
print("🔬 ONNX RUNTIME CUDA DIAGNOSTICS")
print("="*70)
print()

# 1. Python & Environment
print("1️⃣ ENVIRONMENT")
print(f"   Python: {sys.version.split()[0]}")
print(f"   Platform: {sys.platform}")
print()

# 2. CUDA Environment Variables
print("2️⃣ CUDA ENVIRONMENT")
cuda_path = os.environ.get('CUDA_PATH', 'NOT SET')
cuda_home = os.environ.get('CUDA_HOME', 'NOT SET')
path = os.environ.get('PATH', '')

print(f"   CUDA_PATH: {cuda_path}")
print(f"   CUDA_HOME: {cuda_home}")
print(f"   CUDA in PATH: {'cuda' in path.lower()}")

# Find CUDA in PATH
if 'cuda' in path.lower():
    cuda_paths = [p for p in path.split(';') if 'cuda' in p.lower()]
    print(f"   CUDA paths found:")
    for p in cuda_paths[:3]:
        print(f"      {p}")
print()

# 3. PyTorch CUDA
print("3️⃣ PYTORCH CUDA")
try:
    import torch
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU count: {torch.cuda.device_count()}")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# 4. ONNX Runtime
print("4️⃣ ONNX RUNTIME")
try:
    import onnxruntime as ort
    print(f"   Version: {ort.__version__}")
    print(f"   Available providers: {ort.get_available_providers()}")
    
    # Try to get CUDA provider info
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print(f"   ✅ CUDAExecutionProvider: Available")
        
        # Try to get detailed info
        try:
            provider_options = ort.get_provider_options('CUDAExecutionProvider')
            print(f"   Provider options: {provider_options}")
        except:
            pass
    else:
        print(f"   ❌ CUDAExecutionProvider: NOT Available")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# 5. Test CUDA Provider
print("5️⃣ TESTING CUDA PROVIDER")
try:
    import onnxruntime as ort
    import numpy as np
    
    # Create dummy session with CUDA
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
        })
    ]
    
    print("   Creating test session with CUDAExecutionProvider...")
    
    # This needs an ONNX file
    if os.path.exists("yolo11n.onnx"):
        session = ort.InferenceSession("yolo11n.onnx", providers=providers, sess_options=so)
        print(f"   ✅ Session created!")
        print(f"   Active providers: {session.get_providers()}")
    else:
        print("   ⚠️  yolo11n.onnx not found, skipping test")
        
except Exception as e:
    print(f"   ❌ CUDA Provider Error:")
    print(f"      {str(e)}")
    
    # Parse error for hints
    error_str = str(e)
    if "CUDA failure 100" in error_str:
        print()
        print("   💡 DIAGNOSIS: CUDA not detected by ONNX Runtime")
        print()
        print("   Possible causes:")
        print("      1. onnxruntime-gpu built for different CUDA version")
        print("      2. Missing cudnn DLLs")
        print("      3. CUDA not in PATH")
        
print()

# 6. Check DLL dependencies (Windows)
if sys.platform == 'win32':
    print("6️⃣ CHECKING ONNX RUNTIME DLLs")
    try:
        import onnxruntime
        ort_path = os.path.dirname(onnxruntime.__file__)
        print(f"   ONNX Runtime path: {ort_path}")
        
        # List DLLs
        dlls = [f for f in os.listdir(ort_path) if f.endswith('.dll')]
        print(f"   DLLs found: {len(dlls)}")
        
        cuda_dlls = [d for d in dlls if 'cuda' in d.lower() or 'cudnn' in d.lower()]
        if cuda_dlls:
            print(f"   CUDA-related DLLs:")
            for dll in cuda_dlls:
                print(f"      {dll}")
        else:
            print(f"   ⚠️  No CUDA DLLs found in ONNX Runtime package")
            print(f"      This might be CPU-only build!")
    except Exception as e:
        print(f"   Error: {e}")
    print()

print("="*70)
print("📋 SUMMARY")
print("="*70)

# Final recommendation
import onnxruntime as ort
import torch

if 'CUDAExecutionProvider' not in ort.get_available_providers():
    print("❌ PROBLEM: CUDAExecutionProvider not available")
    print()
    print("🔧 SOLUTION:")
    print("   You may have CPU-only onnxruntime installed.")
    print()
    print("   Run these commands:")
    print("   pip uninstall onnxruntime onnxruntime-gpu -y")
    print("   pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/nuget/v3/index.json")
    
elif torch.cuda.is_available() and 'CUDAExecutionProvider' in ort.get_available_providers():
    print("⚠️  PARTIAL: Both have CUDA support but connection failed")
    print()
    print("🔧 SOLUTION:")
    print(f"   PyTorch uses: CUDA {torch.version.cuda}")
    print(f"   ONNX Runtime needs matching CUDA version")
    print()
    print("   Try reinstalling onnxruntime-gpu:")
    print("   pip uninstall onnxruntime-gpu -y")
    
    if torch.version.cuda.startswith('11'):
        print("   pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/nuget/v3/index.json")
    elif torch.version.cuda.startswith('12'):
        print("   pip install onnxruntime-gpu")
        
print()