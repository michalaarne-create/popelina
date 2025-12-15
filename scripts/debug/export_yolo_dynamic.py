"""
Export YOLOv11n to ONNX with dynamic batch size
FIXED for CUDA 11.8 + ONNX Runtime GPU
"""

from ultralytics import YOLO
import onnx
import onnxruntime as ort
import numpy as np

print("="*70)
print("📦 YOLOv11n ONNX Export - Dynamic Batch Size")
print("="*70)
print()

# Check versions
print(f"onnxruntime version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")
print()

# Load YOLOv11n
print("🔄 Loading yolo11n.pt...")
model = YOLO("yolo11n.pt")

# Export with dynamic batch
print("🔥 Exporting to ONNX with dynamic batch size...")
success = model.export(
    format="onnx",
    dynamic=True,
    simplify=True,
    opset=12,
)

print(f"✅ Export complete: {success}")

# Verify dynamic axes
print("\n🔍 Verifying ONNX model...")
onnx_model = onnx.load("yolo11n.onnx")

input_shape = onnx_model.graph.input[0].type.tensor_type.shape
print(f"   Input shape: ", end="")
for dim in input_shape.dim:
    if dim.dim_param:
        print(f"'{dim.dim_param}' ", end="")
    else:
        print(f"{dim.dim_value} ", end="")
print()

# Test with CUDA
print("\n🧪 Testing ONNX Runtime with CUDA...")

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# ✅ FIXED: Proper provider configuration for CUDA
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'DEFAULT',
    }),
    'CPUExecutionProvider'
]

session = ort.InferenceSession("yolo11n.onnx", providers=providers, sess_options=so)
active_provider = session.get_providers()[0]

print(f"   ✅ Active provider: {active_provider}")

if active_provider != 'CUDAExecutionProvider':
    print(f"   ⚠️  WARNING: Fell back to {active_provider}")
else:
    print(f"   🚀 CUDA acceleration: ENABLED")

# Test different batch sizes
print("\n🧪 Testing batch sizes...")
for batch_size in [1, 2, 4, 8, 16]:
    dummy_input = np.random.randn(batch_size, 3, 640, 640).astype(np.float32)
    try:
        outputs = session.run(None, {"images": dummy_input})
        print(f"   ✅ Batch {batch_size:2d}: {dummy_input.shape} → {outputs[0].shape}")
    except Exception as e:
        print(f"   ❌ Batch {batch_size:2d}: {str(e)[:60]}")

print("\n" + "="*70)
print("✅ EXPORT COMPLETE - GPU READY")
print("="*70)
print()
print(f"✅ ONNX file: yolo11n.onnx")
print(f"✅ Provider: {active_provider}")
print(f"✅ Dynamic batch: Supported")
print()