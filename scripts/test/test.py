import torch
import onnxruntime as ort

print("Torch CUDA:", torch.cuda.is_available())
print("ONNX Runtime device:", ort.get_device())
