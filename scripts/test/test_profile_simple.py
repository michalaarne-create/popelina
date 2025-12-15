"""
Simple profiler - bez dependencies
"""

import time
import torch
import numpy as np

print("="*70)
print("🔬 SIMPLE PROFILE TEST")
print("="*70)
print()

# Test 1: Env rendering speed
print("Test 1: Rendering speed")

import cv2

times = []
for _ in range(100):
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    start = time.time()
    
    # Simulate dropdown rendering
    cv2.rectangle(img, (200, 200), (500, 240), (51, 51, 51), 2)
    cv2.rectangle(img, (200, 200), (500, 240), (245, 245, 245), -1)
    
    elapsed = time.time() - start
    times.append(elapsed)

avg_render = np.mean(times) * 1000
print(f"   Average render: {avg_render:.2f} ms")
print()

# Test 2: YOLOv8 speed
if torch.cuda.is_available():
    print("Test 2: YOLOv8n speed")
    
    from ultralytics import YOLO
    
    device = 'cuda'
    model = YOLO('yolov8n.pt')
    backbone = torch.nn.Sequential(*list(model.model.model[:10]))
    backbone = backbone.to(device).eval()
    
    # Warmup
    with torch.no_grad():
        sample = torch.randn(4, 3, 480, 640).to(device)
        for _ in range(10):
            _ = backbone(sample)
        torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(100):
            sample = torch.randn(4, 3, 480, 640).to(device)
            
            start = time.time()
            output = backbone(sample)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            times.append(elapsed)
    
    avg_yolo = np.mean(times) * 1000
    print(f"   Average YOLOv8 (batch=4): {avg_yolo:.2f} ms")
    print(f"   Per sample: {avg_yolo/4:.2f} ms")
    print()

print("✅ Done!")