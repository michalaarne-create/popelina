"""
Test CPU→GPU transfer bottleneck
"""

import torch
import numpy as np
import time

device = 'cuda'

print("🔬 CPU→GPU TRANSFER BENCHMARK")
print("="*70)
print()

# Simulate screen batch
batch_sizes = [16, 64, 128, 256, 512]

for batch_size in batch_sizes:
    # Create batch
    screens_cpu = np.random.randint(0, 255, (batch_size, 480, 640, 3), dtype=np.uint8)
    
    # Measure transfer time
    times = []
    for _ in range(20):
        start = time.time()
        
        # Transfer
        screens_gpu = torch.FloatTensor(screens_cpu).to(device)
        torch.cuda.synchronize()
        
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = np.mean(times) * 1000
    throughput = (batch_size * screens_cpu.nbytes / 1e6) / (avg_time / 1000)  # MB/s
    
    print(f"Batch {batch_size:3d}: {avg_time:6.1f} ms  ({screens_cpu.nbytes/1e6:5.1f} MB, {throughput:6.0f} MB/s)")

print()
print("="*70)

# Expected training overhead
print("📊 TRAINING IMPACT:")
print()
print("With batch_size=512, n_epochs=4:")
total_transfer_per_update = 471 * 4  # MB
print(f"  Transfer per update: {total_transfer_per_update:.0f} MB")
print(f"  If 50ms per batch → {50*16:.0f}ms wasted on transfers!")
print()
print("💡 SOLUTION: Reduce screen resolution!")