"""
Asynchronous environment wrapper with prefetching
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from collections import deque
import threading
from stable_baselines3.common.vec_env import VecEnv


class AsyncPrefetchWrapper:
    """
    Wraps VecEnv to prefetch observations on separate thread.
    GPU can train while CPU collects next batch.
    """
    
    def __init__(self, env: VecEnv, device: str = 'cuda', buffer_size: int = 2):
        self.env = env
        self.device = device
        self.buffer_size = buffer_size
        
        # Prefetch buffer
        self.obs_buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.prefetch_thread = None
        self.running = False
        
        print(f"✅ Async Prefetch: buffer_size={buffer_size}")
    
    def start_prefetch(self):
        """Start background prefetching thread."""
        if not self.running:
            self.running = True
            self.prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
            self.prefetch_thread.start()
    
    def stop_prefetch(self):
        """Stop prefetching."""
        self.running = False
        if self.prefetch_thread:
            self.prefetch_thread.join()
    
    def _prefetch_loop(self):
        """Background thread that prefetches observations."""
        while self.running:
            if len(self.obs_buffer) < self.buffer_size:
                # Get next observation
                obs = self.env.reset()
                
                # Convert to GPU tensors
                obs_gpu = self._to_gpu(obs)
                
                with self.lock:
                    self.obs_buffer.append(obs_gpu)
    
    def _to_gpu(self, obs):
        """Convert observation dict to GPU tensors."""
        obs_gpu = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value).to(self.device, non_blocking=True)
                obs_gpu[key] = tensor
            else:
                obs_gpu[key] = value
        return obs_gpu
    
    def get_prefetched(self):
        """Get prefetched observation."""
        with self.lock:
            if self.obs_buffer:
                return self.obs_buffer.popleft()
        return None
    
    def __getattr__(self, name):
        """Proxy all other attributes to wrapped env."""
        return getattr(self.env, name)