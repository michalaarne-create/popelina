import os, time, threading
from pathlib import Path
from PIL import Image
import utils.region_grow as rg

os.environ.setdefault('REGION_GROW_USE_RAPID_OCR', '1')
os.environ.setdefault('PADDLEOCR_GPU_ID', os.environ.get('PADDLEOCR_GPU_ID', '0'))

img_dir = Path('data') / 'screen' / 'raw screen'
candidates = []
if img_dir.is_dir():
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}:
            candidates.append(p)
img_path = Path(os.environ.get('RG_TEST_IMAGE', ''))
if not img_path.is_file():
    img_path = candidates[0] if candidates else Path(rg.DEFAULT_IMAGE_PATH)
if not img_path.is_file():
    raise FileNotFoundError(f'No test image found (looked for {img_path})')

import pynvml  # type: ignore

samples = []
stop_evt = threading.Event()

GETTERS = [
    'nvmlDeviceGetComputeRunningProcesses_v3',
    'nvmlDeviceGetComputeRunningProcesses',
    'nvmlDeviceGetGraphicsRunningProcesses_v3',
    'nvmlDeviceGetGraphicsRunningProcesses',
]

def gpu_monitor():
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
    except Exception as exc:
        print('[TEST] NVML init failed:', exc)
        return
    try:
        getters = []
        for name in GETTERS:
            fn = getattr(pynvml, name, None)
            if fn is not None:
                getters.append(fn)
        if not getters:
            print('[TEST] NVML has no process getters')
            return
        while not stop_evt.is_set():
            for handle in handles:
                for fn in getters:
                    try:
                        procs = fn(handle)  # type: ignore[misc]
                    except Exception:
                        continue
                    for proc in procs or []:
                        pid = getattr(proc, 'pid', None)
                        mem = getattr(proc, 'usedGpuMemory', 0)
                        if pid == os.getpid() and mem:
                            samples.append(float(mem)/(1024*1024))
            stop_evt.wait(0.05)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

monitor_thread = threading.Thread(target=gpu_monitor, daemon=True)
monitor_thread.start()

rg.DEBUG_OCR = True
print(f'[TEST] Using image: {img_path}')
img = Image.open(img_path).convert('RGB')
start = time.perf_counter()
results = rg.read_ocr_wrapper(img, timer=None)
duration = time.perf_counter() - start
print(f'[TEST] OCR results: {len(results)} in {duration:.2f}s')

stop_evt.set()
monitor_thread.join(timeout=2.0)

if samples:
    print(f'[TEST] GPU memory usage samples (MB): min={min(samples):.1f}, max={max(samples):.1f}, count={len(samples)}')
else:
    print('[TEST] No GPU usage samples recorded')
