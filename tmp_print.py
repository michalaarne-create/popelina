from pathlib import Path
path = Path('popelina_github/scripts/region_grow/numpy_rate/rating.py')
lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
for i, line in enumerate(lines[:80], 1):
    print(f'{i:04d}: {line}')
