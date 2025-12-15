from main import build_hover_from_region_results
from pathlib import Path
import traceback
path = Path('data/screen/region_grow/region_grow/screen_20251121_162423_238811_rg_small.json')
try:
    print(build_hover_from_region_results(path))
except Exception as exc:
    traceback.print_exc()
    print('exc', exc)

