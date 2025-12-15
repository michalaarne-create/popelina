#!/usr/bin/env python3
"""
Prosty skrypt OCR z region growing
- Robi screenshot
- Wykonuje OCR za pomocą EasyOCR
- Dla każdego tekstu wywołuje region_grow
- Rysuje kolorowe regiony na obrazie debugowym
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import easyocr

try:
    import mss
except ImportError:
    print("Zainstaluj mss: pip install mss")
    raise

from region_grow import region_grow


# Kolory dla różnych regionów tekstu
TEXT_REGION_COLORS = [
    (255, 99, 71),    # Tomato
    (135, 206, 235),  # SkyBlue
    (60, 179, 113),   # MediumSeaGreen
    (238, 130, 238),  # Violet
    (255, 215, 0),    # Gold
    (244, 164, 96),   # SandyBrown
    (173, 255, 47),   # GreenYellow
    (70, 130, 180),   # SteelBlue
    (255, 182, 193),  # LightPink
    (32, 178, 170),   # LightSeaGreen
]


def get_color_for_index(idx: int) -> Tuple[int, int, int]:
    """Zwraca kolor dla danego indeksu."""
    return TEXT_REGION_COLORS[idx % len(TEXT_REGION_COLORS)]


def perform_ocr(image: np.ndarray, reader, min_conf: float = 0.3) -> List[Dict[str, Any]]:
    """
    Wykonuje OCR na obrazie i zwraca listę wykrytych tekstów.
    """
    results = reader.readtext(image, detail=True)
    items = []
    
    for entry in results:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
            
        bbox_points, text, conf = entry
        
        if conf < min_conf:
            continue
        
        # Konwertuj bbox (4 punkty) na prostokąt
        xs = [int(pt[0]) for pt in bbox_points]
        ys = [int(pt[1]) for pt in bbox_points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        
        # Środek bbox jako punkt startowy
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        items.append({
            'text': str(text).strip(),
            'conf': float(conf),
            'ocr_bbox': (x1, y1, x2, y2),
            'center': (cx, cy)
        })
    
    return items


def process_with_region_grow(image: np.ndarray, ocr_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dla każdego tekstu z OCR wykonuje region growing.
    """
    for item in ocr_items:
        cx, cy = item['center']
        
        # Wywołaj region_grow z tolerancją 0 (zatrzymuje się na każdej zmianie)
        region = region_grow(image, (cx, cy), tolerance=0, max_expansion=100)
        
        if region:
            item['region_bbox'] = region['bbox']
            item['region_mask'] = region['mask']
        else:
            # Jeśli region_grow nie zadziałał, użyj oryginalnego bbox
            item['region_bbox'] = item['ocr_bbox']
            item['region_mask'] = None
    
    return ocr_items


def draw_debug_image(image: np.ndarray, items: List[Dict[str, Any]], 
                     alpha: float = 0.3) -> np.ndarray:
    """
    Rysuje kolorowe regiony na obrazie debugowym.
    """
    debug = image.copy()
    overlay = image.copy()
    
    for idx, item in enumerate(items):
        color = get_color_for_index(idx)
        
        # Rysuj region (jeśli istnieje)
        if 'region_bbox' in item:
            x1, y1, x2, y2 = item['region_bbox']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # Rysuj oryginalny bbox OCR (cienka linia)
        if 'ocr_bbox' in item:
            x1, y1, x2, y2 = item['ocr_bbox']
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        
        # Dodaj etykietę z tekstem
        text = item.get('text', '')
        if text and 'ocr_bbox' in item:
            x1, y1, _, _ = item['ocr_bbox']
            label = f"{text[:20]}... ({item['conf']:.2f})" if len(text) > 20 else f"{text} ({item['conf']:.2f})"
            cv2.putText(debug, label, (x1, max(y1 - 5, 0)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    
    # Nałóż półprzezroczyste regiony
    cv2.addWeighted(overlay, alpha, debug, 1 - alpha, 0, debug)
    
    return debug


def save_debug_image(image: np.ndarray, tag: str = ""):
    """Zapisuje obraz debugowy."""
    out_dir = Path("debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"ocr_regions_{timestamp}{f'_{tag}' if tag else ''}.png"
    filepath = out_dir / filename
    
    cv2.imwrite(str(filepath), image)
    print(f"[Debug] Zapisano: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="OCR z region growing")
    parser.add_argument("--lang", default="en", help="Język OCR (np. 'en', 'pl')")
    parser.add_argument("--min-conf", type=float, default=0.3, help="Minimalna pewność OCR")
    parser.add_argument("--show", action="store_true", help="Pokaż okno z wynikami")
    parser.add_argument("--loop", action="store_true", help="Tryb ciągły")
    args = parser.parse_args()
    
    # Inicjalizacja
    print("[Info] Inicjalizacja EasyOCR...")
    reader = easyocr.Reader([args.lang], gpu=False)
    
    print("[Info] Inicjalizacja screenshot...")
    sct = mss.mss()
    monitor = sct.monitors[0]  # Główny ekran
    
    def process_frame():
        # Zrób screenshot
        screenshot = sct.grab(monitor)
        image = cv2.cvtColor(np.array(screenshot, dtype=np.uint8), cv2.COLOR_BGRA2BGR)
        
        # OCR
        print("[OCR] Skanowanie...")
        ocr_items = perform_ocr(image, reader, args.min_conf)
        print(f"[OCR] Znaleziono {len(ocr_items)} tekstów")
        
        # Region growing dla każdego tekstu
        print("[Region] Rozszerzanie regionów...")
        items_with_regions = process_with_region_grow(image, ocr_items)
        
        # Wypisz znalezione teksty
        for i, item in enumerate(items_with_regions):
            print(f"  {i:2d}. {item['text']:<30} (conf={item['conf']:.2f})")
            if 'region_bbox' in item:
                x1, y1, x2, y2 = item['region_bbox']
                w, h = x2 - x1, y2 - y1
                print(f"      Region: {w}x{h} px at ({x1},{y1})")
        
        # Rysuj debug
        debug_image = draw_debug_image(image, items_with_regions)
        
        # Zapisz
        save_debug_image(debug_image)
        
        # Pokaż jeśli włączone
        if args.show:
            cv2.imshow("OCR Regions", debug_image)
            cv2.waitKey(1 if args.loop else 0)
        
        return items_with_regions
    
    try:
        if args.loop:
            print("[Info] Tryb ciągły. Naciśnij Ctrl+C aby zakończyć.")
            while True:
                process_frame()
                time.sleep(1)
        else:
            process_frame()
    except KeyboardInterrupt:
        print("\n[Info] Przerwano przez użytkownika")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()