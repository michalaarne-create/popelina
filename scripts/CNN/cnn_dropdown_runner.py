# -*- coding: utf-8 -*-
"""
cnn_dropdown_runner.py
Manager do inference CNN:
- bierze najnowszy JSON z boxami z folderu screen_boxes
- robi screenshot przez mss (cały wirtualny ekran)
- wycina cropy z +10% marginesem, zapisuje do data/temp/OCR_boxes+10%
- batch inference małym CNN (GPU jeśli dostępne)
- zapisuje preds.json z wynikami
Całość ~<1.3 s przy kilkudziesięciu boxach (GPU; na CPU zależnie od liczby boxów).
"""

import os, sys, json, time, glob, shutil, argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# --- mss do szybkich zrzutów
from mss import mss
# używamy tylko do odczytu pozycji (spełnienie wymogu "mss/pynput")
try:
    from pynput.mouse import Controller as MouseController
    _mouse = MouseController()
except Exception:
    _mouse = None

# Torch minimal (bez torchvision)
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ ŚCIEŻKI DOMYŚLNE (zmień jeśli chcesz) --------------------
ROOT_PATH = Path(__file__).resolve().parents[2]
ROOT = str(ROOT_PATH)
JSON_DIR_DEFAULT   = str(ROOT_PATH / "data" / "screen" / "region_grow" / "region_grow")
OUT_CROPS_DEFAULT  = str(ROOT_PATH / "data" / "screen" / "temp" / "OCR_boxes+10%")
MODEL_PATH_DEFAULT = os.path.join(ROOT, "tri_cnn.pt")  # ścieżka do wytrenowanego modelu

IMG_SIZE = 128       # wejście do CNN
PADDING  = 0.10      # +10% po każdej stronie (po stronie dłuższego boku)
THRESH   = 0.50      # próg "tri"
BATCH    = 256

# ------------------ CNN (taka sama jak w trenowaniu: AdaptiveAvgPool->4x4) ---
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap  = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1  = nn.Linear(64 * 4 * 4, 64)
        self.fc2  = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.c1(x)))  # /2
        x = self.pool(F.relu(self.c2(x)))  # /4
        x = self.pool(F.relu(self.c3(x)))  # /8
        x = self.gap(x)                    # ->4x4 niezależnie od wejścia
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ------------------ Utils -----------------------------------------------------
def _now() -> float: return time.perf_counter()

def _read_latest_json(json_dir: str) -> Tuple[str, Any]:
    files = sorted(glob.glob(os.path.join(json_dir, "*.json")), key=os.path.getmtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"Brak plików JSON w {json_dir}")
    path = files[0]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return path, data

def _extract_boxes_from_payload(obj: Any) -> List[List[int]]:
    """
    Elastyczne pobranie boxów:
    - format detektora: {"results":[{"text_box":[x1,y1,x2,y2], ...}, ...]}
    - alternatywnie: {"boxes":[[x1,y1,x2,y2], ...]} albo lista boxów na top-level
    Zwraca listę [x1,y1,x2,y2] w koordynatach ekranu.
    """
    boxes = []

    def is_box(b):
        return isinstance(b, (list, tuple)) and len(b) == 4

    if isinstance(obj, dict):
        if "results" in obj and isinstance(obj["results"], list):
            for r in obj["results"]:
                if isinstance(r, dict):
                    if is_box(r.get("text_box")):
                        boxes.append([int(v) for v in r["text_box"]])
                    elif is_box(r.get("dropdown_box")):
                        boxes.append([int(v) for v in r["dropdown_box"]])
        elif "boxes" in obj and isinstance(obj["boxes"], list):
            for b in obj["boxes"]:
                if is_box(b):
                    boxes.append([int(v) for v in b])
    elif isinstance(obj, list):
        for b in obj:
            if is_box(b):
                boxes.append([int(v) for v in b])

    return boxes

def _clear_dir(path: str):
    p = Path(path)
    if p.exists():
        for child in p.iterdir():
            if child.is_file():
                try: child.unlink()
                except Exception: pass
            elif child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
    else:
        p.mkdir(parents=True, exist_ok=True)

def _tensor_from_pil_or_np(img_arr: np.ndarray) -> torch.Tensor:
    # img_arr: HxWx3 (RGB) uint8
    x = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0
    return x

def _resize_np(img_arr: np.ndarray, size: int) -> np.ndarray:
    # szybkie resize przez OpenCV jeśli jest, w razie czego PIL fallback
    try:
        import cv2
        return cv2.resize(img_arr, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image
        return np.array(Image.fromarray(img_arr).resize((size, size), Image.BILINEAR))

# ------------------ Screenshot + crop -----------------------------------------
def _grab_full_virtual() -> Tuple[np.ndarray, Dict[str, int]]:
    """
    mss.monitors[0] = cały wirtualny ekran (przy multi-monitorach)
    Zwraca: ndarray (H,W,3 RGB) oraz dict z offsetami left/top.
    """
    with mss() as sct:
        mon = sct.monitors[0]  # pełna przestrzeń
        raw = np.array(sct.grab(mon))  # BGRA
        img = raw[..., :3][:, :, ::-1].copy()  # -> RGB
        meta = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}
        return img, meta

def _crop_pad(img: np.ndarray, box: List[int], pad: float, meta: Dict[str,int]) -> np.ndarray:
    """
    img    : screenshot (RGB)
    box    : [x1,y1,x2,y2] w globalnych koordach ekranu
    pad    : ułamek (np. 0.10)
    meta   : offset left/top wirtualnego ekranu
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 -= meta["left"]; x2 -= meta["left"]
    y1 -= meta["top"];  y2 -= meta["top"]

    w = max(1, x2 - x1); h = max(1, y2 - y1)
    d = int(max(w, h) * pad)
    xx1 = max(0, x1 - d)
    yy1 = max(0, y1 - d)
    xx2 = min(img.shape[1], x2 + d)
    yy2 = min(img.shape[0], y2 + d)
    if xx2 <= xx1 or yy2 <= yy1:
        raise ValueError("Pusty crop po pad/clamp")

    return img[yy1:yy2, xx1:xx2, :]

# ------------------ Inference --------------------------------------------------
def run_inference(json_dir: str, out_dir: str, model_path: str,
                  img_size: int = IMG_SIZE, padding: float = PADDING,
                  batch_size: int = BATCH, thresh: float = THRESH) -> Dict[str, Any]:
    t0 = _now()

    # ewentualnie pobierz pozycję myszy (pynput) – „dotknięcie” API (nie wpływa na nic)
    if _mouse:
        _ = _mouse.position  # odczyt bieżącej pozycji (no-op)

    # 1) znajdź najnowszy JSON z boxami
    json_path, payload = _read_latest_json(json_dir)
    boxes = _extract_boxes_from_payload(payload)
    if not boxes:
        raise RuntimeError(f"Brak boxów do sprawdzenia w: {json_path}")

    # 2) screenshot całego wirtualnego ekranu (1 raz)
    scr, meta = _grab_full_virtual()

    # 3) wyczyść tymczasowy folder i zapisz cropy
    _clear_dir(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    crops_np: List[np.ndarray] = []
    kept_boxes: List[List[int]] = []
    for i, b in enumerate(boxes):
        try:
            patch = _crop_pad(scr, b, padding, meta)
            patch = _resize_np(patch, img_size)
            crops_np.append(patch)
            kept_boxes.append([int(v) for v in b])
        except Exception:
            # pomiń nieprawidłowe
            continue

    # brak poprawnych cropów
    if not crops_np:
        raise RuntimeError("Wszystkie boxy wypadły poza ekran / były puste po pad.")

    # zapis cropów (jedna runda w folderze)
    for i, arr in enumerate(crops_np):
        p = os.path.join(out_dir, f"crop_{i:04d}.jpg")
        try:
            from PIL import Image
            Image.fromarray(arr).save(p, quality=92, optimize=True)
        except Exception:
            pass

    t_prep = _now()

    # 4) CNN - ładowanie modelu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCNN().to(device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # 5) batching
    xs = torch.stack([_tensor_from_pil_or_np(arr) for arr in crops_np], dim=0).to(device)
    with torch.no_grad():
        out = []
        for i in range(0, xs.size(0), batch_size):
            logits = model(xs[i:i+batch_size])
            probs = torch.softmax(logits, dim=1)[:, 1]  # klasa 1 = tri
            out.append(probs)
        probs = torch.cat(out, dim=0).float().cpu().numpy()

    preds = (probs >= thresh).astype(np.int32).tolist()

    results = []
    for b, p, pr in zip(kept_boxes, preds, probs.tolist()):
        results.append({"box": b, "is_tri": int(p), "prob_tri": float(pr)})

    # 6) zapis meta/preds
    meta_out = {
        "json_source": json_path,
        "n_boxes_in": len(boxes),
        "n_crops": len(kept_boxes),
        "device": device,
        "img_size": img_size,
        "padding": padding,
        "threshold": thresh,
        "time_ms": round((_now() - t0) * 1000, 1),
        "prep_ms": round((t_prep - t0) * 1000, 1)
    }
    with open(os.path.join(out_dir, "preds.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta_out, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"[CNN] {len(kept_boxes)} cropów | device={device} | total={meta_out['time_ms']} ms")
    return {"meta": meta_out, "results": results}

# ------------------ CLI --------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CNN manager: screenshot->crop(+10%) -> classify triangles")
    ap.add_argument("--json-dir", type=str, default=JSON_DIR_DEFAULT, help="Folder z JSONami (screen_boxes)")
    ap.add_argument("--out-dir",  type=str, default=OUT_CROPS_DEFAULT,  help="Folder na cropy i preds.json (czyści przed runem)")
    ap.add_argument("--model",    type=str, default=MODEL_PATH_DEFAULT, help="Ścieżka do tri_cnn.pt")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--padding",  type=float, default=PADDING)
    ap.add_argument("--batch",    type=int, default=BATCH)
    ap.add_argument("--thresh",   type=float, default=THRESH)
    args = ap.parse_args()

    run_inference(args.json_dir, args.out_dir, args.model, args.img_size, args.padding, args.batch, args.thresh)

if __name__ == "__main__":
    main()
