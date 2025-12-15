# gen_dropdown_dataset.py
# Generator dropdown datasetu (tri/not_tri/full) z presetami,
# bezpieczną perspektywą i sanity-checkiem.
# Wymagania: Pillow, numpy

import os, sys, time, json, random, argparse, zipfile
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ===================== PRESETY (można nadpisywać flagami) =====================
PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny":     {"target_mb": 50,   "tolerance": 0.05, "num_pos": None, "num_neg": None, "min_size": 128, "max_size": 192, "with_full": True,  "zip": False, "seed": 1234},
    "medium":   {"target_mb": 300,  "tolerance": 0.03, "num_pos": None, "num_neg": None, "min_size": 128, "max_size": 256, "with_full": True,  "zip": True,  "seed": 1234},
    # domyślny preset (~1 GB)
    "one_gb":   {"target_mb": 1024, "tolerance": 0.02, "num_pos": None, "num_neg": None, "min_size": 128, "max_size": 192, "with_full": True,  "zip": True,  "seed": 1234},
    # hard-negative mining: tylko negatywy
    "hn_neg":   {"target_mb": 1024, "tolerance": 0.02, "num_pos": 0,    "num_neg": None, "min_size": 128, "max_size": 192, "with_full": True,  "zip": True,  "seed": 4321},
}
DEFAULT_PROFILE = "one_gb"

# ============================ Utils ==========================================
def pick_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = ["DejaVuSans.ttf","Arial.ttf","SegoeUI.ttf","Segoe UI.ttf","LiberationSans-Regular.ttf","NotoSans-Regular.ttf"]
    for name in candidates:
        try: return ImageFont.truetype(name, size)
        except Exception: continue
    return ImageFont.load_default()

def clamp(v, a, b): return max(a, min(b, v))
def rint(a, b): return random.randint(a, b)

def color_jitter(rgb: Tuple[int,int,int], jitter: int=30):
    r,g,b = rgb
    j = lambda c: clamp(c + rint(-jitter, jitter), 0, 255)
    return (j(r), j(g), j(b))

def gradient_bg(size: Tuple[int,int], dark=False):
    w,h = size
    base = Image.new("RGB", (w,h), (0,0,0) if dark else (255,255,255))
    px = base.load()
    c1 = (15,15,18) if dark else (245,246,248)
    c2 = (30,30,35) if dark else (225,228,233)
    for y in range(h):
        t = y/(h-1)
        r = int(c1[0]*(1-t)+c2[0]*t); g = int(c1[1]*(1-t)+c2[1]*t); b = int(c1[2]*(1-t)+c2[2]*t)
        for x in range(w):
            rr=r; gg=g; bb=b
            if (x//32)%2==0:
                rr = clamp(rr+1,0,255); gg = clamp(gg+1,0,255); bb = clamp(bb+1,0,255)
            px[x,y]=(rr,gg,bb)
    return base

def add_noise(img: Image.Image, sigma=(3,10), blend=(0.05,0.25), lines=False):
    """Łagodniejszy szum + rzadkie cienkie linie (bez zalewania obrazu)."""
    w,h = img.size
    arr = np.array(img).astype(np.float32)
    std = random.uniform(*sigma)
    noise = np.random.normal(0, std, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)
    if random.random()<0.7:
        tex = Image.effect_noise((w,h), random.uniform(10,30)).convert("RGB")
        out = Image.blend(out, tex, random.uniform(*blend))
    if lines and random.random()<0.25:
        d = ImageDraw.Draw(out)
        for _ in range(rint(4,12)):
            x1,y1,x2,y2 = rint(0,w-1),rint(0,h-1),rint(0,w-1),rint(0,h-1)
            d.line((x1,y1,x2,y2), fill=color_jitter((220,220,220), 35), width=rint(1,2))
    return out

def jpeg_roundtrip(img: Image.Image, q=None):
    if q is None: q=rint(72, 96)
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=q, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def _border_median_rgb(img: Image.Image):
    a = np.array(img)
    top = a[0, :, :]; bot = a[-1, :, :]; left = a[:, 0, :]; right = a[:, -1, :]
    b = np.concatenate([top, bot, left, right], axis=0)
    return tuple(np.median(b, axis=0).astype(np.uint8).tolist())

def perspective_safe(img: Image.Image, max_warp=10):
    """Perspektywa na powiększonym płótnie + tło z mediany krawędzi → brak 'belek'."""
    w, h = img.size
    pad = max(8, max(w, h)//10)
    bg = _border_median_rgb(img)
    canvas = Image.new("RGB", (w+2*pad, h+2*pad), bg)
    canvas.paste(img, (pad, pad))
    W,H = canvas.size
    dx = int((W*max_warp/100.0) * random.uniform(-0.5, 0.5))
    dy = int((H*max_warp/100.0) * random.uniform(-0.5, 0.5))
    dst = [0+dx,0+dy,  W-dx,0+dy,  W+dx,H-dy,  0-dx,H-dy]
    warped = canvas.transform((W, H), Image.QUAD, dst, resample=Image.BICUBIC)
    x0 = (W - w)//2; y0 = (H - h)//2
    return warped.crop((x0, y0, x0+w, y0+h))

def is_letterboxed(img: Image.Image, band=8, std_thr=4):
    """Wykrywa czarne pasy: niski rozrzut w górnym/dolnym pasie."""
    a = np.array(img).astype(np.int16)
    if a.shape[0] < 2*band: return False
    top = a[:band]; bot = a[-band:]
    return top.std() < std_thr or bot.std() < std_thr

# ---------------------- Glyphs / shapes ----------------------
POS_GLYPHS = ["▼","▾","▿","⯆","⮟","∨","⌄","🔽","V"]   # caret = V
NEG_GLYPHS = ["▲","▵","⯅","⮝","∧","⌃","🔼","^","none",">","<","→","←"]

def draw_triangle(draw: ImageDraw.ImageDraw, cx, cy, size, fill, down=True, mode="filled"):
    h = size
    w = int(size * random.uniform(0.85, 1.25))
    if down:  pts = [(cx - w//2, cy - h//3), (cx + w//2, cy - h//3), (cx, cy + 2*h//3)]
    else:     pts = [(cx - w//2, cy + h//3), (cx + w//2, cy + h//3), (cx, cy - 2*h//3)]
    if mode == "filled":
        draw.polygon(pts, fill=fill)
    else:
        lw = max(1,size//8)
        draw.line((pts[0], pts[1]), fill=fill, width=lw)
        draw.line((pts[1], pts[2]), fill=fill, width=lw)
        draw.line((pts[2], pts[0]), fill=fill, width=lw)

def draw_caret(draw, cx, cy, size, fill, down=True):
    half = size//2
    w = max(2, size//6)
    if down:
        draw.line((cx - half, cy - half//2, cx, cy + half//2), fill=fill, width=w)
        draw.line((cx, cy + half//2, cx + half, cy - half//2), fill=fill, width=w)
    else:
        draw.line((cx - half, cy + half//2, cx, cy - half//2), fill=fill, width=w)
        draw.line((cx, cy - half//2, cx + half, cy + half//2), fill=fill, width=w)

def put_glyph(draw, cx, cy, gsize, glyph, positive=True, dark=False):
    col = color_jitter((230,230,235) if dark else (30,30,30), 40)
    mode = random.choice(["tri_filled","tri_outline","caret","font"])
    if glyph in ["V","^"]: mode = "caret"
    if glyph == "none":    return
    down = (glyph not in ["^","▲","▵","⯅","⮝","∧","⌃"])
    if mode == "tri_filled":  draw_triangle(draw, cx, cy, gsize, col, down=down, mode="filled")
    elif mode == "tri_outline": draw_triangle(draw, cx, cy, gsize, col, down=down, mode="outline")
    elif mode == "caret":      draw_caret(draw, cx, cy, gsize, col, down=down)
    else:
        font = pick_font(rint(max(14, gsize-6), gsize+4))
        draw.text((cx - gsize//3, cy - gsize//3), glyph, font=font, fill=col)

# ---------------------- UI tile / crop synth ----------------------
LANG = ["Select...", "Wybierz", "Poland", "English", "Ryzen 5 5600", "All", "Cena",
        "Sort by", "Filter", "Город", "日本語"]

def synth_tile(w=360, h=140, positive=True, dark=None) -> Tuple[Image.Image, Tuple[int,int,int,int], Tuple[int,int]]:
    """Rysuje UI bez deformacji i zwraca też centrum glifu."""
    if dark is None:
        dark = random.random()<0.45
    img = gradient_bg((w,h), dark=dark)
    d = ImageDraw.Draw(img)

    m = rint(8, 16)
    card = (m, m, w-m, h-m)
    card_fill = color_jitter((30,32,36) if dark else (252,252,252), 18)
    card_out  = color_jitter((70,70,75) if dark else (198,200,205), 30)
    d.rounded_rectangle(card, radius=rint(8,16), fill=card_fill, outline=card_out, width=2)

    pad = rint(8,14)
    box = (card[0]+pad, card[1]+pad, card[2]-pad, card[3]-pad)
    box_fill = color_jitter((45,46,50) if dark else (255,255,255), 10)
    box_out  = color_jitter((120,120,130) if dark else (170,170,175), 25)
    d.rounded_rectangle(box, radius=rint(6,12), fill=box_fill, outline=box_out, width=2)

    if random.random()<0.9:
        font = pick_font(rint(12, 20))
        txt = random.choice(LANG)
        col = color_jitter((230,230,235) if dark else (40,40,40), 30)
        d.text((box[0]+rint(8,16), box[1]+rint(6,16)), txt, fill=col, font=font)

    cx = box[2] - rint(16, 28)
    cy = (box[1] + box[3]) // 2 + rint(-3,3)
    gsize = rint(12, 22)
    glyph = random.choice(POS_GLYPHS if positive else NEG_GLYPHS)
    put_glyph(d, cx, cy, gsize, glyph, positive=positive, dark=dark)

    return img, box, (cx, cy)

def synth_crop(min_size=128, max_size=192, positive=True) -> Image.Image:
    """Kroi wokół glifu, a dopiero potem nakłada efekty."""
    w = rint(340, 520); h = rint(120, 200)  # większy tile -> zapas
    tile, box, (cx, cy) = synth_tile(w, h, positive=positive, dark=None)

    sz0 = rint(min_size, max_size)
    sz = min(sz0, tile.width, tile.height)
    half = sz // 2

    left = clamp(cx - half, 0, tile.width  - sz)
    top  = clamp(cy - half, 0, tile.height - sz)
    crop = tile.crop((left, top, left + sz, top + sz))

    # POST-FX tylko na croppie (obiekt nie "ucieknie")
    if random.random() < 0.30: crop = add_noise(crop, lines=True)
    if random.random() < 0.20: crop = crop.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.9)))
    if random.random() < 0.30: crop = crop.filter(ImageFilter.UnsharpMask(radius=1.0, percent=100, threshold=2))
    if random.random() < 0.15: crop = perspective_safe(crop, max_warp=10)
    if random.random() < 0.30: crop = jpeg_roundtrip(crop, q=rint(78, 95))

    # sanity-check na "belki"; delikatny fallback
    if is_letterboxed(crop, band=8, std_thr=4):
        crop = crop.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=2))
    return crop

# --------------------------- Save helpers ---------------------------
def save_image(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if random.random()<0.85:
        q = rint(80, 98)
        img.save(path, format="JPEG", quality=q, optimize=True)
    else:
        img.save(path.with_suffix(".png"))

def dir_size_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())

# --------------------------- Main gen loop ---------------------------
def generate_dataset(out_dir: Path,
                     target_mb: float = None,
                     tolerance: float = 0.02,
                     num_pos: int = None,
                     num_neg: int = None,
                     min_size: int = 128,
                     max_size: int = 192,
                     with_full: bool = False,
                     seed: int = 1234):
    random.seed(seed); np.random.seed(seed)
    tri_dir  = out_dir / "tri"
    neg_dir  = out_dir / "not_tri"
    full_dir = out_dir / "full"
    if out_dir.exists():
        print(f"[WARN] Output exists: {out_dir} – pliki mogą się nadpisać/mieszać.")
    out_dir.mkdir(parents=True, exist_ok=True)
    tri_dir.mkdir(exist_ok=True)
    neg_dir.mkdir(exist_ok=True)
    if with_full: full_dir.mkdir(exist_ok=True)

    by_target = target_mb is not None and target_mb > 0
    target_bytes = int(target_mb * 1024 * 1024) if by_target else None
    tol_lo = int(target_bytes * (1 - tolerance)) if by_target else None
    tol_hi = int(target_bytes * (1 + tolerance)) if by_target else None

    pos_count = 0; neg_count = 0; full_count = 0
    t0 = time.time()

    def maybe_save_full():
        nonlocal full_count
        if not with_full: return
        if random.random()<0.40:
            tile, _, _ = synth_tile(positive=random.random()<0.6)
            save_image(tile, full_dir / f"tile_{full_count:07d}.jpg")
            full_count += 1

    i = 0
    while True:
        if num_pos is None or pos_count < num_pos:
            imgp = synth_crop(min_size, max_size, positive=True)
            save_image(imgp, tri_dir / f"tri_{pos_count:07d}.jpg")
            pos_count += 1
            maybe_save_full()
        if (num_neg is None) or (neg_count < num_neg):
            imgn = synth_crop(min_size, max_size, positive=False)
            save_image(imgn, neg_dir / f"neg_{neg_count:07d}.jpg")
            neg_count += 1
            maybe_save_full()

        i += 1
        if i % 200 == 0:
            sz = dir_size_bytes(out_dir)
            if by_target:
                print(f"[PROG] size={sz/1024/1024:.1f} MB | pos={pos_count} neg={neg_count} full={full_count}")
                if tol_lo <= sz <= tol_hi or sz > tol_hi:
                    break
            else:
                print(f"[PROG] pos={pos_count} neg={neg_count} full={full_count} size≈{sz/1024/1024:.1f} MB")
                if (num_pos is not None and pos_count >= num_pos) and (num_neg is not None and neg_count >= num_neg):
                    break

    meta = {"pos": pos_count, "neg": neg_count, "full": full_count,
            "min_size": min_size, "max_size": max_size,
            "target_mb": target_mb, "tolerance": tolerance,
            "seed": seed, "time_sec": round(time.time()-t0, 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    final_size = dir_size_bytes(out_dir)
    print(f"[DONE] pos={pos_count} neg={neg_count} full={full_count} | size={final_size/1024/1024:.1f} MB | saved -> {out_dir}")

def zip_folder_no_compress(src: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(src))
    print(f"[ZIP] {zip_path} (≈sum of files)")

# ============================ CLI ============================================
def apply_preset(args) -> Dict[str, Any]:
    preset_name = args.profile or DEFAULT_PROFILE
    if preset_name not in PRESETS:
        raise SystemExit(f"Nieznany profil: {preset_name}. Dostępne: {list(PRESETS.keys())}")
    cfg = dict(PRESETS[preset_name])
    if args.target_mb is not None: cfg["target_mb"] = args.target_mb
    if args.tolerance is not None: cfg["tolerance"] = args.tolerance
    if args.num_pos is not None:   cfg["num_pos"] = args.num_pos
    if args.num_neg is not None:   cfg["num_neg"]  = args.num_neg
    if args.min_size is not None:  cfg["min_size"] = args.min_size
    if args.max_size is not None:  cfg["max_size"] = args.max_size
    if args.with_full is not None: cfg["with_full"] = bool(args.with_full)
    if args.seed is not None:      cfg["seed"] = args.seed
    if args.zip is not None:       cfg["zip"] = bool(args.zip)
    return cfg

def main():
    ap = argparse.ArgumentParser(description="Generate diverse dropdown dataset (tri/not_tri/full) with safe augments.")
    ap.add_argument("--out", type=str, default="CNN_dataset", help="Folder docelowy (domyślnie ./CNN_dataset).")
    ap.add_argument("--profile", type=str, default=None, help=f"Nazwa profilu (domyślnie {DEFAULT_PROFILE}). Dostępne: {list(PRESETS.keys())}")
    # opcjonalne nadpisania:
    ap.add_argument("--target-mb", type=float, default=None)
    ap.add_argument("--tolerance", type=float, default=None)
    ap.add_argument("--num-pos", type=int, default=None)
    ap.add_argument("--num-neg", type=int, default=None)
    ap.add_argument("--min-size", type=int, default=None)
    ap.add_argument("--max-size", type=int, default=None)
    ap.add_argument("--with-full", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--zip", type=int, default=None)
    args = ap.parse_args()

    cfg = apply_preset(args)
    out = Path(args.out)
    print("[INFO] Profile:", args.profile or DEFAULT_PROFILE)
    print("[INFO] Out:", out)
    print("[INFO] Cfg:", json.dumps(cfg, indent=2, ensure_ascii=False))

    generate_dataset(out_dir=out,
                     target_mb=cfg["target_mb"],
                     tolerance=cfg["tolerance"],
                     num_pos=cfg["num_pos"],
                     num_neg=cfg["num_neg"],
                     min_size=cfg["min_size"],
                     max_size=cfg["max_size"],
                     with_full=cfg["with_full"],
                     seed=cfg["seed"])
    if cfg.get("zip", False):
        zip_folder_no_compress(out, out.with_suffix(".zip"))

if __name__ == "__main__":
    main()
