# utils/CNN/CNN_triangles_train.py
# 2-class: (0=not_tri, 1=tri), input 128x128
# GPU augmentacje (Kornia), ImageNet normalizacja, class weights,
# opcjonalny cache do RAM, freeze→unfreeze, confusion matrix, best-model saving.
# Obsługa .zip lub folderu; używa tylko tri/ i not_tri/ (ignoruje full/).

import argparse
import os
import random
import time
from pathlib import Path
import zipfile
from glob import glob

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # pozwala wczytać częściowo uszkodzone pliki

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset, Subset

# torchvision do modelu
from torchvision import models

# ===================== Default Config =====================
ROOT = r"E:\BOT ANK\bot\moje_AI\yolov8\FULL BOT"
DEFAULT_DATA = fr"{ROOT}\data\CNN_dataset"
DEFAULT_EPOCHS = 20
DEFAULT_BATCH = 256
DEFAULT_LR = 1e-3
DEFAULT_VAL_SPLIT = 0.10
DEFAULT_OUT = fr"{ROOT}\tri_cnn.pt"
DEFAULT_SEED = 1234
DEFAULT_CACHE = False          # szybki start (dekodowanie w trakcie)
DEFAULT_WORKERS = 4            # na Windows/SSD zwykle 4–8
DEFAULT_PREFETCH = 4
DEFAULT_FROZEN_EPOCHS = 2      # ile epok trzymamy zamrożony backbone na starcie

# ----------------------- Repro -----------------------
def set_seed(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------- I/O danych -----------------------
def ensure_data_dir(data_arg: str) -> Path:
    p = Path(data_arg)
    if p.suffix.lower() == ".zip":
        out_dir = p.with_suffix("")
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(out_dir)
        return out_dir
    if not p.exists():
        raise FileNotFoundError(f"Brak ścieżki: {p}")
    return p

class FlatFolderDatasetPaths(Dataset):
    """
    Zwraca (uint8 tensor CxHxW po resize 128x128, label).
    cache=True: preload do RAM w __init__ (wolniejszy start, minimalne CPU w trakcie).
    cache=False: dekodowanie on-the-fly (szybki start; polecane z --workers>0).
    """
    def __init__(self, root, label, cache=True, size=(128,128), exts=(".png",".jpg",".jpeg",".bmp",".webp")):
        self.root = Path(root)
        self.label = int(label)
        self.cache = cache
        self.size = size

        paths = []
        for e in exts:
            paths += glob(str(self.root / f"*{e}"))
        self.paths = sorted(paths)
        if not self.paths:
            raise FileNotFoundError(f"Brak obrazów w {self.root}")

        self.cached = None
        if self.cache:
            self.cached = []
            W, H = self.size
            for p in self.paths:
                try:
                    img = Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR)
                except Exception:
                    # jeśli plik padnie, podmień innym aby nie stopować preloada
                    continue
                arr = np.array(img, dtype=np.uint8)              # H,W,3
                ten = torch.from_numpy(arr).permute(2,0,1).contiguous()  # C,H,W
                self.cached.append(ten)
            if not self.cached:
                raise RuntimeError(f"Nie udało się wczytać żadnego obrazu z {self.root}")

    def __len__(self):
        return len(self.paths) if not self.cache else len(self.cached)

    def __getitem__(self, i):
        if self.cache:
            return self.cached[i], self.label
        else:
            W, H = self.size
            p = self.paths[i]
            # 3 próby na wypadek flaky pliku
            for _ in range(3):
                try:
                    img = Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR)
                    arr = np.array(img, dtype=np.uint8)
                    ten = torch.from_numpy(arr).permute(2,0,1).contiguous()
                    return ten, self.label
                except Exception:
                    pass
            # ostatnia deska ratunku — bierz kolejny plik
            j = (i + 1) % len(self.paths)
            img = Image.open(self.paths[j]).convert("RGB").resize((W, H), Image.BILINEAR)
            arr = np.array(img, dtype=np.uint8)
            ten = torch.from_numpy(arr).permute(2,0,1).contiguous()
            return ten, self.label

def build_two_class_dataset(root: Path, val_split: float, seed: int, cache: bool):
    tri_dir = root / "tri"
    neg_dir = root / "not_tri"

    if not tri_dir.is_dir() or not neg_dir.is_dir():
        subs = [d for d in root.iterdir() if d.is_dir()]
        found = False
        for s in subs:
            if (s / "tri").is_dir() and (s / "not_tri").is_dir():
                tri_dir, neg_dir = s / "tri", s / "not_tri"
                found = True
                break
        if not found:
            raise RuntimeError(
                f"Nie znaleziono folderów 'tri' i 'not_tri' pod {root}.\n"
                f"Podkatalogi: {[d.name for d in subs]}"
            )

    pos_all = FlatFolderDatasetPaths(tri_dir, label=1, cache=cache)
    neg_all = FlatFolderDatasetPaths(neg_dir, label=0, cache=cache)

    ds_all = ConcatDataset([neg_all, pos_all])
    full_len = len(ds_all)
    val_len = max(1, int(full_len * val_split))
    train_len = max(1, full_len - val_len)

    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(full_len), [train_len, val_len], generator=g)

    # używamy Subset (picklowalne na Windowsie)
    train_ds = Subset(ds_all, indices=train_idx.indices)
    val_ds   = Subset(ds_all, indices=val_idx.indices)

    return train_ds, val_ds, tri_dir, neg_dir

# ----------------------- Model -----------------------
class ResNet18Tiny(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # kompatybilnie z różnymi wersjami torchvision
        if pretrained:
            try:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                self.backbone = models.resnet18(pretrained=True)
        else:
            try:
                self.backbone = models.resnet18(weights=None)
            except Exception:
                self.backbone = models.resnet18(pretrained=False)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, 2)

    def forward(self, x):
        return self.backbone(x)

# ----------------------- Train/Eval utils -----------------------
def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def to_float01_gpu(xb_uint8, device):
    return xb_uint8.to(device, non_blocking=True).float().div_(255.0)

# ----------------------- Augmentacje (GPU) -----------------------
def build_aug_pipeline(device):
    import kornia.augmentation as K
    # kompatybilny RandomRotation dla różnych wersji Kornia
    try:
        rot = K.RandomRotation(degrees=5, p=1.0, keepdim=True, interpolation='bilinear')
    except TypeError:
        rot = K.RandomRotation(degrees=5, p=1.0, keepdim=True, resample='bilinear')

    aug = torch.nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(0.25, 0.25, 0.0, 0.0, p=1.0),
        rot
    ).to(device)
    return aug

def build_imnet_norm(device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    def norm(x):  # x in [0,1]
        return (x - mean) / std
    return norm

# ----------------------- Pętle treningowe -----------------------
def train_one_epoch(model, loader, opt, crit, device, aug_gpu, norm_fn, scaler):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for xb_u8, yb in loader:
        yb = yb.to(device, non_blocking=True)
        xb = to_float01_gpu(xb_u8, device)
        xb = aug_gpu(xb)
        xb = norm_fn(xb)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            logits = model(xb)
            loss = crit(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, yb) * bs
        n += bs
    return total_loss / max(1, n), total_acc / max(1, n)

@torch.no_grad()
def eval_epoch(model, loader, crit, device, norm_fn):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for xb_u8, yb in loader:
        yb = yb.to(device, non_blocking=True)
        xb = to_float01_gpu(xb_u8, device)
        xb = norm_fn(xb)
        logits = model(xb)
        loss = crit(logits, yb)
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, yb) * bs
        n += bs
    return total_loss / max(1, n), total_acc / max(1, n)

@torch.no_grad()
def confusion_matrix(model, loader, device, norm_fn):
    import numpy as np
    cm = np.zeros((2,2), dtype=int)
    model.eval()
    for xb_u8, yb in loader:
        yb = yb.to(device, non_blocking=True)
        xb = to_float01_gpu(xb_u8, device)
        xb = norm_fn(xb)
        pred = model(xb).argmax(1)
        for t, p in zip(yb.cpu().numpy(), pred.cpu().numpy()):
            cm[t, p] += 1
    return cm

# ----------------------- Helpers -----------------------
def make_loader(ds, batch, shuffle, device, nw, prefetch):
    pin = (device == "cuda")
    kwargs = dict(batch_size=batch, shuffle=shuffle, num_workers=nw, pin_memory=pin, timeout=300)
    if nw > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=max(2, prefetch))
    return DataLoader(ds, **kwargs)

def count_files(folder: Path):
    exts = (".png",".jpg",".jpeg",".bmp",".webp")
    return sum(len(list(folder.glob(f"*{e}"))) for e in exts)

# ----------------------- Main -----------------------
def main():
    parser = argparse.ArgumentParser(description="ResNet18: tri vs not_tri (128x128) — GPU aug, optional RAM cache")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cache", action="store_true", default=DEFAULT_CACHE)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--prefetch", type=int, default=DEFAULT_PREFETCH)
    parser.add_argument("--frozen-epochs", type=int, default=DEFAULT_FROZEN_EPOCHS)
    args = parser.parse_args()

    if args.no_cache:
        args.cache = False

    # ogranicz użycie CPU przez BLAS (opcjonalnie)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    data_root = ensure_data_dir(args.data)
    train_ds, val_ds, tri_dir, neg_dir = build_two_class_dataset(
        data_root, val_split=args.val_split, seed=args.seed, cache=args.cache
    )

    nw = max(0, args.workers)
    train_loader = make_loader(train_ds, args.batch, True,  device, nw, args.prefetch)
    val_loader   = make_loader(val_ds,   args.batch, False, device, max(0, nw//2), max(2, args.prefetch//2))

    # --- model ---
    model = ResNet18Tiny(pretrained=True).to(device)

    # freeze → unfreeze
    for p in model.backbone.layer1.parameters(): p.requires_grad = False
    for p in model.backbone.layer2.parameters(): p.requires_grad = False
    for p in model.backbone.layer3.parameters(): p.requires_grad = False
    for p in model.backbone.layer4.parameters(): p.requires_grad = False
    frozen_epochs = max(0, args.frozen_epochs)

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda', enabled=(device=="cuda"))

    # augmentacje + normalizacja
    aug_gpu = build_aug_pipeline(device)
    norm_fn = build_imnet_norm(device)

    # wagi klas (zbalansowanie)
    neg_n = count_files(neg_dir)
    pos_n = count_files(tri_dir)
    total = max(1, neg_n + pos_n)
    w0 = total / max(1, 2 * neg_n)  # not_tri
    w1 = total / max(1, 2 * pos_n)  # tri
    class_w = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=class_w)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Data root: {data_root}")
    print(f"[INFO] Using folders: tri= {tri_dir}, not_tri= {neg_dir}")
    print(f"[INFO] Train/Val sizes: {len(train_ds)}/{len(val_ds)}")
    print(f"[INFO] Cache to RAM: {args.cache} | workers={nw} | pin_memory={(device=='cuda')}")
    print(f"[INFO] Class weights: not_tri={w0:.3f}, tri={w1:.3f}")

    best_val = float("inf")
    best_path = Path(args.out)
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # unfreeze po warmupie
        if epoch == frozen_epochs + 1:
            for p in model.parameters(): p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=args.lr * 0.3)  # mniejszy LR po odblokowaniu
            print("[UNFREEZE] Odblokowano backbone, LR *= 0.3")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, crit, device, aug_gpu, norm_fn, scaler)
        va_loss, va_acc = eval_epoch(model, val_loader, crit, device, norm_fn)
        sched.step(va_loss)

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} | lr {opt.param_groups[0]['lr']:.2e}")

        cm = confusion_matrix(model, val_loader, device, norm_fn)
        print("Confusion Matrix (rows=true, cols=pred):")
        print(cm)

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": va_loss,
                        "val_acc": va_acc,
                        "input_size": (128,128)}, best_path)
            print(f"[SAVE] Best so far -> {best_path} (val_loss={va_loss:.4f})")

    print(f"[DONE] Total time: {time.time() - t0:.1f}s | Best model: {best_path.resolve()}")

if __name__ == "__main__":
    main()
