from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLASSES: List[str] = [
    "single",
    "multi",
    "dropdown",
    "dropdown_scroll",
    "slider",
    "text",
    "triple",
    "mixed",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
FEATURE_NAMES: List[str] = [
    "question_count",
    "has_next",
    "has_select",
    "has_input",
    "option_count",
    "vertical_regularity",
    "scroll_needed",
    "multi_hint",
    "scroll_hint",
    "triple_hint",
    "mix_hint",
    "text_hint",
    "marker_total",
    "marker_circle",
    "marker_square",
    "marker_unknown",
    "marker_mean_conf",
    "is_multi_block_layout",
    "is_exact_triple_count",
    "distinct_block_type_count",
    "all_blocks_same_type",
    "has_mixed_controls",
    "avg_options_per_block",
    "min_options_per_block",
    "max_options_per_block",
    "single_block_ratio",
    "multi_block_ratio",
    "dropdown_block_ratio",
    "text_block_ratio",
    "slider_block_ratio",
    "has_slider",
    "all_blocks_have_triple_marker",
    "all_blocks_have_mix_marker",
]

_PROMPT_MULTI = (
    "zaznacz",
    "wybierz wszystkie",
    "wielokrot",
    "all that apply",
    "select all",
)
_PROMPT_SCROLL = ("scroll", "przewi", "duzo opcji")
_PROMPT_TRIPLE = ("(1/3)", "(2/3)", "(3/3)", "1/3", "2/3", "3/3")
_PROMPT_MIX = ("(mix)", "mix")
_PROMPT_TEXT = ("wpisz", "podaj", "type", "enter")


def _try_tqdm(total: int, desc: str):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(total=total, desc=desc, ncols=100)
    except Exception:
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if isinstance(obj, dict):
                rows.append(obj)
        return rows
    payload = _load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [r for r in payload["rows"] if isinstance(r, dict)]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def _resolve(path: str, manifest_path: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (manifest_path.parent.parent / p).resolve() if "labels/" in path.replace("\\", "/") else (manifest_path.parent / p).resolve()


def _norm_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _prompt_flags(prompt: str) -> Dict[str, bool]:
    norm = _norm_text(prompt)
    return {
        "multi_hint": any(tok in norm for tok in _PROMPT_MULTI),
        "scroll_hint": any(tok in norm for tok in _PROMPT_SCROLL),
        "triple_hint": any(tok in norm for tok in _PROMPT_TRIPLE),
        "mix_hint": any(tok in norm for tok in _PROMPT_MIX),
        "text_hint": any(tok in norm for tok in _PROMPT_TEXT),
    }


def _extract_features(label: Dict[str, Any]) -> Dict[str, float]:
    blocks = label.get("blocks") if isinstance(label.get("blocks"), list) else []
    block_types = [str(b.get("type") or "") for b in blocks if isinstance(b, dict)]
    has_next = bool(label.get("has_next"))
    require_scroll = bool(label.get("require_scroll"))
    option_count = 0
    option_counts: List[int] = []
    prompt_flags = {
        "multi_hint": False,
        "scroll_hint": False,
        "triple_hint": False,
        "mix_hint": False,
        "text_hint": False,
    }
    block_triple_markers = 0
    block_mix_markers = 0
    for b in blocks:
        if not isinstance(b, dict):
            continue
        opts_n = len(b.get("options") or [])
        option_count += opts_n
        option_counts.append(opts_n)
        flags = _prompt_flags(str(b.get("prompt") or ""))
        for key, value in flags.items():
            prompt_flags[key] = bool(prompt_flags[key] or value)
        if flags.get("triple_hint"):
            block_triple_markers += 1
        if flags.get("mix_hint"):
            block_mix_markers += 1

    has_select = any(t in {"dropdown", "dropdown_scroll"} for t in block_types)
    has_input = any(t == "text" for t in block_types)
    multi_hint = any(t == "multi" for t in block_types)
    scroll_hint = any(t == "dropdown_scroll" for t in block_types) or require_scroll
    triple_hint = bool(prompt_flags["triple_hint"])
    mix_hint = bool(prompt_flags["mix_hint"])
    text_hint = any(t == "text" for t in block_types)
    distinct_block_type_count = float(len(set(t for t in block_types if t)))
    all_blocks_same_type = 1.0 if len(set(t for t in block_types if t)) <= 1 and len(block_types) >= 1 else 0.0
    has_mixed_controls = 1.0 if distinct_block_type_count >= 2.0 else 0.0
    q_count = float(len(blocks) if blocks else 1)
    avg_options_per_block = float(option_count) / max(1.0, q_count)
    min_options_per_block = float(min(option_counts)) if option_counts else 0.0
    max_options_per_block = float(max(option_counts)) if option_counts else 0.0
    single_blocks = sum(1 for t in block_types if t == "single")
    multi_blocks = sum(1 for t in block_types if t == "multi")
    dropdown_blocks = sum(1 for t in block_types if t in {"dropdown", "dropdown_scroll"})
    text_blocks = sum(1 for t in block_types if t == "text")
    slider_blocks = sum(1 for t in block_types if t == "slider")

    if block_types:
        if all(t == "single" for t in block_types):
            marker_circle = 1.0
            marker_square = 0.0
        elif all(t == "multi" for t in block_types):
            marker_circle = 0.0
            marker_square = 1.0
        else:
            marker_circle = 0.45
            marker_square = 0.45
    else:
        marker_circle = 0.0
        marker_square = 0.0
    marker_total = 1.0 if any(t in {"single", "multi"} for t in block_types) else 0.0
    marker_unknown = max(0.0, 1.0 - marker_circle - marker_square) if marker_total > 0 else 0.0
    marker_mean_conf = 0.9 if marker_total > 0 else 0.0

    f = {
        "question_count": q_count,
        "has_next": 1.0 if has_next else 0.0,
        "has_select": 1.0 if has_select else 0.0,
        "has_input": 1.0 if has_input else 0.0,
        "option_count": float(option_count),
        "vertical_regularity": 1.0 if option_count >= 2 else 0.0,
        "scroll_needed": 1.0 if require_scroll else 0.0,
        "multi_hint": 1.0 if (multi_hint or prompt_flags["multi_hint"]) else 0.0,
        "scroll_hint": 1.0 if (scroll_hint or prompt_flags["scroll_hint"]) else 0.0,
        "triple_hint": 1.0 if triple_hint else 0.0,
        "mix_hint": 1.0 if mix_hint else 0.0,
        "text_hint": 1.0 if (text_hint or prompt_flags["text_hint"]) else 0.0,
        "marker_total": marker_total,
        "marker_circle": marker_circle,
        "marker_square": marker_square,
        "marker_unknown": marker_unknown,
        "marker_mean_conf": marker_mean_conf,
        "is_multi_block_layout": 1.0 if q_count >= 2.0 else 0.0,
        "is_exact_triple_count": 1.0 if int(q_count) == 3 else 0.0,
        "distinct_block_type_count": distinct_block_type_count,
        "all_blocks_same_type": all_blocks_same_type,
        "has_mixed_controls": has_mixed_controls,
        "avg_options_per_block": avg_options_per_block,
        "min_options_per_block": min_options_per_block,
        "max_options_per_block": max_options_per_block,
        "single_block_ratio": float(single_blocks) / max(1.0, q_count),
        "multi_block_ratio": float(multi_blocks) / max(1.0, q_count),
        "dropdown_block_ratio": float(dropdown_blocks) / max(1.0, q_count),
        "text_block_ratio": float(text_blocks) / max(1.0, q_count),
        "slider_block_ratio": float(slider_blocks) / max(1.0, q_count),
        "has_slider": 1.0 if slider_blocks > 0 else 0.0,
        "all_blocks_have_triple_marker": 1.0 if block_triple_markers == int(q_count) and q_count >= 1 else 0.0,
        "all_blocks_have_mix_marker": 1.0 if block_mix_markers == int(q_count) and q_count >= 1 else 0.0,
    }
    return f


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(np.sum(e, axis=1, keepdims=True), 1e-12, None)


def _one_hot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((y.shape[0], k), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y)) if y.size else 0.0


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def build_rollout_payload(*, out_model: Path, out_payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = out_payload.get("meta") if isinstance(out_payload.get("meta"), dict) else {}
    return {
        "release_family": "quiz_type_rollout",
        "selected_model_path": str(out_model),
        "selected_variant": "rollout",
        "model_type": str(out_payload.get("model_type") or ""),
        "classes": list(out_payload.get("classes") or []),
        "manifest": str(meta.get("manifest") or ""),
        "val_acc": float(meta.get("val_acc") or 0.0),
    }


_BINARY_FEATURE_INDICES: List[int] = [
    i for i, fn in enumerate(FEATURE_NAMES)
    if fn in {
        "has_next", "has_select", "has_input", "vertical_regularity",
        "scroll_needed", "multi_hint", "scroll_hint", "triple_hint",
        "mix_hint", "text_hint", "marker_total", "is_multi_block_layout",
        "is_exact_triple_count", "all_blocks_same_type", "has_mixed_controls",
        "has_slider", "all_blocks_have_triple_marker", "all_blocks_have_mix_marker",
    }
]


def _augment_batch(
    X: np.ndarray,
    rng: np.random.Generator,
    *,
    dropout: float = 0.15,
    noise_std: float = 0.08,
    flip_prob: float = 0.05,
) -> np.ndarray:
    """Apply feature-level augmentation to simulate OCR/DOM errors.

    - dropout: probability of zeroing out each feature (simulates missing detection)
    - noise_std: std of additive Gaussian noise on continuous features
    - flip_prob: probability of flipping binary features (simulates false pos/neg)
    """
    X_aug = X.copy()
    n, d = X_aug.shape
    if dropout > 0.0:
        mask = rng.random((n, d)) > dropout
        X_aug *= mask
    if noise_std > 0.0:
        X_aug += rng.normal(0.0, noise_std, size=(n, d))
    if flip_prob > 0.0 and _BINARY_FEATURE_INDICES:
        bin_idx = np.array(_BINARY_FEATURE_INDICES, dtype=np.intp)
        flip_mask = rng.random((n, len(bin_idx))) < flip_prob
        binary_cols = X_aug[:, bin_idx]
        binary_cols = np.where(flip_mask, 1.0 - binary_cols, binary_cols)
        X_aug[:, bin_idx] = binary_cols
    return X_aug


def _forward_linear(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X @ W + b


def _forward_mlp(
    X: np.ndarray,
    params: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z1 = X @ params["W1"] + params["b1"]
    h1 = _relu(z1)
    if "W2h" in params:
        z2h = h1 @ params["W2h"] + params["b2h"]
        h2 = _relu(z2h)
        z_out = h2 @ params["W3"] + params["b3"]
    else:
        z_out = h1 @ params["W2"] + params["b2"]
    return z1, h1, z_out


def train_linear(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = X_train.shape
    k = len(CLASSES)
    W = rng.normal(loc=0.0, scale=0.01, size=(d, k))
    b = np.zeros((k,), dtype=np.float64)

    pbar = _try_tqdm(epochs, "train")
    for ep in range(1, epochs + 1):
        logits = X_train @ W + b
        probs = _softmax(logits)
        y_oh = _one_hot(y_train, k)
        diff = (probs - y_oh) / max(1, n)
        grad_W = X_train.T @ diff + l2 * W
        grad_b = np.sum(diff, axis=0)
        W -= lr * grad_W
        b -= lr * grad_b

        tr_pred = np.argmax(probs, axis=1)
        tr_acc = _acc(tr_pred, y_train)
        val_probs = _softmax(X_val @ W + b) if X_val.size else np.zeros((0, k))
        val_pred = np.argmax(val_probs, axis=1) if X_val.size else np.array([], dtype=np.int64)
        val_acc = _acc(val_pred, y_val) if X_val.size else 0.0

        if pbar is not None:
            pbar.set_postfix({"train_acc": f"{tr_acc:.4f}", "val_acc": f"{val_acc:.4f}"})
            pbar.update(1)
        else:
            bar_w = 28
            done = int(math.floor(ep * bar_w / max(1, epochs)))
            bar = "#" * done + "-" * (bar_w - done)
            print(f"[{bar}] {ep}/{epochs} train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")
    if pbar is not None:
        pbar.close()
    return W, b


def train_mlp(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
    hidden1: int,
    hidden2: int = 0,
    aug_dropout: float = 0.0,
    aug_noise_std: float = 0.0,
    aug_flip_prob: float = 0.0,
    patience: int = 0,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = X_train.shape
    k = len(CLASSES)
    h1_size = max(4, int(hidden1))
    h2_size = max(0, int(hidden2))
    use_aug = (aug_dropout > 0.0 or aug_noise_std > 0.0 or aug_flip_prob > 0.0)
    use_h2 = h2_size >= 4

    if use_h2:
        params: Dict[str, np.ndarray] = {
            "W1": rng.normal(loc=0.0, scale=0.08, size=(d, h1_size)),
            "b1": np.zeros((h1_size,), dtype=np.float64),
            "W2h": rng.normal(loc=0.0, scale=0.08, size=(h1_size, h2_size)),
            "b2h": np.zeros((h2_size,), dtype=np.float64),
            "W3": rng.normal(loc=0.0, scale=0.08, size=(h2_size, k)),
            "b3": np.zeros((k,), dtype=np.float64),
        }
    else:
        params = {
            "W1": rng.normal(loc=0.0, scale=0.08, size=(d, h1_size)),
            "b1": np.zeros((h1_size,), dtype=np.float64),
            "W2": rng.normal(loc=0.0, scale=0.08, size=(h1_size, k)),
            "b2": np.zeros((k,), dtype=np.float64),
        }

    best_params = {k_: v.copy() for k_, v in params.items()}
    best_val_acc = -1.0
    wait = 0
    effective_patience = max(0, int(patience))

    pbar = _try_tqdm(epochs, "train")
    for ep in range(1, epochs + 1):
        X_input = _augment_batch(X_train, rng, dropout=aug_dropout, noise_std=aug_noise_std, flip_prob=aug_flip_prob) if use_aug else X_train

        z1 = X_input @ params["W1"] + params["b1"]
        a1 = _relu(z1)

        if use_h2:
            z2h = a1 @ params["W2h"] + params["b2h"]
            a2 = _relu(z2h)
            logits = a2 @ params["W3"] + params["b3"]
        else:
            logits = a1 @ params["W2"] + params["b2"]

        probs = _softmax(logits)
        y_oh = _one_hot(y_train, k)
        diff = (probs - y_oh) / max(1, n)

        if use_h2:
            grad_W3 = a2.T @ diff + l2 * params["W3"]
            grad_b3 = np.sum(diff, axis=0)
            da2 = diff @ params["W3"].T
            dz2h = da2 * (z2h > 0.0)
            grad_W2h = a1.T @ dz2h + l2 * params["W2h"]
            grad_b2h = np.sum(dz2h, axis=0)
            da1 = dz2h @ params["W2h"].T
            dz1 = da1 * (z1 > 0.0)
            grad_W1 = X_input.T @ dz1 + l2 * params["W1"]
            grad_b1 = np.sum(dz1, axis=0)
            params["W3"] -= lr * grad_W3
            params["b3"] -= lr * grad_b3
            params["W2h"] -= lr * grad_W2h
            params["b2h"] -= lr * grad_b2h
        else:
            grad_W2 = a1.T @ diff + l2 * params["W2"]
            grad_b2 = np.sum(diff, axis=0)
            da1 = diff @ params["W2"].T
            dz1 = da1 * (z1 > 0.0)
            grad_W1 = X_input.T @ dz1 + l2 * params["W1"]
            grad_b1 = np.sum(dz1, axis=0)
            params["W2"] -= lr * grad_W2
            params["b2"] -= lr * grad_b2

        params["W1"] -= lr * grad_W1
        params["b1"] -= lr * grad_b1

        tr_pred = np.argmax(probs, axis=1)
        tr_acc = _acc(tr_pred, y_train)
        if X_val.size:
            _, _, val_logits = _forward_mlp(X_val, params)
            val_probs = _softmax(val_logits)
            val_pred = np.argmax(val_probs, axis=1)
            val_acc = _acc(val_pred, y_val)
        else:
            val_acc = 0.0

        if val_acc > best_val_acc + 1e-9:
            best_val_acc = val_acc
            best_params = {k_: v.copy() for k_, v in params.items()}
            wait = 0
        else:
            wait += 1

        if pbar is not None:
            pbar.set_postfix({"train_acc": f"{tr_acc:.4f}", "val_acc": f"{val_acc:.4f}", "best": f"{best_val_acc:.4f}"})
            pbar.update(1)
        else:
            bar_w = 28
            done = int(math.floor(ep * bar_w / max(1, epochs)))
            bar = "#" * done + "-" * (bar_w - done)
            print(f"[{bar}] {ep}/{epochs} train_acc={tr_acc:.4f} val_acc={val_acc:.4f} best={best_val_acc:.4f}")

        if effective_patience > 0 and wait >= effective_patience:
            if pbar is not None:
                pbar.set_postfix({"early_stop": ep, "best_val_acc": f"{best_val_acc:.4f}"})
            else:
                print(f"  early stop at epoch {ep}, best_val_acc={best_val_acc:.4f}")
            break

    if pbar is not None:
        pbar.close()
    return best_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Train quiz_type linear model from generated dataset labels.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "data" / "benchmarks" / "random_www_quiz_100k" / "manifests" / "dataset_manifest.jsonl"),
    )
    parser.add_argument(
        "--out-model",
        type=str,
        default=str(ROOT / "data" / "models" / "quiz_type_robust_v3.json"),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260313)
    parser.add_argument("--model-type", type=str, default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32, help="Second hidden layer size; 0=disable (2-layer MLP).")
    parser.add_argument("--aug-dropout", type=float, default=0.15, help="Feature dropout probability per sample (robustness).")
    parser.add_argument("--aug-noise-std", type=float, default=0.08, help="Gaussian noise std added to features (robustness).")
    parser.add_argument("--aug-flip-prob", type=float, default=0.05, help="Binary feature flip probability (robustness).")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience in epochs; 0=disable.")
    parser.add_argument("--out-rollout", type=str, default="", help="Optional rollout contract JSON that points runtime to the trained model.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = _read_manifest(manifest_path)
    rng = random.Random(int(args.seed))
    rng.shuffle(rows)

    X_rows: List[List[float]] = []
    y_rows: List[int] = []
    skipped = 0
    for row in rows:
        gt = str(row.get("expected_global_type") or "").strip()
        if gt not in CLASS_TO_IDX:
            skipped += 1
            continue
        label_path_raw = str(row.get("label_path") or "").strip()
        if not label_path_raw:
            skipped += 1
            continue
        label_path = _resolve(label_path_raw, manifest_path)
        if not label_path.exists():
            skipped += 1
            continue
        try:
            label = _load_json(label_path)
        except Exception:
            skipped += 1
            continue
        feats = _extract_features(label)
        X_rows.append([float(feats.get(n, 0.0)) for n in FEATURE_NAMES])
        y_rows.append(CLASS_TO_IDX[gt])

    if not X_rows:
        raise RuntimeError("No training rows after filtering.")

    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.int64)
    n = X.shape[0]
    split = int(max(1, min(n - 1, round(n * (1.0 - float(args.val_ratio))))))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    X_train_n = (X_train - mu) / sigma
    X_val_n = (X_val - mu) / sigma if X_val.size else X_val

    print(
        json.dumps(
            {
                "rows_total": len(rows),
                "rows_used": int(n),
                "rows_skipped": int(skipped),
                "train_rows": int(X_train.shape[0]),
                "val_rows": int(X_val.shape[0]),
                "epochs": int(args.epochs),
            },
            ensure_ascii=False,
        )
    )

    model_type = str(args.model_type or "mlp").strip().lower()
    if model_type == "mlp":
        params = train_mlp(
            X_train=X_train_n,
            y_train=y_train,
            X_val=X_val_n,
            y_val=y_val,
            epochs=int(args.epochs),
            lr=float(args.lr),
            l2=float(args.l2),
            seed=int(args.seed),
            hidden1=int(args.hidden1),
            hidden2=int(args.hidden2),
            aug_dropout=float(args.aug_dropout),
            aug_noise_std=float(args.aug_noise_std),
            aug_flip_prob=float(args.aug_flip_prob),
            patience=int(args.patience),
        )
        probs_val = _softmax(_forward_mlp(X_val_n, params)[2]) if X_val_n.size else np.zeros((0, len(CLASSES)))
    else:
        W, b = train_linear(
            X_train=X_train_n,
            y_train=y_train,
            X_val=X_val_n,
            y_val=y_val,
            epochs=int(args.epochs),
            lr=float(args.lr),
            l2=float(args.l2),
            seed=int(args.seed),
        )
        probs_val = _softmax(_forward_linear(X_val_n, W, b)) if X_val_n.size else np.zeros((0, len(CLASSES)))
    val_pred = np.argmax(probs_val, axis=1) if X_val_n.size else np.array([], dtype=np.int64)
    val_acc = _acc(val_pred, y_val) if X_val_n.size else 0.0

    out_payload = {
        "version": "v1",
        "model_type": model_type,
        "description": f"{model_type.upper()} quiz_type model trained from random_www_quiz labels.",
        "classes": CLASSES,
        "feature_names": FEATURE_NAMES,
        "normalization": {
            "mean": {fn: float(mu[i]) for i, fn in enumerate(FEATURE_NAMES)},
            "std": {fn: float(sigma[i]) for i, fn in enumerate(FEATURE_NAMES)},
        },
        "meta": {
            "train_rows": int(X_train.shape[0]),
            "val_rows": int(X_val.shape[0]),
            "val_acc": float(val_acc),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "l2": float(args.l2),
            "seed": int(args.seed),
            "manifest": str(manifest_path),
            "hidden1": int(args.hidden1),
            "hidden2": int(args.hidden2),
            "aug_dropout": float(args.aug_dropout),
            "aug_noise_std": float(args.aug_noise_std),
            "aug_flip_prob": float(args.aug_flip_prob),
            "patience": int(args.patience),
        },
    }
    if model_type == "mlp":
        layers_payload: Dict[str, Any] = {
            "hidden1": int(args.hidden1),
            "W1": params["W1"].tolist(),
            "b1": params["b1"].tolist(),
        }
        if "W2h" in params:
            layers_payload["hidden2"] = int(args.hidden2)
            layers_payload["W2h"] = params["W2h"].tolist()
            layers_payload["b2h"] = params["b2h"].tolist()
            layers_payload["W3"] = params["W3"].tolist()
            layers_payload["b3"] = params["b3"].tolist()
        else:
            layers_payload["W2"] = params["W2"].tolist()
            layers_payload["b2"] = params["b2"].tolist()
        out_payload["layers"] = layers_payload
    else:
        weights: Dict[str, Dict[str, float]] = {}
        for fi, fn in enumerate(FEATURE_NAMES):
            weights[fn] = {cn: float(W[fi, ci]) for ci, cn in enumerate(CLASSES)}
        bias = {cn: float(b[ci]) for ci, cn in enumerate(CLASSES)}
        out_payload["bias"] = bias
        out_payload["weights"] = weights

    out_path = Path(args.out_model).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_rollout:
        rollout_path = Path(args.out_rollout).resolve()
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        rollout_path.write_text(
            json.dumps(build_rollout_payload(out_model=out_path, out_payload=out_payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps({"out_model": str(out_path), "val_acc": float(val_acc)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
