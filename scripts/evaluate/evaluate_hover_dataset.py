"""Offline hover-policy evaluator.

Loads the trained PPO hover agent, simulates a complete episode to obtain all
predicted dot locations, and projects them onto raw screenshots. The output
screenshots are saved with yellow dots indicating every placement the agent
decided to make.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "envs").exists()),
    Path(__file__).resolve().parent,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from envs.hover_env import HoverEnvMultiLineV2
from models.feature_extractor_onnx_optimized import YOLOv11ONNXExtractorOptimized  # noqa: F401


@dataclass
class DotRecord:
    sequence: int
    step: int
    line_id: str
    x_env: float
    y_env: float
    x_img: int
    y_img: int


class HoverEnvWithScreenshots(HoverEnvMultiLineV2):
    """Environment wrapper that matches the training observation structure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "state": spaces.Box(low=-2.0, high=2.0, shape=(12,), dtype=np.float32),
            }
        )

    def _render_screenshot(self) -> np.ndarray:
        img = np.ones((480, 640, 3), dtype=np.uint8) * 240
        for line in self.lines:
            y = int(line["y1"])
            x1, x2 = int(line["x1"]), int(line["x2"])
            cv2.line(img, (x1, y), (x2, y), (100, 100, 100), 2)
        for line in self.lines:
            for x, y in self.line_dots[line["id"]]:
                cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
        if self.current_pos is not None:
            cv2.circle(
                img,
                (int(self.current_pos[0]), int(self.current_pos[1])),
                6,
                (0, 255, 0),
                2,
            )
        return img

    def reset(self, seed=None, options=None):
        obs_state, info = super().reset(seed=seed, options=options)
        return {"screen": self._render_screenshot(), "state": obs_state}, info

    def step(self, action):
        obs_state, reward, terminated, truncated, info = super().step(action)
        return {"screen": self._render_screenshot(), "state": obs_state}, reward, terminated, truncated, info


def list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
    return files


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_scale(env_size: Tuple[int, int], image_size: Tuple[int, int]) -> Tuple[float, float]:
    env_w, env_h = env_size
    img_w, img_h = image_size
    return img_w / env_w, img_h / env_h


def collect_new_dots(
    env: HoverEnvWithScreenshots,
    prev_counts: Dict[str, int],
    step_idx: int,
) -> Iterable[Tuple[str, Tuple[float, float], int]]:
    for line in env.lines:
        line_id = line["id"]
        dots = env.line_dots[line_id]
        start = prev_counts.get(line_id, 0)
        if start >= len(dots):
            continue
        for dot in dots[start:]:
            yield line_id, tuple(dot), step_idx


def annotate_image(image: np.ndarray, dot_records: List[DotRecord], radius: int = 6) -> np.ndarray:
    annotated = image.copy()
    for record in dot_records:
        cv2.circle(
            annotated,
            (record.x_img, record.y_img),
            radius,
            (0, 255, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return annotated


def run_episode(
    model: PPO,
    env: HoverEnvWithScreenshots,
    *,
    deterministic: bool,
    max_steps: int,
    seed: int,
) -> List[Tuple[str, Tuple[float, float], int]]:
    obs, _ = env.reset(seed=seed)

    dot_counts = {line["id"]: len(env.line_dots[line["id"]]) for line in env.lines}
    sequence: List[Tuple[str, Tuple[float, float], int]] = []

    # Record initial dots produced during reset (step index 0).
    for line_id, count in dot_counts.items():
        dots = env.line_dots[line_id]
        for idx in range(count):
            sequence.append((line_id, tuple(dots[idx]), 0))

    for step_idx in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)

        new_counts = {line["id"]: len(env.line_dots[line["id"]]) for line in env.lines}
        for line_id, dot, _ in collect_new_dots(env, dot_counts, step_idx):
            sequence.append((line_id, dot, step_idx))
        dot_counts = new_counts

        if terminated or truncated:
            break

    return sequence


def process_image(
    image_path: Path,
    output_path: Path,
    json_path: Path,
    *,
    model: PPO,
    env: HoverEnvWithScreenshots,
    deterministic: bool,
    max_steps: int,
    seed: int,
    dot_radius: int,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    h_img, w_img = image.shape[:2]
    scale_x, scale_y = compute_scale((env.screen_w, env.screen_h), (w_img, h_img))

    sequence = run_episode(model, env, deterministic=deterministic, max_steps=max_steps, seed=seed)

    dot_records: List[DotRecord] = []
    for idx, (line_id, (x_env, y_env), step_idx) in enumerate(sequence):
        x_img = int(round(x_env * scale_x))
        y_img = int(round(y_env * scale_y))
        dot_records.append(
            DotRecord(
                sequence=idx,
                step=step_idx,
                line_id=line_id,
                x_env=float(x_env),
                y_env=float(y_env),
                x_img=int(np.clip(x_img, 0, w_img - 1)),
                y_img=int(np.clip(y_img, 0, h_img - 1)),
            )
        )

    annotated = annotate_image(image, dot_records, radius=dot_radius)
    cv2.imwrite(str(output_path), annotated)

    with json_path.open("w", encoding="utf-8") as fh:
        payload = {
            "source": str(image_path),
            "annotated": str(output_path),
            "width": w_img,
            "height": h_img,
            "dots": [asdict(record) for record in dot_records],
        }
        json.dump(payload, fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate hover screenshots with PPO policy predictions.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(
            Path("models") / "saved" / "phase2_transfer" / "best_model" / "best_model.zip"
        ),
        help="Path to the trained PPO zip file.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(Path("data") / "test_hover"),
        help="Directory with raw screenshots to annotate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("data") / "test_hover_evaluated"),
        help="Directory where annotated screenshots will be written.",
    )
    parser.add_argument(
        "--lines-file",
        type=str,
        default=str(Path("data") / "text_lines.json"),
        help="Line definition JSON used by the hover environment.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=320,
        help="Maximum number of environment steps per evaluation.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=8,
        help="Radius (in pixels) for the yellow dots drawn on output images.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy predictions (default: stochastic).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Base seed used when resetting the environment for each image.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force inference on the CPU instead of CUDA/DML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    print(f"[HoverEval] Loading model from {args.model}")
    print(f"[HoverEval] Device: {device}")
    model = PPO.load(args.model, device=device)

    env = HoverEnvWithScreenshots(lines_file=args.lines_file)
    env.screen_w = 1920  # ensure explicit attributes for scaling
    env.screen_h = 1080

    images = list_images(input_dir)
    if not images:
        print(f"[HoverEval] No images found in {input_dir}")
        return

    print(f"[HoverEval] Found {len(images)} image(s). Processing...")
    for idx, image_path in enumerate(images):
        seed = args.seed + idx
        output_path = output_dir / image_path.name
        json_path = output_dir / f"{image_path.stem}.json"
        print(f"[HoverEval]  ({idx + 1}/{len(images)}) {image_path.name} -> {output_path.name}")
        process_image(
            image_path,
            output_path,
            json_path,
            model=model,
            env=env,
            deterministic=args.deterministic,
            max_steps=args.max_steps,
            seed=seed,
            dot_radius=args.radius,
        )

    print("[HoverEval] Done.")


if __name__ == "__main__":
    main()
