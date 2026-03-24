import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eit.models import get_model


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_project_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_frames(npz_path: Path, input_key: str | None = None) -> tuple[np.ndarray, str]:
    if not npz_path.exists():
        raise FileNotFoundError(f"❌ 数据文件不存在: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as loaded:
        if input_key is not None:
            if input_key not in loaded:
                raise KeyError(f"❌ 输入文件中不存在键 {input_key!r}。可用键: {list(loaded.keys())}")
            key = input_key
        elif "frames" in loaded:
            key = "frames"
        elif "input_data" in loaded:
            key = "input_data"
        elif "input" in loaded:
            key = "input"
        else:
            key = list(loaded.keys())[0]

        frames = np.asarray(loaded[key], dtype=np.float32)

    if frames.ndim == 4:
        frames = frames.reshape(-1, frames.shape[-2], frames.shape[-1])
    if frames.ndim != 3:
        raise ValueError(f"❌ 推理输入期望为 [T, H, W] 或 [N, T, H, W]，实际得到 {frames.shape}")

    return frames, key


def normalize_frames(
    frames: np.ndarray,
    mode: str,
    low_q: float = 0.1,
    high_q: float = 99.9,
    mask_threshold: float = 1e-6,
) -> np.ndarray:
    frames = frames.astype(np.float32, copy=False)
    mode = mode.lower()

    if mode == "none":
        return frames

    normalized = np.zeros_like(frames, dtype=np.float32)

    if mode == "global":
        valid_mask = frames > mask_threshold
        if np.any(valid_mask):
            valid_pixels = frames[valid_mask]
            vmin = float(np.percentile(valid_pixels, low_q))
            vmax = float(np.percentile(valid_pixels, high_q))
        else:
            vmin, vmax = 0.0, 1.0

        if vmax <= vmin:
            vmax = vmin + 1e-6

        normalized = np.clip((frames - vmin) / (vmax - vmin), 0.0, 1.0)
        normalized[frames <= mask_threshold] = 0.0
        return normalized

    if mode == "per_frame":
        for index, frame in enumerate(frames):
            mask = frame > mask_threshold
            if not np.any(mask):
                continue

            roi = frame[mask]
            vmin = float(np.percentile(roi, low_q))
            vmax = float(np.percentile(roi, high_q))
            if vmax <= vmin:
                vmax = vmin + 1e-6

            frame_norm = np.clip((frame - vmin) / (vmax - vmin), 0.0, 1.0)
            frame_norm[~mask] = 0.0
            normalized[index] = frame_norm
        return normalized

    raise ValueError(f"❌ 不支持的归一化模式: {mode}，可选值为 none / global / per_frame")


def resize_frames(frames: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_size
    resized = [
        cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        for frame in frames
    ]
    return np.asarray(resized, dtype=np.float32)


def resolve_checkpoint_path(cfg: dict, override: str | None) -> Path:
    if override is not None:
        return resolve_project_path(override)

    inference_cfg = cfg.get("inference", {})
    if inference_cfg.get("checkpoint_path"):
        return resolve_project_path(inference_cfg["checkpoint_path"])

    save_dir = resolve_project_path(cfg["save_dir"])
    return save_dir / "best_ssim.pth"


def resolve_output_path(cfg: dict, input_path: Path, override: str | None) -> Path:
    if override is not None:
        return resolve_project_path(override)

    inference_cfg = cfg.get("inference", {})
    if inference_cfg.get("output_path"):
        return resolve_project_path(inference_cfg["output_path"])

    experiment_name = cfg.get("experiment_name") or Path(cfg["save_dir"]).name
    output_name = f"{experiment_name}_{input_path.stem}_infer.npz"
    return PROJECT_ROOT / "outputs" / "inference" / output_name


def build_window_indices(center_idx: int, total_frames: int, pad: int) -> list[int]:
    return [
        min(max(frame_idx, 0), total_frames - 1)
        for frame_idx in range(center_idx - pad, center_idx + pad + 1)
    ]


def save_preview(processed_frames: np.ndarray, predictions: np.ndarray, output_path: Path) -> None:
    preview_path = output_path.with_name(f"{output_path.stem}_compare.png")
    idx = len(predictions) // 2

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(processed_frames[idx], cmap="jet")
    plt.title("Model Input")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(predictions[idx], cmap="jet")
    plt.title("Enhanced Output")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(preview_path)
    plt.close()


def inference(
    cfg: dict,
    checkpoint_path: Path,
    input_path: Path,
    output_path: Path,
    normalization: str,
    low_q: float,
    high_q: float,
    input_key: str | None,
) -> None:
    device = get_best_device()
    print(f"🚀 Inference Device: {device}")

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    model = get_model(model_cfg).to(device)

    model_h = int(data_cfg.get("img_size_h", 176))
    model_w = int(data_cfg.get("img_size_w", 256))
    n_frames = int(model_cfg["params"]["n_frames"])

    print(f"📥 Loading Weights: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"❌ 权重文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    raw_frames, resolved_key = load_frames(input_path, input_key=input_key)
    origin_h, origin_w = raw_frames.shape[-2:]
    print(f"📂 Loading Data: {input_path}")
    print(f"   Input Key: {resolved_key}")
    print(f"   Raw Shape: {raw_frames.shape}")
    print(f"   Output Shape: ({origin_h}, {origin_w})")

    normalized_frames = normalize_frames(
        raw_frames,
        mode=normalization,
        low_q=low_q,
        high_q=high_q,
    )
    processed_frames = resize_frames(normalized_frames, target_size=(model_h, model_w))
    input_tensor = torch.from_numpy(processed_frames).float()

    print(
        "🔄 Preprocessing: "
        f"normalization={normalization}, model_input=({model_h}, {model_w}), window={n_frames}"
    )

    pad = n_frames // 2
    enhanced_results = []
    with torch.no_grad():
        for idx in tqdm(range(len(input_tensor)), unit="fr"):
            window_indices = build_window_indices(idx, len(input_tensor), pad)
            seq = input_tensor[window_indices].unsqueeze(0).to(device)
            pred = model(seq)
            pred_np = pred.squeeze().detach().cpu().numpy()

            if pred_np.ndim != 2:
                raise ValueError(f"❌ 单源推理期望模型输出 2D 图像，实际得到 {pred_np.shape}")

            final_img = cv2.resize(pred_np, (origin_w, origin_h), interpolation=cv2.INTER_LINEAR)
            enhanced_results.append(final_img.astype(np.float32))

    enhanced_results = np.asarray(enhanced_results, dtype=np.float32)

    print(f"💾 Saving Results: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, frames=enhanced_results)
    save_preview(
        processed_frames=resize_frames(processed_frames, target_size=(origin_h, origin_w)),
        predictions=enhanced_results,
        output_path=output_path,
    )
    print(f"✅ Inference Done! Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一的单源 EIT 推理入口。")
    parser.add_argument("--config", type=str, default="configs/experiments/exp001_baseline_unet.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="默认自动使用对应实验的 best_ssim.pth")
    parser.add_argument("--input", type=str, default=None, help="输入 npz 路径；若未指定则尝试读取配置中的 inference.input_path")
    parser.add_argument("--output", type=str, default=None, help="输出 npz 路径；默认自动生成到 outputs/inference/")
    parser.add_argument("--input-key", type=str, default=None, help="npz 中的输入数组键名，默认自动推断")
    parser.add_argument(
        "--normalization",
        type=str,
        default=None,
        choices=["none", "global", "per_frame"],
        help="推理前归一化策略，默认读取配置中的 inference.normalization，若缺失则为 none",
    )
    parser.add_argument("--low-q", type=float, default=None, help="分位数归一化下界")
    parser.add_argument("--high-q", type=float, default=None, help="分位数归一化上界")
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    with open(config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    inference_cfg = cfg.get("inference", {})
    input_path = resolve_project_path(args.input or inference_cfg.get("input_path"))
    if input_path is None:
        raise ValueError("❌ 请通过 --input 指定推理输入，或在配置中补充 inference.input_path。")

    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint)
    output_path = resolve_output_path(cfg, input_path, args.output)
    normalization = args.normalization or inference_cfg.get("normalization", "none")
    low_q = float(args.low_q if args.low_q is not None else inference_cfg.get("low_q", 0.1))
    high_q = float(args.high_q if args.high_q is not None else inference_cfg.get("high_q", 99.9))

    inference(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        input_path=input_path,
        output_path=output_path,
        normalization=normalization,
        low_q=low_q,
        high_q=high_q,
        input_key=args.input_key or inference_cfg.get("input_key"),
    )
