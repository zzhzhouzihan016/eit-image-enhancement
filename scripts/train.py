import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import wandb
except ImportError:
    wandb = None

from eit.dataset_dual_source import (
    LCTSCDualSourceDataset,
    LCTSCReconSequenceDataset,
    build_case_splits_from_manifest,
    build_record_key,
    build_slice_group_key,
)
from eit.dataset_structeit import (
    StructEITCacheDataset,
    StructEITSequenceDataset,
    discover_structeit_records,
    resolve_structeit_cache_root,
    limit_case_ids_per_source,
    split_structeit_case_ids,
)
from eit.models import get_model
from eit.utils.seed import set_seed


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def as_spatial_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    if tensor.ndim == 4:
        if tensor.shape[1] == 1:
            return tensor
        return tensor.reshape(-1, 1, tensor.shape[-2], tensor.shape[-1])
    if tensor.ndim == 5:
        batch_size, time_steps, channels, height, width = tensor.shape
        return tensor.reshape(batch_size * time_steps, channels, height, width)
    raise ValueError(f"不支持的张量形状: {tuple(tensor.shape)}")


def select_middle_frame(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim == 4:
        if tensor.shape[1] == 1:
            return tensor[:, 0]
        return tensor[:, tensor.shape[1] // 2]
    if tensor.ndim == 5:
        return tensor[:, tensor.shape[1] // 2, 0]
    raise ValueError(f"不支持的张量形状: {tuple(tensor.shape)}")


def to_uint8_image(frame: torch.Tensor, value_range: tuple[float, float] | None = None) -> np.ndarray:
    array = frame.detach().cpu().numpy().astype(np.float32)
    if value_range is None:
        vmin = float(array.min())
        vmax = float(array.max())
    else:
        vmin, vmax = value_range

    if vmax <= vmin:
        vmax = vmin + 1e-6

    normalized = np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def forward_model(model: nn.Module, model_inputs):
    if isinstance(model_inputs, dict):
        if set(model_inputs.keys()) == {"recon"}:
            return model(model_inputs["recon"])
        return model(**model_inputs)
    return model(model_inputs)


def prepare_batch(batch, device: torch.device):
    extras: dict[str, Any] = {}
    if isinstance(batch, dict):
        recon = batch["recon"].to(device)
        target = batch["target"].to(device)
        model_inputs = {"recon": recon}
        if "voltage" in batch:
            model_inputs["voltage"] = batch["voltage"].to(device)
        if "mask" in batch:
            extras["mask"] = batch["mask"].to(device)
        if "meta" in batch:
            extras["meta"] = batch["meta"]
        return model_inputs, target, recon, target.size(0), extras

    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    return inputs, targets, inputs, targets.size(0), extras


def expand_mask_like(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None

    if reference.ndim == 4:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim != 4:
            raise ValueError(f"mask 期望维度为 2/3/4，实际为 {tuple(mask.shape)}")

        if mask.shape[0] != reference.shape[0]:
            raise ValueError("mask 与预测张量的 batch 维不一致。")
        if mask.shape[1] == 1 and reference.shape[1] != 1:
            mask = mask.expand(-1, reference.shape[1], -1, -1)
        return mask

    if reference.ndim == 3:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError(f"mask 期望维度为 2/3，实际为 {tuple(mask.shape)}")
        if mask.shape[0] != reference.shape[0]:
            raise ValueError("mask 与预测张量的 batch 维不一致。")
        return mask

    raise ValueError(f"当前仅支持 3D/4D 预测张量，实际为 {tuple(reference.shape)}")


def extract_roi_bounds(mask_2d: torch.Tensor, margin: int = 0, threshold: float = 0.5) -> tuple[int, int, int, int] | None:
    positive = torch.nonzero(mask_2d > threshold, as_tuple=False)
    if positive.numel() == 0:
        return None

    y0 = max(int(positive[:, 0].min().item()) - margin, 0)
    y1 = min(int(positive[:, 0].max().item()) + margin + 1, int(mask_2d.shape[0]))
    x0 = max(int(positive[:, 1].min().item()) - margin, 0)
    x1 = min(int(positive[:, 1].max().item()) + margin + 1, int(mask_2d.shape[1]))
    return y0, y1, x0, x1


class EITMetrics:
    @staticmethod
    def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
        diff = preds.reshape(preds.shape[0], -1) - targets.reshape(targets.shape[0], -1)
        mse = torch.mean(diff**2, dim=1)
        return torch.mean(torch.sqrt(mse + 1e-8)).item()

    @staticmethod
    def psnr(preds: torch.Tensor, targets: torch.Tensor, data_range: float | None = None) -> float:
        p = preds.reshape(preds.shape[0], -1)
        t = targets.reshape(targets.shape[0], -1)
        mse = torch.mean((p - t) ** 2, dim=1)

        if data_range is None:
            ranges = torch.clamp(t.max(dim=1).values - t.min(dim=1).values, min=1e-6)
        else:
            ranges = torch.full_like(mse, float(data_range))

        psnr_val = 10 * torch.log10((ranges**2) / (mse + 1e-8))
        return torch.mean(psnr_val).item()

    @staticmethod
    def cc(preds: torch.Tensor, targets: torch.Tensor) -> float:
        p = preds.reshape(preds.shape[0], -1)
        t = targets.reshape(targets.shape[0], -1)
        p_mean = p - torch.mean(p, dim=1, keepdim=True)
        t_mean = t - torch.mean(t, dim=1, keepdim=True)
        num = torch.sum(p_mean * t_mean, dim=1)
        den = torch.sqrt(torch.sum(p_mean**2, dim=1)) * torch.sqrt(torch.sum(t_mean**2, dim=1))
        return torch.mean(num / (den + 1e-8)).item()

    @staticmethod
    def rie(preds: torch.Tensor, targets: torch.Tensor) -> float:
        p = preds.reshape(preds.shape[0], -1)
        t = targets.reshape(targets.shape[0], -1)
        return torch.mean(torch.norm(p - t, p=2, dim=1) / (torch.norm(t, p=2, dim=1) + 1e-8)).item()

    @staticmethod
    def ssim(
        preds: torch.Tensor,
        targets: torch.Tensor,
        window_size: int = 11,
        window_sigma: float = 1.5,
        data_range: float | None = None,
    ) -> float:
        preds_4d = as_spatial_batch(preds)
        targets_4d = as_spatial_batch(targets)

        channel = preds_4d.size(1)
        gauss = torch.tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * window_sigma**2)) for x in range(window_size)],
            device=preds_4d.device,
            dtype=preds_4d.dtype,
        )
        gauss = gauss / gauss.sum()
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(channel, 1, window_size, window_size)

        mu1 = F.conv2d(preds_4d, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(targets_4d, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(preds_4d * preds_4d, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(targets_4d * targets_4d, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(preds_4d * targets_4d, window, padding=window_size // 2, groups=channel) - mu1_mu2

        if data_range is None:
            value_range = float((targets_4d.max() - targets_4d.min()).clamp_min(1e-6).item())
        else:
            value_range = float(data_range)

        c1 = (0.01 * value_range) ** 2
        c2 = (0.03 * value_range) ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
        )
        return ssim_map.mean().item()

    @classmethod
    def roi_metrics(
        cls,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None,
        margin: int = 6,
        mask_threshold: float = 0.5,
    ) -> tuple[float, float, float, int]:
        if masks is None:
            return float("nan"), float("nan"), float("nan"), 0

        if masks.ndim == 4 and masks.shape[1] == 1:
            masks_2d = masks[:, 0]
        elif masks.ndim == 3:
            masks_2d = masks
        else:
            raise ValueError(f"ROI metric 期望 mask 为 [B,H,W] 或 [B,1,H,W]，实际为 {tuple(masks.shape)}")

        roi_l1_values: list[float] = []
        roi_psnr_values: list[float] = []
        roi_ssim_values: list[float] = []

        for sample_index in range(preds.shape[0]):
            bounds = extract_roi_bounds(masks_2d[sample_index], margin=margin, threshold=mask_threshold)
            if bounds is None:
                continue
            y0, y1, x0, x1 = bounds
            pred_crop = preds[sample_index : sample_index + 1, ..., y0:y1, x0:x1]
            target_crop = targets[sample_index : sample_index + 1, ..., y0:y1, x0:x1]

            roi_l1_values.append(torch.mean(torch.abs(pred_crop - target_crop)).item())
            roi_psnr_values.append(cls.psnr(pred_crop, target_crop))
            roi_ssim_values.append(cls.ssim(pred_crop, target_crop))

        valid_count = len(roi_l1_values)
        if valid_count == 0:
            return float("nan"), float("nan"), float("nan"), 0

        return (
            float(sum(roi_l1_values) / valid_count),
            float(sum(roi_psnr_values) / valid_count),
            float(sum(roi_ssim_values) / valid_count),
            valid_count,
        )


class MetricTracker:
    def __init__(self, prefix: str = "val"):
        self.prefix = prefix
        self.reset()

    def reset(self) -> None:
        self.val_loss = 0.0
        self.rmse = 0.0
        self.psnr = 0.0
        self.ssim = 0.0
        self.cc = 0.0
        self.rie = 0.0
        self.roi_l1 = 0.0
        self.roi_psnr = 0.0
        self.roi_ssim = 0.0
        self.count = 0
        self.roi_count = 0

    def update(
        self,
        loss: float,
        rmse: float,
        p: float,
        s: float,
        c: float,
        r: float,
        n: int = 1,
        roi_l1: float | None = None,
        roi_psnr: float | None = None,
        roi_ssim: float | None = None,
        roi_n: int = 0,
    ) -> None:
        self.val_loss += loss * n
        self.rmse += rmse * n
        self.psnr += p * n
        self.ssim += s * n
        self.cc += c * n
        self.rie += r * n
        self.count += n

        if roi_n > 0 and roi_l1 is not None and roi_psnr is not None and roi_ssim is not None:
            self.roi_l1 += roi_l1 * roi_n
            self.roi_psnr += roi_psnr * roi_n
            self.roi_ssim += roi_ssim * roi_n
            self.roi_count += roi_n

    def avg(self) -> dict[str, float]:
        metrics = {
            f"{self.prefix}/loss": self.val_loss / self.count,
            f"{self.prefix}/rmse": self.rmse / self.count,
            f"{self.prefix}/psnr": self.psnr / self.count,
            f"{self.prefix}/ssim": self.ssim / self.count,
            f"{self.prefix}/cc": self.cc / self.count,
            f"{self.prefix}/rie": self.rie / self.count,
            f"{self.prefix}/roi_count": float(self.roi_count),
        }
        if self.roi_count > 0:
            metrics[f"{self.prefix}/roi_l1"] = self.roi_l1 / self.roi_count
            metrics[f"{self.prefix}/roi_psnr"] = self.roi_psnr / self.roi_count
            metrics[f"{self.prefix}/roi_ssim"] = self.roi_ssim / self.roi_count
        else:
            metrics[f"{self.prefix}/roi_l1"] = float("nan")
            metrics[f"{self.prefix}/roi_psnr"] = float("nan")
            metrics[f"{self.prefix}/roi_ssim"] = float("nan")
        return metrics


class EdgeLoss(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        kernel = torch.tensor([[0.1, 0.3, 0.1], [0.3, -1.6, 0.3], [0.1, 0.3, 0.1]], device=device)
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_4d = as_spatial_batch(pred)
        target_4d = as_spatial_batch(target)
        pred_edges = F.conv2d(pred_4d, self.kernel, padding=1)
        target_edges = F.conv2d(target_4d, self.kernel, padding=1)
        return self.l1(pred_edges, target_edges)


class TemporalDifferenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 4 or target.ndim != 4 or pred.shape[1] <= 1:
            return pred.new_tensor(0.0)
        return self.l1(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])


class ROILoss(nn.Module):
    def __init__(self, mask_threshold: float = 0.5):
        super().__init__()
        self.mask_threshold = float(mask_threshold)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        expanded_mask = expand_mask_like(mask, pred)
        if expanded_mask is None:
            return pred.new_tensor(0.0)

        roi_mask = (expanded_mask > self.mask_threshold).to(dtype=pred.dtype)
        denom = roi_mask.sum()
        if float(denom.item()) <= 0.0:
            return pred.new_tensor(0.0)

        return torch.sum(torch.abs(pred - target) * roi_mask) / denom


class TensorNoiseScaleAugment:
    def __init__(
        self,
        noise_std: float = 0.0,
        scale_range: tuple[float, float] = (1.0, 1.0),
        dropout_prob: float = 0.0,
    ) -> None:
        self.noise_std = float(noise_std)
        self.scale_min = float(scale_range[0])
        self.scale_max = float(scale_range[1])
        self.dropout_prob = float(dropout_prob)

        if self.scale_min <= 0 or self.scale_max <= 0:
            raise ValueError("scale_range 必须为正数区间。")
        if self.scale_min > self.scale_max:
            raise ValueError("scale_range 的最小值不能大于最大值。")
        if not 0.0 <= self.dropout_prob < 1.0:
            raise ValueError("dropout_prob 必须在 [0, 1) 区间。")

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.clone()

        if self.scale_min != 1.0 or self.scale_max != 1.0:
            scale = float(torch.empty(1).uniform_(self.scale_min, self.scale_max).item())
            out = out * scale

        if self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std

        if self.dropout_prob > 0:
            keep_mask = (torch.rand_like(out) >= self.dropout_prob).to(out.dtype)
            out = out * keep_mask

        return out


class TensorStandardizeTransform:
    def __init__(self, mean: Any, std: Any, eps: float = 1e-6) -> None:
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.clamp(torch.as_tensor(std, dtype=torch.float32), min=float(eps))

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self.std.to(device=tensor.device, dtype=tensor.dtype)

        while mean.ndim < tensor.ndim:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)

        return (tensor - mean) / std


class TensorTransformPipeline:
    def __init__(self, *transforms) -> None:
        self.transforms = [transform for transform in transforms if transform is not None]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor
        for transform in self.transforms:
            out = transform(out)
        return out


def parse_scale_range(value: Any, default: tuple[float, float] = (1.0, 1.0)) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        scalar = float(value)
        return scalar, scalar
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError("scale_range 需要是标量或长度为 2 的列表/元组。")


def compose_tensor_transforms(*transforms):
    pipeline = TensorTransformPipeline(*transforms)
    return pipeline if pipeline.transforms else None


def resolve_noise_indices(noise_mode: str, fixed_noise_index: int, noise_indices) -> tuple[int, ...]:
    if noise_mode == "fixed":
        return (int(fixed_noise_index),)
    if noise_mode == "expand":
        if noise_indices is None:
            return (0, 1, 2, 3, 4)
        return tuple(int(index) for index in noise_indices)
    raise ValueError("noise_mode 仅支持 'fixed' 或 'expand'。")


def select_records_by_keys(records, record_keys: list[str] | None):
    if record_keys is None:
        return list(records)

    allowed_keys = set(record_keys)
    return [
        record
        for record in records
        if build_record_key(record.case_id, record.slice_index, record.sample_name) in allowed_keys
    ]


def build_normalization_cache_path(
    dataset_root: Path,
    dataset_label: str,
    manifest_name: str,
    records,
    noise_indices: tuple[int, ...],
) -> Path:
    cache_dir = dataset_root / "normalization_cache"
    record_keys = sorted(build_record_key(record.case_id, record.slice_index, record.sample_name) for record in records)
    digest_source = "\n".join(record_keys).encode("utf-8")
    record_digest = hashlib.sha1(digest_source).hexdigest()[:12]
    manifest_stem = Path(manifest_name).stem
    noise_signature = "-".join(str(index) for index in noise_indices)
    filename = f"{dataset_label}_{manifest_stem}_{record_digest}_noise_{noise_signature}.npz"
    return cache_dir / filename


def compute_lctsc_input_stats(
    records,
    noise_indices: tuple[int, ...],
    include_voltage: bool,
    cache_path: Path | None = None,
) -> dict[str, np.ndarray]:
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            stats = {key: cached[key] for key in cached.files}
        print(f"📐 Loaded normalization stats from cache: {cache_path}")
        return stats

    if not records:
        raise ValueError("计算标准化统计量时没有可用训练记录。")

    recon_sum = 0.0
    recon_sq_sum = 0.0
    recon_count = 0

    voltage_sum = None
    voltage_sq_sum = None
    voltage_count = 0

    for record in records:
        with np.load(record.npz_path, allow_pickle=False) as loaded:
            recon_all = loaded["input_recon"]
            voltage_all = loaded["valid208_voltage_noisy"] if include_voltage else None

            for noise_index in noise_indices:
                recon = np.asarray(recon_all[noise_index], dtype=np.float64)
                recon_sum += float(recon.sum())
                recon_sq_sum += float(np.square(recon).sum())
                recon_count += int(recon.size)

                if include_voltage:
                    voltage = np.asarray(voltage_all[noise_index], dtype=np.float64)
                    voltage_flat = voltage.reshape(-1, voltage.shape[-1])
                    if voltage_sum is None:
                        voltage_sum = np.zeros(voltage_flat.shape[-1], dtype=np.float64)
                        voltage_sq_sum = np.zeros(voltage_flat.shape[-1], dtype=np.float64)
                    voltage_sum += voltage_flat.sum(axis=0)
                    voltage_sq_sum += np.square(voltage_flat).sum(axis=0)
                    voltage_count += int(voltage_flat.shape[0])

    recon_mean = np.asarray(recon_sum / max(recon_count, 1), dtype=np.float32)
    recon_var = max(recon_sq_sum / max(recon_count, 1) - float(recon_mean) ** 2, 1e-12)
    recon_std = np.asarray(math.sqrt(recon_var), dtype=np.float32)

    stats: dict[str, np.ndarray] = {
        "recon_mean": recon_mean,
        "recon_std": recon_std,
    }

    if include_voltage:
        if voltage_sum is None or voltage_sq_sum is None:
            raise ValueError("期望计算 voltage 标准化统计量，但未收集到任何 voltage 数据。")
        voltage_mean = voltage_sum / max(voltage_count, 1)
        voltage_var = np.maximum(voltage_sq_sum / max(voltage_count, 1) - np.square(voltage_mean), 1e-12)
        stats["voltage_mean"] = voltage_mean.astype(np.float32)
        stats["voltage_std"] = np.sqrt(voltage_var).astype(np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **stats)
        print(f"📐 Saved normalization stats to cache: {cache_path}")

    return stats


def build_lctsc_normalization_transforms(
    data_cfg: dict,
    dataset_cls,
    manifest_name: str,
    records,
    record_keys: list[str] | None,
    noise_mode: str,
    fixed_noise_index: int,
    noise_indices,
):
    norm_cfg = data_cfg.get("normalization", {})
    if not bool(norm_cfg.get("enable", False)):
        return None, None, {"enabled": False}

    strategy = str(norm_cfg.get("strategy", "train_split")).lower()
    if strategy != "train_split":
        raise ValueError("当前 normalization.strategy 仅支持 'train_split'。")

    dataset_root = Path(data_cfg["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root

    selected_records = select_records_by_keys(records, record_keys)
    selected_noise_indices = resolve_noise_indices(noise_mode, fixed_noise_index, noise_indices)
    include_voltage = dataset_cls is LCTSCDualSourceDataset

    cache_path = None
    if bool(norm_cfg.get("cache", True)):
        dataset_label = "dual_source" if include_voltage else "recon_only"
        cache_path = build_normalization_cache_path(
            dataset_root=dataset_root,
            dataset_label=dataset_label,
            manifest_name=manifest_name,
            records=selected_records,
            noise_indices=selected_noise_indices,
        )

    print(
        "📐 Computing normalization stats: "
        f"records={len(selected_records)}, noise_indices={list(selected_noise_indices)}, "
        f"include_voltage={include_voltage}"
    )
    stats = compute_lctsc_input_stats(
        records=selected_records,
        noise_indices=selected_noise_indices,
        include_voltage=include_voltage,
        cache_path=cache_path,
    )

    recon_transform = TensorStandardizeTransform(stats["recon_mean"], stats["recon_std"])
    voltage_transform = None
    if include_voltage:
        voltage_transform = TensorStandardizeTransform(stats["voltage_mean"], stats["voltage_std"])

    summary = {
        "enabled": True,
        "strategy": strategy,
        "cache_path": str(cache_path) if cache_path is not None else "",
        "record_count": len(selected_records),
        "noise_indices": list(selected_noise_indices),
        "recon_mean": float(np.asarray(stats["recon_mean"]).item()),
        "recon_std": float(np.asarray(stats["recon_std"]).item()),
    }
    if include_voltage:
        summary["voltage_mean_avg"] = float(np.mean(stats["voltage_mean"]))
        summary["voltage_std_avg"] = float(np.mean(stats["voltage_std"]))

    print(
        "📐 Normalization summary: "
        f"recon_mean={summary['recon_mean']:.6e}, recon_std={summary['recon_std']:.6e}"
        + (
            f", voltage_std_avg={summary['voltage_std_avg']:.6e}"
            if include_voltage
            else ""
        )
    )

    return recon_transform, voltage_transform, summary


def build_lctsc_augmentation_transforms(data_cfg: dict, dataset_cls):
    aug_cfg = data_cfg.get("augmentation", {})
    if not bool(aug_cfg.get("enable", False)):
        return None, None

    recon_transform = TensorNoiseScaleAugment(
        noise_std=float(aug_cfg.get("recon_noise_std", 0.0)),
        scale_range=parse_scale_range(aug_cfg.get("recon_scale_range"), default=(1.0, 1.0)),
        dropout_prob=float(aug_cfg.get("recon_dropout_prob", 0.0)),
    )

    voltage_transform = None
    if dataset_cls is LCTSCDualSourceDataset:
        voltage_transform = TensorNoiseScaleAugment(
            noise_std=float(aug_cfg.get("voltage_noise_std", 0.0)),
            scale_range=parse_scale_range(aug_cfg.get("voltage_scale_range"), default=(1.0, 1.0)),
            dropout_prob=float(aug_cfg.get("voltage_dropout_prob", 0.0)),
        )

    return recon_transform, voltage_transform


def build_structeit_normalization_cache_path(
    dataset_root: Path,
    input_key: str,
    records,
) -> Path:
    cache_dir = dataset_root / "normalization_cache"
    record_digest = hashlib.sha1("\n".join(sorted(record.case_id for record in records)).encode("utf-8")).hexdigest()[:12]
    filename = f"structeit_{input_key}_{record_digest}.npz"
    return cache_dir / filename


def compute_structeit_input_stats(
    records,
    input_key: str,
    cache_path: Path | None = None,
) -> dict[str, np.ndarray]:
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            stats = {key: cached[key] for key in cached.files}
        print(f"📐 Loaded StructEIT normalization stats from cache: {cache_path}")
        return stats

    if not records:
        raise ValueError("计算 StructEIT 标准化统计量时没有可用训练记录。")

    recon_sum = 0.0
    recon_sq_sum = 0.0
    recon_count = 0

    for record in records:
        with np.load(record.npz_path, allow_pickle=False) as loaded:
            recon = np.asarray(loaded[input_key], dtype=np.float64)
        recon_sum += float(recon.sum())
        recon_sq_sum += float(np.square(recon).sum())
        recon_count += int(recon.size)

    recon_mean = np.asarray(recon_sum / max(recon_count, 1), dtype=np.float32)
    recon_var = max(recon_sq_sum / max(recon_count, 1) - float(recon_mean) ** 2, 1e-12)
    recon_std = np.asarray(math.sqrt(recon_var), dtype=np.float32)
    stats = {
        "recon_mean": recon_mean,
        "recon_std": recon_std,
    }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **stats)
        print(f"📐 Saved StructEIT normalization stats to cache: {cache_path}")

    return stats


def build_structeit_normalization_transform(data_cfg: dict, train_records, input_key: str):
    norm_cfg = data_cfg.get("normalization", {})
    if not bool(norm_cfg.get("enable", False)):
        return None, {"enabled": False}

    strategy = str(norm_cfg.get("strategy", "train_split")).lower()
    if strategy != "train_split":
        raise ValueError("当前 StructEIT normalization.strategy 仅支持 'train_split'。")

    dataset_root = Path(data_cfg["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root

    cache_path = None
    if bool(norm_cfg.get("cache", True)):
        cache_path = build_structeit_normalization_cache_path(
            dataset_root=dataset_root,
            input_key=input_key,
            records=train_records,
        )

    print(
        "📐 Computing StructEIT normalization stats: "
        f"records={len(train_records)}, input_key={input_key}"
    )
    stats = compute_structeit_input_stats(
        records=train_records,
        input_key=input_key,
        cache_path=cache_path,
    )

    recon_transform = TensorStandardizeTransform(stats["recon_mean"], stats["recon_std"])
    summary = {
        "enabled": True,
        "strategy": strategy,
        "cache_path": str(cache_path) if cache_path is not None else "",
        "record_count": len(train_records),
        "recon_mean": float(np.asarray(stats["recon_mean"]).item()),
        "recon_std": float(np.asarray(stats["recon_std"]).item()),
    }
    print(
        "📐 StructEIT normalization summary: "
        f"recon_mean={summary['recon_mean']:.6e}, recon_std={summary['recon_std']:.6e}"
    )
    return recon_transform, summary


def _resolve_structeit_case_splits(data_cfg: dict, seed: int) -> dict[str, list[str]]:
    case_split_cfg = data_cfg.get("case_split", {})
    train_case_ids = case_split_cfg.get("train_case_ids")
    val_case_ids = case_split_cfg.get("val_case_ids")
    test_case_ids = case_split_cfg.get("test_case_ids")
    input_key = str(data_cfg.get("input_key", "greit_img"))

    if train_case_ids is None or val_case_ids is None:
        records = discover_structeit_records(
            dataset_root=data_cfg["dataset_root"],
            input_key=input_key,
        )
        raw_splits = split_structeit_case_ids(
            case_ids=[record.case_id for record in records],
            train_ratio=float(case_split_cfg.get("train_ratio", 0.7)),
            val_ratio=float(case_split_cfg.get("val_ratio", 0.15)),
            seed=seed,
        )
        splits = {
            "train": list(raw_splits["train"]),
            "val": list(raw_splits["val"]),
            "test": list(raw_splits["test"]),
        }
    else:
        splits = {
            "train": list(train_case_ids),
            "val": list(val_case_ids),
            "test": list(test_case_ids or []),
        }

    debug_limit_cfg = case_split_cfg.get("debug_limit_per_source")
    if debug_limit_cfg is not None:
        for split_name in ("train", "val", "test"):
            if isinstance(debug_limit_cfg, dict):
                split_limit = debug_limit_cfg.get(split_name)
            else:
                split_limit = debug_limit_cfg
            if split_limit is not None:
                splits[split_name] = limit_case_ids_per_source(splits[split_name], int(split_limit))

    return splits


def build_structeit_baseline_dataloaders(cfg: dict, seed: int):
    data_cfg = cfg["data"]
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))
    dataset_root = data_cfg["dataset_root"]
    input_key = str(data_cfg.get("input_key", "greit_img"))
    target_key = str(data_cfg.get("target_key", "target_img"))
    target_mode = str(data_cfg.get("target_mode", "middle"))
    window_size = int(data_cfg.get("window_size", cfg["model"].get("params", {}).get("n_frames", 1)))
    train_frame_stride = int(data_cfg.get("frame_stride", 1))
    eval_frame_stride = int(data_cfg.get("eval_frame_stride", train_frame_stride))
    foreground_mask_cfg = data_cfg.get("foreground_mask", {})
    include_mask = bool(foreground_mask_cfg.get("enable", True))
    mask_threshold_abs = float(foreground_mask_cfg.get("threshold_abs", 1e-6))

    case_splits = _resolve_structeit_case_splits(data_cfg, seed)
    probe_train_ds = StructEITSequenceDataset(
        dataset_root=dataset_root,
        case_ids=case_splits["train"],
        input_key=input_key,
        target_key=target_key,
        window_size=window_size,
        frame_stride=train_frame_stride,
        target_mode=target_mode,
        include_mask=include_mask,
        mask_threshold_abs=mask_threshold_abs,
    )

    norm_recon_transform, normalization_info = build_structeit_normalization_transform(
        data_cfg=data_cfg,
        train_records=probe_train_ds.records,
        input_key=input_key,
    )
    aug_recon_transform, _ = build_lctsc_augmentation_transforms(data_cfg, StructEITSequenceDataset)
    train_recon_transform = compose_tensor_transforms(norm_recon_transform, aug_recon_transform)
    eval_recon_transform = norm_recon_transform

    train_ds = StructEITSequenceDataset(
        dataset_root=dataset_root,
        case_ids=case_splits["train"],
        input_key=input_key,
        target_key=target_key,
        window_size=window_size,
        frame_stride=train_frame_stride,
        target_mode=target_mode,
        recon_transform=train_recon_transform,
        include_mask=include_mask,
        mask_threshold_abs=mask_threshold_abs,
    )
    val_ds = StructEITSequenceDataset(
        dataset_root=dataset_root,
        case_ids=case_splits["val"],
        input_key=input_key,
        target_key=target_key,
        window_size=window_size,
        frame_stride=eval_frame_stride,
        target_mode=target_mode,
        recon_transform=eval_recon_transform,
        include_mask=include_mask,
        mask_threshold_abs=mask_threshold_abs,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loaders = {"val": DataLoader(val_ds, shuffle=False, **loader_kwargs)}

    split_info: dict[str, Any] = {
        "split_mode": "structeit_case_split",
        "train_cases": case_splits["train"],
        "val_cases": case_splits["val"],
        "test_cases": case_splits["test"],
        "input_key": input_key,
        "target_key": target_key,
        "window_size": window_size,
        "train_frame_stride": train_frame_stride,
        "eval_frame_stride": eval_frame_stride,
        "target_mode": target_mode,
        "foreground_mask": {
            "enable": include_mask,
            "threshold_abs": mask_threshold_abs,
        },
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "normalization": normalization_info,
    }

    print(
        "📊 StructEIT Split: "
        f"TrainCases={len(case_splits['train'])}, ValCases={len(case_splits['val'])}, "
        f"TestCases={len(case_splits['test'])} | "
        f"TrainSamples={len(train_ds)}, ValSamples={len(val_ds)} | "
        f"TrainStride={train_frame_stride}, EvalStride={eval_frame_stride}"
    )
    return train_loader, val_loaders, split_info


def build_structeit_cached_dataloaders(cfg: dict, seed: int):
    del seed
    data_cfg = cfg["data"]
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))
    cache_root = resolve_structeit_cache_root(data_cfg["cache_root"])

    global_metadata_path = cache_root / "cache_metadata.json"
    if not global_metadata_path.exists():
        raise FileNotFoundError(f"找不到 StructEIT 缓存元信息: {global_metadata_path}")

    with open(global_metadata_path, "r", encoding="utf-8") as file:
        global_metadata = json.load(file)

    normalization = global_metadata.get("normalization") or {}
    if not normalization:
        raise ValueError(f"{global_metadata_path} 中缺少 normalization 信息。")

    recon_transform = TensorStandardizeTransform(
        normalization["recon_mean"],
        normalization["recon_std"],
    )
    shard_cache_size = int(data_cfg.get("shard_cache_size", 1))

    train_ds = StructEITCacheDataset(
        cache_root=cache_root,
        split="train",
        recon_transform=recon_transform,
        cache_size=shard_cache_size,
    )
    val_ds = StructEITCacheDataset(
        cache_root=cache_root,
        split="val",
        recon_transform=recon_transform,
        cache_size=shard_cache_size,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loaders = {"val": DataLoader(val_ds, shuffle=False, **loader_kwargs)}

    split_info: dict[str, Any] = {
        "split_mode": "structeit_cached",
        "cache_root": str(cache_root),
        "normalization": normalization,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "cache_metadata": global_metadata,
    }

    print(
        "📦 StructEIT Cached Split: "
        f"TrainSamples={len(train_ds)}, ValSamples={len(val_ds)} | "
        f"batch={batch_size}, workers={num_workers}, shard_cache_size={shard_cache_size}"
    )
    return train_loader, val_loaders, split_info


def inspect_single_source_npz(npz_path: str | Path, frames_per_seq_hint: int | None = None) -> tuple[int, int]:
    with np.load(npz_path, allow_pickle=False) as raw:
        if "input_data" in raw:
            inputs = raw["input_data"]
        else:
            inputs = raw["input"]

        if inputs.ndim == 4:
            return int(inputs.shape[0]), int(inputs.shape[1])
        if inputs.ndim == 3:
            total_frames = int(inputs.shape[0])
            if frames_per_seq_hint is None:
                return 1, total_frames
            if total_frames % int(frames_per_seq_hint) != 0:
                raise ValueError(
                    "single-source 3D 输入无法按 frames_per_seq 正确分组: "
                    f"total_frames={total_frames}, frames_per_seq={frames_per_seq_hint}"
                )
            return total_frames // int(frames_per_seq_hint), int(frames_per_seq_hint)
        raise ValueError(f"不支持的 single-source 输入维度: {inputs.shape}")


def build_single_source_dataloaders(cfg: dict, seed: int):
    from eit.dataset import EITSequenceDataset

    data_cfg = cfg["data"]
    train_path = Path(data_cfg["train_path"])
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path
    frames_per_seq = data_cfg.get("frames_per_seq")
    num_samples, detected_frames_per_seq = inspect_single_source_npz(train_path, frames_per_seq_hint=frames_per_seq)

    if frames_per_seq is None:
        frames_per_seq = detected_frames_per_seq
    elif int(frames_per_seq) != int(detected_frames_per_seq):
        print(
            "⚠️ Single-source config 中的 frames_per_seq "
            f"({frames_per_seq}) 与数据实际值 ({detected_frames_per_seq}) 不一致，已自动修正。"
        )
        frames_per_seq = detected_frames_per_seq

    img_h = int(data_cfg.get("img_size_h", 176))
    img_w = int(data_cfg.get("img_size_w", 256))
    n_frames = int(cfg["model"]["params"]["n_frames"])
    target_mode = str(data_cfg.get("target_mode", "middle")).lower()
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))
    val_ratio = float(data_cfg.get("val_ratio", 0.05))

    target_blur_cfg = data_cfg.get("target_blur", {})
    apply_target_blur = bool(target_blur_cfg.get("enable", False))
    target_blur_kernel_size = int(target_blur_cfg.get("kernel_size", 5))
    target_blur_sigma = target_blur_cfg.get("sigma", 1.0)

    aug_cfg = data_cfg.get("augmentation", {})
    use_augmentation = bool(aug_cfg.get("enable", True))
    hflip_prob = float(aug_cfg.get("hflip_prob", 0.5))
    rotate_prob = float(aug_cfg.get("rotate_prob", 0.3))
    rotate_deg = float(aug_cfg.get("rotate_deg", 5.0))

    print(
        "🔧 Single-source Dataset Config: "
        f"samples={num_samples}, frames_per_seq={frames_per_seq}, size=({img_h}, {img_w}), "
        f"val_ratio={val_ratio:.2f}, target_mode={target_mode}"
    )

    if num_samples < 2:
        raise ValueError("single-source 数据至少需要 2 个原始序列，才能进行 train/val 划分。")
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("data.val_ratio 建议设置在 (0, 0.5) 区间内。")

    generator = torch.Generator()
    generator.manual_seed(seed)
    sample_indices = torch.randperm(num_samples, generator=generator).tolist()
    val_sample_count = max(1, int(round(num_samples * val_ratio)))
    if val_sample_count >= num_samples:
        val_sample_count = num_samples - 1

    val_sample_ids = sorted(sample_indices[:val_sample_count])
    train_sample_ids = sorted(sample_indices[val_sample_count:])

    train_ds = EITSequenceDataset(
        train_path,
        n_frames=n_frames,
        frames_per_seq=frames_per_seq,
        target_mode=target_mode,
        target_size=(img_h, img_w),
        use_augmentation=use_augmentation,
        sample_ids=train_sample_ids,
        apply_target_blur=apply_target_blur,
        target_blur_kernel_size=target_blur_kernel_size,
        target_blur_sigma=target_blur_sigma,
        augment_hflip_prob=hflip_prob,
        augment_rotate_prob=rotate_prob,
        augment_rotate_deg=rotate_deg,
    )
    val_ds = EITSequenceDataset(
        train_path,
        n_frames=n_frames,
        frames_per_seq=frames_per_seq,
        target_mode=target_mode,
        target_size=(img_h, img_w),
        use_augmentation=False,
        sample_ids=val_sample_ids,
        apply_target_blur=False,
    )
    print(
        "📊 Single-source Split (by sequence): "
        f"TrainSeq={len(train_sample_ids)}, ValSeq={len(val_sample_ids)} | "
        f"TrainWindows={len(train_ds)}, ValWindows={len(val_ds)}"
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    split_info = {
        "split_mode": "single_sequence",
        "train_sequence_ids": train_sample_ids,
        "val_sequence_ids": val_sample_ids,
        "target_mode": target_mode,
    }
    return train_loader, {"val": val_loader}, split_info


def _resolve_lctsc_case_splits(data_cfg: dict, seed: int) -> tuple[str, dict[str, list[str]]]:
    manifest_name = data_cfg.get("manifest_name", "global_samples_manifest.csv")
    case_split_cfg = data_cfg.get("case_split", {})
    train_case_ids = case_split_cfg.get("train_case_ids")
    val_inter_case_ids = case_split_cfg.get("val_inter_case_ids")
    if val_inter_case_ids is None:
        val_inter_case_ids = case_split_cfg.get("val_case_ids")
    test_case_ids = case_split_cfg.get("test_case_ids")

    if train_case_ids is None or val_inter_case_ids is None:
        raw_splits = build_case_splits_from_manifest(
            dataset_root=data_cfg["dataset_root"],
            manifest_name=manifest_name,
            train_ratio=float(case_split_cfg.get("train_ratio", 0.7)),
            val_ratio=float(case_split_cfg.get("val_ratio", 0.15)),
            seed=seed,
        )
        splits = {
            "train": list(raw_splits["train"]),
            "val_inter": list(raw_splits["val"]),
            "test": list(raw_splits["test"]),
        }
    else:
        splits = {
            "train": list(train_case_ids),
            "val_inter": list(val_inter_case_ids),
            "test": list(test_case_ids or []),
        }

    return manifest_name, splits


def split_train_records_for_intra_val(
    records,
    ratio: float,
    group_by: str,
    seed: int,
) -> dict[str, Any]:
    if not 0.0 <= ratio < 1.0:
        raise ValueError("intra_val.ratio 必须在 [0, 1) 区间内。")

    all_record_keys = sorted(build_record_key(r.case_id, r.slice_index, r.sample_name) for r in records)

    result = {
        "enabled": False,
        "group_by": group_by,
        "ratio": ratio,
        "train_record_keys": all_record_keys,
        "val_intra_record_keys": [],
        "train_group_keys": [],
        "val_intra_group_keys": [],
    }

    if ratio <= 0 or len(records) < 2:
        return result

    if group_by not in {"sample", "slice"}:
        raise ValueError("intra_val.group_by 仅支持 'sample' 或 'slice'。")

    grouped_record_keys: dict[str, list[str]] = {}
    for record in records:
        record_key = build_record_key(record.case_id, record.slice_index, record.sample_name)
        if group_by == "slice":
            group_key = build_slice_group_key(record.case_id, record.slice_index)
        else:
            group_key = record_key
        grouped_record_keys.setdefault(group_key, []).append(record_key)

    group_keys = sorted(grouped_record_keys)
    if len(group_keys) < 2:
        return result

    shuffled_group_keys = list(group_keys)
    random.Random(seed).shuffle(shuffled_group_keys)

    val_group_count = max(1, int(round(len(shuffled_group_keys) * ratio)))
    if val_group_count >= len(shuffled_group_keys):
        val_group_count = len(shuffled_group_keys) - 1
    if val_group_count <= 0:
        return result

    val_intra_group_keys = sorted(shuffled_group_keys[:val_group_count])
    train_group_keys = sorted(shuffled_group_keys[val_group_count:])
    val_intra_record_keys = sorted(
        record_key for group_key in val_intra_group_keys for record_key in grouped_record_keys[group_key]
    )
    train_record_keys = sorted(
        record_key for group_key in train_group_keys for record_key in grouped_record_keys[group_key]
    )

    if not train_record_keys or not val_intra_record_keys:
        return result

    result.update(
        {
            "enabled": True,
            "train_record_keys": train_record_keys,
            "val_intra_record_keys": val_intra_record_keys,
            "train_group_keys": train_group_keys,
            "val_intra_group_keys": val_intra_group_keys,
        }
    )
    return result


def _build_dataset_kwargs(
    dataset_cls,
    dataset_root: str,
    manifest_name: str,
    case_ids: list[str],
    noise_mode: str,
    fixed_noise_index: int,
    noise_indices,
    mask_key: str,
    recon_transform=None,
    voltage_transform=None,
    record_keys: list[str] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dataset_root": dataset_root,
        "manifest_name": manifest_name,
        "case_ids": case_ids,
        "record_keys": record_keys,
        "noise_mode": noise_mode,
        "fixed_noise_index": fixed_noise_index,
        "noise_indices": noise_indices,
        "mask_key": mask_key,
        "recon_transform": recon_transform,
    }
    if dataset_cls is LCTSCDualSourceDataset:
        kwargs["voltage_transform"] = voltage_transform
    return kwargs


def _build_lctsc_sequence_dataloaders(cfg: dict, seed: int, dataset_cls, dataset_label: str):
    data_cfg = cfg["data"]
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))
    dataset_root = data_cfg["dataset_root"]
    mask_key = str(data_cfg.get("mask_key", "target_pathology_mask"))
    manifest_name, case_splits = _resolve_lctsc_case_splits(data_cfg, seed)

    noise_cfg = data_cfg.get("noise", {})
    noise_mode = noise_cfg.get("mode", "fixed")
    fixed_noise_index = int(noise_cfg.get("fixed_index", 2))
    noise_indices = noise_cfg.get("indices")

    probe_train_ds = dataset_cls(
        **_build_dataset_kwargs(
            dataset_cls=dataset_cls,
            dataset_root=dataset_root,
            manifest_name=manifest_name,
            case_ids=case_splits["train"],
            noise_mode=noise_mode,
            fixed_noise_index=fixed_noise_index,
            noise_indices=noise_indices,
            mask_key=mask_key,
        )
    )

    intra_val_cfg = data_cfg.get("intra_val", {})
    intra_val_ratio = float(intra_val_cfg.get("ratio", 0.0))
    intra_val_group_by = str(intra_val_cfg.get("group_by", "slice"))
    intra_val_seed = int(intra_val_cfg.get("seed", seed))
    intra_val_split = split_train_records_for_intra_val(
        probe_train_ds.records,
        ratio=intra_val_ratio if bool(intra_val_cfg.get("enable", False)) else 0.0,
        group_by=intra_val_group_by,
        seed=intra_val_seed,
    )

    train_record_keys = intra_val_split["train_record_keys"] if intra_val_split["enabled"] else None
    norm_recon_transform, norm_voltage_transform, normalization_info = build_lctsc_normalization_transforms(
        data_cfg=data_cfg,
        dataset_cls=dataset_cls,
        manifest_name=manifest_name,
        records=probe_train_ds.records,
        record_keys=train_record_keys,
        noise_mode=noise_mode,
        fixed_noise_index=fixed_noise_index,
        noise_indices=noise_indices,
    )

    aug_recon_transform, aug_voltage_transform = build_lctsc_augmentation_transforms(data_cfg, dataset_cls)
    train_recon_transform = compose_tensor_transforms(norm_recon_transform, aug_recon_transform)
    train_voltage_transform = compose_tensor_transforms(norm_voltage_transform, aug_voltage_transform)
    eval_recon_transform = norm_recon_transform
    eval_voltage_transform = norm_voltage_transform

    train_ds = dataset_cls(
        **_build_dataset_kwargs(
            dataset_cls=dataset_cls,
            dataset_root=dataset_root,
            manifest_name=manifest_name,
            case_ids=case_splits["train"],
            record_keys=train_record_keys,
            noise_mode=noise_mode,
            fixed_noise_index=fixed_noise_index,
            noise_indices=noise_indices,
            mask_key=mask_key,
            recon_transform=train_recon_transform,
            voltage_transform=train_voltage_transform,
        )
    )

    val_loaders: dict[str, DataLoader] = {}
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if case_splits["val_inter"]:
        val_inter_ds = dataset_cls(
            **_build_dataset_kwargs(
                dataset_cls=dataset_cls,
                dataset_root=dataset_root,
                manifest_name=manifest_name,
                case_ids=case_splits["val_inter"],
                noise_mode=noise_mode,
                fixed_noise_index=fixed_noise_index,
                noise_indices=noise_indices,
                mask_key=mask_key,
                recon_transform=eval_recon_transform,
                voltage_transform=eval_voltage_transform,
            )
        )
        val_loaders["val_inter"] = DataLoader(val_inter_ds, shuffle=False, **loader_kwargs)
    else:
        val_inter_ds = None

    if intra_val_split["enabled"]:
        val_intra_ds = dataset_cls(
            **_build_dataset_kwargs(
                dataset_cls=dataset_cls,
                dataset_root=dataset_root,
                manifest_name=manifest_name,
                case_ids=case_splits["train"],
                record_keys=intra_val_split["val_intra_record_keys"],
                noise_mode=noise_mode,
                fixed_noise_index=fixed_noise_index,
                noise_indices=noise_indices,
                mask_key=mask_key,
                recon_transform=eval_recon_transform,
                voltage_transform=eval_voltage_transform,
            )
        )
        val_loaders["val_intra"] = DataLoader(val_intra_ds, shuffle=False, **loader_kwargs)
    else:
        val_intra_ds = None

    if not val_loaders:
        raise ValueError("当前未构建任何验证集，请检查 case_split 或 intra_val 配置。")

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)

    split_info: dict[str, Any] = {
        "split_mode": "lctsc_case_split",
        "train_cases": case_splits["train"],
        "val_inter_cases": case_splits["val_inter"],
        "test_cases": case_splits["test"],
        "mask_key": mask_key,
        "train_samples": len(train_ds),
        "val_inter_samples": len(val_inter_ds) if val_inter_ds is not None else 0,
        "val_intra": {
            "enabled": bool(intra_val_split["enabled"]),
            "group_by": intra_val_split["group_by"],
            "ratio": intra_val_split["ratio"],
            "train_group_count": len(intra_val_split["train_group_keys"]),
            "val_intra_group_count": len(intra_val_split["val_intra_group_keys"]),
            "train_record_count": len(intra_val_split["train_record_keys"]),
            "val_intra_record_count": len(intra_val_split["val_intra_record_keys"]),
            "val_intra_samples": len(val_intra_ds) if val_intra_ds is not None else 0,
        },
        "normalization": normalization_info,
    }

    print(
        f"📊 {dataset_label} Split: "
        f"TrainCases={len(case_splits['train'])}, ValInterCases={len(case_splits['val_inter'])}, "
        f"TestCases={len(case_splits['test'])} | "
        f"TrainSamples={len(train_ds)}, "
        f"ValInterSamples={len(val_inter_ds) if val_inter_ds is not None else 0}, "
        f"ValIntraSamples={len(val_intra_ds) if val_intra_ds is not None else 0}"
    )
    if intra_val_split["enabled"]:
        print(
            "   ↳ Intra-val split: "
            f"group_by={intra_val_split['group_by']}, ratio={intra_val_split['ratio']:.2f}, "
            f"train_groups={len(intra_val_split['train_group_keys'])}, "
            f"val_groups={len(intra_val_split['val_intra_group_keys'])}"
        )

    return train_loader, val_loaders, split_info


def build_dual_source_dataloaders(cfg: dict, seed: int):
    return _build_lctsc_sequence_dataloaders(
        cfg,
        seed,
        dataset_cls=LCTSCDualSourceDataset,
        dataset_label="Dual-source",
    )


def build_recon_sequence_dataloaders(cfg: dict, seed: int):
    return _build_lctsc_sequence_dataloaders(
        cfg,
        seed,
        dataset_cls=LCTSCReconSequenceDataset,
        dataset_label="Single-source LCTSC",
    )


def build_dataloaders(cfg: dict, seed: int):
    dataset_type = cfg["data"].get("dataset_type", "single_sequence")
    if dataset_type == "dual_source_lctsc":
        return build_dual_source_dataloaders(cfg, seed)
    if dataset_type == "single_source_lctsc_seq":
        return build_recon_sequence_dataloaders(cfg, seed)
    if dataset_type == "structeit_single_source":
        return build_structeit_baseline_dataloaders(cfg, seed)
    if dataset_type == "structeit_cached":
        return build_structeit_cached_dataloaders(cfg, seed)
    return build_single_source_dataloaders(cfg, seed)


def visualize_and_save(model, loader, device, save_dir: Path, epoch: int, num_samples: int = 3):
    model.eval()
    vis_dir = save_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    wandb_images = []

    try:
        batch = next(iter(loader))
        model_inputs, targets, vis_inputs, _, _ = prepare_batch(batch, device)
        with torch.no_grad():
            preds = forward_model(model, model_inputs)

        vis_input_frames = select_middle_frame(vis_inputs)
        vis_target_frames = select_middle_frame(targets)
        vis_pred_frames = select_middle_frame(preds)

        for index in range(min(num_samples, vis_input_frames.shape[0])):
            img_in = vis_input_frames[index]
            img_tar = vis_target_frames[index]
            img_pred = vis_pred_frames[index]

            target_range = (
                float(torch.min(torch.stack([img_tar.min(), img_pred.min()]))),
                float(torch.max(torch.stack([img_tar.max(), img_pred.max()]))),
            )

            input_uint8 = to_uint8_image(img_in)
            target_uint8 = to_uint8_image(img_tar, value_range=target_range)
            pred_uint8 = to_uint8_image(img_pred, value_range=target_range)

            target_height, target_width = target_uint8.shape
            if input_uint8.shape != (target_height, target_width):
                input_uint8 = cv2.resize(input_uint8, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            vis_in = cv2.applyColorMap(input_uint8, cv2.COLORMAP_JET)
            vis_tar = cv2.applyColorMap(target_uint8, cv2.COLORMAP_JET)
            vis_pred = cv2.applyColorMap(pred_uint8, cv2.COLORMAP_JET)

            combined = np.hstack((vis_in, vis_tar, vis_pred))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined, "Input", (10, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(combined, "Target", (10 + combined.shape[1] // 3, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(combined, "Pred", (10 + 2 * combined.shape[1] // 3, 20), font, 0.5, (255, 255, 255), 1)

            save_path = vis_dir / f"epoch_{epoch}_sample_{index}.jpg"
            cv2.imwrite(str(save_path), combined)

            if wandb is not None:
                combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
                wandb_images.append(wandb.Image(combined_rgb, caption=f"Ep{epoch}-S{index}"))

    except Exception as exc:
        print(f"⚠️ 可视化失败: {exc}")

    return wandb_images


def init_wandb(cfg: dict) -> bool:
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enable", False):
        return False

    if wandb is None:
        raise ImportError("❌ 当前环境未安装 wandb，请先安装或在配置中关闭 wandb。")

    init_kwargs = {
        "project": wandb_cfg["project"],
        "name": wandb_cfg["name"],
        "config": cfg,
        "tags": wandb_cfg.get("tags", []),
    }
    if wandb_cfg.get("entity"):
        init_kwargs["entity"] = wandb_cfg["entity"]
    if wandb_cfg.get("mode"):
        init_kwargs["mode"] = wandb_cfg["mode"]

    try:
        wandb.init(**init_kwargs)
        return True
    except Exception as exc:
        print(f"⚠️ WandB 初始化失败，已自动关闭本次 WandB 同步: {exc}")
        print("   如果要在线同步，请将 wandb.entity 改为你的 team entity。")
        print("   如果只想先本地跑通，也可以在配置里设置 wandb.enable: False 或 wandb.mode: offline。")
        wandb_cfg["enable"] = False
        return False


def save_case_splits(save_dir: Path, splits: dict[str, Any] | None) -> None:
    if splits is None:
        return

    split_path = save_dir / "case_splits.yaml"
    with open(split_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(splits, file, allow_unicode=True, sort_keys=False)


def compute_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    l1_crit: nn.Module,
    roi_crit: ROILoss,
    edge_crit: EdgeLoss,
    temporal_crit: TemporalDifferenceLoss,
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parts = {
        "l1": l1_crit(preds, targets),
        "roi": roi_crit(preds, targets, mask),
        "edge": edge_crit(preds, targets),
        "temporal": temporal_crit(preds, targets),
    }

    total_loss = preds.new_tensor(0.0)
    for name, value in parts.items():
        total_loss = total_loss + float(loss_weights.get(name, 0.0)) * value
    return total_loss, parts


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prefix: str,
    l1_crit: nn.Module,
    roi_crit: ROILoss,
    edge_crit: EdgeLoss,
    temporal_crit: TemporalDifferenceLoss,
    loss_weights: dict[str, float],
    roi_margin: int,
    roi_mask_threshold: float,
) -> dict[str, float]:
    tracker = MetricTracker(prefix=prefix)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            model_inputs, targets, _, batch_size, extras = prepare_batch(batch, device)
            preds = forward_model(model, model_inputs)
            mask = extras.get("mask")

            val_loss, _ = compute_loss(
                preds=preds,
                targets=targets,
                mask=mask,
                l1_crit=l1_crit,
                roi_crit=roi_crit,
                edge_crit=edge_crit,
                temporal_crit=temporal_crit,
                loss_weights=loss_weights,
            )
            roi_l1, roi_psnr, roi_ssim, roi_n = EITMetrics.roi_metrics(
                preds,
                targets,
                mask,
                margin=roi_margin,
                mask_threshold=roi_mask_threshold,
            )

            tracker.update(
                val_loss.item(),
                EITMetrics.rmse(preds, targets),
                EITMetrics.psnr(preds, targets),
                EITMetrics.ssim(preds, targets),
                EITMetrics.cc(preds, targets),
                EITMetrics.rie(preds, targets),
                n=batch_size,
                roi_l1=roi_l1,
                roi_psnr=roi_psnr,
                roi_ssim=roi_ssim,
                roi_n=roi_n,
            )

    return tracker.avg()


def metric_fieldnames_for_prefix(prefix: str) -> list[str]:
    return [
        f"{prefix}/loss",
        f"{prefix}/rmse",
        f"{prefix}/psnr",
        f"{prefix}/ssim",
        f"{prefix}/cc",
        f"{prefix}/rie",
        f"{prefix}/roi_l1",
        f"{prefix}/roi_psnr",
        f"{prefix}/roi_ssim",
        f"{prefix}/roi_count",
    ]


def choose_monitor(train_cfg: dict, val_loaders: dict[str, DataLoader]) -> tuple[str, str, str]:
    default_split = "val_inter" if "val_inter" in val_loaders else next(iter(val_loaders.keys()))
    monitor_split = str(train_cfg.get("monitor_split", default_split))
    if monitor_split not in val_loaders:
        raise ValueError(f"monitor_split={monitor_split} 不存在，可选值: {list(val_loaders.keys())}")

    monitor_metric = str(train_cfg.get("monitor_metric", "ssim"))
    if monitor_metric not in {"loss", "rmse", "psnr", "ssim", "cc", "rie", "roi_l1", "roi_psnr", "roi_ssim"}:
        raise ValueError(
            "monitor_metric 仅支持 "
            "{'loss', 'rmse', 'psnr', 'ssim', 'cc', 'rie', 'roi_l1', 'roi_psnr', 'roi_ssim'}。"
        )
    monitor_mode = str(train_cfg.get("monitor_mode", "max")).lower()
    if monitor_mode not in {"max", "min"}:
        raise ValueError("monitor_mode 仅支持 'max' 或 'min'。")

    return monitor_split, monitor_metric, monitor_mode


def is_improved(current: float, best: float | None, mode: str, min_delta: float = 0.0) -> bool:
    if not math.isfinite(current):
        return False
    if best is None:
        return True
    if mode == "max":
        return current > best + min_delta
    return current < best - min_delta


def format_metric_value(value: float, precision: int = 4) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


def print_split_metrics(split_name: str, metrics: dict[str, float]) -> None:
    prefix = split_name
    print(
        f"  {split_name} -> "
        f"L:{format_metric_value(metrics[f'{prefix}/loss'])} | "
        f"RMSE:{format_metric_value(metrics[f'{prefix}/rmse'])} | "
        f"PSNR:{format_metric_value(metrics[f'{prefix}/psnr'], precision=2)} | "
        f"SSIM:{format_metric_value(metrics[f'{prefix}/ssim'])} | "
        f"CC:{format_metric_value(metrics[f'{prefix}/cc'])} | "
        f"ROI-L1:{format_metric_value(metrics[f'{prefix}/roi_l1'])} | "
        f"ROI-SSIM:{format_metric_value(metrics[f'{prefix}/roi_ssim'])}"
    )


def sanitize_metrics_for_wandb(metrics: dict[str, float]) -> dict[str, float]:
    sanitized: dict[str, float] = {}
    for key, value in metrics.items():
        if math.isfinite(value):
            sanitized[key] = value
    return sanitized


def train_pipeline(cfg: dict) -> None:
    save_dir = Path(cfg["save_dir"])
    if not save_dir.is_absolute():
        save_dir = PROJECT_ROOT / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    wandb_enabled = init_wandb(cfg)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = get_best_device()
    print(f"🚀 Device: {device} | Output: {save_dir}")

    train_loader, val_loaders, split_info = build_dataloaders(cfg, seed)
    save_case_splits(save_dir, split_info)

    model = get_model(cfg["model"]).to(device)

    train_cfg = cfg["train"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    scheduler_name = str(train_cfg.get("scheduler", "CosineAnnealingLR"))
    if scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg["epochs"]),
            eta_min=1e-6,
        )
    else:
        scheduler = None

    roi_cfg = train_cfg.get("roi", {})
    roi_margin = int(roi_cfg.get("margin", 6))
    roi_mask_threshold = float(roi_cfg.get("mask_threshold", 0.5))

    l1_crit = nn.L1Loss()
    roi_crit = ROILoss(mask_threshold=roi_mask_threshold)
    edge_crit = EdgeLoss(device)
    temporal_crit = TemporalDifferenceLoss()

    raw_loss_weights = train_cfg.get("loss_weights", {})
    loss_weights = {
        "l1": float(raw_loss_weights.get("l1", 1.0)),
        "roi": float(raw_loss_weights.get("roi", 0.0)),
        "edge": float(raw_loss_weights.get("edge", 0.0)),
        "temporal": float(raw_loss_weights.get("temporal", 0.0)),
    }

    monitor_split, monitor_metric, monitor_mode = choose_monitor(train_cfg, val_loaders)
    monitor_key = f"{monitor_split}/{monitor_metric}"
    best_monitor_value: float | None = None
    best_split_ssim: dict[str, float | None] = {split_name: None for split_name in val_loaders}

    early_stop_cfg = train_cfg.get("early_stopping", {})
    early_stop_enable = bool(early_stop_cfg.get("enable", False))
    early_stop_patience = int(early_stop_cfg.get("patience", 30))
    early_stop_min_delta = float(early_stop_cfg.get("min_delta", 0.0))
    early_stop_bad_epochs = 0

    csv_file = save_dir / "results.csv"
    csv_fields = [
        "epoch",
        "train/loss",
        "train/lr",
        "train/l1",
        "train/roi",
        "train/edge",
        "train/temporal",
    ]
    for split_name in val_loaders:
        csv_fields.extend(metric_fieldnames_for_prefix(split_name))
    csv_fields.append("monitor/value")

    with open(csv_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()

    logging_cfg = cfg.get("logging", {})
    save_freq = int(logging_cfg.get("save_freq", 10))
    val_freq = int(logging_cfg.get("val_freq", 1))
    epochs = int(train_cfg["epochs"])
    amp_cfg = train_cfg.get("amp", {})
    amp_enabled = device.type == "cuda" and bool(amp_cfg.get("enable", True))
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print(
        f"🎯 Monitor: {monitor_key} ({monitor_mode}) | "
        f"LossWeights: l1={loss_weights['l1']}, roi={loss_weights['roi']}, "
        f"edge={loss_weights['edge']}, temporal={loss_weights['temporal']} | "
        f"AMP={'on' if amp_enabled else 'off'}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0
        train_part_acc = {"l1": 0.0, "roi": 0.0, "edge": 0.0, "temporal": 0.0}
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{epochs}", unit="bt")

        for batch in pbar:
            model_inputs, targets, _, _, extras = prepare_batch(batch, device)
            mask = extras.get("mask")

            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.cuda.amp.autocast if amp_enabled else nullcontext
            with amp_context():
                preds = forward_model(model, model_inputs)
                loss, loss_parts = compute_loss(
                    preds=preds,
                    targets=targets,
                    mask=mask,
                    l1_crit=l1_crit,
                    roi_crit=roi_crit,
                    edge_crit=edge_crit,
                    temporal_crit=temporal_crit,
                    loss_weights=loss_weights,
                )

            if amp_enabled:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss_acc += loss.item()
            for name in train_part_acc:
                train_part_acc[name] += float(loss_parts[name].item())
            pbar.set_postfix({"L": f"{loss.item():.4f}"})

        train_loss_avg = train_loss_acc / max(len(train_loader), 1)
        train_part_avg = {f"train/{name}": train_part_acc[name] / max(len(train_loader), 1) for name in train_part_acc}
        if scheduler is not None:
            scheduler.step()

        should_validate = val_freq <= 1 or epoch % val_freq == 0 or epoch == 1 or epoch == epochs
        eval_metrics: dict[str, float] = {}
        if should_validate:
            for split_name, loader in val_loaders.items():
                split_metrics = evaluate_loader(
                    model=model,
                    loader=loader,
                    device=device,
                    prefix=split_name,
                    l1_crit=l1_crit,
                    roi_crit=roi_crit,
                    edge_crit=edge_crit,
                    temporal_crit=temporal_crit,
                    loss_weights=loss_weights,
                    roi_margin=roi_margin,
                    roi_mask_threshold=roi_mask_threshold,
                )
                eval_metrics.update(split_metrics)
                print_split_metrics(split_name, split_metrics)
        else:
            print(f"  ⏭️ Skip validation at epoch {epoch} (val_freq={val_freq})")

        row = {
            "epoch": epoch,
            "train/loss": train_loss_avg,
            "train/lr": optimizer.param_groups[0]["lr"],
            **train_part_avg,
            **eval_metrics,
            "monitor/value": eval_metrics.get(monitor_key, best_monitor_value if best_monitor_value is not None else float("nan")),
        }
        with open(csv_file, "a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=csv_fields)
            writer.writerow(row)

        primary_vis_loader = val_loaders.get(monitor_split)
        if primary_vis_loader is None:
            primary_vis_loader = next(iter(val_loaders.values()))

        wandb_images = None
        if should_validate and epoch % save_freq == 0:
            wandb_images = visualize_and_save(model, primary_vis_loader, device, save_dir, epoch)

        if wandb_enabled:
            log_data = {
                "train/loss": train_loss_avg,
                "train/lr": optimizer.param_groups[0]["lr"],
                **train_part_avg,
                **sanitize_metrics_for_wandb(eval_metrics),
                "epoch": epoch,
            }
            if wandb_images:
                log_data["examples"] = wandb_images
            wandb.log(log_data)

        if should_validate:
            for split_name in val_loaders:
                split_ssim_key = f"{split_name}/ssim"
                split_ssim = eval_metrics.get(split_ssim_key, float("nan"))
                if is_improved(split_ssim, best_split_ssim[split_name], mode="max", min_delta=0.0):
                    best_split_ssim[split_name] = split_ssim
                    torch.save(model.state_dict(), save_dir / f"best_{split_name}_ssim.pth")
                    print(f"  🔥 Best {split_name} SSIM Updated!")

            current_monitor = eval_metrics.get(monitor_key, float("nan"))
            if is_improved(current_monitor, best_monitor_value, mode=monitor_mode, min_delta=early_stop_min_delta):
                best_monitor_value = current_monitor
                early_stop_bad_epochs = 0
                torch.save(model.state_dict(), save_dir / "best.pth")
                if monitor_metric == "ssim":
                    torch.save(model.state_dict(), save_dir / "best_ssim.pth")
                print(f"  ✅ Monitor Improved: {monitor_key}={format_metric_value(current_monitor)}")
            else:
                early_stop_bad_epochs += 1

        torch.save(model.state_dict(), save_dir / "last.pth")

        if early_stop_enable and early_stop_bad_epochs >= early_stop_patience:
            print(
                f"⏹️ Early stopping triggered at epoch {epoch} | "
                f"monitor={monitor_key}, patience={early_stop_patience}"
            )
            break

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/exp003_early_cbam_unet.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    train_pipeline(cfg)
