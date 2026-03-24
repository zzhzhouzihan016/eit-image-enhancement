from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _parse_sigma_range(value: float | Sequence[float] | None) -> tuple[float, float]:
    if value is None:
        return 1.0, 1.0
    if isinstance(value, (int, float)):
        sigma = float(value)
        return sigma, sigma
    if len(value) != 2:
        raise ValueError("target_blur_sigma 必须是浮点数或长度为 2 的序列。")
    low, high = float(value[0]), float(value[1])
    return min(low, high), max(low, high)


def _build_gaussian_kernel(
    kernel_size: int,
    sigma: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel_1d = torch.exp(-(coords**2) / max(2.0 * sigma * sigma, 1e-6))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.unsqueeze(0).unsqueeze(0)


def _apply_gaussian_blur(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"高斯模糊输入期望为 [C, H, W]，实际得到 {tuple(image.shape)}")

    kernel = _build_gaussian_kernel(
        kernel_size=kernel_size,
        sigma=sigma,
        dtype=image.dtype,
        device=image.device,
    )
    channels = image.shape[0]
    weight = kernel.expand(channels, 1, kernel_size, kernel_size)
    blurred = F.conv2d(
        image.unsqueeze(0),
        weight,
        padding=kernel_size // 2,
        groups=channels,
    )
    return blurred.squeeze(0)


def _rotate_sequence(tensor: torch.Tensor, angle_deg: float) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"旋转输入期望为 [N, H, W]，实际得到 {tuple(tensor.shape)}")
    if abs(angle_deg) < 1e-6:
        return tensor

    batch = tensor.unsqueeze(1)
    angle_rad = np.deg2rad(angle_deg)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    theta = batch.new_tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
    ).unsqueeze(0).repeat(batch.shape[0], 1, 1)

    grid = F.affine_grid(theta, batch.size(), align_corners=False)
    rotated = F.grid_sample(
        batch,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return rotated.squeeze(1)


class EITSequenceDataset(Dataset):
    def __init__(
        self,
        npz_path,
        n_frames: int = 5,
        frames_per_seq: int | None = None,
        target_size: tuple[int, int] = (176, 256),
        use_augmentation: bool = False,
        sample_ids: Sequence[int] | None = None,
        apply_target_blur: bool = False,
        target_blur_kernel_size: int = 5,
        target_blur_sigma: float | Sequence[float] | None = None,
        augment_hflip_prob: float = 0.5,
        augment_rotate_prob: float = 0.3,
        augment_rotate_deg: float = 5.0,
    ):
        super().__init__()
        self.use_augmentation = use_augmentation
        self.target_size = tuple(target_size)
        self.apply_target_blur = apply_target_blur
        self.target_blur_kernel_size = int(target_blur_kernel_size)
        self.target_blur_sigma = _parse_sigma_range(target_blur_sigma)
        self.augment_hflip_prob = float(augment_hflip_prob)
        self.augment_rotate_prob = float(augment_rotate_prob)
        self.augment_rotate_deg = float(augment_rotate_deg)

        if self.target_blur_kernel_size <= 0 or self.target_blur_kernel_size % 2 == 0:
            raise ValueError("target_blur_kernel_size 必须是正奇数。")

        print(f"📂 Loading dataset: {npz_path}")
        raw = np.load(npz_path)

        if "input_data" in raw:
            in_data = raw["input_data"]
            gt_data = raw["target_data"]
        else:
            in_data = raw["input"]
            gt_data = raw["target"]

        if in_data.ndim == 4:
            num_samples, detected_frames_per_seq, height, width = in_data.shape
            if frames_per_seq is None:
                frames_per_seq = detected_frames_per_seq
                print(f"   ℹ️ 自动检测到每样本帧数: {frames_per_seq}")
            elif detected_frames_per_seq != frames_per_seq:
                print(
                    "   ⚠️ 警告: 数据实际帧数 "
                    f"({detected_frames_per_seq}) 与参数 ({frames_per_seq}) 不一致，已自动修正。"
                )
                frames_per_seq = detected_frames_per_seq

            in_data = in_data.reshape(-1, height, width)
            gt_data = gt_data.reshape(-1, height, width)
        elif in_data.ndim == 3:
            if frames_per_seq is None:
                frames_per_seq = in_data.shape[0]
                print(f"   ℹ️ 检测到单序列输入，frames_per_seq 自动设为 {frames_per_seq}")
            num_samples = max(in_data.shape[0] // frames_per_seq, 1)
        else:
            raise ValueError(f"❌ 不支持的数据维度: {in_data.shape}")

        self.inputs = torch.from_numpy(in_data).float()
        self.targets = torch.from_numpy(gt_data).float()

        print(f"   Original Size: {self.inputs.shape}")
        self.inputs = self._resize_tensor(self.inputs, self.target_size)
        self.targets = self._resize_tensor(self.targets, self.target_size)
        print(f"   Resized Size: {self.inputs.shape}")

        self.n_frames = int(n_frames)
        self.frames_per_seq = int(frames_per_seq)
        self.total_frames = len(self.inputs)

        if self.frames_per_seq < self.n_frames:
            raise ValueError(f"❌ frames_per_seq={self.frames_per_seq} 小于 n_frames={self.n_frames}")
        if self.total_frames % self.frames_per_seq != 0:
            raise ValueError(
                f"❌ total_frames={self.total_frames} 不能被 frames_per_seq={self.frames_per_seq} 整除。"
            )

        self.num_samples = self.total_frames // self.frames_per_seq
        if sample_ids is None:
            self.sample_ids = tuple(range(self.num_samples))
        else:
            normalized_ids = sorted({int(sample_id) for sample_id in sample_ids})
            if not normalized_ids:
                raise ValueError("❌ sample_ids 为空，没有可用样本。")
            if normalized_ids[0] < 0 or normalized_ids[-1] >= self.num_samples:
                raise ValueError(f"❌ sample_ids 超出范围，合法范围是 [0, {self.num_samples - 1}]。")
            self.sample_ids = tuple(normalized_ids)

        self.valid_indices: list[int] = []
        for sample_id in self.sample_ids:
            seq_start = sample_id * self.frames_per_seq
            valid_starts = range(seq_start, seq_start + self.frames_per_seq - self.n_frames + 1)
            self.valid_indices.extend(valid_starts)

        print(
            "✅ Dataset Ready: "
            f"TotalSamples={self.num_samples}, SelectedSamples={len(self.sample_ids)}, "
            f"ValidSequences={len(self.valid_indices)}"
        )

    @staticmethod
    def _resize_tensor(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        resized = F.interpolate(
            tensor.unsqueeze(1),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(1)

    def __len__(self):
        return len(self.valid_indices)

    def _maybe_blur_target(self, target_img: torch.Tensor) -> torch.Tensor:
        if not self.apply_target_blur:
            return target_img

        sigma_min, sigma_max = self.target_blur_sigma
        sigma = sigma_min if sigma_min == sigma_max else random.uniform(sigma_min, sigma_max)
        return _apply_gaussian_blur(target_img, self.target_blur_kernel_size, sigma)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        in_seq = self.inputs[real_idx : real_idx + self.n_frames]
        mid_idx = real_idx + (self.n_frames // 2)
        target_img = self.targets[mid_idx].unsqueeze(0)

        target_img = self._maybe_blur_target(target_img)

        if self.use_augmentation:
            if random.random() < self.augment_hflip_prob:
                in_seq = torch.flip(in_seq, dims=[-1])
                target_img = torch.flip(target_img, dims=[-1])

            if random.random() < self.augment_rotate_prob and self.augment_rotate_deg > 0:
                angle = random.uniform(-self.augment_rotate_deg, self.augment_rotate_deg)
                in_seq = _rotate_sequence(in_seq, angle)
                target_img = _rotate_sequence(target_img, angle)

        return in_seq, target_img
