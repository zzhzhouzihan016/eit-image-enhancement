from __future__ import annotations

import numpy as np


def window_hu(image_hu: np.ndarray, window_width: float = 1500.0, window_level: float = -600.0) -> np.ndarray:
    lower = window_level - window_width / 2.0
    upper = window_level + window_width / 2.0
    clipped = np.clip(image_hu, lower, upper)
    normalized = (clipped - lower) / max(upper - lower, 1e-6)
    return (normalized * 255.0).astype(np.uint8)


def gray_to_rgb(gray_u8: np.ndarray) -> np.ndarray:
    if gray_u8.ndim != 2:
        raise ValueError(f"输入必须是二维灰度图，当前形状: {gray_u8.shape}")
    return np.repeat(gray_u8[..., None], 3, axis=-1)


def select_middle_slice(num_slices: int) -> int:
    if num_slices <= 0:
        raise ValueError("切片数必须大于 0")
    return num_slices // 2


def select_rib_proxy_slice(volume_hu: np.ndarray) -> int:
    if volume_hu.ndim != 3:
        raise ValueError(f"输入体数据必须是 3D，当前形状: {volume_hu.shape}")

    lung_like = ((volume_hu >= -950) & (volume_hu <= -250)).sum(axis=(1, 2)).astype(np.float32)
    if lung_like.max() <= 0:
        return select_middle_slice(volume_hu.shape[0])

    candidate_mask = lung_like >= lung_like.max() * 0.30
    candidate_indices = np.where(candidate_mask)[0]
    if len(candidate_indices) == 0:
        return select_middle_slice(volume_hu.shape[0])

    proxy_rank = int(round((len(candidate_indices) - 1) * 0.65))
    return int(candidate_indices[proxy_rank])


def build_contact_indices(center_index: int, num_slices: int, count: int = 9) -> list[int]:
    count = max(3, count)
    half = count // 2
    if num_slices <= count:
        return list(range(num_slices))

    start = max(0, center_index - half)
    end = min(num_slices, start + count)
    start = max(0, end - count)
    return list(range(start, end))


def build_index_range(center_index: int, num_slices: int, range_size: int = 21) -> list[int]:
    return build_contact_indices(center_index=center_index, num_slices=num_slices, count=range_size)
