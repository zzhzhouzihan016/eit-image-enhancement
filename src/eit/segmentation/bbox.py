from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class BoundingBoxResult:
    bbox_xyxy: List[int]
    binary_mask: np.ndarray
    contour: Optional[np.ndarray] = None


def to_uint8_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"不支持的图像形状: {image.shape}")

    if gray.dtype == np.uint8:
        return gray

    gray = gray.astype(np.float32)
    gray_min = float(gray.min())
    gray_max = float(gray.max())
    if gray_max <= gray_min:
        return np.zeros_like(gray, dtype=np.uint8)
    gray = (gray - gray_min) / (gray_max - gray_min)
    return (gray * 255.0).astype(np.uint8)


def _pad_bbox(x: int, y: int, w: int, h: int, width: int, height: int, padding: int) -> List[int]:
    xmin = max(0, x - padding)
    ymin = max(0, y - padding)
    xmax = min(width - 1, x + w - 1 + padding)
    ymax = min(height - 1, y + h - 1 + padding)
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def _estimate_air_intensity(gray: np.ndarray, corner_patch_size: int = 32) -> float:
    patch_size = max(8, min(corner_patch_size, gray.shape[0] // 6, gray.shape[1] // 6))
    corner_patches = [
        gray[:patch_size, :patch_size],
        gray[:patch_size, -patch_size:],
        gray[-patch_size:, :patch_size],
        gray[-patch_size:, -patch_size:],
    ]
    return float(np.median(np.concatenate([patch.ravel() for patch in corner_patches])))


def _shrink_bbox_for_prompt(
    bbox_xyxy: List[int],
    width: int,
    height: int,
    shrink_ratio_x: float,
    shrink_ratio_y_top: float,
    shrink_ratio_y_bottom: float,
) -> List[int]:
    xmin, ymin, xmax, ymax = bbox_xyxy
    bbox_width = xmax - xmin + 1
    bbox_height = ymax - ymin + 1

    shrink_x = max(4, int(round(bbox_width * shrink_ratio_x)))
    shrink_y_top = max(4, int(round(bbox_height * shrink_ratio_y_top)))
    shrink_y_bottom = max(4, int(round(bbox_height * shrink_ratio_y_bottom)))

    xmin = min(max(0, xmin + shrink_x), width - 2)
    xmax = max(min(width - 1, xmax - shrink_x), xmin + 1)
    ymin = min(max(0, ymin + shrink_y_top), height - 2)
    ymax = max(min(height - 1, ymax - shrink_y_bottom), ymin + 1)
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def find_torso_bbox(
    image: np.ndarray,
    air_margin: int = 25,
    padding: int = 10,
    close_kernel_size: int = 21,
    open_kernel_size: int = 5,
    border_touch_tolerance: int = 4,
    prompt_shrink_ratio_x: float = 0.06,
    prompt_shrink_ratio_y_top: float = 0.02,
    prompt_shrink_ratio_y_bottom: float = 0.03,
) -> BoundingBoxResult:
    gray = to_uint8_grayscale(image)
    height, width = gray.shape

    air_intensity = _estimate_air_intensity(gray)
    threshold = min(254, max(1, int(round(air_intensity + air_margin))))

    binary_mask = (gray > threshold).astype(np.uint8) * 255
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))

    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("未能从图像中找到 torso 轮廓，请检查输入图像是否已经去掉黑背景。")

    torso_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(torso_contour)
    bbox_xyxy = _pad_bbox(x, y, w, h, width, height, padding)

    touches_border = (
        x <= border_touch_tolerance
        or y <= border_touch_tolerance
        or (x + w) >= (width - border_touch_tolerance)
        or (y + h) >= (height - border_touch_tolerance)
    )
    covers_most_image = (w / width) >= 0.95 or (h / height) >= 0.90
    if touches_border or covers_most_image:
        bbox_xyxy = _shrink_bbox_for_prompt(
            bbox_xyxy=bbox_xyxy,
            width=width,
            height=height,
            shrink_ratio_x=prompt_shrink_ratio_x,
            shrink_ratio_y_top=prompt_shrink_ratio_y_top,
            shrink_ratio_y_bottom=prompt_shrink_ratio_y_bottom,
        )

    return BoundingBoxResult(bbox_xyxy=bbox_xyxy, binary_mask=binary_mask, contour=torso_contour)


def find_lung_bbox(
    image: np.ndarray,
    torso_mask: np.ndarray,
    padding: int = 8,
    dark_percentile: float = 45.0,
    min_component_area_ratio: float = 0.002,
) -> BoundingBoxResult:
    gray = to_uint8_grayscale(image)
    height, width = gray.shape
    torso_binary = (torso_mask > 0).astype(np.uint8)

    if torso_binary.sum() == 0:
        raise RuntimeError("torso_mask 为空，无法估计肺部候选框。")

    torso_values = gray[torso_binary > 0]
    threshold = np.percentile(torso_values, dark_percentile)
    lung_candidates = ((gray <= threshold) & (torso_binary > 0)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lung_candidates = cv2.morphologyEx(lung_candidates, cv2.MORPH_OPEN, kernel, iterations=1)
    lung_candidates = cv2.morphologyEx(lung_candidates, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lung_candidates, connectivity=8)
    min_area = height * width * min_component_area_ratio

    kept_labels = []
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        x = stats[label_idx, cv2.CC_STAT_LEFT]
        y = stats[label_idx, cv2.CC_STAT_TOP]
        w = stats[label_idx, cv2.CC_STAT_WIDTH]
        h = stats[label_idx, cv2.CC_STAT_HEIGHT]

        if area < min_area:
            continue
        if y <= 0 or (y + h) >= height:
            continue
        kept_labels.append((label_idx, area))

    if not kept_labels:
        raise RuntimeError("未能从 torso 内部自动估计肺部候选区域。")

    kept_labels = [label for label, _ in sorted(kept_labels, key=lambda item: item[1], reverse=True)[:2]]
    lung_mask = np.isin(labels, kept_labels).astype(np.uint8) * 255

    ys, xs = np.where(lung_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("肺部候选区域为空，无法生成肺部候选框。")

    x = int(xs.min())
    y = int(ys.min())
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    bbox_xyxy = _pad_bbox(x, y, w, h, width, height, padding)

    return BoundingBoxResult(bbox_xyxy=bbox_xyxy, binary_mask=lung_mask, contour=None)
