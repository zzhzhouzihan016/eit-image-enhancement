from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform

from .bbox import BoundingBoxResult, find_lung_bbox, find_torso_bbox, to_uint8_grayscale


def _maybe_add_medsam_path() -> bool:
    env_root = os.environ.get("MEDSAM_ROOT")
    candidates: List[Path] = []
    if env_root:
        candidates.append(Path(env_root))

    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        candidate = parent / "archives" / "MedSAM-main"
        if candidate.exists():
            candidates.append(candidate)
            break

    for candidate in candidates:
        segment_anything_dir = candidate / "segment_anything"
        if segment_anything_dir.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return True

    return False


@dataclass
class MedSAMSegmentationResult:
    image_rgb: np.ndarray
    torso_bbox: List[int]
    torso_mask: np.ndarray
    torso_bbox_mask: np.ndarray
    lung_bbox: Optional[List[int]] = None
    lung_mask: Optional[np.ndarray] = None
    lung_bbox_mask: Optional[np.ndarray] = None


@dataclass
class _MaskCandidate:
    prompt_box: List[int]
    mask: np.ndarray
    score: float


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image_u8 = to_uint8_grayscale(image)
        return np.repeat(image_u8[..., None], 3, axis=-1)

    if image.ndim == 3 and image.shape[2] == 3:
        if image.dtype == np.uint8:
            return image
        image_u8 = to_uint8_grayscale(image)
        return np.repeat(image_u8[..., None], 3, axis=-1)

    raise ValueError(f"不支持的图像形状: {image.shape}")


def build_filled_mask_from_contour(contour: Optional[np.ndarray], image_shape_hw: Tuple[int, int]) -> np.ndarray:
    if contour is None:
        raise RuntimeError("torso contour 为空，无法生成填充后的 torso mask。")

    filled_mask = np.zeros(image_shape_hw, dtype=np.uint8)
    cv2.drawContours(filled_mask, [contour], contourIdx=-1, color=1, thickness=-1)
    return filled_mask


def _ensure_odd_kernel_size(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    if value % 2 == 0:
        value += 1
    return value


def _keep_largest_components(mask: np.ndarray, component_count: int) -> np.ndarray:
    binary_mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    components = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        components.append((area, label_idx))

    if not components:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    output_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    for _, label_idx in sorted(components, reverse=True)[:component_count]:
        output_mask[labels == label_idx] = 1
    return output_mask


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    binary_mask = (mask > 0).astype(np.uint8) * 255
    height, width = binary_mask.shape

    flood_filled = binary_mask.copy()
    flood_fill_workspace = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood_filled, flood_fill_workspace, seedPoint=(0, 0), newVal=255)
    holes = cv2.bitwise_not(flood_filled)
    filled = cv2.bitwise_or(binary_mask, holes)
    return (filled > 0).astype(np.uint8)


def _clean_lung_mask_for_thorax(mask: np.ndarray) -> np.ndarray:
    cleaned = _keep_largest_components(mask, component_count=2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx((cleaned * 255).astype(np.uint8), cv2.MORPH_CLOSE, close_kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)
    cleaned = _keep_largest_components(cleaned > 0, component_count=2)
    return cleaned.astype(np.uint8)


def _build_lung_hull_mask(lung_mask: np.ndarray) -> np.ndarray:
    lung_binary = (lung_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours((lung_binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("未能从 lung_mask 中提取肺部轮廓。")

    stacked_points = np.vstack(contours)
    hull = cv2.convexHull(stacked_points)
    hull_mask = np.zeros_like(lung_binary, dtype=np.uint8)
    cv2.drawContours(hull_mask, [hull], contourIdx=-1, color=1, thickness=-1)
    return hull_mask


def build_thorax_mask_from_body_and_lungs(
    body_mask: np.ndarray,
    lung_mask: np.ndarray,
    inner_margin_ratio: float = 0.067,
    lung_expand_ratio: float = 0.113,
    smoothing_ratio: float = 0.07,
) -> np.ndarray:
    body_binary = _keep_largest_components(body_mask, component_count=1)
    lung_binary = _clean_lung_mask_for_thorax(lung_mask)

    if body_binary.sum() == 0:
        raise RuntimeError("body_mask 为空，无法构建胸腔区域。")
    if lung_binary.sum() == 0:
        raise RuntimeError("lung_mask 为空，无法构建胸腔区域。")

    body_contours, _ = cv2.findContours((body_binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not body_contours:
        raise RuntimeError("未能从 body_mask 中提取外轮廓。")

    x, y, w, h = cv2.boundingRect(max(body_contours, key=cv2.contourArea))
    reference_size = max(32, min(w, h))

    inner_margin_px = max(12, int(round(reference_size * inner_margin_ratio)))
    lung_expand_px = _ensure_odd_kernel_size(int(round(reference_size * lung_expand_ratio)), minimum=25)
    smoothing_px = _ensure_odd_kernel_size(int(round(reference_size * smoothing_ratio)), minimum=15)

    body_dist = cv2.distanceTransform((body_binary * 255).astype(np.uint8), cv2.DIST_L2, 5)
    inner_region = (body_dist > float(inner_margin_px)).astype(np.uint8)

    lung_hull_mask = _build_lung_hull_mask(lung_binary)
    lung_expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lung_expand_px, lung_expand_px))
    expanded_lung_hull = cv2.dilate((lung_hull_mask * 255).astype(np.uint8), lung_expand_kernel, iterations=1)
    expanded_lung_hull = cv2.morphologyEx(expanded_lung_hull, cv2.MORPH_CLOSE, lung_expand_kernel, iterations=1)
    expanded_lung_hull = (expanded_lung_hull > 0).astype(np.uint8)

    thorax_candidate = (inner_region & expanded_lung_hull & body_binary).astype(np.uint8)
    thorax_candidate = np.where(lung_binary > 0, 1, thorax_candidate).astype(np.uint8)

    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smoothing_px, smoothing_px))
    thorax_candidate = cv2.morphologyEx((thorax_candidate * 255).astype(np.uint8), cv2.MORPH_CLOSE, smooth_kernel, iterations=1)
    thorax_candidate = cv2.morphologyEx(thorax_candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    thorax_candidate = _fill_mask_holes(thorax_candidate)
    thorax_candidate = ((thorax_candidate > 0) & (body_binary > 0)).astype(np.uint8)
    thorax_candidate = _keep_largest_components(thorax_candidate, component_count=1)

    if thorax_candidate.sum() == 0:
        raise RuntimeError("胸腔区域提取失败，结果为空。")

    return thorax_candidate.astype(np.uint8)


def _adjust_bbox(
    bbox_xyxy: List[int],
    image_shape_hw: Tuple[int, int],
    shrink_left: float = 0.0,
    shrink_top: float = 0.0,
    shrink_right: float = 0.0,
    shrink_bottom: float = 0.0,
) -> List[int]:
    height, width = image_shape_hw
    xmin, ymin, xmax, ymax = bbox_xyxy
    bbox_width = xmax - xmin + 1
    bbox_height = ymax - ymin + 1

    xmin = min(max(0, xmin + int(round(bbox_width * shrink_left))), width - 2)
    xmax = max(min(width - 1, xmax - int(round(bbox_width * shrink_right))), xmin + 1)
    ymin = min(max(0, ymin + int(round(bbox_height * shrink_top))), height - 2)
    ymax = max(min(height - 1, ymax - int(round(bbox_height * shrink_bottom))), ymin + 1)
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def _generate_torso_prompt_boxes(bbox_xyxy: List[int], image_shape_hw: Tuple[int, int]) -> List[List[int]]:
    candidates = [
        bbox_xyxy,
        _adjust_bbox(bbox_xyxy, image_shape_hw, shrink_top=0.01, shrink_bottom=0.04),
        _adjust_bbox(bbox_xyxy, image_shape_hw, shrink_left=0.03, shrink_top=0.02, shrink_right=0.03, shrink_bottom=0.05),
        _adjust_bbox(bbox_xyxy, image_shape_hw, shrink_left=0.05, shrink_top=0.03, shrink_right=0.05, shrink_bottom=0.06),
    ]

    unique_candidates: List[List[int]] = []
    seen = set()
    for candidate in candidates:
        candidate_key = tuple(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        unique_candidates.append(candidate)
    return unique_candidates


def _score_torso_mask(mask: np.ndarray) -> float:
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float("-inf")

    height, width = mask.shape
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    area_ratio = float(mask_u8.sum()) / float(height * width)
    width_ratio = float(w) / float(width)
    height_ratio = float(h) / float(height)
    center_x = float(x + (w / 2.0)) / float(width)
    center_y = float(y + (h / 2.0)) / float(height)

    score = 0.0
    score -= abs(area_ratio - 0.42)
    score -= abs(width_ratio - 0.75)
    score -= abs(height_ratio - 0.80)
    score -= 0.75 * abs(center_x - 0.50)
    score -= 1.00 * abs(center_y - 0.52)

    if y <= int(0.03 * height):
        score -= 0.20
    if (y + h) < int(0.65 * height):
        score -= 0.25

    return score


def _select_best_torso_candidate(
    segmenter: "MedSAMSegmenter",
    image_embedding: torch.Tensor,
    image_shape_hw: Tuple[int, int],
    base_bbox_xyxy: List[int],
) -> _MaskCandidate:
    best_candidate: Optional[_MaskCandidate] = None
    for prompt_box in _generate_torso_prompt_boxes(base_bbox_xyxy, image_shape_hw):
        mask = segmenter.predict_mask_from_embedding(image_embedding, image_shape_hw, prompt_box)
        score = _score_torso_mask(mask)
        candidate = _MaskCandidate(prompt_box=prompt_box, mask=mask, score=score)
        if best_candidate is None or candidate.score > best_candidate.score:
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("未能为 torso 生成有效的 MedSAM 候选结果。")

    return best_candidate


class MedSAMSegmenter:
    def __init__(self, checkpoint_path: Union[str, Path], model_type: str = "vit_b", device: Optional[str] = None):
        try:
            from segment_anything import sam_model_registry
        except ImportError as exc:
            if _maybe_add_medsam_path():
                from segment_anything import sam_model_registry
            else:
                raise ImportError(
                    "未找到 `segment_anything`。请先安装官方 SAM/MedSAM 依赖，例如："
                    "`pip install git+https://github.com/facebookresearch/segment-anything.git`，"
                    "或设置 MEDSAM_ROOT 指向 MedSAM 仓库。"
                ) from exc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, int, int]:
        image_rgb = ensure_rgb(image)
        height, width, _ = image_rgb.shape

        image_1024 = transform.resize(
            image_rgb,
            (1024, 1024),
            order=3,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)
        image_1024 = (image_1024 - image_1024.min()) / np.clip(
            image_1024.max() - image_1024.min(),
            a_min=1e-8,
            a_max=None,
        )
        image_tensor = torch.tensor(image_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        return image_rgb, image_tensor, height, width

    @torch.no_grad()
    def encode_image(self, image: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, int, int]:
        image_rgb, image_tensor, height, width = self.preprocess_image(image)
        image_embedding = self.model.image_encoder(image_tensor)
        return image_rgb, image_embedding, height, width

    @torch.no_grad()
    def predict_mask_from_embedding(
        self,
        image_embedding: torch.Tensor,
        image_shape_hw: Tuple[int, int],
        box_xyxy: List[int],
    ) -> np.ndarray:
        height, width = image_shape_hw

        box_np = np.array([box_xyxy], dtype=np.float32)
        box_1024 = box_np / np.array([width, height, width, height], dtype=np.float32) * 1024.0
        box_torch = torch.as_tensor(box_1024, dtype=torch.float32, device=self.device)[:, None, :]

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )

        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)
        low_res_pred = F.interpolate(
            low_res_pred,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        low_res_pred = low_res_pred.squeeze().cpu().numpy()
        return (low_res_pred > 0.5).astype(np.uint8)

    @torch.no_grad()
    def predict_mask_from_box(self, image: np.ndarray, box_xyxy: List[int]) -> np.ndarray:
        _, image_embedding, height, width = self.encode_image(image)
        return self.predict_mask_from_embedding(image_embedding, (height, width), box_xyxy)


def segment_torso_and_lungs(
    image: np.ndarray,
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None,
    model_type: str = "vit_b",
    predict_lungs: bool = True,
) -> MedSAMSegmentationResult:
    image_rgb = ensure_rgb(image)
    torso_bbox_result: BoundingBoxResult = find_torso_bbox(image_rgb)
    body_mask = build_filled_mask_from_contour(
        contour=torso_bbox_result.contour,
        image_shape_hw=image_rgb.shape[:2],
    )

    segmenter = MedSAMSegmenter(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device,
    )

    _, image_embedding, height, width = segmenter.encode_image(image_rgb)

    result = MedSAMSegmentationResult(
        image_rgb=image_rgb,
        torso_bbox=torso_bbox_result.bbox_xyxy,
        torso_mask=body_mask,
        torso_bbox_mask=(torso_bbox_result.binary_mask > 0).astype(np.uint8),
    )

    if predict_lungs:
        lung_bbox_result = find_lung_bbox(image_rgb, body_mask)
        lung_mask = segmenter.predict_mask_from_embedding(image_embedding, (height, width), lung_bbox_result.bbox_xyxy)
        thorax_mask = build_thorax_mask_from_body_and_lungs(body_mask=body_mask, lung_mask=lung_mask)
        result.torso_mask = thorax_mask
        result.lung_bbox = lung_bbox_result.bbox_xyxy
        result.lung_mask = lung_mask
        result.lung_bbox_mask = lung_bbox_result.binary_mask

    return result


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    overlay = image_rgb.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + color_arr * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_bbox(image_rgb: np.ndarray, bbox_xyxy: List[int], color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    output = image_rgb.copy()
    xmin, ymin, xmax, ymax = bbox_xyxy
    cv2.rectangle(output, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)
    return output
