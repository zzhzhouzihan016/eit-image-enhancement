from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class FemFromMaskResult:
    material_mask: np.ndarray
    outer_contour: np.ndarray
    nodes: np.ndarray
    elements: np.ndarray
    centroids: np.ndarray
    conductivities: np.ndarray


@dataclass
class ThoraxFemResult:
    cleaned_torso_mask: np.ndarray
    cleaned_lung_mask: np.ndarray
    outer_contour: np.ndarray
    lung_contours: List[np.ndarray]
    nodes: np.ndarray
    elements: np.ndarray
    centroids: np.ndarray
    conductivities: np.ndarray
    region_labels: np.ndarray


def load_mask(path: Union[str, Path]) -> np.ndarray:
    mask_path = Path(path).expanduser().resolve()

    if mask_path.suffix.lower() == ".npy":
        return np.load(mask_path)

    if mask_path.suffix.lower() == ".npz":
        loaded = np.load(mask_path)
        if not loaded.files:
            raise ValueError(f"掩膜文件为空: {mask_path}")
        return loaded[loaded.files[0]]

    image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"无法读取掩膜文件: {mask_path}")
    return image


def ensure_binary_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask_u8 = mask.astype(np.uint8)
    if mask_u8.max() <= 1:
        return np.where(mask_u8 > 0, 255, 0).astype(np.uint8)
    return np.where(mask_u8 > threshold, 255, 0).astype(np.uint8)


def build_material_mask(torso_mask: np.ndarray, lung_mask: np.ndarray) -> np.ndarray:
    torso_binary = ensure_binary_mask(torso_mask) > 0
    lung_binary = ensure_binary_mask(lung_mask) > 0
    material_binary = torso_binary & (~lung_binary)
    return np.where(material_binary, 255, 0).astype(np.uint8)


def keep_largest_components(mask: np.ndarray, component_count: int) -> np.ndarray:
    binary_mask = ensure_binary_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary_mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary_mask)

    components = []
    for label_idx in range(1, num_labels):
        components.append((int(stats[label_idx, cv2.CC_STAT_AREA]), label_idx))

    kept_mask = np.zeros_like(binary_mask)
    for _, label_idx in sorted(components, reverse=True)[:component_count]:
        kept_mask[labels == label_idx] = 255
    return kept_mask


def clean_torso_mask(
    mask: np.ndarray,
    close_kernel_size: int = 9,
    open_kernel_size: int = 5,
) -> np.ndarray:
    cleaned = keep_largest_components(mask, component_count=1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return keep_largest_components(cleaned, component_count=1)


def clean_lung_mask(
    mask: np.ndarray,
    close_kernel_size: int = 7,
    open_kernel_size: int = 5,
) -> np.ndarray:
    cleaned = keep_largest_components(mask, component_count=2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return keep_largest_components(cleaned, component_count=2)


def _segment_orientation(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    return float((point_b[0] - point_a[0]) * (point_c[1] - point_a[1]) - (point_b[1] - point_a[1]) * (point_c[0] - point_a[0]))


def _on_segment(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> bool:
    return (
        min(point_a[0], point_b[0]) <= point_c[0] <= max(point_a[0], point_b[0])
        and min(point_a[1], point_b[1]) <= point_c[1] <= max(point_a[1], point_b[1])
    )


def _segments_intersect(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray, point_d: np.ndarray) -> bool:
    orient_1 = _segment_orientation(point_a, point_b, point_c)
    orient_2 = _segment_orientation(point_a, point_b, point_d)
    orient_3 = _segment_orientation(point_c, point_d, point_a)
    orient_4 = _segment_orientation(point_c, point_d, point_b)

    if ((orient_1 > 0 and orient_2 < 0) or (orient_1 < 0 and orient_2 > 0)) and (
        (orient_3 > 0 and orient_4 < 0) or (orient_3 < 0 and orient_4 > 0)
    ):
        return True

    epsilon = 1e-8
    if abs(orient_1) < epsilon and _on_segment(point_a, point_b, point_c):
        return True
    if abs(orient_2) < epsilon and _on_segment(point_a, point_b, point_d):
        return True
    if abs(orient_3) < epsilon and _on_segment(point_c, point_d, point_a):
        return True
    if abs(orient_4) < epsilon and _on_segment(point_c, point_d, point_b):
        return True
    return False


def _polygon_has_self_intersection(points: np.ndarray) -> bool:
    point_count = points.shape[0]
    if point_count < 4:
        return False

    for start_idx in range(point_count):
        segment_a_start = points[start_idx]
        segment_a_end = points[(start_idx + 1) % point_count]
        for other_idx in range(start_idx + 1, point_count):
            if other_idx in {start_idx, (start_idx + 1) % point_count}:
                continue
            if (other_idx + 1) % point_count in {start_idx, (start_idx + 1) % point_count}:
                continue

            segment_b_start = points[other_idx]
            segment_b_end = points[(other_idx + 1) % point_count]
            if _segments_intersect(segment_a_start, segment_a_end, segment_b_start, segment_b_end):
                return True

    return False


def polygon_signed_area(points: np.ndarray) -> float:
    points_xy = points.astype(np.float64)
    return 0.5 * float(
        np.sum(points_xy[:, 0] * np.roll(points_xy[:, 1], -1) - np.roll(points_xy[:, 0], -1) * points_xy[:, 1])
    )


def ensure_counterclockwise(points: np.ndarray) -> np.ndarray:
    return points if polygon_signed_area(points) > 0 else points[::-1].copy()


def ensure_clockwise(points: np.ndarray) -> np.ndarray:
    return points if polygon_signed_area(points) < 0 else points[::-1].copy()


def simplify_contour_points(contour: np.ndarray, epsilon_ratio: float) -> np.ndarray:
    perimeter = cv2.arcLength(contour, True)
    candidate_ratios = []
    for ratio in [epsilon_ratio, 0.008, 0.005, 0.003, 0.002]:
        ratio = float(ratio)
        if ratio > 0 and ratio not in candidate_ratios:
            candidate_ratios.append(ratio)

    for ratio in candidate_ratios:
        epsilon = max(1.0, ratio * perimeter)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        contour_points = simplified[:, 0, :].astype(np.float64)
        if contour_points.shape[0] < 3:
            continue
        if not _polygon_has_self_intersection(contour_points):
            return contour_points

    contour_points = contour[:, 0, :].astype(np.float64)
    if contour_points.shape[0] < 3:
        raise RuntimeError("轮廓简化后顶点少于 3 个，无法生成网格。")
    return contour_points


def extract_outer_torso_contour(mask: np.ndarray, epsilon_ratio: float = 0.005) -> np.ndarray:
    binary_mask = ensure_binary_mask(mask)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("未能从掩膜中提取 torso 外轮廓。")

    outer_contour = max(contours, key=cv2.contourArea)
    return simplify_contour_points(outer_contour, epsilon_ratio=epsilon_ratio)


def extract_lung_contours(mask: np.ndarray, epsilon_ratio: float = 0.01) -> List[np.ndarray]:
    binary_mask = ensure_binary_mask(mask)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) < 2:
        raise RuntimeError("未能提取到两片肺叶的外轮廓。")

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    simplified_contours = [simplify_contour_points(contour, epsilon_ratio=epsilon_ratio) for contour in sorted_contours]
    simplified_contours.sort(key=lambda points: float(points[:, 0].mean()))
    return simplified_contours


def _to_cartesian_coordinates(points_xy: np.ndarray, image_height: int) -> np.ndarray:
    points_cart = points_xy.astype(np.float64).copy()
    points_cart[:, 1] = (image_height - 1) - points_cart[:, 1]
    return points_cart


def _to_image_coordinates(points_xy: np.ndarray, image_height: int) -> np.ndarray:
    points_img = points_xy.astype(np.float64).copy()
    points_img[:, 1] = (image_height - 1) - points_img[:, 1]
    return points_img


def generate_unstructured_mesh(
    contour_points: np.ndarray,
    mask_shape_hw: Tuple[int, int],
    mesh_size: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import pygmsh
    except ImportError as exc:
        raise ImportError(
            "未安装 `pygmsh`。请先安装 `pygmsh` 和其依赖 `gmsh`，例如：`pip install pygmsh gmsh`。"
        ) from exc

    height, _ = mask_shape_hw
    contour_cart = _to_cartesian_coordinates(contour_points, image_height=height)
    polygon_points = np.column_stack(
        [
            contour_cart[:, 0],
            contour_cart[:, 1],
            np.zeros(contour_cart.shape[0], dtype=np.float64),
        ]
    )

    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(polygon_points, mesh_size=float(mesh_size))
        mesh = geom.generate_mesh(dim=2)

    triangle_cells = mesh.cells_dict.get("triangle")
    if triangle_cells is None or len(triangle_cells) == 0:
        raise RuntimeError("pygmsh 未生成三角单元。")

    nodes_cart = mesh.points[:, :2].astype(np.float64)
    nodes_img = _to_image_coordinates(nodes_cart, image_height=height)
    elements = triangle_cells.astype(np.int32)
    return nodes_img, elements


def compute_triangle_centroids(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    return nodes[elements].mean(axis=1).astype(np.float64)


def assign_conductivities_from_mask(
    centroids: np.ndarray,
    material_mask: np.ndarray,
    white_conductivity: float = 0.38,
    black_conductivity: float = 0.24,
) -> np.ndarray:
    binary_mask = ensure_binary_mask(material_mask)
    height, width = binary_mask.shape

    pixel_x = np.rint(centroids[:, 0]).astype(np.int32)
    pixel_y = np.rint(centroids[:, 1]).astype(np.int32)
    pixel_x = np.clip(pixel_x, 0, width - 1)
    pixel_y = np.clip(pixel_y, 0, height - 1)

    pixel_values = binary_mask[pixel_y, pixel_x]
    return np.where(pixel_values > 0, white_conductivity, black_conductivity).astype(np.float32)


def build_fem_from_material_mask(
    material_mask: np.ndarray,
    epsilon_ratio: float = 0.005,
    mesh_size: float = 12.0,
    white_conductivity: float = 0.38,
    black_conductivity: float = 0.24,
) -> FemFromMaskResult:
    binary_mask = ensure_binary_mask(material_mask)
    outer_contour = extract_outer_torso_contour(binary_mask, epsilon_ratio=epsilon_ratio)
    nodes, elements = generate_unstructured_mesh(
        contour_points=outer_contour,
        mask_shape_hw=binary_mask.shape,
        mesh_size=mesh_size,
    )
    centroids = compute_triangle_centroids(nodes, elements)
    conductivities = assign_conductivities_from_mask(
        centroids=centroids,
        material_mask=binary_mask,
        white_conductivity=white_conductivity,
        black_conductivity=black_conductivity,
    )

    return FemFromMaskResult(
        material_mask=binary_mask,
        outer_contour=outer_contour,
        nodes=nodes,
        elements=elements,
        centroids=centroids,
        conductivities=conductivities,
    )


def _add_occ_surface_from_contour(contour_points: np.ndarray, mesh_size: float) -> int:
    import gmsh

    point_tags = []
    for x_coord, y_coord in contour_points:
        point_tags.append(gmsh.model.occ.addPoint(float(x_coord), float(y_coord), 0.0, float(mesh_size)))

    line_tags = []
    for idx in range(len(point_tags)):
        start_tag = point_tags[idx]
        end_tag = point_tags[(idx + 1) % len(point_tags)]
        line_tags.append(gmsh.model.occ.addLine(start_tag, end_tag))

    curve_loop = gmsh.model.occ.addCurveLoop(line_tags)
    return gmsh.model.occ.addPlaneSurface([curve_loop])


def _collect_gmsh_triangles(image_height: int) -> Tuple[np.ndarray, np.ndarray]:
    import gmsh

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    if len(node_tags) == 0:
        raise RuntimeError("gmsh 未生成任何节点。")

    nodes_cart = node_coords.reshape(-1, 3)[:, :2].astype(np.float64)
    tag_to_index = {int(node_tag): idx for idx, node_tag in enumerate(node_tags.tolist())}

    triangle_blocks = []
    for dim, surface_tag in gmsh.model.getEntities(2):
        element_types, _, element_nodes = gmsh.model.mesh.getElements(dim, surface_tag)
        for element_type, node_block in zip(element_types, element_nodes):
            if int(element_type) != 2:
                continue
            triangles = np.asarray(node_block, dtype=np.int64).reshape(-1, 3)
            triangle_indices = np.vectorize(tag_to_index.__getitem__)(triangles).astype(np.int32)
            triangle_blocks.append(triangle_indices)

    if not triangle_blocks:
        raise RuntimeError("gmsh 未生成任何三角单元。")

    elements = np.vstack(triangle_blocks).astype(np.int32)
    nodes_img = _to_image_coordinates(nodes_cart, image_height=image_height)
    return nodes_img, elements


def assign_lung_conductivities(
    centroids: np.ndarray,
    lung_mask: np.ndarray,
    soft_tissue_conductivity: float = 0.38,
    lung_conductivity: float = 0.24,
) -> Tuple[np.ndarray, np.ndarray]:
    lung_binary = ensure_binary_mask(lung_mask)
    height, width = lung_binary.shape

    pixel_x = np.rint(centroids[:, 0]).astype(np.int32)
    pixel_y = np.rint(centroids[:, 1]).astype(np.int32)
    pixel_x = np.clip(pixel_x, 0, width - 1)
    pixel_y = np.clip(pixel_y, 0, height - 1)

    is_lung = lung_binary[pixel_y, pixel_x] > 0
    conductivities = np.where(is_lung, lung_conductivity, soft_tissue_conductivity).astype(np.float32)
    region_labels = np.where(is_lung, 1, 0).astype(np.int32)
    return conductivities, region_labels


def filter_elements_inside_mask(
    nodes: np.ndarray,
    elements: np.ndarray,
    reference_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    centroids = compute_triangle_centroids(nodes, elements)
    binary_mask = ensure_binary_mask(reference_mask)
    height, width = binary_mask.shape

    pixel_x = np.rint(centroids[:, 0]).astype(np.int32)
    pixel_y = np.rint(centroids[:, 1]).astype(np.int32)
    pixel_x = np.clip(pixel_x, 0, width - 1)
    pixel_y = np.clip(pixel_y, 0, height - 1)
    keep = binary_mask[pixel_y, pixel_x] > 0
    return elements[keep], keep


def build_thorax_fem_from_masks(
    torso_mask: np.ndarray,
    lung_mask: np.ndarray,
    torso_epsilon_ratio: float = 0.003,
    lung_epsilon_ratio: float = 0.01,
    outer_mesh_size: float = 12.0,
    lung_mesh_size: float = 8.0,
    soft_tissue_conductivity: float = 0.38,
    lung_conductivity: float = 0.24,
) -> ThoraxFemResult:
    try:
        import gmsh
    except ImportError as exc:
        raise ImportError(
            "未安装 `gmsh`。请先安装 `gmsh` 和 `pygmsh`，例如：`pip install gmsh pygmsh`。"
        ) from exc

    cleaned_torso_mask = clean_torso_mask(torso_mask)
    cleaned_lung_mask = clean_lung_mask(lung_mask)

    outer_contour_img = extract_outer_torso_contour(cleaned_torso_mask, epsilon_ratio=torso_epsilon_ratio)
    lung_contours_img = extract_lung_contours(cleaned_lung_mask, epsilon_ratio=lung_epsilon_ratio)

    image_height = cleaned_torso_mask.shape[0]
    outer_contour_cart = ensure_counterclockwise(_to_cartesian_coordinates(outer_contour_img, image_height=image_height))
    lung_contours_cart = [
        ensure_clockwise(_to_cartesian_coordinates(lung_contour, image_height=image_height))
        for lung_contour in lung_contours_img
    ]

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("thorax_fem")

        outer_surface = _add_occ_surface_from_contour(outer_contour_cart, mesh_size=outer_mesh_size)
        lung_surfaces = [
            _add_occ_surface_from_contour(lung_contour, mesh_size=lung_mesh_size)
            for lung_contour in lung_contours_cart
        ]

        gmsh.model.occ.fragment([(2, outer_surface)], [(2, surface_tag) for surface_tag in lung_surfaces])
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)

        nodes_img, elements = _collect_gmsh_triangles(image_height=image_height)
    finally:
        gmsh.finalize()

    elements, _ = filter_elements_inside_mask(
        nodes=nodes_img,
        elements=elements,
        reference_mask=cleaned_torso_mask,
    )
    centroids = compute_triangle_centroids(nodes_img, elements)
    conductivities, region_labels = assign_lung_conductivities(
        centroids=centroids,
        lung_mask=cleaned_lung_mask,
        soft_tissue_conductivity=soft_tissue_conductivity,
        lung_conductivity=lung_conductivity,
    )

    return ThoraxFemResult(
        cleaned_torso_mask=cleaned_torso_mask,
        cleaned_lung_mask=cleaned_lung_mask,
        outer_contour=outer_contour_img,
        lung_contours=lung_contours_img,
        nodes=nodes_img,
        elements=elements,
        centroids=centroids,
        conductivities=conductivities,
        region_labels=region_labels,
    )
