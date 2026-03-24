"""轮廓提取与有限元网格生成模块。"""

from .fem_from_mask import (
    FemFromMaskResult,
    ThoraxFemResult,
    assign_conductivities_from_mask,
    assign_lung_conductivities,
    build_fem_from_material_mask,
    build_thorax_fem_from_masks,
    build_material_mask,
    clean_lung_mask,
    clean_torso_mask,
    compute_triangle_centroids,
    extract_outer_torso_contour,
    extract_lung_contours,
    generate_unstructured_mesh,
    load_mask,
)

__all__ = [
    "FemFromMaskResult",
    "ThoraxFemResult",
    "assign_conductivities_from_mask",
    "assign_lung_conductivities",
    "build_fem_from_material_mask",
    "build_thorax_fem_from_masks",
    "build_material_mask",
    "clean_lung_mask",
    "clean_torso_mask",
    "compute_triangle_centroids",
    "extract_lung_contours",
    "extract_outer_torso_contour",
    "generate_unstructured_mesh",
    "load_mask",
]
