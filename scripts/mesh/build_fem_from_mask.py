import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eit.mesh import build_fem_from_material_mask, build_material_mask, build_thorax_fem_from_masks, load_mask


def resolve_default_output_dir(mask_path: Optional[Path], torso_mask_path: Optional[Path]) -> Path:
    reference_path = mask_path or torso_mask_path
    if reference_path is None:
        return PROJECT_ROOT / "data" / "interim" / "ct_cases" / "mesh" / "fem"

    if reference_path.parent.name == "medsam" and reference_path.parent.parent.name == "segmentation":
        case_dir = reference_path.parent.parent.parent
        return case_dir / "mesh" / "fem"

    return reference_path.parent / "mesh_fem"


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8))


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def save_mesh_preview(
    background_mask: np.ndarray,
    outer_contour: np.ndarray,
    lung_contours: list[np.ndarray],
    nodes: np.ndarray,
    elements: np.ndarray,
    path: Path,
) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(background_mask, cmap="gray")
    plt.triplot(nodes[:, 0], nodes[:, 1], elements, color="#00D5FF", linewidth=0.35, alpha=0.8)
    plt.plot(
        np.r_[outer_contour[:, 0], outer_contour[0, 0]],
        np.r_[outer_contour[:, 1], outer_contour[0, 1]],
        color="yellow",
        linewidth=2.0,
    )
    for lung_contour in lung_contours:
        plt.plot(
            np.r_[lung_contour[:, 0], lung_contour[0, 0]],
            np.r_[lung_contour[:, 1], lung_contour[0, 1]],
            color="#FF7F0E",
            linewidth=1.8,
        )
    plt.title("Thorax + lung FEM mesh")
    plt.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_conductivity_preview(
    nodes: np.ndarray,
    elements: np.ndarray,
    conductivities: np.ndarray,
    path: Path,
) -> None:
    plt.figure(figsize=(8, 8))
    trip = plt.tripcolor(
        nodes[:, 0],
        nodes[:, 1],
        elements,
        facecolors=conductivities,
        cmap="coolwarm",
        edgecolors="k",
        linewidth=0.08,
    )
    plt.gca().invert_yaxis()
    plt.colorbar(trip, fraction=0.046, pad=0.04, label="Conductivity")
    plt.title("Element conductivity mapping")
    plt.axis("equal")
    plt.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 torso/lung 掩膜构建带双肺内边界的 EIT 2D 有限元网格。")
    parser.add_argument("--mask", type=str, default=None, help="输入的材料掩膜：torso 白、lung/background 黑。仅保留兼容模式。")
    parser.add_argument("--torso-mask", type=str, default=None, help="torso 掩膜路径。推荐使用。")
    parser.add_argument("--lung-mask", type=str, default=None, help="lung 掩膜路径。推荐使用。")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录；默认保存到病例目录下的 mesh/fem/")
    parser.add_argument("--epsilon-ratio", type=float, default=0.005, help="兼容模式轮廓简化系数，默认 0.005")
    parser.add_argument("--mesh-size", type=float, default=12.0, help="兼容模式网格目标边长，单位像素，默认 12")
    parser.add_argument("--white-conductivity", type=float, default=0.38, help="非肺区域电导率，默认 0.38")
    parser.add_argument("--black-conductivity", type=float, default=0.24, help="肺区域电导率，默认 0.24")
    parser.add_argument("--torso-epsilon-ratio", type=float, default=0.003, help="torso 外轮廓简化系数，默认 0.003")
    parser.add_argument("--lung-epsilon-ratio", type=float, default=0.010, help="肺轮廓简化系数，默认 0.010")
    parser.add_argument("--outer-mesh-size", type=float, default=12.0, help="胸廓区域目标网格边长，默认 12")
    parser.add_argument("--lung-mesh-size", type=float, default=8.0, help="肺边界附近目标网格边长，默认 8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mask is None and (args.torso_mask is None or args.lung_mask is None):
        raise ValueError("请提供 `--mask`，或同时提供 `--torso-mask` 和 `--lung-mask`。")

    mask_path = Path(args.mask).expanduser().resolve() if args.mask else None
    torso_mask_path = Path(args.torso_mask).expanduser().resolve() if args.torso_mask else None
    lung_mask_path = Path(args.lung_mask).expanduser().resolve() if args.lung_mask else None

    if mask_path is not None:
        material_mask = load_mask(mask_path)
        mask_source = "combined_mask"
    else:
        torso_mask = load_mask(torso_mask_path)
        lung_mask = load_mask(lung_mask_path)
        material_mask = build_material_mask(torso_mask=torso_mask, lung_mask=lung_mask)
        mask_source = "torso_minus_lung"

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else resolve_default_output_dir(mask_path=mask_path, torso_mask_path=torso_mask_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if torso_mask_path is not None and lung_mask_path is not None:
        remove_if_exists(output_dir / "material_mask.npy")
        remove_if_exists(output_dir / "material_mask.png")
        thorax_result = build_thorax_fem_from_masks(
            torso_mask=torso_mask,
            lung_mask=lung_mask,
            torso_epsilon_ratio=args.torso_epsilon_ratio,
            lung_epsilon_ratio=args.lung_epsilon_ratio,
            outer_mesh_size=args.outer_mesh_size,
            lung_mesh_size=args.lung_mesh_size,
            soft_tissue_conductivity=args.white_conductivity,
            lung_conductivity=args.black_conductivity,
        )

        np.save(output_dir / "cleaned_torso_mask.npy", thorax_result.cleaned_torso_mask.astype(np.uint8))
        np.save(output_dir / "cleaned_lung_mask.npy", thorax_result.cleaned_lung_mask.astype(np.uint8))
        np.save(output_dir / "outer_contour.npy", thorax_result.outer_contour.astype(np.float64))
        for idx, contour in enumerate(thorax_result.lung_contours):
            np.save(output_dir / f"lung_contour_{idx:02d}.npy", contour.astype(np.float64))
        np.save(output_dir / "nodes.npy", thorax_result.nodes.astype(np.float64))
        np.save(output_dir / "elements.npy", thorax_result.elements.astype(np.int32))
        np.save(output_dir / "triangle_centroids.npy", thorax_result.centroids.astype(np.float64))
        np.save(output_dir / "conductivity.npy", thorax_result.conductivities.astype(np.float32))
        np.save(output_dir / "region_labels.npy", thorax_result.region_labels.astype(np.int32))
        np.savez_compressed(
            output_dir / "fem_mesh_data.npz",
            cleaned_torso_mask=thorax_result.cleaned_torso_mask.astype(np.uint8),
            cleaned_lung_mask=thorax_result.cleaned_lung_mask.astype(np.uint8),
            outer_contour=thorax_result.outer_contour.astype(np.float64),
            nodes=thorax_result.nodes.astype(np.float64),
            elements=thorax_result.elements.astype(np.int32),
            triangle_centroids=thorax_result.centroids.astype(np.float64),
            conductivity=thorax_result.conductivities.astype(np.float32),
            region_labels=thorax_result.region_labels.astype(np.int32),
        )

        save_mask_png(thorax_result.cleaned_torso_mask, output_dir / "cleaned_torso_mask.png")
        save_mask_png(thorax_result.cleaned_lung_mask, output_dir / "cleaned_lung_mask.png")
        save_mesh_preview(
            background_mask=thorax_result.cleaned_torso_mask,
            outer_contour=thorax_result.outer_contour,
            lung_contours=thorax_result.lung_contours,
            nodes=thorax_result.nodes,
            elements=thorax_result.elements,
            path=output_dir / "mesh_preview.png",
        )
        save_conductivity_preview(
            nodes=thorax_result.nodes,
            elements=thorax_result.elements,
            conductivities=thorax_result.conductivities,
            path=output_dir / "conductivity_preview.png",
        )

        metadata: dict[str, Any] = {
            "mode": "thorax_with_lungs",
            "mask_source": mask_source,
            "mask_path": str(mask_path) if mask_path is not None else None,
            "torso_mask_path": str(torso_mask_path) if torso_mask_path is not None else None,
            "lung_mask_path": str(lung_mask_path) if lung_mask_path is not None else None,
            "torso_epsilon_ratio": float(args.torso_epsilon_ratio),
            "lung_epsilon_ratio": float(args.lung_epsilon_ratio),
            "outer_mesh_size": float(args.outer_mesh_size),
            "lung_mesh_size": float(args.lung_mesh_size),
            "white_conductivity": float(args.white_conductivity),
            "black_conductivity": float(args.black_conductivity),
            "node_count": int(thorax_result.nodes.shape[0]),
            "element_count": int(thorax_result.elements.shape[0]),
            "outer_contour_vertex_count": int(thorax_result.outer_contour.shape[0]),
            "lung_contour_vertex_counts": [int(contour.shape[0]) for contour in thorax_result.lung_contours],
        }
        node_count = thorax_result.nodes.shape[0]
        element_count = thorax_result.elements.shape[0]
        conductivities = thorax_result.conductivities
    else:
        remove_if_exists(output_dir / "cleaned_torso_mask.npy")
        remove_if_exists(output_dir / "cleaned_torso_mask.png")
        remove_if_exists(output_dir / "cleaned_lung_mask.npy")
        remove_if_exists(output_dir / "cleaned_lung_mask.png")
        remove_if_exists(output_dir / "region_labels.npy")
        remove_if_exists(output_dir / "lung_contour_00.npy")
        remove_if_exists(output_dir / "lung_contour_01.npy")
        result = build_fem_from_material_mask(
            material_mask=material_mask,
            epsilon_ratio=args.epsilon_ratio,
            mesh_size=args.mesh_size,
            white_conductivity=args.white_conductivity,
            black_conductivity=args.black_conductivity,
        )

        np.save(output_dir / "material_mask.npy", result.material_mask.astype(np.uint8))
        np.save(output_dir / "outer_contour.npy", result.outer_contour.astype(np.float64))
        np.save(output_dir / "nodes.npy", result.nodes.astype(np.float64))
        np.save(output_dir / "elements.npy", result.elements.astype(np.int32))
        np.save(output_dir / "triangle_centroids.npy", result.centroids.astype(np.float64))
        np.save(output_dir / "conductivity.npy", result.conductivities.astype(np.float32))
        np.savez_compressed(
            output_dir / "fem_mesh_data.npz",
            material_mask=result.material_mask.astype(np.uint8),
            outer_contour=result.outer_contour.astype(np.float64),
            nodes=result.nodes.astype(np.float64),
            elements=result.elements.astype(np.int32),
            triangle_centroids=result.centroids.astype(np.float64),
            conductivity=result.conductivities.astype(np.float32),
        )

        save_mask_png(result.material_mask, output_dir / "material_mask.png")
        save_mesh_preview(
            background_mask=result.material_mask,
            outer_contour=result.outer_contour,
            lung_contours=[],
            nodes=result.nodes,
            elements=result.elements,
            path=output_dir / "mesh_preview.png",
        )
        save_conductivity_preview(
            nodes=result.nodes,
            elements=result.elements,
            conductivities=result.conductivities,
            path=output_dir / "conductivity_preview.png",
        )

        metadata = {
            "mode": "legacy_single_mask",
            "mask_source": mask_source,
            "mask_path": str(mask_path) if mask_path is not None else None,
            "torso_mask_path": str(torso_mask_path) if torso_mask_path is not None else None,
            "lung_mask_path": str(lung_mask_path) if lung_mask_path is not None else None,
            "mesh_size": float(args.mesh_size),
            "epsilon_ratio": float(args.epsilon_ratio),
            "white_conductivity": float(args.white_conductivity),
            "black_conductivity": float(args.black_conductivity),
            "node_count": int(result.nodes.shape[0]),
            "element_count": int(result.elements.shape[0]),
            "contour_vertex_count": int(result.outer_contour.shape[0]),
        }
        node_count = result.nodes.shape[0]
        element_count = result.elements.shape[0]
        conductivities = result.conductivities

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("✅ FEM 网格构建完成")
    print(f"   输出目录: {output_dir}")
    print(f"   节点数: {node_count}")
    print(f"   三角单元数: {element_count}")
    print(
        "   电导率统计: "
        f"muscle={int(np.sum(conductivities == args.white_conductivity))}, "
        f"lung={int(np.sum(conductivities == args.black_conductivity))}"
    )


if __name__ == "__main__":
    main()
