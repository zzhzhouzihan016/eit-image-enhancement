from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data_ct" / "interim" / "lctsc_fem_curated" / "cases"
DEFAULT_ACCEPTED_MANIFEST = PROJECT_ROOT / "data_ct" / "interim" / "lctsc_fem_curated" / "accepted_manifest.csv"
DEFAULT_PROCESSED_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "train_sim" / "lctsc_cem_pathology_jac32_gt64" / "global_samples_manifest.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "roi_demo_lung_only"
DEFAULT_SAMPLE_SPECS = (
    "LCTSC-Test-S1-102:44",
    "LCTSC-Test-S1-201:62",
    "LCTSC-Train-S3-010:68",
)


@dataclass(frozen=True)
class SliceAssets:
    case_id: str
    slice_index: int
    slice_dir: Path
    ct_hu: np.ndarray
    body_mask: np.ndarray
    lung_mask: np.ndarray
    pixel_spacing_mm: tuple[float, float]
    fem_outer_contour_img: np.ndarray
    fem_lung_contours_img: tuple[np.ndarray, ...]


def parse_sample_spec(spec: str) -> tuple[str, int]:
    if ":" not in spec:
        raise ValueError(f"Invalid sample spec: {spec}. Expected CASE_ID:SLICE_INDEX.")
    case_id, slice_part = spec.rsplit(":", 1)
    return case_id, int(slice_part)


def ensure_binary(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.max() <= 1:
        return (mask_u8 > 0).astype(np.uint8)
    return (mask_u8 > threshold).astype(np.uint8)


def cartesian_metric_to_image_coordinates(
    points_xy_m: np.ndarray,
    image_shape_hw: tuple[int, int],
    pixel_spacing_mm: tuple[float, float],
) -> np.ndarray:
    height, width = image_shape_hw
    spacing_y_mm, spacing_x_mm = pixel_spacing_mm
    spacing_x_m = float(spacing_x_mm) / 1000.0
    spacing_y_m = float(spacing_y_mm) / 1000.0

    points_img = np.asarray(points_xy_m, dtype=np.float64).copy()
    points_img[:, 0] = points_img[:, 0] / max(spacing_x_m, 1e-12)
    points_img[:, 1] = (height - 1) - (points_img[:, 1] / max(spacing_y_m, 1e-12))
    points_img[:, 0] = np.clip(points_img[:, 0], 0.0, float(width - 1))
    points_img[:, 1] = np.clip(points_img[:, 1], 0.0, float(height - 1))
    return points_img


def load_fem_contours(
    slice_dir: Path,
    image_shape_hw: tuple[int, int],
    pixel_spacing_mm: tuple[float, float],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    fem_dir = slice_dir / "fem"
    outer_contour = cartesian_metric_to_image_coordinates(
        np.load(fem_dir / "outer_contour_m.npy"),
        image_shape_hw=image_shape_hw,
        pixel_spacing_mm=pixel_spacing_mm,
    )

    lung_contours: list[np.ndarray] = []
    for contour_path in sorted(fem_dir.glob("lung_contour_*_m.npy")):
        lung_contours.append(
            cartesian_metric_to_image_coordinates(
                np.load(contour_path),
                image_shape_hw=image_shape_hw,
                pixel_spacing_mm=pixel_spacing_mm,
            )
        )

    if len(lung_contours) < 2:
        raise RuntimeError(f"{slice_dir} 缺少双肺轮廓文件。")

    return outer_contour, tuple(lung_contours)


def load_slice_assets(data_root: Path, case_id: str, slice_index: int) -> SliceAssets:
    slice_dir = data_root / case_id / "slices" / f"slice_{slice_index:03d}"
    if not slice_dir.exists():
        raise FileNotFoundError(f"Slice directory not found: {slice_dir}")

    ct_hu = np.load(slice_dir / "ct_hu.npy")
    body_mask = ensure_binary(np.load(slice_dir / "body_mask.npy"))
    lung_mask = ensure_binary(np.load(slice_dir / "lung_mask.npy"))
    slice_meta = json.loads((slice_dir / "slice_metadata.json").read_text(encoding="utf-8"))
    spacing_xy_mm = tuple(float(v) for v in slice_meta["spacing_xy_mm"])
    outer_contour, lung_contours = load_fem_contours(
        slice_dir=slice_dir,
        image_shape_hw=ct_hu.shape,
        pixel_spacing_mm=(spacing_xy_mm[1], spacing_xy_mm[0]),
    )

    return SliceAssets(
        case_id=case_id,
        slice_index=slice_index,
        slice_dir=slice_dir,
        ct_hu=ct_hu,
        body_mask=body_mask,
        lung_mask=lung_mask,
        pixel_spacing_mm=(spacing_xy_mm[1], spacing_xy_mm[0]),
        fem_outer_contour_img=outer_contour,
        fem_lung_contours_img=lung_contours,
    )


def load_slice_keys_from_manifest(manifest_path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            keys.add((row["case_id"], int(row["slice_index"])))
    return keys


def normalize_ct_to_rgb(ct_hu: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    body_bool = body_mask.astype(bool)
    body_values = ct_hu[body_bool]

    if body_values.size == 0:
        vmin = float(np.min(ct_hu))
        vmax = float(np.max(ct_hu))
    else:
        vmin, vmax = np.percentile(body_values, [1.0, 99.0]).astype(np.float32)

    if float(vmax) <= float(vmin):
        vmax = vmin + 1.0

    normalized = np.clip((ct_hu.astype(np.float32) - float(vmin)) / float(vmax - vmin), 0.0, 1.0)
    normalized[~body_bool] = 0.0
    gray_u8 = (normalized * 255.0).astype(np.uint8)
    return np.repeat(gray_u8[..., None], 3, axis=-1)


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    output = image_rgb.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    mask_bool = ensure_binary(mask).astype(bool)
    output[mask_bool] = output[mask_bool] * (1.0 - alpha) + color_arr * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def draw_contours(
    image_rgb: np.ndarray,
    contours: tuple[np.ndarray, ...] | list[np.ndarray],
    color: tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    canvas = image_rgb.copy()
    contour_list = [np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2) for points in contours]
    cv2.polylines(canvas, contour_list, isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return canvas


def rasterize_contours(
    contours: tuple[np.ndarray, ...] | list[np.ndarray],
    shape_hw: tuple[int, int],
    thickness: int,
) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    contour_list = [np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2) for points in contours]
    cv2.polylines(mask, contour_list, isClosed=True, color=1, thickness=thickness, lineType=cv2.LINE_AA)
    return ensure_binary(mask)


def resize_binary_mask(mask: np.ndarray, size_hw: tuple[int, int], threshold: float) -> np.ndarray:
    height, width = size_hw
    resized = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    return (resized >= threshold).astype(np.uint8)


def build_roi_candidates(
    fem_lung_contours_img: tuple[np.ndarray, ...],
    image_shape_hw: tuple[int, int],
    dst_shape_hw: tuple[int, int],
) -> dict[str, np.ndarray]:
    reference_size = min(image_shape_hw)
    contour_band_thickness = max(12, int(round(reference_size * 0.026)))

    lung_contour_band = rasterize_contours(
        contours=fem_lung_contours_img,
        shape_hw=image_shape_hw,
        thickness=contour_band_thickness,
    )
    lung_contour_band_64 = resize_binary_mask(lung_contour_band, size_hw=dst_shape_hw, threshold=0.08)

    return {
        "lung_contour_band": lung_contour_band,
        "lung_contour_band_64": lung_contour_band_64,
        "contour_band_thickness": np.asarray(contour_band_thickness, dtype=np.int32),
    }


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (ensure_binary(mask) * 255).astype(np.uint8))


def save_panel(
    assets: SliceAssets,
    roi_candidates: dict[str, np.ndarray],
    output_dir: Path,
    dst_shape_hw: tuple[int, int],
    in_accepted_manifest: bool,
    in_processed_manifest: bool,
) -> None:
    ct_rgb = normalize_ct_to_rgb(assets.ct_hu, assets.body_mask)
    lung_overlay = overlay_mask(ct_rgb, assets.lung_mask, color=(80, 200, 255), alpha=0.42)

    fem_reference_overlay = draw_contours(
        image_rgb=ct_rgb,
        contours=(assets.fem_outer_contour_img,),
        color=(255, 210, 90),
        thickness=2,
    )
    fem_reference_overlay = draw_contours(
        image_rgb=fem_reference_overlay,
        contours=assets.fem_lung_contours_img,
        color=(120, 255, 150),
        thickness=3,
    )

    roi_overlay = overlay_mask(ct_rgb, roi_candidates["lung_contour_band"], color=(255, 90, 90), alpha=0.62)

    ct_rgb_64 = cv2.resize(ct_rgb, (dst_shape_hw[1], dst_shape_hw[0]), interpolation=cv2.INTER_AREA)
    ct_rgb_64_big = cv2.resize(ct_rgb_64, (ct_rgb.shape[1], ct_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    roi_64_big = cv2.resize(
        roi_candidates["lung_contour_band_64"],
        (ct_rgb.shape[1], ct_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    roi_overlay_64 = overlay_mask(ct_rgb_64_big, roi_64_big, color=(255, 90, 90), alpha=0.68)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    panels = [
        (ct_rgb, "CT slice"),
        (lung_overlay, "Lung mask overlay"),
        (fem_reference_overlay, "FEM contours reference"),
        (roi_overlay, "Proposed ROI: lung contour band"),
        (roi_overlay_64, "Proposed ROI at 64x64"),
        (np.full_like(ct_rgb, 16), "Notes"),
    ]

    for axis, (image, title) in zip(axes.flat, panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    notes = [
        f"case_id = {assets.case_id}",
        f"slice_index = {assets.slice_index:03d}",
        f"accepted_manifest = {in_accepted_manifest}",
        f"processed_manifest = {in_processed_manifest}",
        f"contour band thickness = {int(roi_candidates['contour_band_thickness'])} px",
        "ROI only uses bilateral lung contours from FEM.",
        "Outer thorax contour is shown only as reference.",
        "This version does not include thorax in ROI.",
    ]
    axes.flat[-1].text(0.05, 0.95, "\n".join(notes), va="top", ha="left", color="white", fontsize=12)

    fig.suptitle(f"Lung contour ROI demo for {assets.case_id} slice {assets.slice_index:03d}", fontsize=16)
    fig.tight_layout()
    panel_path = output_dir / "roi_lung_only_panel.png"
    fig.savefig(panel_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_metadata(
    assets: SliceAssets,
    roi_candidates: dict[str, np.ndarray],
    output_dir: Path,
    dst_shape_hw: tuple[int, int],
    in_accepted_manifest: bool,
    in_processed_manifest: bool,
) -> None:
    metadata = {
        "case_id": assets.case_id,
        "slice_index": assets.slice_index,
        "slice_dir": str(assets.slice_dir),
        "ct_shape_hw": list(assets.ct_hu.shape),
        "roi_target_shape_hw": list(dst_shape_hw),
        "in_accepted_manifest": in_accepted_manifest,
        "in_processed_manifest": in_processed_manifest,
        "lung_pixels": int(ensure_binary(assets.lung_mask).sum()),
        "roi_pixels": int(ensure_binary(roi_candidates["lung_contour_band"]).sum()),
        "roi_64_pixels": int(ensure_binary(roi_candidates["lung_contour_band_64"]).sum()),
        "contour_band_thickness_px": int(roi_candidates["contour_band_thickness"]),
        "roi_definition": "bilateral FEM lung contour band only",
        "thorax_reference_only": True,
    }
    (output_dir / "roi_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_outputs_for_sample(
    data_root: Path,
    output_root: Path,
    case_id: str,
    slice_index: int,
    dst_shape_hw: tuple[int, int],
    accepted_slice_keys: set[tuple[str, int]],
    processed_slice_keys: set[tuple[str, int]],
) -> Path:
    assets = load_slice_assets(data_root=data_root, case_id=case_id, slice_index=slice_index)
    roi_candidates = build_roi_candidates(
        fem_lung_contours_img=assets.fem_lung_contours_img,
        image_shape_hw=assets.ct_hu.shape,
        dst_shape_hw=dst_shape_hw,
    )

    output_dir = output_root / case_id / f"slice_{slice_index:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ct_rgb = normalize_ct_to_rgb(assets.ct_hu, assets.body_mask)
    cv2.imwrite(str(output_dir / "ct_rgb.png"), cv2.cvtColor(ct_rgb, cv2.COLOR_RGB2BGR))
    save_mask_png(assets.body_mask, output_dir / "body_mask.png")
    save_mask_png(assets.lung_mask, output_dir / "lung_mask.png")
    save_mask_png(roi_candidates["lung_contour_band"], output_dir / "proposed_lung_contour_roi_mask.png")
    save_mask_png(roi_candidates["lung_contour_band_64"], output_dir / "proposed_lung_contour_roi_mask_64.png")

    fem_reference_mask = rasterize_contours((assets.fem_outer_contour_img,), assets.ct_hu.shape, thickness=2)
    fem_lung_reference_mask = rasterize_contours(assets.fem_lung_contours_img, assets.ct_hu.shape, thickness=3)
    save_mask_png(fem_reference_mask, output_dir / "fem_outer_contour_reference_mask.png")
    save_mask_png(fem_lung_reference_mask, output_dir / "fem_lung_contour_reference_mask.png")

    sample_key = (case_id, slice_index)
    in_accepted_manifest = sample_key in accepted_slice_keys
    in_processed_manifest = sample_key in processed_slice_keys

    save_panel(
        assets=assets,
        roi_candidates=roi_candidates,
        output_dir=output_dir,
        dst_shape_hw=dst_shape_hw,
        in_accepted_manifest=in_accepted_manifest,
        in_processed_manifest=in_processed_manifest,
    )
    save_metadata(
        assets=assets,
        roi_candidates=roi_candidates,
        output_dir=output_dir,
        dst_shape_hw=dst_shape_hw,
        in_accepted_manifest=in_accepted_manifest,
        in_processed_manifest=in_processed_manifest,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize lung-contour ROI candidates on curated CT slices.")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT), help="Curated CT slice root.")
    parser.add_argument("--accepted-manifest", type=str, default=str(DEFAULT_ACCEPTED_MANIFEST), help="Accepted slice manifest.")
    parser.add_argument("--processed-manifest", type=str, default=str(DEFAULT_PROCESSED_MANIFEST), help="Processed dataset manifest.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument(
        "--sample",
        action="append",
        default=None,
        help="Sample spec in the form CASE_ID:SLICE_INDEX. Can be repeated.",
    )
    parser.add_argument("--target-size", type=int, default=64, help="Low-resolution ROI size for training.")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    accepted_manifest = Path(args.accepted_manifest).expanduser().resolve()
    processed_manifest = Path(args.processed_manifest).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    sample_specs = tuple(args.sample) if args.sample else DEFAULT_SAMPLE_SPECS
    dst_shape_hw = (int(args.target_size), int(args.target_size))

    accepted_slice_keys = load_slice_keys_from_manifest(accepted_manifest)
    processed_slice_keys = load_slice_keys_from_manifest(processed_manifest)

    generated_dirs: list[str] = []
    for spec in sample_specs:
        case_id, slice_index = parse_sample_spec(spec)
        out_dir = build_outputs_for_sample(
            data_root=data_root,
            output_root=output_root,
            case_id=case_id,
            slice_index=slice_index,
            dst_shape_hw=dst_shape_hw,
            accepted_slice_keys=accepted_slice_keys,
            processed_slice_keys=processed_slice_keys,
        )
        generated_dirs.append(str(out_dir))

    print("Generated ROI demo outputs:")
    for directory in generated_dirs:
        print(directory)


if __name__ == "__main__":
    main()
