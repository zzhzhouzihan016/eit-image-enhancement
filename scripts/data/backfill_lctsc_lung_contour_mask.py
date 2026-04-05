from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "train_sim" / "lctsc_cem_pathology_jac32_gt64"
DEFAULT_PROCESSED_MANIFEST = DEFAULT_DATASET_ROOT / "global_samples_manifest.csv"
DEFAULT_CURATED_ROOT = PROJECT_ROOT / "data_ct" / "interim" / "lctsc_fem_curated" / "cases"
DEFAULT_ACCEPTED_MANIFEST = PROJECT_ROOT / "data_ct" / "interim" / "lctsc_fem_curated" / "accepted_manifest.csv"
DEFAULT_OUTPUT_KEY = "target_lung_contour_mask"


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


def rasterize_contours(
    contours: list[np.ndarray] | tuple[np.ndarray, ...],
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


def load_manifest_rows_by_slice(manifest_path: Path) -> dict[tuple[str, int], list[dict[str, str]]]:
    grouped_rows: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            grouped_rows[(row["case_id"], int(row["slice_index"]))].append(row)
    return dict(grouped_rows)


def load_slice_keys_from_manifest(manifest_path: Path | None) -> set[tuple[str, int]]:
    if manifest_path is None or not manifest_path.exists():
        return set()

    keys: set[tuple[str, int]] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            keys.add((row["case_id"], int(row["slice_index"])))
    return keys


def build_lung_contour_mask_for_slice(
    curated_slice_dir: Path,
    target_shape_hw: tuple[int, int],
    contour_band_thickness: int | None,
    downsample_threshold: float,
) -> tuple[np.ndarray, dict[str, int]]:
    ct_hu = np.load(curated_slice_dir / "ct_hu.npy")
    image_shape_hw = tuple(int(v) for v in ct_hu.shape)
    slice_meta = json.loads((curated_slice_dir / "slice_metadata.json").read_text(encoding="utf-8"))
    spacing_xy_mm = tuple(float(v) for v in slice_meta["spacing_xy_mm"])
    pixel_spacing_yx_mm = (spacing_xy_mm[1], spacing_xy_mm[0])

    fem_dir = curated_slice_dir / "fem"
    lung_contours_img: list[np.ndarray] = []
    for contour_path in sorted(fem_dir.glob("lung_contour_*_m.npy")):
        lung_contours_img.append(
            cartesian_metric_to_image_coordinates(
                np.load(contour_path),
                image_shape_hw=image_shape_hw,
                pixel_spacing_mm=pixel_spacing_yx_mm,
            )
        )

    if len(lung_contours_img) < 2:
        raise RuntimeError(f"{curated_slice_dir} 缺少双肺轮廓文件，无法构建 ROI。")

    if contour_band_thickness is None:
        thickness = max(12, int(round(min(image_shape_hw) * 0.026)))
    else:
        thickness = int(contour_band_thickness)

    src_mask = rasterize_contours(lung_contours_img, shape_hw=image_shape_hw, thickness=thickness)
    dst_mask = resize_binary_mask(src_mask, size_hw=target_shape_hw, threshold=downsample_threshold)

    meta = {
        "src_height": int(image_shape_hw[0]),
        "src_width": int(image_shape_hw[1]),
        "dst_height": int(target_shape_hw[0]),
        "dst_width": int(target_shape_hw[1]),
        "contour_band_thickness_px": int(thickness),
        "src_mask_pixels": int(src_mask.sum()),
        "dst_mask_pixels": int(dst_mask.sum()),
    }
    return dst_mask.astype(np.uint8), meta


def update_npz_with_mask(npz_path: Path, output_key: str, mask: np.ndarray, overwrite: bool) -> str:
    with np.load(npz_path, allow_pickle=False) as loaded:
        if output_key in loaded.files and not overwrite:
            return "skipped_existing"
        payload = {name: loaded[name] for name in loaded.files}

    payload[output_key] = mask.astype(np.uint8)
    temp_path = npz_path.with_name(f"{npz_path.stem}.tmp.npz")
    np.savez_compressed(temp_path, **payload)
    temp_path.replace(npz_path)
    return "updated"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill bilateral lung contour ROI masks into processed LCTSC NPZ samples.")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT), help="Processed dataset root.")
    parser.add_argument("--processed-manifest", type=str, default=str(DEFAULT_PROCESSED_MANIFEST), help="Processed sample manifest.")
    parser.add_argument("--curated-root", type=str, default=str(DEFAULT_CURATED_ROOT), help="Curated FEM slice root.")
    parser.add_argument("--accepted-manifest", type=str, default=str(DEFAULT_ACCEPTED_MANIFEST), help="Accepted slice manifest.")
    parser.add_argument("--output-key", type=str, default=DEFAULT_OUTPUT_KEY, help="Output NPZ field name.")
    parser.add_argument("--target-size", type=int, default=64, help="Target ROI mask size.")
    parser.add_argument("--downsample-threshold", type=float, default=0.08, help="Threshold after area downsampling.")
    parser.add_argument("--contour-band-thickness", type=int, default=None, help="Optional fixed contour thickness in source pixels.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output field when it already exists.")
    parser.add_argument("--limit-slices", type=int, default=None, help="Only process the first N unique slices, for debugging.")
    parser.add_argument("--dry-run", action="store_true", help="Only scan and report without modifying NPZ files.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    processed_manifest = Path(args.processed_manifest).expanduser().resolve()
    curated_root = Path(args.curated_root).expanduser().resolve()
    accepted_manifest = Path(args.accepted_manifest).expanduser().resolve() if args.accepted_manifest else None
    target_shape_hw = (int(args.target_size), int(args.target_size))

    if not dataset_root.exists():
        raise FileNotFoundError(f"Processed dataset root 不存在: {dataset_root}")
    if not processed_manifest.exists():
        raise FileNotFoundError(f"Processed manifest 不存在: {processed_manifest}")
    if not curated_root.exists():
        raise FileNotFoundError(f"Curated root 不存在: {curated_root}")

    rows_by_slice = load_manifest_rows_by_slice(processed_manifest)
    accepted_slice_keys = load_slice_keys_from_manifest(accepted_manifest)
    slice_keys = sorted(rows_by_slice.keys())
    if args.limit_slices is not None:
        slice_keys = slice_keys[: int(args.limit_slices)]

    summary = {
        "dataset_root": str(dataset_root),
        "processed_manifest": str(processed_manifest),
        "curated_root": str(curated_root),
        "accepted_manifest": str(accepted_manifest) if accepted_manifest is not None else "",
        "output_key": args.output_key,
        "target_shape_hw": list(target_shape_hw),
        "downsample_threshold": float(args.downsample_threshold),
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "slice_count": len(slice_keys),
        "sample_count": int(sum(len(rows_by_slice[key]) for key in slice_keys)),
        "updated_npz_count": 0,
        "skipped_existing_count": 0,
        "missing_curated_slices": [],
        "nonaccepted_processed_slices": [],
        "slice_examples": [],
    }

    print(
        f"Processing {summary['slice_count']} slices / {summary['sample_count']} samples "
        f"into output key `{args.output_key}`..."
    )

    for slice_idx, slice_key in enumerate(slice_keys, start=1):
        case_id, slice_index = slice_key
        curated_slice_dir = curated_root / case_id / "slices" / f"slice_{slice_index:03d}"
        if not curated_slice_dir.exists():
            summary["missing_curated_slices"].append({"case_id": case_id, "slice_index": slice_index})
            continue

        if accepted_slice_keys and slice_key not in accepted_slice_keys:
            summary["nonaccepted_processed_slices"].append({"case_id": case_id, "slice_index": slice_index})

        mask_64, mask_meta = build_lung_contour_mask_for_slice(
            curated_slice_dir=curated_slice_dir,
            target_shape_hw=target_shape_hw,
            contour_band_thickness=args.contour_band_thickness,
            downsample_threshold=float(args.downsample_threshold),
        )

        if len(summary["slice_examples"]) < 10:
            summary["slice_examples"].append(
                {
                    "case_id": case_id,
                    "slice_index": slice_index,
                    **mask_meta,
                }
            )

        for row in rows_by_slice[slice_key]:
            npz_path = dataset_root / row["npz_path"]
            if args.dry_run:
                continue

            status = update_npz_with_mask(
                npz_path=npz_path,
                output_key=args.output_key,
                mask=mask_64,
                overwrite=bool(args.overwrite),
            )
            if status == "updated":
                summary["updated_npz_count"] += 1
            elif status == "skipped_existing":
                summary["skipped_existing_count"] += 1

        if slice_idx % 20 == 0 or slice_idx == len(slice_keys):
            print(
                f"  [{slice_idx:03d}/{len(slice_keys):03d}] "
                f"{case_id} slice_{slice_index:03d} | "
                f"updated={summary['updated_npz_count']} skipped={summary['skipped_existing_count']}"
            )

    summary_path = dataset_root / f"{args.output_key}_backfill_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Summary saved to: {summary_path}")
    print(
        f"updated={summary['updated_npz_count']}, "
        f"skipped_existing={summary['skipped_existing_count']}, "
        f"missing_curated_slices={len(summary['missing_curated_slices'])}, "
        f"nonaccepted_processed_slices={len(summary['nonaccepted_processed_slices'])}"
    )


if __name__ == "__main__":
    main()
