from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pydicom


@dataclass
class DicomSeries:
    volume_hu: np.ndarray
    dicom_paths: List[Path]
    z_positions: List[float]
    spacing_xy: Tuple[float, float]
    slice_thickness: Optional[float]
    patient_id: Optional[str]
    study_uid: Optional[str]
    series_uid: Optional[str]
    rows: int
    cols: int


def _read_dicom(path: Path):
    return pydicom.dcmread(str(path), force=True)


def _is_image_slice(ds: Any) -> bool:
    return hasattr(ds, "PixelData") and hasattr(ds, "Rows") and hasattr(ds, "Columns")


def _slice_sort_key(ds: Any, fallback_index: int) -> tuple[float, int]:
    if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient is not None:
        try:
            return float(ds.ImagePositionPatient[2]), fallback_index
        except Exception:
            pass

    if hasattr(ds, "SliceLocation"):
        try:
            return float(ds.SliceLocation), fallback_index
        except Exception:
            pass

    if hasattr(ds, "InstanceNumber"):
        try:
            return float(ds.InstanceNumber), fallback_index
        except Exception:
            pass

    return float(fallback_index), fallback_index


def _to_hu(ds: Any) -> np.ndarray:
    pixel_array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return pixel_array * slope + intercept


def load_dicom_series(dicom_dir: Union[str, Path]) -> DicomSeries:
    dicom_dir = Path(dicom_dir)
    if not dicom_dir.exists():
        raise FileNotFoundError(f"找不到 DICOM 目录: {dicom_dir}")

    dicom_files = sorted([path for path in dicom_dir.rglob("*") if path.is_file()])
    if not dicom_files:
        raise FileNotFoundError(f"目录内没有文件: {dicom_dir}")

    raw_items: List[Tuple[Tuple[float, int], Path, Any]] = []
    for index, path in enumerate(dicom_files):
        try:
            ds = _read_dicom(path)
        except Exception:
            continue
        if not _is_image_slice(ds):
            continue
        raw_items.append((_slice_sort_key(ds, index), path, ds))

    if not raw_items:
        raise ValueError(f"目录内未找到可堆叠的 DICOM 图像切片: {dicom_dir}")

    raw_items.sort(key=lambda item: item[0])

    first_ds = raw_items[0][2]
    rows = int(first_ds.Rows)
    cols = int(first_ds.Columns)

    volume_slices: List[np.ndarray] = []
    z_positions: List[float] = []
    used_paths: List[Path] = []

    for sort_key, path, ds in raw_items:
        if int(ds.Rows) != rows or int(ds.Columns) != cols:
            continue
        volume_slices.append(_to_hu(ds))
        z_positions.append(float(sort_key[0]))
        used_paths.append(path)

    if not volume_slices:
        raise ValueError("没有找到尺寸一致的 DICOM 图像切片。")

    pixel_spacing = getattr(first_ds, "PixelSpacing", [1.0, 1.0])
    spacing_xy = (float(pixel_spacing[0]), float(pixel_spacing[1]))
    slice_thickness = None
    if hasattr(first_ds, "SliceThickness"):
        try:
            slice_thickness = float(first_ds.SliceThickness)
        except Exception:
            slice_thickness = None

    volume_hu = np.stack(volume_slices, axis=0).astype(np.float32)

    return DicomSeries(
        volume_hu=volume_hu,
        dicom_paths=used_paths,
        z_positions=z_positions,
        spacing_xy=spacing_xy,
        slice_thickness=slice_thickness,
        patient_id=getattr(first_ds, "PatientID", None),
        study_uid=getattr(first_ds, "StudyInstanceUID", None),
        series_uid=getattr(first_ds, "SeriesInstanceUID", None),
        rows=rows,
        cols=cols,
    )
