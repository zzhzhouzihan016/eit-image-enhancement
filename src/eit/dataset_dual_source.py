from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils.paths import resolve_from_root


DEFAULT_DATASET_ROOT = "data/processed/train_sim/lctsc_cem_pathology_jac32_gt64"
DEFAULT_MANIFEST_NAME = "global_samples_manifest.csv"
DEFAULT_NOISE_INDEX = 2
DEFAULT_EXPANDED_NOISE_INDICES = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class DualSourceRecord:
    case_id: str
    slice_index: int
    sample_name: str
    pathology_label: str
    lung_side: str | None
    severity_level: int
    replicate_index: int
    frames: int
    sample_dir: Path
    npz_path: Path
    metadata_path: Path


def _build_sample_metadata(
    record: DualSourceRecord,
    noise_index: int,
    noise_levels: np.ndarray | None,
) -> dict[str, Any]:
    return {
        "noise_db": float(noise_levels[noise_index]) if noise_levels is not None else float("nan"),
        "noise_index": noise_index,
        "case_id": record.case_id,
        "slice_index": record.slice_index,
        "sample_name": record.sample_name,
        "pathology_label": record.pathology_label,
        "lung_side": record.lung_side or "",
        "severity_level": record.severity_level,
        "replicate_index": record.replicate_index,
    }


def _build_record_meta(record: DualSourceRecord) -> dict[str, Any]:
    return {
        "case_id": record.case_id,
        "slice_index": record.slice_index,
        "sample_name": record.sample_name,
        "record_key": build_record_key(record.case_id, record.slice_index, record.sample_name),
        "slice_group_key": build_slice_group_key(record.case_id, record.slice_index),
        "pathology_label": record.pathology_label,
        "lung_side": record.lung_side or "",
        "severity_level": record.severity_level,
        "replicate_index": record.replicate_index,
        "frames": record.frames,
        "sample_dir": str(record.sample_dir),
        "npz_path": str(record.npz_path),
        "metadata_path": str(record.metadata_path),
    }


def resolve_dual_source_dataset_root(path_like: str | Path | None = None) -> Path:
    if path_like is None:
        return resolve_from_root(DEFAULT_DATASET_ROOT)
    return resolve_from_root(path_like)


def build_relative_sample_dir(case_id: str, slice_index: int, sample_name: str) -> Path:
    return Path("cases") / case_id / "slices" / f"slice_{slice_index:03d}" / sample_name


def build_record_key(case_id: str, slice_index: int, sample_name: str) -> str:
    return f"{case_id}::slice_{int(slice_index):03d}::{sample_name}"


def build_slice_group_key(case_id: str, slice_index: int) -> str:
    return f"{case_id}::slice_{int(slice_index):03d}"


def discover_case_ids(
    dataset_root: str | Path | None = None,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
) -> list[str]:
    root = resolve_dual_source_dataset_root(dataset_root)
    manifest_path = root / manifest_name

    with open(manifest_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        case_ids = sorted({row["case_id"] for row in reader})

    return case_ids


def split_case_ids(
    case_ids: Sequence[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须在 0 和 1 之间。")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio 必须在 0 和 1 之间。")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio 必须小于 1。")

    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)

    num_cases = len(shuffled)
    train_end = int(num_cases * train_ratio)
    val_end = train_end + int(num_cases * val_ratio)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def build_case_splits_from_manifest(
    dataset_root: str | Path | None = None,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    case_ids = discover_case_ids(dataset_root=dataset_root, manifest_name=manifest_name)
    return split_case_ids(case_ids, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)


def _parse_optional_str(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    if normalized == "":
        return None
    if normalized.lower() in {"none", "null", "nan"}:
        return None
    return normalized


def _parse_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _resolve_record_paths(dataset_root: Path, row: dict[str, str]) -> tuple[Path, Path, Path]:
    relative_sample_dir = build_relative_sample_dir(
        case_id=row["case_id"],
        slice_index=_parse_int(row["slice_index"]),
        sample_name=row["sample_name"],
    )

    sample_dir = dataset_root / relative_sample_dir
    npz_path = sample_dir / "sequence_data.npz"
    metadata_path = sample_dir / "metadata.json"

    return sample_dir, npz_path, metadata_path


class LCTSCDualSourceDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path | None = None,
        manifest_name: str = DEFAULT_MANIFEST_NAME,
        case_ids: Sequence[str] | None = None,
        exclude_case_ids: Sequence[str] | None = None,
        record_keys: Sequence[str] | None = None,
        exclude_record_keys: Sequence[str] | None = None,
        noise_mode: str = "fixed",
        fixed_noise_index: int = DEFAULT_NOISE_INDEX,
        noise_indices: Sequence[int] | None = None,
        recon_key: str = "input_recon",
        voltage_key: str = "valid208_voltage_noisy",
        target_key: str = "target_delta_sigma",
        mask_key: str = "target_pathology_mask",
        recon_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        voltage_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        mask_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        return_meta: bool = True,
        include_mask: bool = True,
        verify_files: bool = True,
    ) -> None:
        super().__init__()

        self.dataset_root = resolve_dual_source_dataset_root(dataset_root)
        self.manifest_path = self.dataset_root / manifest_name
        self.noise_mode = noise_mode
        self.fixed_noise_index = fixed_noise_index
        self.noise_indices = tuple(noise_indices) if noise_indices is not None else None
        self.recon_key = recon_key
        self.voltage_key = voltage_key
        self.target_key = target_key
        self.mask_key = mask_key
        self.recon_transform = recon_transform
        self.voltage_transform = voltage_transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform
        self.return_meta = return_meta
        self.include_mask = include_mask

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest 文件不存在: {self.manifest_path}")

        self.records = self._load_records(
            case_ids=case_ids,
            exclude_case_ids=exclude_case_ids,
            record_keys=record_keys,
            exclude_record_keys=exclude_record_keys,
            verify_files=verify_files,
        )
        self.index: list[tuple[DualSourceRecord, int]] = self._build_index()
        self.case_ids = sorted({record.case_id for record in self.records})

        if not self.index:
            raise ValueError("没有可用的训练样本，请检查 case_ids、噪声设置或数据路径。")

    def _load_records(
        self,
        case_ids: Sequence[str] | None,
        exclude_case_ids: Sequence[str] | None,
        record_keys: Sequence[str] | None,
        exclude_record_keys: Sequence[str] | None,
        verify_files: bool,
    ) -> list[DualSourceRecord]:
        include_set = set(case_ids) if case_ids is not None else None
        exclude_set = set(exclude_case_ids or [])
        include_record_set = set(record_keys) if record_keys is not None else None
        exclude_record_set = set(exclude_record_keys or [])

        records: list[DualSourceRecord] = []
        with open(self.manifest_path, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                case_id = row["case_id"]
                if include_set is not None and case_id not in include_set:
                    continue
                if case_id in exclude_set:
                    continue
                record_key = build_record_key(
                    case_id=row["case_id"],
                    slice_index=_parse_int(row["slice_index"]),
                    sample_name=row["sample_name"],
                )
                if include_record_set is not None and record_key not in include_record_set:
                    continue
                if record_key in exclude_record_set:
                    continue

                sample_dir, npz_path, metadata_path = _resolve_record_paths(self.dataset_root, row)
                if verify_files and not npz_path.exists():
                    raise FileNotFoundError(f"找不到样本 npz 文件: {npz_path}")
                if verify_files and not metadata_path.exists():
                    raise FileNotFoundError(f"找不到样本 metadata 文件: {metadata_path}")

                records.append(
                    DualSourceRecord(
                        case_id=case_id,
                        slice_index=_parse_int(row["slice_index"]),
                        sample_name=row["sample_name"],
                        pathology_label=row.get("pathology_label", ""),
                        lung_side=_parse_optional_str(row.get("lung_side")),
                        severity_level=_parse_int(row.get("severity_level")),
                        replicate_index=_parse_int(row.get("replicate_index")),
                        frames=_parse_int(row.get("frames")),
                        sample_dir=sample_dir,
                        npz_path=npz_path,
                        metadata_path=metadata_path,
                    )
                )

        return records

    def _build_index(self) -> list[tuple[DualSourceRecord, int]]:
        if self.noise_mode not in {"fixed", "expand"}:
            raise ValueError("noise_mode 仅支持 'fixed' 或 'expand'。")

        if self.noise_mode == "fixed":
            noise_indices = (self.fixed_noise_index,)
        else:
            noise_indices = self.noise_indices or DEFAULT_EXPANDED_NOISE_INDICES

        index: list[tuple[DualSourceRecord, int]] = []
        for record in self.records:
            for noise_index in noise_indices:
                index.append((record, int(noise_index)))

        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record, noise_index = self.index[idx]

        with np.load(record.npz_path, allow_pickle=False) as loaded:
            recon_all = loaded[self.recon_key]
            voltage_all = loaded[self.voltage_key]
            target = loaded[self.target_key]
            noise_levels = loaded["noise_levels_db"] if "noise_levels_db" in loaded.files else None

            if recon_all.ndim != 4:
                raise ValueError(f"{record.npz_path} 中 {self.recon_key} 期望为 4 维，实际为 {recon_all.shape}")
            if voltage_all.ndim != 3:
                raise ValueError(f"{record.npz_path} 中 {self.voltage_key} 期望为 3 维，实际为 {voltage_all.shape}")
            if target.ndim != 3:
                raise ValueError(f"{record.npz_path} 中 {self.target_key} 期望为 3 维，实际为 {target.shape}")
            if noise_index < 0 or noise_index >= recon_all.shape[0]:
                raise IndexError(f"noise_index={noise_index} 超出范围，当前样本仅有 {recon_all.shape[0]} 个噪声级别。")
            if voltage_all.shape[0] != recon_all.shape[0]:
                raise ValueError(f"{record.npz_path} 中重建图与电压的噪声维不一致。")

            recon = torch.from_numpy(np.asarray(recon_all[noise_index], dtype=np.float32))
            voltage = torch.from_numpy(np.asarray(voltage_all[noise_index], dtype=np.float32))
            target_tensor = torch.from_numpy(np.asarray(target, dtype=np.float32))

            if self.recon_transform is not None:
                recon = self.recon_transform(recon)
            if self.voltage_transform is not None:
                voltage = self.voltage_transform(voltage)
            if self.target_transform is not None:
                target_tensor = self.target_transform(target_tensor)

            sample: dict[str, Any] = {
                "recon": recon,
                "voltage": voltage,
                "target": target_tensor,
                **_build_sample_metadata(record, noise_index, noise_levels),
            }

            if self.include_mask:
                if self.mask_key not in loaded.files:
                    raise KeyError(f"{record.npz_path} 中缺少 mask 字段: {self.mask_key}")
                mask_tensor = torch.from_numpy(np.asarray(loaded[self.mask_key], dtype=np.float32))
                if self.mask_transform is not None:
                    mask_tensor = self.mask_transform(mask_tensor)
                sample["mask"] = mask_tensor

        if self.return_meta:
            sample["meta"] = _build_record_meta(record)

        return sample


class LCTSCReconSequenceDataset(LCTSCDualSourceDataset):
    """
    LCTSC 20-frame 单源数据集：
    - recon: [T, 32, 32]
    - target: [T, 64, 64]

    复用与双源版本完全一致的 manifest、case split 与 noise 采样逻辑，
    只是不再读取 voltage，便于做公平的单/双源对照实验。
    """

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        manifest_name: str = DEFAULT_MANIFEST_NAME,
        case_ids: Sequence[str] | None = None,
        exclude_case_ids: Sequence[str] | None = None,
        record_keys: Sequence[str] | None = None,
        exclude_record_keys: Sequence[str] | None = None,
        noise_mode: str = "fixed",
        fixed_noise_index: int = DEFAULT_NOISE_INDEX,
        noise_indices: Sequence[int] | None = None,
        recon_key: str = "input_recon",
        target_key: str = "target_delta_sigma",
        mask_key: str = "target_pathology_mask",
        recon_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        mask_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        return_meta: bool = True,
        include_mask: bool = True,
        verify_files: bool = True,
    ) -> None:
        super().__init__(
            dataset_root=dataset_root,
            manifest_name=manifest_name,
            case_ids=case_ids,
            exclude_case_ids=exclude_case_ids,
            record_keys=record_keys,
            exclude_record_keys=exclude_record_keys,
            noise_mode=noise_mode,
            fixed_noise_index=fixed_noise_index,
            noise_indices=noise_indices,
            recon_key=recon_key,
            voltage_key="valid208_voltage_noisy",
            target_key=target_key,
            mask_key=mask_key,
            recon_transform=recon_transform,
            voltage_transform=None,
            target_transform=target_transform,
            mask_transform=mask_transform,
            return_meta=return_meta,
            include_mask=include_mask,
            verify_files=verify_files,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record, noise_index = self.index[idx]

        with np.load(record.npz_path, allow_pickle=False) as loaded:
            recon_all = loaded[self.recon_key]
            target = loaded[self.target_key]
            noise_levels = loaded["noise_levels_db"] if "noise_levels_db" in loaded.files else None

            if recon_all.ndim != 4:
                raise ValueError(f"{record.npz_path} 中 {self.recon_key} 期望为 4 维，实际为 {recon_all.shape}")
            if target.ndim != 3:
                raise ValueError(f"{record.npz_path} 中 {self.target_key} 期望为 3 维，实际为 {target.shape}")
            if noise_index < 0 or noise_index >= recon_all.shape[0]:
                raise IndexError(f"noise_index={noise_index} 超出范围，当前样本仅有 {recon_all.shape[0]} 个噪声级别。")

            recon = torch.from_numpy(np.asarray(recon_all[noise_index], dtype=np.float32))
            target_tensor = torch.from_numpy(np.asarray(target, dtype=np.float32))

            if self.recon_transform is not None:
                recon = self.recon_transform(recon)
            if self.target_transform is not None:
                target_tensor = self.target_transform(target_tensor)

            sample: dict[str, Any] = {
                "recon": recon,
                "target": target_tensor,
                **_build_sample_metadata(record, noise_index, noise_levels),
            }

            if self.include_mask:
                if self.mask_key not in loaded.files:
                    raise KeyError(f"{record.npz_path} 中缺少 mask 字段: {self.mask_key}")
                mask_tensor = torch.from_numpy(np.asarray(loaded[self.mask_key], dtype=np.float32))
                if self.mask_transform is not None:
                    mask_tensor = self.mask_transform(mask_tensor)
                sample["mask"] = mask_tensor

        if self.return_meta:
            sample["meta"] = _build_record_meta(record)

        return sample
