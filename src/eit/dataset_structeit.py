from __future__ import annotations

import ast
import random
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils.paths import resolve_from_root


DEFAULT_DATASET_ROOT = "data/processed/train_sim/struct_eit"


@dataclass(frozen=True)
class StructEITRecord:
    case_id: str
    source_group: str
    npz_path: Path
    num_frames: int


def resolve_structeit_dataset_root(path_like: str | Path | None = None) -> Path:
    if path_like is None:
        return resolve_from_root(DEFAULT_DATASET_ROOT)
    return resolve_from_root(path_like)


def infer_structeit_source_group(case_id: str) -> str:
    if case_id.startswith("LIDC-IDRI-"):
        return "LIDC-IDRI"
    if case_id.startswith("LUNG1-"):
        return "LUNG1"
    return case_id.split("-", 1)[0]


def _read_npy_header(raw: bytes) -> tuple[tuple[int, ...], str, bool]:
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("给定内容不是合法的 NPY 头。")

    major = raw[6]
    if major == 1:
        header_len = struct.unpack("<H", raw[8:10])[0]
        start = 10
    elif major in {2, 3}:
        header_len = struct.unpack("<I", raw[8:12])[0]
        start = 12
    else:
        raise ValueError(f"不支持的 NPY 版本: {major}")

    header = raw[start : start + header_len].decode("latin1")
    metadata = ast.literal_eval(header)
    return tuple(metadata["shape"]), str(metadata["descr"]), bool(metadata["fortran_order"])


def inspect_npz_array_shape(npz_path: str | Path, key: str) -> tuple[int, ...]:
    npz_path = Path(npz_path)
    member_name = f"{key}.npy"

    with zipfile.ZipFile(npz_path, "r") as archive:
        try:
            with archive.open(member_name) as file:
                raw = file.read(512)
        except KeyError as exc:
            raise KeyError(f"{npz_path} 中缺少字段 {key}") from exc

    shape, _, _ = _read_npy_header(raw)
    return shape


def discover_structeit_records(
    dataset_root: str | Path | None = None,
    input_key: str = "greit_img",
    case_ids: Sequence[str] | None = None,
    verify_files: bool = True,
) -> list[StructEITRecord]:
    root = resolve_structeit_dataset_root(dataset_root)
    include_case_ids = set(case_ids) if case_ids is not None else None

    records: list[StructEITRecord] = []
    for npz_path in sorted(root.glob("*_dataset.npz")):
        if verify_files and not npz_path.exists():
            raise FileNotFoundError(f"找不到 StructEIT 样本文件: {npz_path}")

        case_id = npz_path.name.removesuffix("_dataset.npz")
        if include_case_ids is not None and case_id not in include_case_ids:
            continue

        shape = inspect_npz_array_shape(npz_path, key=input_key)
        if len(shape) != 3:
            raise ValueError(f"{npz_path} 中 {input_key} 期望为 3 维，实际为 {shape}")

        records.append(
            StructEITRecord(
                case_id=case_id,
                source_group=infer_structeit_source_group(case_id),
                npz_path=npz_path,
                num_frames=int(shape[0]),
            )
        )

    if not records:
        raise ValueError(f"在 {root} 下未找到可用的 StructEIT 样本。")
    return records


def discover_structeit_case_ids(
    dataset_root: str | Path | None = None,
    input_key: str = "greit_img",
) -> list[str]:
    return [record.case_id for record in discover_structeit_records(dataset_root=dataset_root, input_key=input_key)]


def split_structeit_case_ids(
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

    grouped_case_ids: dict[str, list[str]] = {}
    for case_id in case_ids:
        grouped_case_ids.setdefault(infer_structeit_source_group(case_id), []).append(case_id)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for group_case_ids in grouped_case_ids.values():
        shuffled = list(group_case_ids)
        rng.shuffle(shuffled)

        num_cases = len(shuffled)
        train_end = int(num_cases * train_ratio)
        val_end = train_end + int(num_cases * val_ratio)

        splits["train"].extend(shuffled[:train_end])
        splits["val"].extend(shuffled[train_end:val_end])
        splits["test"].extend(shuffled[val_end:])

    for split_name in splits:
        splits[split_name] = sorted(splits[split_name])
    return splits


def limit_case_ids_per_source(case_ids: Sequence[str], per_source_limit: int | None) -> list[str]:
    if per_source_limit is None or per_source_limit <= 0:
        return list(case_ids)

    remaining: dict[str, int] = {}
    limited: list[str] = []
    for case_id in case_ids:
        source_group = infer_structeit_source_group(case_id)
        count = remaining.get(source_group, 0)
        if count >= per_source_limit:
            continue
        remaining[source_group] = count + 1
        limited.append(case_id)
    return limited


class StructEITSequenceDataset(Dataset):
    """
    StructEIT 单源序列数据集。

    默认输出：
    - recon: [T, H, W]
    - target: [1, H, W] 或 [T, H, W]
    """

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        case_ids: Sequence[str] | None = None,
        input_key: str = "greit_img",
        target_key: str = "target_img",
        window_size: int = 1,
        frame_stride: int = 1,
        target_mode: str = "middle",
        recon_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        include_mask: bool = True,
        mask_threshold_abs: float = 1e-6,
        return_meta: bool = True,
        verify_files: bool = True,
    ) -> None:
        super().__init__()

        if window_size <= 0:
            raise ValueError("window_size 必须是正整数。")
        if frame_stride <= 0:
            raise ValueError("frame_stride 必须是正整数。")

        self.dataset_root = resolve_structeit_dataset_root(dataset_root)
        self.input_key = input_key
        self.target_key = target_key
        self.window_size = int(window_size)
        self.frame_stride = int(frame_stride)
        self.target_mode = str(target_mode).lower()
        self.recon_transform = recon_transform
        self.target_transform = target_transform
        self.include_mask = include_mask
        self.mask_threshold_abs = float(mask_threshold_abs)
        self.return_meta = return_meta

        if self.target_mode not in {"middle", "last", "sequence"}:
            raise ValueError("target_mode 仅支持 'middle'、'last' 或 'sequence'。")

        self.records = discover_structeit_records(
            dataset_root=self.dataset_root,
            input_key=self.input_key,
            case_ids=case_ids,
            verify_files=verify_files,
        )
        self.index = self._build_index()
        self.case_ids = sorted(record.case_id for record in self.records)

        if not self.index:
            raise ValueError("没有可用的 StructEIT 训练样本，请检查 case_ids 或 window_size。")

    def _build_index(self) -> list[tuple[StructEITRecord, int]]:
        index: list[tuple[StructEITRecord, int]] = []
        for record in self.records:
            if record.num_frames < self.window_size:
                continue
            for start in range(0, record.num_frames - self.window_size + 1, self.frame_stride):
                index.append((record, start))
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record, start_index = self.index[idx]
        stop_index = start_index + self.window_size

        with np.load(record.npz_path, allow_pickle=False) as loaded:
            recon_all = loaded[self.input_key]
            target_all = loaded[self.target_key]

            if recon_all.ndim != 3:
                raise ValueError(f"{record.npz_path} 中 {self.input_key} 期望为 3 维，实际为 {recon_all.shape}")
            if target_all.ndim != 3:
                raise ValueError(f"{record.npz_path} 中 {self.target_key} 期望为 3 维，实际为 {target_all.shape}")
            if recon_all.shape[0] != target_all.shape[0]:
                raise ValueError(f"{record.npz_path} 中输入与目标的时间维不一致。")

            recon_window = torch.from_numpy(np.asarray(recon_all[start_index:stop_index], dtype=np.float32))
            if self.target_mode == "sequence":
                target_tensor = torch.from_numpy(np.asarray(target_all[start_index:stop_index], dtype=np.float32))
                target_frame_index: int | None = None
            else:
                if self.target_mode == "middle":
                    target_index = start_index + self.window_size // 2
                else:
                    target_index = stop_index - 1
                target_tensor = torch.from_numpy(np.asarray(target_all[target_index], dtype=np.float32)).unsqueeze(0)
                target_frame_index = int(target_index)

        if self.recon_transform is not None:
            recon_window = self.recon_transform(recon_window)
        if self.target_transform is not None:
            target_tensor = self.target_transform(target_tensor)

        sample: dict[str, Any] = {
            "recon": recon_window,
            "target": target_tensor,
        }
        if self.include_mask:
            sample["mask"] = (torch.abs(target_tensor) > self.mask_threshold_abs).to(dtype=torch.float32)

        if self.return_meta:
            sample["meta"] = {
                "case_id": record.case_id,
                "source_group": record.source_group,
                "npz_path": str(record.npz_path),
                "window_start": int(start_index),
                "window_stop": int(stop_index),
                "window_size": int(self.window_size),
                "frame_stride": int(self.frame_stride),
                "target_mode": self.target_mode,
                "target_frame_index": target_frame_index,
                "num_frames": int(record.num_frames),
                "mask_threshold_abs": float(self.mask_threshold_abs),
            }

        return sample
