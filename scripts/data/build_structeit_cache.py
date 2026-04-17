from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eit.dataset_structeit import (  # noqa: E402
    StructEITRecord,
    discover_structeit_records,
    infer_structeit_source_group,
    resolve_structeit_cache_root,
    resolve_structeit_dataset_root,
    split_structeit_case_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 StructEIT 原始 case npz 展开为训练缓存 shard。")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/processed/train_sim/struct_eit",
        help="StructEIT 原始数据根目录。",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/cache/structeit_baseline_stride2",
        help="缓存输出根目录。",
    )
    parser.add_argument("--input-key", type=str, default="greit_img", help="输入字段名。")
    parser.add_argument("--target-key", type=str, default="target_img", help="目标字段名。")
    parser.add_argument("--window-size", type=int, default=1, help="时间窗长度。")
    parser.add_argument("--frame-stride", type=int, default=2, help="缓存时的帧采样步长。")
    parser.add_argument(
        "--target-mode",
        type=str,
        default="middle",
        choices=["middle", "last", "sequence"],
        help="目标帧选择方式。",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="训练集 case 比例。")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集 case 比例。")
    parser.add_argument("--seed", type=int, default=42, help="划分随机种子。")
    parser.add_argument("--shard-size", type=int, default=4096, help="每个 shard 的样本数。")
    parser.add_argument(
        "--mask-threshold-abs",
        type=float,
        default=1e-6,
        help="自动前景 mask 的绝对值阈值。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出目录已存在则覆盖。",
    )
    return parser.parse_args()


def _target_index_for_window(start_index: int, stop_index: int, window_size: int, target_mode: str) -> int:
    if target_mode == "middle":
        return start_index + window_size // 2
    if target_mode == "last":
        return stop_index - 1
    raise ValueError(f"sequence 模式没有单独 target index: {target_mode}")


def _stack_and_save_shard(
    shard_path: Path,
    recon_list: list[np.ndarray],
    target_list: list[np.ndarray],
    mask_list: list[np.ndarray],
    case_ids: list[str],
    source_groups: list[str],
    window_starts: list[int],
    target_frame_indices: list[int],
) -> dict[str, Any]:
    payload = {
        "recon": torch.from_numpy(np.stack(recon_list, axis=0)),
        "target": torch.from_numpy(np.stack(target_list, axis=0)),
        "mask": torch.from_numpy(np.stack(mask_list, axis=0)),
        "case_ids": list(case_ids),
        "source_groups": list(source_groups),
        "window_starts": list(window_starts),
        "target_frame_indices": list(target_frame_indices),
    }
    temp_path = shard_path.with_suffix(f"{shard_path.suffix}.tmp")
    try:
        torch.save(payload, temp_path)
    except RuntimeError as exc:
        # Some remote/container filesystems are unstable with the default zip
        # serializer for large tensors, so fall back to the legacy format.
        if temp_path.exists():
            temp_path.unlink()
        torch.save(payload, temp_path, _use_new_zipfile_serialization=False)
    os.replace(temp_path, shard_path)

    return {
        "filename": shard_path.name,
        "num_samples": len(recon_list),
        "recon_shape": list(payload["recon"].shape),
        "target_shape": list(payload["target"].shape),
        "mask_shape": list(payload["mask"].shape),
    }


def build_split_cache(
    split_name: str,
    records: list[StructEITRecord],
    split_dir: Path,
    input_key: str,
    target_key: str,
    window_size: int,
    frame_stride: int,
    target_mode: str,
    shard_size: int,
    mask_threshold_abs: float,
    accumulate_recon_stats: bool = False,
) -> tuple[dict[str, Any], dict[str, float] | None]:
    split_dir.mkdir(parents=True, exist_ok=True)

    shard_infos: list[dict[str, Any]] = []
    recon_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    case_ids: list[str] = []
    source_groups: list[str] = []
    window_starts: list[int] = []
    target_frame_indices: list[int] = []
    shard_index = 0
    num_samples = 0

    recon_sum = 0.0
    recon_sq_sum = 0.0
    recon_count = 0

    progress = tqdm(records, desc=f"Build {split_name}", unit="case")
    for record in progress:
        with np.load(record.npz_path, allow_pickle=False) as loaded:
            recon_all = np.asarray(loaded[input_key], dtype=np.float32)
            target_all = np.asarray(loaded[target_key], dtype=np.float32)

        if recon_all.ndim != 3 or target_all.ndim != 3:
            raise ValueError(f"{record.npz_path} 中字段维度异常。")
        if recon_all.shape[0] != target_all.shape[0]:
            raise ValueError(f"{record.npz_path} 中输入与目标时间维不一致。")

        for start_index in range(0, recon_all.shape[0] - window_size + 1, frame_stride):
            stop_index = start_index + window_size
            recon_window = recon_all[start_index:stop_index]
            if target_mode == "sequence":
                target_tensor = target_all[start_index:stop_index]
                target_frame_index = stop_index - 1
            else:
                target_frame_index = _target_index_for_window(start_index, stop_index, window_size, target_mode)
                target_tensor = target_all[target_frame_index][None, ...]

            mask_tensor = (np.abs(target_tensor) > mask_threshold_abs).astype(np.uint8)

            recon_list.append(recon_window.astype(np.float32, copy=False))
            target_list.append(target_tensor.astype(np.float32, copy=False))
            mask_list.append(mask_tensor)
            case_ids.append(record.case_id)
            source_groups.append(record.source_group)
            window_starts.append(int(start_index))
            target_frame_indices.append(int(target_frame_index))
            num_samples += 1

            if accumulate_recon_stats:
                recon_sum += float(recon_window.sum())
                recon_sq_sum += float(np.square(recon_window, dtype=np.float64).sum())
                recon_count += int(recon_window.size)

            if len(recon_list) >= shard_size:
                shard_path = split_dir / f"shard_{shard_index:04d}.pt"
                shard_infos.append(
                    _stack_and_save_shard(
                        shard_path=shard_path,
                        recon_list=recon_list,
                        target_list=target_list,
                        mask_list=mask_list,
                        case_ids=case_ids,
                        source_groups=source_groups,
                        window_starts=window_starts,
                        target_frame_indices=target_frame_indices,
                    )
                )
                shard_index += 1
                recon_list, target_list, mask_list = [], [], []
                case_ids, source_groups, window_starts, target_frame_indices = [], [], [], []

    if recon_list:
        shard_path = split_dir / f"shard_{shard_index:04d}.pt"
        shard_infos.append(
            _stack_and_save_shard(
                shard_path=shard_path,
                recon_list=recon_list,
                target_list=target_list,
                mask_list=mask_list,
                case_ids=case_ids,
                source_groups=source_groups,
                window_starts=window_starts,
                target_frame_indices=target_frame_indices,
            )
        )

    split_metadata = {
        "split": split_name,
        "num_cases": len(records),
        "num_samples": num_samples,
        "window_size": window_size,
        "frame_stride": frame_stride,
        "target_mode": target_mode,
        "input_key": input_key,
        "target_key": target_key,
        "mask_threshold_abs": mask_threshold_abs,
        "case_ids": [record.case_id for record in records],
        "shards": shard_infos,
    }
    with open(split_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(split_metadata, file, ensure_ascii=False, indent=2)

    stats = None
    if accumulate_recon_stats:
        recon_mean = recon_sum / max(recon_count, 1)
        recon_var = max(recon_sq_sum / max(recon_count, 1) - recon_mean**2, 1e-12)
        stats = {
            "recon_mean": float(recon_mean),
            "recon_std": float(math.sqrt(recon_var)),
            "recon_count": int(recon_count),
        }
    return split_metadata, stats


def main() -> None:
    args = parse_args()
    dataset_root = resolve_structeit_dataset_root(args.dataset_root)
    output_root = resolve_structeit_cache_root(args.output_root)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"缓存输出目录已存在: {output_root}。如需覆盖，请添加 --overwrite。")
        for path in sorted(output_root.glob("**/*"), reverse=True):
            if path.is_file():
                path.unlink()
        for path in sorted(output_root.glob("**/*"), reverse=True):
            if path.is_dir():
                path.rmdir()
    output_root.mkdir(parents=True, exist_ok=True)

    records = discover_structeit_records(dataset_root=dataset_root, input_key=args.input_key)
    splits = split_structeit_case_ids(
        case_ids=[record.case_id for record in records],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    record_map = {record.case_id: record for record in records}

    split_records = {
        split_name: [record_map[case_id] for case_id in case_ids]
        for split_name, case_ids in splits.items()
    }

    train_metadata, train_stats = build_split_cache(
        split_name="train",
        records=split_records["train"],
        split_dir=output_root / "train",
        input_key=args.input_key,
        target_key=args.target_key,
        window_size=args.window_size,
        frame_stride=args.frame_stride,
        target_mode=args.target_mode,
        shard_size=args.shard_size,
        mask_threshold_abs=args.mask_threshold_abs,
        accumulate_recon_stats=True,
    )
    val_metadata, _ = build_split_cache(
        split_name="val",
        records=split_records["val"],
        split_dir=output_root / "val",
        input_key=args.input_key,
        target_key=args.target_key,
        window_size=args.window_size,
        frame_stride=args.frame_stride,
        target_mode=args.target_mode,
        shard_size=args.shard_size,
        mask_threshold_abs=args.mask_threshold_abs,
        accumulate_recon_stats=False,
    )
    test_metadata, _ = build_split_cache(
        split_name="test",
        records=split_records["test"],
        split_dir=output_root / "test",
        input_key=args.input_key,
        target_key=args.target_key,
        window_size=args.window_size,
        frame_stride=args.frame_stride,
        target_mode=args.target_mode,
        shard_size=args.shard_size,
        mask_threshold_abs=args.mask_threshold_abs,
        accumulate_recon_stats=False,
    )

    global_metadata = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "input_key": args.input_key,
        "target_key": args.target_key,
        "window_size": int(args.window_size),
        "frame_stride": int(args.frame_stride),
        "target_mode": args.target_mode,
        "mask_threshold_abs": float(args.mask_threshold_abs),
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "shard_size": int(args.shard_size),
        "num_cases_total": len(records),
        "num_cases_by_source": {
            "LIDC-IDRI": sum(1 for record in records if infer_structeit_source_group(record.case_id) == "LIDC-IDRI"),
            "LUNG1": sum(1 for record in records if infer_structeit_source_group(record.case_id) == "LUNG1"),
        },
        "normalization": train_stats,
        "splits": {
            "train": train_metadata,
            "val": val_metadata,
            "test": test_metadata,
        },
    }
    with open(output_root / "cache_metadata.json", "w", encoding="utf-8") as file:
        json.dump(global_metadata, file, ensure_ascii=False, indent=2)

    print("✅ StructEIT 缓存构建完成")
    print(
        "   "
        f"train={train_metadata['num_samples']} | "
        f"val={val_metadata['num_samples']} | "
        f"test={test_metadata['num_samples']} | "
        f"recon_mean={train_stats['recon_mean']:.6e} | "
        f"recon_std={train_stats['recon_std']:.6e}"
    )


if __name__ == "__main__":
    main()
