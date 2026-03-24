import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("data/processed/train_sim/lctsc_cem_pathology_jac32_gt64")
DEFAULT_MANIFESTS = ("global_samples_manifest.csv", "samples_manifest.csv")


def build_relative_sample_dir(case_id: str, slice_index: int, sample_name: str) -> Path:
    return Path("cases") / case_id / "slices" / f"slice_{slice_index:03d}" / sample_name


def rewrite_manifest(manifest_path: Path, create_backup: bool = True) -> tuple[int, int]:
    with open(manifest_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        return 0, 0

    for required in ("sample_dir", "npz_path", "metadata_path"):
        if required not in fieldnames:
            fieldnames.append(required)

    changed_rows = 0
    for row in rows:
        relative_sample_dir = build_relative_sample_dir(
            case_id=row["case_id"],
            slice_index=int(row["slice_index"]),
            sample_name=row["sample_name"],
        ).as_posix()
        relative_npz_path = f"{relative_sample_dir}/sequence_data.npz"
        relative_metadata_path = f"{relative_sample_dir}/metadata.json"

        original_triplet = (
            row.get("sample_dir", ""),
            row.get("npz_path", ""),
            row.get("metadata_path", ""),
        )

        row["sample_dir"] = relative_sample_dir
        row["npz_path"] = relative_npz_path
        row["metadata_path"] = relative_metadata_path

        if original_triplet != (
            relative_sample_dir,
            relative_npz_path,
            relative_metadata_path,
        ):
            changed_rows += 1

    if create_backup and changed_rows > 0:
        backup_path = manifest_path.with_name(f"{manifest_path.stem}.windows_backup.csv")
        if not backup_path.exists():
            shutil.copy2(manifest_path, backup_path)

    with open(manifest_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), changed_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="将 LCTSC manifest 中的 Windows 绝对路径改写为相对路径。")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="数据集根目录。",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不保留原始 manifest 备份。",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")

    for manifest_name in DEFAULT_MANIFESTS:
        manifest_path = dataset_root / manifest_name
        if not manifest_path.exists():
            print(f"跳过缺失文件: {manifest_path}")
            continue

        total_rows, changed_rows = rewrite_manifest(
            manifest_path=manifest_path,
            create_backup=not args.no_backup,
        )
        print(f"{manifest_path.name}: total_rows={total_rows}, changed_rows={changed_rows}")


if __name__ == "__main__":
    main()
