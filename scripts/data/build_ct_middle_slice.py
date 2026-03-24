import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eit.io.dicom import load_dicom_series
from eit.preprocess.ct import (
    build_index_range,
    gray_to_rgb,
    select_middle_slice,
    select_rib_proxy_slice,
    window_hu,
)


def save_slice_png(image_u8: np.ndarray, save_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(image_u8, cmap="gray")
    plt.title(title)
    plt.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def save_contact_sheet(
    volume_u8: np.ndarray,
    indices: List[int],
    selected_indices: List[int],
    save_path: Path,
    title: str,
) -> None:
    n_cols = min(5, len(indices))
    n_rows = int(np.ceil(len(indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, indices):
        ax.imshow(volume_u8[idx], cmap="gray")
        label = f"slice={idx}"
        if idx in selected_indices:
            label += " ✓"
            for spine in ax.spines.values():
                spine.set_edgecolor("lime")
                spine.set_linewidth(3)
        ax.set_title(label)
        ax.axis("off")

    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_coronal_preview(
    volume_u8: np.ndarray,
    range_indices: List[int],
    selected_indices: List[int],
    save_path: Path,
) -> None:
    mid_col = volume_u8.shape[2] // 2
    coronal = np.transpose(volume_u8[:, :, mid_col], (1, 0))

    plt.figure(figsize=(10, 6))
    plt.imshow(coronal, cmap="gray", aspect="auto")
    range_start = min(range_indices)
    range_end = max(range_indices)
    plt.axvspan(range_start, range_end, color="yellow", alpha=0.15, label="candidate range")
    for idx in selected_indices:
        plt.axvline(idx, color="lime", linestyle="-", linewidth=1.4)
    plt.title("Coronal preview with candidate range and selected axial slices")
    plt.xlabel("Axial slice index")
    plt.ylabel("Image row")
    plt.legend(loc="upper right")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close()


def style_axis(ax: Axes, idx: int, selected: bool) -> None:
    label = f"slice={idx}"
    if selected:
        label += " ✓"
    ax.set_title(label, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("lime" if selected else "#666666")
        spine.set_linewidth(3 if selected else 1)


def interactive_select_slices(volume_u8: np.ndarray, candidate_indices: List[int], preselected: List[int]) -> List[int]:
    selected = set(preselected)
    n_cols = min(5, len(candidate_indices))
    n_rows = int(np.ceil(len(candidate_indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    axis_to_index: Dict[Axes, int] = {}

    for ax, idx in zip(axes, candidate_indices):
        ax.imshow(volume_u8[idx], cmap="gray")
        style_axis(ax, idx, idx in selected)
        axis_to_index[ax] = idx

    for ax in axes[len(candidate_indices):]:
        ax.axis("off")

    fig.suptitle("点击切片进行勾选；按 Enter 确认，按 c 清空，按 a 全选，按 q 退出", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def redraw() -> None:
        for ax, idx in axis_to_index.items():
            style_axis(ax, idx, idx in selected)
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes not in axis_to_index:
            return
        idx = axis_to_index[event.inaxes]
        if idx in selected:
            selected.remove(idx)
        else:
            selected.add(idx)
        redraw()

    def on_key(event) -> None:
        if event.key == "enter":
            plt.close(fig)
        elif event.key == "c":
            selected.clear()
            redraw()
        elif event.key == "a":
            selected.update(candidate_indices)
            redraw()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return sorted(selected)


def resolve_selected_index(volume_hu: np.ndarray, selector: str, slice_index: Optional[int]) -> int:
    if selector == "index":
        if slice_index is None:
            raise ValueError("当 selector=index 时，必须提供 --slice-index")
        if not (0 <= slice_index < volume_hu.shape[0]):
            raise ValueError(f"--slice-index 超出范围: {slice_index}, 合法范围为 [0, {volume_hu.shape[0] - 1}]")
        return int(slice_index)

    if selector == "rib_proxy":
        return select_rib_proxy_slice(volume_hu)

    return select_middle_slice(volume_hu.shape[0])


def build_case_output_dir(output_root: Path, case_id: str) -> Path:
    case_output_dir = output_root / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)
    return case_output_dir


def build_case_stage_dirs(case_output_dir: Path) -> Dict[str, Path]:
    selection_dir = case_output_dir / "selection"
    slice_dir = case_output_dir / "slice"
    selection_dir.mkdir(parents=True, exist_ok=True)
    slice_dir.mkdir(parents=True, exist_ok=True)
    return {
        "selection": selection_dir,
        "slice": slice_dir,
    }


def export_selected_slices(
    slice_output_dir: Path,
    volume_hu: np.ndarray,
    volume_u8: np.ndarray,
    selected_indices: List[int],
) -> None:
    selections_dir = slice_output_dir / "selected_slices"
    selections_dir.mkdir(parents=True, exist_ok=True)

    for idx in selected_indices:
        stem = f"slice_{idx:04d}"
        slice_hu = volume_hu[idx].astype(np.float32)
        slice_rgb = gray_to_rgb(volume_u8[idx]).astype(np.uint8)

        np.save(selections_dir / f"{stem}_hu.npy", slice_hu)
        np.save(selections_dir / f"{stem}_rgb.npy", slice_rgb)
        save_slice_png(volume_u8[idx], selections_dir / f"{stem}.png", title=f"Selected slice {idx}")

    if len(selected_indices) == 1:
        idx = selected_indices[0]
        np.save(slice_output_dir / "selected_slice_hu.npy", volume_hu[idx].astype(np.float32))
        np.save(slice_output_dir / "selected_slice_rgb.npy", gray_to_rgb(volume_u8[idx]).astype(np.uint8))
        save_slice_png(volume_u8[idx], slice_output_dir / "selected_slice_preview.png", title=f"Selected slice {idx}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="读取 DICOM 序列，转换为 HU，应用肺窗，并导出选定切片及预览图。"
    )
    parser.add_argument("--dicom-dir", type=str, required=True, help="单个 CT 序列所在目录，可位于外接硬盘")
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/interim/ct_cases",
        help="输出根目录，默认保存到项目内的 data/interim/ct_cases",
    )
    parser.add_argument("--case-id", type=str, default=None, help="病例标识；默认取 DICOM 目录名")
    parser.add_argument(
        "--selector",
        type=str,
        choices=["middle", "rib_proxy", "index"],
        default="middle",
        help="切片选择方式：middle 为体数据中间层；rib_proxy 为近似胸中上部肺野代理层；index 为手动指定",
    )
    parser.add_argument("--slice-index", type=int, default=None, help="当 selector=index 时使用")
    parser.add_argument("--window-width", type=float, default=1500.0, help="肺窗窗宽，默认 1500")
    parser.add_argument("--window-level", type=float, default=-600.0, help="肺窗窗位，默认 -600")
    parser.add_argument("--range-size", type=int, default=21, help="候选范围包含的切片数，建议 15~31")
    parser.add_argument("--no-interactive", action="store_true", help="关闭交互窗口，仅按默认选中逻辑导出")
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir).expanduser().resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()

    case_id = args.case_id or dicom_dir.name
    case_output_dir = build_case_output_dir(output_root, case_id)
    stage_dirs = build_case_stage_dirs(case_output_dir)
    selection_output_dir = stage_dirs["selection"]
    slice_output_dir = stage_dirs["slice"]

    print(f"📂 读取 DICOM 序列: {dicom_dir}")
    series = load_dicom_series(dicom_dir)
    volume_hu = series.volume_hu
    volume_u8 = window_hu(volume_hu, window_width=args.window_width, window_level=args.window_level)

    middle_index = select_middle_slice(volume_hu.shape[0])
    rib_proxy_index = select_rib_proxy_slice(volume_hu)
    default_index = resolve_selected_index(volume_hu, args.selector, args.slice_index)
    candidate_indices = build_index_range(default_index, volume_hu.shape[0], range_size=args.range_size)
    selected_indices = [default_index]

    save_contact_sheet(
        volume_u8,
        candidate_indices,
        selected_indices,
        selection_output_dir / "candidate_range_before.png",
        title=f"Candidate range before interactive selection | selector={args.selector}",
    )
    save_coronal_preview(
        volume_u8,
        candidate_indices,
        selected_indices,
        selection_output_dir / "coronal_before.png",
    )

    if not args.no_interactive:
        selected_indices = interactive_select_slices(volume_u8, candidate_indices, preselected=selected_indices)

    if not selected_indices:
        selected_indices = [default_index]
        print(f"⚠️ 未勾选任何切片，已回退到默认切片: {default_index}")

    export_selected_slices(slice_output_dir, volume_hu, volume_u8, selected_indices)
    save_contact_sheet(
        volume_u8,
        candidate_indices,
        selected_indices,
        selection_output_dir / "candidate_range_after.png",
        title="Candidate range after selection",
    )
    save_coronal_preview(
        volume_u8,
        candidate_indices,
        selected_indices,
        selection_output_dir / "coronal_after.png",
    )

    meta = {
        "dicom_dir": str(dicom_dir),
        "case_id": case_id,
        "selector": args.selector,
        "default_index": int(default_index),
        "selected_indices": [int(idx) for idx in selected_indices],
        "middle_index": int(middle_index),
        "rib_proxy_index": int(rib_proxy_index),
        "candidate_indices": [int(idx) for idx in candidate_indices],
        "volume_shape": list(map(int, volume_hu.shape)),
        "spacing_xy_mm": [float(series.spacing_xy[0]), float(series.spacing_xy[1])],
        "slice_thickness_mm": series.slice_thickness,
        "window_width": float(args.window_width),
        "window_level": float(args.window_level),
        "range_size": int(args.range_size),
        "patient_id": series.patient_id,
        "study_uid": series.study_uid,
        "series_uid": series.series_uid,
        "rows": int(series.rows),
        "cols": int(series.cols),
        "used_dicom_count": len(series.dicom_paths),
    }
    with open(case_output_dir / "case_metadata.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)

    print("✅ 处理完成")
    print(f"   体数据形状: {volume_hu.shape}")
    print(f"   中间层索引: {middle_index}")
    print(f"   rib_proxy 索引: {rib_proxy_index}")
    print(f"   默认切片索引: {default_index}")
    print(f"   候选范围: {candidate_indices[0]} ~ {candidate_indices[-1]}")
    print(f"   最终保留切片: {selected_indices}")
    print(f"   导出格式: 每张切片同时保存为 PNG 和 NPY")
    print(f"   输出目录: {case_output_dir}")


if __name__ == "__main__":
    main()
