import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eit.segmentation.medsam import draw_bbox, overlay_mask, segment_torso_and_lungs


def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        return array

    if path.suffix.lower() == ".npz":
        loaded = np.load(path)
        first_key = loaded.files[0]
        return loaded[first_key]

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resolve_default_output_dir(image_path: Path) -> Path:
    if image_path.parent.name == "selected_slices" and image_path.parent.parent.name == "slice":
        case_dir = image_path.parent.parent.parent
    elif image_path.parent.name == "slice":
        case_dir = image_path.parent.parent
    else:
        case_dir = image_path.parent
    return case_dir / "segmentation" / "medsam"


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def save_overlay(image_rgb: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="对 2D CT slice 使用 MedSAM 分割 torso 和 lungs。")
    parser.add_argument("--image", type=str, required=True, help="输入图像，可为 .npy / .npz / .png")
    parser.add_argument("--checkpoint", type=str, required=True, help="MedSAM 权重文件，例如 medsam_vit_b.pth")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录；默认保存到病例目录下的 segmentation/medsam/",
    )
    parser.add_argument("--device", type=str, default=None, help="推理设备，例如 cuda / cpu")
    parser.add_argument("--model-type", type=str, default="vit_b", help="SAM backbone 类型，默认 vit_b")
    parser.add_argument("--torso-only", action="store_true", help="仅分割 torso，不分割 lungs")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else resolve_default_output_dir(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    result = segment_torso_and_lungs(
        image=image,
        checkpoint_path=args.checkpoint,
        device=args.device,
        model_type=args.model_type,
        predict_lungs=not args.torso_only,
    )

    torso_overlay = overlay_mask(result.image_rgb, result.torso_mask, color=(255, 80, 80), alpha=0.35)
    torso_overlay = draw_bbox(torso_overlay, result.torso_bbox, color=(0, 255, 0), thickness=2)
    save_overlay(torso_overlay, output_dir / "torso_overlay.png", title="Torso segmentation")

    np.save(output_dir / "torso_mask.npy", result.torso_mask.astype(np.uint8))
    save_mask_png(result.torso_mask, output_dir / "torso_mask.png")

    meta: dict[str, Any] = {
        "input_image": str(image_path),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "model_type": args.model_type,
        "torso_bbox_xyxy": result.torso_bbox,
        "predict_lungs": not args.torso_only,
    }

    if result.lung_mask is not None and result.lung_bbox is not None:
        lung_overlay = overlay_mask(result.image_rgb, result.lung_mask, color=(80, 180, 255), alpha=0.35)
        lung_overlay = draw_bbox(lung_overlay, result.lung_bbox, color=(255, 255, 0), thickness=2)
        save_overlay(lung_overlay, output_dir / "lung_overlay.png", title="Lung segmentation")

        combined = overlay_mask(result.image_rgb, result.torso_mask, color=(255, 80, 80), alpha=0.20)
        combined = overlay_mask(combined, result.lung_mask, color=(80, 180, 255), alpha=0.35)
        combined = draw_bbox(combined, result.torso_bbox, color=(0, 255, 0), thickness=2)
        combined = draw_bbox(combined, result.lung_bbox, color=(255, 255, 0), thickness=2)
        save_overlay(combined, output_dir / "combined_overlay.png", title="Torso + Lung segmentation")

        np.save(output_dir / "lung_mask.npy", result.lung_mask.astype(np.uint8))
        save_mask_png(result.lung_mask, output_dir / "lung_mask.png")
        if result.lung_bbox_mask is not None:
            save_mask_png(result.lung_bbox_mask > 0, output_dir / "lung_bbox_candidate_mask.png")

        meta["lung_bbox_xyxy"] = result.lung_bbox

    save_mask_png(result.torso_bbox_mask > 0, output_dir / "torso_bbox_candidate_mask.png")

    with open(output_dir / "torso_bbox.json", "w", encoding="utf-8") as file:
        json.dump({"bbox_xyxy": result.torso_bbox}, file, ensure_ascii=False, indent=2)
    if result.lung_bbox is not None:
        with open(output_dir / "lung_bbox.json", "w", encoding="utf-8") as file:
            json.dump({"bbox_xyxy": result.lung_bbox}, file, ensure_ascii=False, indent=2)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)

    print("✅ MedSAM 分割完成")
    print(f"   输入图像: {image_path}")
    print(f"   输出目录: {output_dir}")
    print(f"   Torso bbox: {result.torso_bbox}")
    if result.lung_bbox is not None:
        print(f"   Lung bbox: {result.lung_bbox}")


if __name__ == "__main__":
    main()
