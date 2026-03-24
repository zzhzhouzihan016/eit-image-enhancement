import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eit.dataset import EITSequenceDataset

# ================= 配置 =================
# 指向你最新的 .npz 文件路径
NPZ_PATH = PROJECT_ROOT / "data/processed/train_sim/npz/eit_pathology_dataHR_200seq.npz"


# =======================================

def visualize_blur_effect():
    print(f"📂 Loading dataset from {NPZ_PATH}...")

    # 实例化数据集 (开启数据增强以检查 Flip/Rotate，同时也会应用 Gaussian Blur)
    dataset = EITSequenceDataset(
        npz_path=NPZ_PATH,
        n_frames=5,
        frames_per_seq=None,  # 自动检测
        target_size=(176, 256),
        use_augmentation=True  # 开启以检查 flip/rotate 是否正常
    )

    print(f"✅ Dataset loaded. Total samples: {len(dataset)}")

    # 随机取几个样本
    indices = np.random.choice(len(dataset), 5, replace=False)

    plt.figure(figsize=(15, 10), facecolor='black')

    for i, idx in enumerate(indices):
        # 获取数据 (dataset.__getitem__ 会自动应用 Blur 和 Augmentation)
        # in_seq: [5, 176, 256], target: [1, 176, 256]
        in_seq, target_blurred = dataset[idx]

        # 为了对比，我们需要获取“未模糊”的原始 Target
        # 小技巧：直接去 dataset.targets 里拿原始数据
        real_idx = dataset.valid_indices[idx]
        mid_idx = real_idx + (dataset.n_frames // 2)
        target_sharp = dataset.targets[mid_idx]  # 这是原始的、锐利的 tensor

        # 如果 dataset 开启了 augmentation，我们取到的 target_blurred 是经过翻转的
        # 为了对比公平，我们这里只看 target_blurred 的效果，不强求和 target_sharp 像素对齐
        # (因为 target_sharp 没做 flip，target_blurred 做了 flip，形状可能不一样)
        # **但在训练中，Input 和 Target 是同步变换的，所以不用担心。**

        # 转换为 numpy 用于画图
        img_in = in_seq[-1].numpy()  # 取序列最后一帧 Input
        img_sharp = target_sharp.numpy()
        img_blur = target_blurred.squeeze().numpy()

        # 画 Input
        plt.subplot(5, 3, i * 3 + 1)
        plt.imshow(img_in, cmap='jet')
        plt.title(f"Input (Sim) #{idx}", color='white')
        plt.axis('off')

        # 画 原始 Target (Sharp)
        plt.subplot(5, 3, i * 3 + 2)
        plt.imshow(img_sharp, cmap='gray')
        plt.title("Original Target (Sharp)", color='white')
        plt.axis('off')

        # 画 处理后 Target (Blurred + Aug)
        plt.subplot(5, 3, i * 3 + 3)
        plt.imshow(img_blur, cmap='gray')
        plt.title("Training Target (Blurred)", color='white')
        plt.axis('off')

    plt.tight_layout()
    output_path = PROJECT_ROOT / "outputs/figures/check_dataset_blur.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, facecolor='black')
    print(f"📸 检查完成！请查看 {output_path}")
    print("   - 确认 'Training Target' 是否保留了肺部的主要形状？")
    print("   - 确认边缘是否变得柔和（像一团云）？")
    print("   - 如果开启了增强，确认 Input 和 Target 是否有旋转/翻转的变化？")


if __name__ == "__main__":
    visualize_blur_effect()
