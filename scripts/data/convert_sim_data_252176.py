import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import cv2
import h5py
from pathlib import Path
from tqdm import tqdm  # 用于显示进度条
from skimage.exposure import match_histograms

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ================= 配置区 =================
MAT_PATH = PROJECT_ROOT / "data/processed/train_sim/mat/eit_pathology_dataHR_200seq.mat"
REF_REAL_DATA_PATH = PROJECT_ROOT / "data/processed/npz_norm/250kHzHR/250kHzHR_normalized_0p1_99p9.npz"
SAVE_PATH = PROJECT_ROOT / "data/processed/train_sim/npz/eit_pathology_dataHR_200seq.npz"
DISPLAY_SIZE = (252, 176)


# =========================================

def create_blue_black_cmap():
    colors = ["black", "#000080", "#0066FF", "#00FFFF", "white"]
    nodes = [0.0, 0.2, 0.5, 0.8, 1.0]
    return mcolors.LinearSegmentedColormap.from_list("blue_black", list(zip(nodes, colors)))


def convert_and_process():
    print(f"🚀 开始极速转换流程: {MAT_PATH}")

    if not os.path.exists(MAT_PATH):
        raise FileNotFoundError(f"❌ 找不到文件: {MAT_PATH}")

    # ================= 1. 读取参考真实数据 =================
    use_hist_match = False
    ref_img_flat = None  # 预先准备展平的参考图

    if os.path.exists(REF_REAL_DATA_PATH):
        print(f"✅ 加载真实参考数据: {REF_REAL_DATA_PATH}")
        ref_data = np.load(REF_REAL_DATA_PATH)
        ref_frames = ref_data['frames'] if 'frames' in ref_data else ref_data['input_data']
        # 取中间一帧作为全局灰度分布参考
        ref_img = ref_frames[len(ref_frames) // 2]
        if ref_img.ndim == 3: ref_img = ref_img[0]

        # [关键优化] 提前将参考图展平，避免循环中重复操作
        ref_img_flat = ref_img.ravel()
        use_hist_match = True
    else:
        print(f"⚠️ 未找到真实参考数据，将跳过直方图匹配。")

    # ================= 2. Chunk-by-Chunk 内存优化处理 =================
    print("📂 正在打开 MATLAB v7.3 大文件 (h5py 流式读取)...")
    with h5py.File(MAT_PATH, 'r') as f:
        # MATLAB 中的 [N, T, H, W] 在 HDF5 中是反过来的 [W, H, T, N]
        shape_in = f['input_data'].shape
        W, H, T, N = shape_in
        print(f"   检测到数据维度: 样本数(N)={N}, 帧数(T)={T}, 高(H)={H}, 宽(W)={W}")

        # 提前在内存中分配一块连续空间
        print("   预分配内存以防止 OOM...")
        final_inputs = np.empty((N, T, H, W), dtype=np.float32)
        final_targets = np.empty((N, T, H, W), dtype=np.float32)

        # 开始切片式遍历病人
        print("⚡ 开始流水线处理 (清洗 -> 归一化 -> 匹配)...")
        for i in tqdm(range(N), desc="处理样本进度", unit="病人"):
            # 1. 读取单个样本并转置为 [T, H, W]
            x = f['input_data'][:, :, :, i].transpose(2, 1, 0)
            y = f['target_data'][:, :, :, i].transpose(2, 1, 0)

            # 2. 清洗 NaN/Inf (copy=False 原地修改，省内存)
            np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # 3. 极性翻转
            if np.abs(np.min(x)) > np.max(x):
                x = -x

            # 4. 预归一化 Input (每帧独立归一化到 0-1)
            mins_x = x.min(axis=(1, 2), keepdims=True)
            maxs_x = x.max(axis=(1, 2), keepdims=True)
            ranges_x = maxs_x - mins_x
            ranges_x[ranges_x < 1e-9] = 1.0
            x = (x - mins_x) / ranges_x

            # 5. 直方图匹配 (Sim2Real 核心提速与修复)
            if use_hist_match:
                # [核心修复] 将 3D 数据展平为 1D 进行匹配，彻底避开维度/通道检查错误
                # 这样匹配的是整体分布，且保留了 32 帧内部的相对亮度结构
                x_flat = x.ravel()
                matched_flat = match_histograms(x_flat, ref_img_flat, channel_axis=None)
                x = matched_flat.reshape(x.shape)

            # 6. 处理 Target (只需归一化)
            mins_y = y.min(axis=(1, 2), keepdims=True)
            maxs_y = y.max(axis=(1, 2), keepdims=True)
            ranges_y = maxs_y - mins_y
            ranges_y[ranges_y < 1e-9] = 1.0
            y = (y - mins_y) / ranges_y

            # 7. 写入预分配的数组
            final_inputs[i] = x
            final_targets[i] = y

    # ================= 3. 保存到磁盘 =================
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    print(f"💾 正在将处理好的 4D 张量压缩保存至: {SAVE_PATH} (需要一点时间...)")
    np.savez_compressed(SAVE_PATH, input_data=final_inputs, target_data=final_targets)

    # ================= 4. 画图验证 =================
    print("📸 生成效果验证图 check_sim2real.png ...")
    idx_n, idx_t = 0, T // 2

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), facecolor='black')

    # Plot 1: Input Image
    im1 = axs[0].imshow(final_inputs[idx_n, idx_t], cmap=create_blue_black_cmap(), vmin=0, vmax=1)
    axs[0].set_title("Processed Input (Sim2Real)", color='white')
    axs[0].axis('off')

    # Plot 2: Target Image
    im2 = axs[1].imshow(final_targets[idx_n, idx_t], cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("Target (Ground Truth)", color='white')
    axs[1].axis('off')

    # Plot 3: Histograms
    axs[2].set_facecolor('black')
    if use_hist_match:
        # 画出展平后的分布对比
        axs[2].hist(ref_img_flat, bins=50, density=True, color='cyan', histtype='step', label='Real Target Dist')
    axs[2].hist(final_inputs[idx_n, idx_t].flatten(), bins=50, density=True, color='lime', linestyle='--',
                histtype='step', label='Sim Input Dist')
    axs[2].set_title("Pixel Intensity Distribution", color='white')
    axs[2].tick_params(colors='white')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('check_sim2real.png', facecolor='black')
    print("🎉 转换大功告成！已修复维度错误！")


if __name__ == "__main__":
    convert_and_process()
