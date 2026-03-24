import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import cv2
from pathlib import Path
from skimage.exposure import match_histograms

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ================= 配置区 =================
MAT_PATH = PROJECT_ROOT / "data/processed/train_sim/mat/eit_pathology_data_100seq.mat"
REF_REAL_DATA_PATH = PROJECT_ROOT / "data/processed/npz_norm/250kHz/250kHz_normalized_0p1_99p9.npz"
SAVE_PATH = PROJECT_ROOT / "data/processed/train_sim/npz/eit_pathology_data_100seq.npz"
DISPLAY_SIZE = (126, 88)


# =========================================

def create_blue_black_cmap():
    colors = ["black", "#000080", "#0066FF", "#00FFFF", "white"]
    nodes = [0.0, 0.2, 0.5, 0.8, 1.0]
    return mcolors.LinearSegmentedColormap.from_list("blue_black", list(zip(nodes, colors)))


def convert_with_roi_matching():
    print(f"1. 加载数据...")
    mat = scipy.io.loadmat(MAT_PATH)
    sim_inputs = mat['input_data']
    sim_targets = mat['target_data']

    # 维度展平
    total_frames = sim_inputs.shape[0] * sim_inputs.shape[1]
    h, w = sim_inputs.shape[2], sim_inputs.shape[3]
    sim_inputs = sim_inputs.reshape(total_frames, h, w)
    sim_targets = sim_targets.reshape(total_frames, h, w)

    # === [关键修复 1]：处理 EIT 差分负值问题 ===
    print("   正在检查仿真数据极性...")
    # 如果大部分像素是负的，说明是吸气导致电导率下降，我们需要取反，让肺部变亮
    # 简单判断：如果最大值很小，而最小值很负
    if np.max(sim_inputs) < np.abs(np.min(sim_inputs)):
        print("   检测到负极性信号 (吸气)，正在执行翻转 (-1)...")
        sim_inputs = -sim_inputs

    # === [关键修复 2]：预归一化到 0-1 ===
    # 为了让后面的 Mask 和直方图匹配能正常工作，必须先拉伸
    print("   执行预归一化 (Pre-Normalization)...")
    # 对每张图单独归一化，防止某些帧信号太弱被淹没
    sim_inputs_norm = []
    for img in sim_inputs:
        _min, _max = img.min(), img.max()
        if _max > _min:
            img = (img - _min) / (_max - _min)
        else:
            img = np.zeros_like(img)
        sim_inputs_norm.append(img)
    sim_inputs = np.array(sim_inputs_norm)

    # 加载真实数据
    real_data = np.load(REF_REAL_DATA_PATH)['frames']

    print("2. 构建真实数据 ROI 参考分布...")
    ref_idx = len(real_data) // 2
    reference_full = real_data[ref_idx]
    real_roi_pixels = reference_full[reference_full > 0]
    ref_roi_reshaped = real_roi_pixels[:, np.newaxis]

    print("3. 执行 ROI-Masked 直方图匹配...")
    inputs_matched = []

    for i in range(len(sim_inputs)):
        src_img = sim_inputs[i]

        # 提取 ROI (现在数据在0-1之间，用 1e-6 就安全了)
        mask = src_img > 1e-6

        if np.sum(mask) == 0:
            inputs_matched.append(src_img)
            continue

        src_roi_pixels = src_img[mask]
        src_roi_reshaped = src_roi_pixels[:, np.newaxis]

        # 直方图匹配
        matched_roi_pixels = match_histograms(src_roi_reshaped, ref_roi_reshaped, channel_axis=None)
        matched_roi_pixels = matched_roi_pixels.flatten()

        # 填回
        matched_img = np.zeros_like(src_img)
        matched_img[mask] = matched_roi_pixels
        inputs_matched.append(matched_img)

        if i % 500 == 0:
            print(f"   已处理 {i}/{total_frames} 帧...")

    inputs_matched = np.array(inputs_matched)

    # 后处理 Target
    tar_min, tar_max = sim_targets.min(), sim_targets.max()
    if tar_max > tar_min:
        sim_targets = (sim_targets - tar_min) / (tar_max - tar_min)

    inputs_matched = np.clip(inputs_matched, 0, 1)
    inputs_matched = inputs_matched.astype(np.float32)
    sim_targets = sim_targets.astype(np.float32)

    print(f"4. 保存: {SAVE_PATH}")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez_compressed(SAVE_PATH, input=inputs_matched, target=sim_targets)

    # --- 修正后的可视化 (更稳健) ---
    print("5. 生成修正版验证图 (check_roi_matching_v3.png)...")

    # 找一个信号最强的帧 (标准差最大)，确保有东西看
    stds = [np.std(im) for im in inputs_matched[:100]]
    idx = np.argmax(stds)
    print(f"   选取第 {idx} 帧进行可视化 (信号最强)...")

    img_for_show = cv2.resize(inputs_matched[idx], DISPLAY_SIZE)

    fig = plt.figure(figsize=(12, 5), facecolor='black')

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_facecolor('black')
    plt.hist(real_roi_pixels.flatten(), bins=100, density=True,
             histtype='step', color='#00FFFF', linewidth=2, label='Real ROI')

    # [关键] 动态阈值，只要大于0就算，防止阈值太高画不出图
    sim_roi_show = inputs_matched[idx][inputs_matched[idx] > 1e-6]

    if len(sim_roi_show) > 0:
        plt.hist(sim_roi_show.flatten(), bins=100, density=True,
                 histtype='step', color='#00FF00', linewidth=2, linestyle='--', label='Sim ROI')
    else:
        print("⚠️ 警告：可视化帧的 ROI 为空！")

    plt.legend(loc='upper right', facecolor='black', labelcolor='white')
    plt.title("ROI Pixel Distribution", color='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2)

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_facecolor('black')
    my_cmap = create_blue_black_cmap()
    im = plt.imshow(img_for_show, cmap=my_cmap, vmin=0, vmax=1, aspect='equal')
    plt.title("Matched Simulation Image", color='white')
    cbar = plt.colorbar(im, ax=ax2)
    cbar.ax.yaxis.set_tick_params(color='white')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('eit_pathology_data_100seq.png', facecolor='black')
    print("✅ 完成！")


if __name__ == "__main__":
    convert_with_roi_matching()
