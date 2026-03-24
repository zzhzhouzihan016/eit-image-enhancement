import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ================= 配置区 =================
# 建议使用你刚才生成的 V4 仿真数据，或者真实数据
# DATA_PATH = PROJECT_ROOT / "data/processed/train_sim/npz/simulation_train_v4_imbalanced_176x252.npz"
DATA_PATH = PROJECT_ROOT / "data/processed/npz_norm/250kHzHR/250kHzHR_normalized_0p1_99p9.npz"

# 如果是仿真数据(N, T, H, W)，设为 True；如果是真实长序列(Total, H, W)，设为 False
IS_SIMULATION_FORMAT = False

# 截取多少帧来分析 (太长了看不清，取 100-200 帧正好包含几个周期)
ANALYZE_LEN = 150


# =======================================

def normalize_signal(s):
    """归一化到 0-1 方便画图对比"""
    return (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-9)


def verify_idea():
    print(f"1. 加载数据: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print("❌ 文件不存在")
        return

    data = np.load(DATA_PATH)

    # 统一数据格式为 [Time, H, W]
    if IS_SIMULATION_FORMAT:
        # 如果是仿真数据，取第一个样本的序列
        # 假设 key 是 'input_data'
        raw = data['input_data']  # [N, T, H, W]
        seq = raw[0]  # 取第0个样本 -> [T, H, W]
    else:
        # 如果是真实数据
        key = 'frames' if 'frames' in data else 'input_data'
        seq = data[key]  # [Total, H, W]

    # 截取一段来分析
    seq = seq[:ANALYZE_LEN]
    T, H, W = seq.shape
    print(f"   分析序列维度: {seq.shape} (T={T}, H={H}, W={W})")

    # ================= 验证步骤 1: 提取曲线 =================
    print("2. 计算全局与局部曲线...")

    # A. 全局像素和曲线 (Global Sum Curve) -> 导师说的“正弦曲线”
    global_curve = np.sum(seq, axis=(1, 2))

    # B. 寻找典型像素
    # 计算每个像素在时间轴上的方差
    variance_map = np.var(seq, axis=0)

    # 找方差最大的点 -> 肯定是肺部呼吸最剧烈的地方
    lung_y, lung_x = np.unravel_index(np.argmax(variance_map), (H, W))
    lung_pixel_curve = seq[:, lung_y, lung_x]

    # 找方差很小但不是0的点 -> 背景/噪声
    # 这里我们找方差在 10% 分位数左右的点，避开纯黑背景
    valid_mask = variance_map > 1e-6
    if np.sum(valid_mask) > 0:
        flat_indices = np.where(valid_mask.flatten())[0]
        # 随机挑一个低方差的点
        # 先排序，取比较靠前的
        sorted_indices = np.argsort(variance_map.flatten()[flat_indices])
        noise_idx = flat_indices[sorted_indices[len(sorted_indices) // 5]]  # 取前 20% 弱的点
        noise_y, noise_x = np.unravel_index(noise_idx, (H, W))
    else:
        noise_y, noise_x = 0, 0

    noise_pixel_curve = seq[:, noise_y, noise_x]

    # ================= 验证步骤 2: 计算相关性热力图 =================
    print("3. 计算全图相关性 (Correlation Map)...")

    # 向量化计算 Pearson 相关系数
    # 1. 展平: [T, H, W] -> [T, N_pixels]
    pixels_flat = seq.reshape(T, -1)

    # 2. 归一化 (减均值，除标准差)
    # 注意加 1e-9 防止除零
    pixels_mean = pixels_flat.mean(axis=0)
    pixels_std = pixels_flat.std(axis=0) + 1e-9
    pixels_norm = (pixels_flat - pixels_mean) / pixels_std

    global_mean = global_curve.mean()
    global_std = global_curve.std() + 1e-9
    global_norm = (global_curve - global_mean) / global_std

    # 3. 点积计算相关性: Mean(A * B)
    # global_norm: [T], pixels_norm: [T, N]
    # 结果: [N]
    correlation_flat = np.mean(pixels_norm * global_norm[:, np.newaxis], axis=0)

    # 4. 变回图像形状
    correlation_map = correlation_flat.reshape(H, W)

    # ================= 验证步骤 3: 绘图展示 =================
    print("4. 生成验证报告图...")
    plt.figure(figsize=(14, 9))

    # --- 子图 1: 时域波形对比 ---
    plt.subplot(2, 2, 1)
    plt.plot(normalize_signal(global_curve), 'k-', linewidth=2, label='Global Sum (Hypothesis)', alpha=0.6)
    plt.plot(normalize_signal(lung_pixel_curve), 'r--', label='Lung Pixel')
    plt.plot(normalize_signal(noise_pixel_curve), 'g:', label='Noise Pixel')
    plt.title("Time Domain Analysis")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlabel("Frame")

    # --- 子图 2: 频域分析 (FFT) ---
    plt.subplot(2, 2, 2)

    def plot_fft(sig, color, label):
        fft_vals = np.fft.fft(sig)
        freqs = np.fft.fftfreq(len(sig))
        # 只取正半轴
        pos_mask = freqs > 0
        plt.plot(freqs[pos_mask], np.abs(fft_vals[pos_mask]), color=color, label=label)

    plot_fft(normalize_signal(global_curve), 'black', 'Global Sum')
    plot_fft(normalize_signal(lung_pixel_curve), 'red', 'Lung Pixel')
    plot_fft(normalize_signal(noise_pixel_curve), 'green', 'Noise Pixel')
    plt.title("Frequency Domain Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("Frequency")

    # --- 子图 3: 相关性热力图 (最关键的证据) ---
    plt.subplot(2, 1, 2)
    # 使用 jet 配色：红色代表强相关(肺)，蓝色代表不相关(噪声)，深蓝/青色代表负相关
    im = plt.imshow(correlation_map, cmap='jet', vmin=-1, vmax=1)
    plt.colorbar(im, label="Pearson Correlation Coeff")
    plt.title("Spatial Correlation Map\n(Correlation between 'Global Sum' and 'Each Pixel')")
    plt.xlabel("If this looks like lungs, the hypothesis is TRUE!")

    # 标记出我们选的点
    plt.plot(lung_x, lung_y, 'rx', markersize=10, markeredgewidth=2, label='Lung Sample')
    plt.plot(noise_x, noise_y, 'gx', markersize=10, markeredgewidth=2, label='Noise Sample')
    plt.legend()

    plt.tight_layout()
    output_path = PROJECT_ROOT / "outputs/figures/hypothesis_verification.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"✅ 验证完成！请查看生成的 '{output_path}'")


if __name__ == "__main__":
    verify_idea()
