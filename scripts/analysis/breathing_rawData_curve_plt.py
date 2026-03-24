import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ================= 配置 =================
FILE_PATH = PROJECT_ROOT / "data/raw/npz/250kHz/250kHz_all_frames.npz"


# =======================================

def analyze_breathing_curve():
    # 1. 加载数据
    try:
        data = np.load(FILE_PATH)
        # 自动获取第一个 key (通常是 'frames' 或 'arr_0')
        key = data.files[0]
        frames = data[key]
        print(f"成功加载数据，Key为: {key}, 形状: {frames.shape}")
    except FileNotFoundError:
        print("❌ 找不到文件，请检查路径")
        return

    # frames shape 通常是 [N_frames, Height, Width]
    n_frames = frames.shape[0]

    # 2. 逐帧计算灰度总和 (Sum of Grayscale)
    # axis=(1, 2) 表示在 高(H) 和 宽(W) 维度上求和，只保留 时间(T) 维度
    # 使用 dtype=np.float64 防止求和时溢出
    grayscale_sums = np.sum(frames, axis=(1, 2), dtype=np.float64)

    # 3. 绘制曲线
    plt.figure(figsize=(12, 5))

    # 绘制时间-强度曲线
    plt.plot(range(n_frames), grayscale_sums, color='blue', linewidth=1.5, label='Global Impedance Change')

    # 标注图表
    plt.title(f"Breathing Pattern Analysis (File: {key})")
    plt.xlabel("Time (Frames)")
    plt.ylabel("Sum of Pixel Intensity (Arbitrary Unit)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 4. 简单分析：这像正弦波吗？
    plt.show()


if __name__ == "__main__":
    analyze_breathing_curve()
