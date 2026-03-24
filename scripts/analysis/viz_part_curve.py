import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ================= 配置 =================
FILE_PATH = PROJECT_ROOT / "data/raw/npz/250kHz/250kHz_all_frames.npz"
ZOOM_WINDOW = 400  # 局部放大看前多少帧


# =======================================

def view_segments():
    print(f"1. 加载数据: {FILE_PATH}")
    try:
        data = np.load(FILE_PATH)
        key = data.files[0]  # 自动获取第一个键值
        frames = data[key]
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    print("2. 计算全帧灰度总和...")
    # sum axis=(1,2) 表示把一张图所有像素加起来，变成一个数
    grayscale_sums = np.sum(frames, axis=(1, 2), dtype=np.float64)
    total_frames = len(grayscale_sums)

    print("3. 绘图...")
    plt.figure(figsize=(12, 10))

    # --- 子图 1: 全局视图 ---
    plt.subplot(2, 1, 1)
    plt.plot(grayscale_sums, color='#1f77b4', linewidth=1)
    plt.title(f"Global View (All {total_frames} Frames)")
    plt.xlabel("Frame Index")
    plt.ylabel("Intensity Sum")
    plt.grid(True, alpha=0.3)
    # 标记出我们即将放大的区域
    plt.axvspan(0, ZOOM_WINDOW, color='red', alpha=0.1, label='Zoomed Area')
    plt.legend()

    # --- 子图 2: 局部放大视图 ---
    plt.subplot(2, 1, 2)
    # 只取前 ZOOM_WINDOW 帧
    segment = grayscale_sums[:ZOOM_WINDOW]
    plt.plot(range(ZOOM_WINDOW), segment, color='#d62728', linewidth=2)
    plt.title(f"Zoomed View (First {ZOOM_WINDOW} Frames)")
    plt.xlabel("Frame Index")
    plt.ylabel("Intensity Sum")

    # 设置更细密的网格，方便数格子
    plt.grid(True, which='major', alpha=0.6)
    plt.grid(True, which='minor', alpha=0.2, linestyle='--')
    plt.minorticks_on()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    view_segments()
