import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ================= 配置 =================
FILE_PATH = PROJECT_ROOT / "data/raw/npz/250kHz/250kHz_all_frames.npz"

# 寻峰参数 (关键!)
# distance: 两个波峰之间最小隔多少帧？(防止把噪点当波峰)
# 如果呼吸很慢，这个值可以设大点，比如 30 或 40
MIN_DISTANCE = 30
# prominence: 波峰突起程度，过滤掉太矮的假波峰
PROMINENCE = 10

# 设备采样率 (用于推算每分钟呼吸次数)
FPS_EST = 20


# =======================================

def auto_calculate_period():
    print(f"1. 加载数据: {FILE_PATH}")
    try:
        data = np.load(FILE_PATH)
        key = data.files[0]
        frames = data[key]
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 计算曲线
    grayscale_sums = np.sum(frames, axis=(1, 2), dtype=np.float64)

    # 2. 自动寻找波峰
    print("2. 正在寻找呼吸波峰...")
    peaks, properties = find_peaks(grayscale_sums, distance=MIN_DISTANCE, prominence=PROMINENCE)

    if len(peaks) < 2:
        print("⚠️ 未检测到足够的波峰，请尝试减小 MIN_DISTANCE 或 PROMINENCE 参数。")
        return

    # 3. 计算统计数据
    # np.diff 计算相邻两个峰之间的距离
    cycle_periods = np.diff(peaks)
    avg_period = np.mean(cycle_periods)
    std_period = np.std(cycle_periods)

    # 推算每分钟呼吸次数 (RPM)
    bpm = 60 / (avg_period / FPS_EST)

    print("\n" + "=" * 40)
    print("📊 呼吸周期分析报告")
    print("=" * 40)
    print(f"检测到的呼吸次数 : {len(peaks)}")
    print(f"平均周期 (帧数)  : {avg_period:.2f} 帧")
    print(f"周期标准差       : {std_period:.2f} (越低越稳定)")
    print(f"最短周期         : {np.min(cycle_periods)} 帧")
    print(f"最长周期         : {np.max(cycle_periods)} 帧")
    print("-" * 40)
    print(f"推算呼吸频率     : {bpm:.2f} 次/分 (基于 {FPS_EST}fps)")
    print("=" * 40 + "\n")

    # 4. 绘图验证 (只画前 600 帧以免太挤)
    plot_len = min(600, len(grayscale_sums))
    valid_peaks = peaks[peaks < plot_len]

    plt.figure(figsize=(15, 6))
    plt.plot(range(plot_len), grayscale_sums[:plot_len], label='Breathing Signal')
    # 用红叉标记找到的峰
    plt.plot(valid_peaks, grayscale_sums[valid_peaks], "x", color='red', markersize=10, label='Detected Peaks')

    plt.title(f"Automatic Cycle Detection (Avg: {avg_period:.1f} frames/cycle)")
    plt.xlabel("Frame Index")
    plt.ylabel("Intensity Sum")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    auto_calculate_period()
