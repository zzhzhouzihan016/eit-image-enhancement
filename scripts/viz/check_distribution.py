import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 路径配置
REAL_PATH = PROJECT_ROOT / "data/processed/npz_norm/250kHz/250kHz_normalized_0p1_99p9.npz"
SIM_PATH = PROJECT_ROOT / "data/processed/train_sim/npz/simulation_train.npz"


def check_hist():
    print("加载数据...")
    real_data = np.load(REAL_PATH)['frames']
    sim_data = np.load(SIM_PATH)['input']  # 注意是 input (模糊的那一组)

    # 展平并只取 ROI (大于0的部分)
    real_pixels = real_data[real_data > 0].flatten()
    sim_pixels = sim_data[sim_data > 0].flatten()

    plt.figure(figsize=(10, 6))

    # 画直方图
    plt.hist(real_pixels, bins=100, alpha=0.5, label='Real Data', density=True, color='blue', range=(0, 1))
    plt.hist(sim_pixels, bins=100, alpha=0.5, label='Sim Data (Input)', density=True, color='orange', range=(0, 1))

    plt.title("Distribution Mismatch Check: Real vs Simulation")
    plt.xlabel("Pixel Value (Normalized 0-1)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = PROJECT_ROOT / "outputs/figures/distribution_check.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ 诊断图已生成: {output_path}")
    print("如果两个波峰完全不重合，说明仿真参数（电导率设置）需要调整！")


if __name__ == "__main__":
    check_hist()
