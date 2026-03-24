import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ================= 配置 =================
MAT_PATH = PROJECT_ROOT / "data/processed/train_sim/mat/eit_simulation_data.mat"
SAVE_PATH = PROJECT_ROOT / "data/processed/train_sim/npz/simulation_train.npz"


# =======================================

def convert_and_visualize():
    print(f"1. 加载 MATLAB 数据: {MAT_PATH}")
    if not os.path.exists(MAT_PATH):
        print("错误: 找不到 .mat 文件，请检查路径")
        return

    mat = scipy.io.loadmat(MAT_PATH)

    # MATLAB 里的维度是 [Sequence, Frame, H, W]
    # 我们需要把它展平成 [Total_Frames, H, W] 以便训练
    # 假设有 10 个序列，每个 50 帧 -> 总共 500 帧

    inputs = mat['input_data']  # 模糊图
    targets = mat['target_data']  # 清晰图

    print(f"   原始维度: Input={inputs.shape}, Target={targets.shape}")

    # 展平 (Flatten) 序列维度和帧维度
    # reshape: [S, F, H, W] -> [S*F, H, W]
    total_frames = inputs.shape[0] * inputs.shape[1]
    h, w = inputs.shape[2], inputs.shape[3]

    inputs = inputs.reshape(total_frames, h, w)
    targets = targets.reshape(total_frames, h, w)

    print(f"2. 归一化处理...")
    # 对 Input (模糊图) 使用 robust 归一化 (0.1% - 99.9%) - 模拟真实数据的处理
    in_min = np.percentile(inputs, 0.1)
    in_max = np.percentile(inputs, 99.9)
    inputs = (inputs - in_min) / (in_max - in_min)
    inputs = np.clip(inputs, 0, 1)

    # 对 Target (真值) 使用简单 Min-Max 归一化 (因为它很干净)
    tar_min = targets.min()
    tar_max = targets.max()
    targets = (targets - tar_min) / (tar_max - tar_min)

    # 转为 float32
    inputs = inputs.astype(np.float32)
    targets = targets.astype(np.float32)

    print(f"3. 保存为 .npz: {SAVE_PATH}")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez_compressed(SAVE_PATH, input=inputs, target=targets)

    # --- 可视化检查 (这一步很重要！) ---
    print("4. 可视化检查 (保存为 check_sim.png)...")
    idx = 25  # 随便选一帧看看

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(inputs[idx], cmap='jet')
    plt.title(f'Input (Blurred)\nShape: {inputs.shape}')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(targets[idx], cmap='jet')
    plt.title(f'Target (Ground Truth)\nShape: {targets.shape}')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('check_sim.png')
    print("✅ 完成！请查看 'check_sim.png' 确认肺部位置是否对应。")


if __name__ == "__main__":
    convert_and_visualize()
