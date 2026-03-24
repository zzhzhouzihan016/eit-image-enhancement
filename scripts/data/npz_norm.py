import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ================= 配置区 =================
# 原始数据路径
INPUT_PATH = PROJECT_ROOT / "data/raw/npz/250kHzHR/250kHzHR.npz"
# 输出保存路径
OUTPUT_PATH = PROJECT_ROOT / "data/processed/npz_norm/250kHzHR/250kHzHR_normalized_0p1_99p9.npz"

# 实验确定的最优参数
BEST_LOW_Q = 0.1
BEST_HIGH_Q = 99.9


# =========================================

def normalize_dataset(input_path, output_path, low_q, high_q):
    print(f"1. 加载数据: {input_path}")
    if not os.path.exists(input_path):
        print("错误：找不到输入文件！")
        return

    data = np.load(input_path)
    # 假设你的 npz 里数组叫 'frames' 或 'data'，请根据实际情况修改
    # 如果不确定 key 是什么，可以用 list(data.keys()) 查看
    key = list(data.keys())[0]
    frames = data[key]
    print(f"   数据形状: {frames.shape}, 数据类型: {frames.dtype}")

    # 2. 提取 ROI (假设 > 0 为有效区域)
    # 注意：这里我们基于“全局”还是“单帧”计算分位数？
    # 对于 EIT 连续成像，通常建议对“每一帧”单独归一化（能够适应呼吸幅度的变化），
    # 或者对“全量数据”统一归一化（保留呼吸幅度的绝对差异）。
    # *建议*：对于提升清晰度任务，通常【单帧归一化】效果更好，因为对比度被拉得更满。

    print(f"2. 开始归一化处理 (Low={low_q}%, High={high_q}%) ...")

    frames_norm = np.zeros_like(frames, dtype=np.float32)

    for i in range(len(frames)):
        frame = frames[i]

        # 提取 ROI 掩膜
        mask = frame > 0
        roi = frame[mask]

        if roi.size > 0:
            # 计算当前帧的分位数界限
            vmin = np.percentile(roi, low_q)
            vmax = np.percentile(roi, high_q)

            # 防止分母为0
            if vmax <= vmin:
                vmax = vmin + 1e-6

            # 截断 (Clip) + 线性映射 (0-1)
            # 先截断，再计算，公式：(x - vmin) / (vmax - vmin)
            # 这一步等价于先 map 再 clip(0,1)
            norm_roi = (roi - vmin) / (vmax - vmin)
            norm_roi = np.clip(norm_roi, 0, 1)

            # 赋值回图像
            frames_norm[i][mask] = norm_roi

        # 背景区域本身就是 0 (初始化时 zeros_like)，所以不需要额外操作
        # 如果原始背景是 -1，这里已经被 0 覆盖了，完美。

        if i % 500 == 0:
            print(f"   已处理 {i}/{len(frames)} 帧...")

    print("3. 处理完成，正在保存...")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存为压缩的 npz
    np.savez_compressed(output_path, frames=frames_norm)
    print(f"✅ 成功保存至: {output_path}")
    print(f"   输出类型: {frames_norm.dtype}")
    print(f"   数值范围: [{frames_norm.min():.4f}, {frames_norm.max():.4f}]")


if __name__ == "__main__":
    normalize_dataset(INPUT_PATH, OUTPUT_PATH, BEST_LOW_Q, BEST_HIGH_Q)
