import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ================= 配置区 =================
# 1. 训练数据路径
INPUT_NPZ = PROJECT_ROOT / "data/processed/train_sim/npz/test.npz"

# 2. 输出视频路径
OUTPUT_VIDEO = PROJECT_ROOT / "outputs/videos/train_sim/test.mp4"

# 3. 视频参数
FPS = 10
# 单图尺寸 (126x88 的 3倍放大) -> (378, 264)
# 最终视频宽度会是 378 * 2 = 756
DISPLAY_SIZE = (378, 264)


# =========================================

def create_pure_blue_black_lut():
    """纯净蓝黑渐变"""
    colors = [
        (0.0, 0.0, 0.0),  # Black
        (0.0, 0.0, 0.4),  # Dark Blue
        (0.0, 0.2, 0.8),  # Medium Blue
        (0.0, 0.6, 1.0),  # Bright Blue
        (0.8, 1.0, 1.0)  # Whitish Cyan
    ]
    nodes = [0.0, 0.15, 0.4, 0.7, 1.0]
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_black", list(zip(nodes, colors)))

    lut = np.empty((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        rgba = cmap(i / 255.0)
        lut[i, 0, 0] = int(rgba[2] * 255)  # B
        lut[i, 0, 1] = int(rgba[1] * 255)  # G
        lut[i, 0, 2] = int(rgba[0] * 255)  # R
    return lut


def make_check_video():
    print(f"1. 加载训练数据: {INPUT_NPZ}")
    if not os.path.exists(INPUT_NPZ):
        print("❌ 找不到文件")
        return

    # 自动识别 Key
    data_dict = np.load(INPUT_NPZ)
    keys = list(data_dict.keys())
    print(f"   文件包含 Keys: {keys}")

    # 优先找 'input' 和 'target'
    if 'input_data' in keys:
        frames_in = data_dict['input_data']
    elif 'input' in keys:
        frames_in = data_dict['input']
    elif 'frames' in keys:
        frames_in = data_dict['frames']
    else:
        print(f"❌ 无法识别图像数据 Key。请检查你的键名是否为 'input_data'。当前包含: {keys}")
        return

    # 尝试找 Target
    frames_tar = None
    if 'target_data' in keys:
        frames_tar = data_dict['target_data']
        print("   ✅ 找到 Target (target_data) 数据，将生成对比视频")
    elif 'target' in keys:
        frames_tar = data_dict['target']
        print("   ✅ 找到 Target 数据，将生成对比视频")
    else:
        print("   ⚠️ 未找到 Target，仅生成 Input 视频")

    # 准备视频写入
    lut = create_pure_blue_black_lut()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    # 视频宽度取决于是否双屏
    vid_w = DISPLAY_SIZE[0] * 2 if frames_tar is not None else DISPLAY_SIZE[0]
    vid_h = DISPLAY_SIZE[1]

    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (vid_w, vid_h))

    # 全局归一化参数 (取前500帧估算)
    # 训练数据通常已经是 0-1 之间了，但为了显示稳定还是再算一下
    print("2. 计算动态范围...")
    sample = frames_in[:500]
    g_min = np.percentile(sample, 0.5)
    g_max = np.percentile(sample, 99.5)
    if g_max <= g_min: g_max = g_min + 1e-6
    print(f"   Input Range: [{g_min:.4f}, {g_max:.4f}]")

    total_frames = len(frames_in)
    print(f"   Total Frames: {total_frames}")

    print("3. 开始生成...")
    for i in range(total_frames):
        # --- 处理 Input (左图) ---
        img_in = frames_in[i]
        # 归一化 & 转 uint8
        img_in = (img_in - g_min) / (g_max - g_min)
        img_in = np.clip(img_in, 0, 1)
        img_in_u8 = (img_in * 255).astype(np.uint8)

        # 伪彩 & Resize
        img_in_bgr = cv2.cvtColor(img_in_u8, cv2.COLOR_GRAY2BGR)
        img_in_color = cv2.LUT(img_in_bgr, lut)
        img_in_show = cv2.resize(img_in_color, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)

        # 标注
        cv2.putText(img_in_show, "Input (Sim)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        final_frame = img_in_show

        # --- 处理 Target (右图) ---
        if frames_tar is not None:
            img_tar = frames_tar[i]
            # Target 通常本来就是 0-1
            img_tar = np.clip(img_tar, 0, 1)
            img_tar_u8 = (img_tar * 255).astype(np.uint8)

            img_tar_bgr = cv2.cvtColor(img_tar_u8, cv2.COLOR_GRAY2BGR)
            img_tar_color = cv2.LUT(img_tar_bgr, lut)
            img_tar_show = cv2.resize(img_tar_color, DISPLAY_SIZE, interpolation=cv2.INTER_NEAREST)  # Target用最近邻，保持锐利

            cv2.putText(img_tar_show, "Target (GT)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 左右拼接
            final_frame = np.hstack((img_in_show, img_tar_show))

        out.write(final_frame)

        if i % 200 == 0:
            print(f"   已处理 {i}/{total_frames}...")

    out.release()
    print(f"✅ 检查视频已生成: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    make_check_video()
