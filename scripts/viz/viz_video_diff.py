import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.ndimage import minimum_filter1d, gaussian_filter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ================= 配置区 =================
REAL_DATA_PATH = PROJECT_ROOT / "data/processed/npz_norm/250kHzHR/250kHzHR_normalized_0p1_99p9.npz"
RESULT_NPZ_PATH = PROJECT_ROOT / "outputs/inference/st_u_net_legacy_results/exp2_unet_cbam/exp002_cbam_unet_run1_result.npz"
SAVE_VIDEO_NAME = PROJECT_ROOT / "outputs/inference/st_u_net_legacy_results/exp2_unet_cbam/exp002_cbam_unet_run1_result_TriColor.mp4"

FPS = 15
VIDEO_SCALE = 4

# 核心算法参数 (保持之前的优化)
BASELINE_WINDOW = 30  # 动态基线窗口
SPATIAL_SIGMA = 1.0  # 空间平滑
ROI_STD_THRESHOLD = 0.02


# ================= 🎨 极简三色配色表 (Tri-Color) =================
def create_tricolor_lut():
    """
    极简三色表: 黑 -> 蓝 -> 白
    去除所有青色、杂色，提升视觉纯净度。
    """
    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    # BGR 顺序
    anchors = [
        (0, [0, 0, 0]),  # 0%   黑色 (背景)
        (30, [0, 0, 0]),  # 12%  黑色 (死区扩展，进一步压制噪声)
        # (40, [80, 0, 0]),  # 15%  深蓝 (起步)
        (160, [255, 0, 0]),  # 60%  纯蓝 (主体通气)
        (255, [255, 255, 255])  # 100% 白色 (峰值)
    ]

    # 线性插值填充
    for i in range(len(anchors) - 1):
        s_i, s_c = anchors[i]
        e_i, e_c = anchors[i + 1]
        L = e_i - s_i
        for j in range(L):
            r = j / L
            lut[s_i + j, 0] = [int(s_c[k] + (e_c[k] - s_c[k]) * r) for k in range(3)]

    lut[255, 0] = [255, 255, 255]
    return lut


TRICOLOR_LUT = create_tricolor_lut()


# ================= 核心算法 (Pro版逻辑保持不变) =================
def apply_pro_differential(frames):
    print("   -> 步骤 1: 计算滑动动态基线...")
    baseline = minimum_filter1d(frames, size=BASELINE_WINDOW, axis=0, mode='nearest')

    print("   -> 步骤 2: 计算差分...")
    diff_frames = frames - baseline
    diff_frames = np.clip(diff_frames, 0, None)

    print("   -> 步骤 3: 空间平滑...")
    pixel_std = np.std(diff_frames, axis=0)
    mask = pixel_std > ROI_STD_THRESHOLD

    smoothed_frames = np.zeros_like(diff_frames)
    for i in range(len(diff_frames)):
        frame = diff_frames[i]
        frame[~mask] = 0
        smoothed_frames[i] = gaussian_filter(frame, sigma=SPATIAL_SIGMA)

    print("   -> 步骤 4: 动态归一化...")
    global_max = np.percentile(smoothed_frames, 99.9)
    if global_max < 1e-9: global_max = 1.0

    # [新增] Gamma 矫正: 让中间的蓝色层次更丰富，不那么容易直接变白
    norm_frames = smoothed_frames / global_max
    norm_frames = np.clip(norm_frames, 0, 1)
    norm_frames = np.power(norm_frames, 0.8)  # Gamma < 1 提亮暗部，Gamma > 1 压暗

    return norm_frames


def apply_custom_colormap(gray_float):
    gray_u8 = (gray_float * 255).astype(np.uint8)
    if gray_u8.ndim == 2:
        gray_3c = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    else:
        gray_3c = gray_u8
    return cv2.LUT(gray_3c, TRICOLOR_LUT)


def make_video(raw_data, enh_data):
    print(f"🎥 启动极简三色视频生成: {SAVE_VIDEO_NAME}")
    os.makedirs(os.path.dirname(SAVE_VIDEO_NAME), exist_ok=True)

    print("\n>>> 处理原始数据...")
    raw_processed = apply_pro_differential(raw_data)

    print("\n>>> 处理增强数据...")
    enh_processed = apply_pro_differential(enh_data)

    total_frames = min(len(raw_data), len(enh_data))
    h, w = raw_data.shape[1], raw_data.shape[2]
    new_w, new_h = w * VIDEO_SCALE, h * VIDEO_SCALE
    video_size = (new_w * 2, new_h)

    out = cv2.VideoWriter(SAVE_VIDEO_NAME, cv2.VideoWriter_fourcc(*'mp4v'), FPS, video_size)

    for i in range(total_frames):
        raw_color = apply_custom_colormap(raw_processed[i])
        enh_color = apply_custom_colormap(enh_processed[i])

        combined = np.hstack((raw_color, enh_color))
        combined_resized = cv2.resize(combined, video_size, interpolation=cv2.INTER_NEAREST)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_resized, "Raw (Tri-Color)", (20, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(combined_resized, "ST-UNet (Tri-Color)", (new_w + 20, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(combined_resized, f"Frame: {i}", (20, new_h - 20), font, 0.8, (150, 150, 150), 2)

        out.write(combined_resized)
        if i % 100 == 0 and i > 0: print(f"   已写入 {i}/{total_frames} 帧...")

    out.release()
    print(f"✅ 完成！视频已保存至: {SAVE_VIDEO_NAME}")


def main():
    if not os.path.exists(REAL_DATA_PATH) or not os.path.exists(RESULT_NPZ_PATH):
        return print("❌ 文件不存在")

    try:
        raw_data = np.load(REAL_DATA_PATH)['frames']
        enh_data = np.load(RESULT_NPZ_PATH)['frames']
    except KeyError:
        raw_loaded = np.load(REAL_DATA_PATH)
        key_raw = 'frames' if 'frames' in raw_loaded else list(raw_loaded.keys())[0]
        raw_data = raw_loaded[key_raw]

        enh_loaded = np.load(RESULT_NPZ_PATH)
        key_enh = 'frames' if 'frames' in enh_loaded else list(enh_loaded.keys())[0]
        enh_data = enh_loaded[key_enh]

    print(f"数据加载成功: Raw {raw_data.shape}, Enh {enh_data.shape}")
    make_video(raw_data, enh_data)


if __name__ == "__main__":
    main()
