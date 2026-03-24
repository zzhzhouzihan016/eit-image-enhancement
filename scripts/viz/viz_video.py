import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ================= 配置区 =================
# 1. 模式选择: 'image' (单帧图) 或 'video' (完整MP4)
MODE = 'video'

REAL_DATA_PATH = PROJECT_ROOT / "data/processed/npz_norm/250kHzHR/250kHzHR_normalized_0p1_99p9.npz"
RESULT_NPZ_PATH = PROJECT_ROOT / "outputs/inference/st_u_net_legacy_results/exp2_unet_cbam/exp002_cbam_unet_run1_result.npz"

# 输出文件路径
SAVE_IMG_NAME = PROJECT_ROOT / "outputs/inference/st_u_net_legacy_results/exp2_unet_cbam/exp002_cbam_unet_run1_result.png"
SAVE_VIDEO_NAME = PROJECT_ROOT / "outputs/inference/st_u_net_legacy_results/exp2_unet_cbam/exp002_cbam_unet_run1_result.mp4"

# 3. 参数配置
FRAME_IDX = 582  # 'image'模式下看哪一帧
FPS = 15  # 'video'模式下的帧率
VIDEO_SCALE = 4  # 视频放大倍数

# 4. 归一化参数
LOW_Q = 0.1
HIGH_Q = 99.9


# ================= 🎨 德尔格配色表 (Dräger Style) =================

def create_draeger_lut():
    """
    创建类似 Dräger PulmoVista 的伪彩查找表 (LUT)
    渐变逻辑: 黑色(0) -> 深蓝 -> 亮蓝 -> 青色 -> 白色(255)
    注意: OpenCV 使用 BGR 格式
    """
    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    # Anchor points: (Position, [B, G, R])
    anchors = [
        (0, [0, 0, 0]),  # Black (Background)
        (30, [50, 0, 0]),  # Very Dark Blue
        (80, [180, 0, 0]),  # Dark Blue
        (140, [255, 100, 0]),  # Blue
        (200, [255, 255, 0]),  # Cyan (Blue + Green)
        (255, [255, 255, 255])  # White (Peak)
    ]

    for i in range(len(anchors) - 1):
        start_idx, start_color = anchors[i]
        end_idx, end_color = anchors[i + 1]

        length = end_idx - start_idx
        for j in range(length):
            idx = start_idx + j
            ratio = j / length
            b = start_color[0] + (end_color[0] - start_color[0]) * ratio
            g = start_color[1] + (end_color[1] - start_color[1]) * ratio
            r = start_color[2] + (end_color[2] - start_color[2]) * ratio
            lut[idx, 0] = [int(b), int(g), int(r)]

    lut[255, 0] = [255, 255, 255]
    return lut


# 全局初始化 LUT
DRAEGER_LUT = create_draeger_lut()


# ========================================================

def normalize_raw_frame(frame):
    """归一化逻辑"""
    mask = frame > 0
    roi = frame[mask]
    if roi.size > 0:
        vmin = np.percentile(roi, LOW_Q)
        vmax = np.percentile(roi, HIGH_Q)
        if vmax <= vmin: vmax = vmin + 1e-6
        frame_norm = (frame - vmin) / (vmax - vmin)
        frame_norm = np.clip(frame_norm, 0, 1)
        frame_norm[~mask] = 0
        return frame_norm
    else:
        return np.zeros_like(frame)


def apply_custom_colormap(gray_u8):
    """
    [关键修复] 应用自定义 LUT
    OpenCV 的 LUT 函数要求：如果 LUT 是 3 通道的，输入图片也必须是 3 通道的。
    """
    # 1. 将单通道灰度图转换为 3 通道 BGR 图 (虽然看起来还是灰的)
    if gray_u8.ndim == 2:
        gray_3c = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    else:
        gray_3c = gray_u8

    # 2. 现在可以应用 3 通道 LUT 了
    # 这会将 R=G=B=Value 的像素映射到 LUT[Value] 对应的颜色
    return cv2.LUT(gray_3c, DRAEGER_LUT)


def make_image_plot(raw, enh, idx):
    """生成单帧对比图"""
    raw_u8 = (raw * 255).astype(np.uint8)
    enh_u8 = (enh * 255).astype(np.uint8)

    # 这里的 apply_custom_colormap 现在会返回正确的彩色图
    raw_color_bgr = apply_custom_colormap(raw_u8)
    enh_color_bgr = apply_custom_colormap(enh_u8)

    # 转 RGB 给 matplotlib 显示
    raw_color_rgb = cv2.cvtColor(raw_color_bgr, cv2.COLOR_BGR2RGB)
    enh_color_rgb = cv2.cvtColor(enh_color_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 5), facecolor='black')

    plt.subplot(1, 2, 1)
    plt.imshow(raw_color_rgb)
    plt.title(f"Original (Dräger Style) - Frame {idx}", color='white')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(enh_color_rgb)
    plt.title(f"Enhanced (ST-UNet) - Frame {idx}", color='white')
    plt.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(SAVE_IMG_NAME), exist_ok=True)
    plt.savefig(SAVE_IMG_NAME, dpi=300, facecolor='black')
    print(f"✅ 图片模式完成: 已保存至 {SAVE_IMG_NAME}")


def make_video(raw_data, enh_data):
    """生成 MP4 对比视频"""
    print(f"🎥 正在生成视频: {SAVE_VIDEO_NAME}")
    os.makedirs(os.path.dirname(SAVE_VIDEO_NAME), exist_ok=True)

    total_frames = min(len(raw_data), len(enh_data))
    h, w = raw_data.shape[1], raw_data.shape[2]

    new_w = w * VIDEO_SCALE
    new_h = h * VIDEO_SCALE
    video_size = (new_w * 2, new_h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(SAVE_VIDEO_NAME, fourcc, FPS, video_size)

    for i in range(total_frames):
        # 1. 准备原始帧
        raw_norm = normalize_raw_frame(raw_data[i])
        raw_u8 = (raw_norm * 255).astype(np.uint8)
        raw_color = apply_custom_colormap(raw_u8)  # 使用修复后的函数

        # 2. 准备增强帧
        enh_norm = np.clip(enh_data[i], 0, 1)
        enh_u8 = (enh_norm * 255).astype(np.uint8)
        enh_color = apply_custom_colormap(enh_u8)  # 使用修复后的函数

        # 3. 拼接
        combined = np.hstack((raw_color, enh_color))

        # 4. 放大
        combined_resized = cv2.resize(combined, video_size, interpolation=cv2.INTER_NEAREST)

        # 5. 添加文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_resized, "Original", (20, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(combined_resized, "Enhanced (ST-UNet)", (new_w + 20, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(combined_resized, f"Frame: {i}", (20, new_h - 20), font, 0.8, (200, 200, 200), 2)

        out.write(combined_resized)

        if i % 100 == 0:
            print(f"   已处理 {i}/{total_frames} 帧...")

    out.release()
    print(f"✅ 视频模式完成: 已保存至 {SAVE_VIDEO_NAME}")


def main():
    print(f"🚀 启动德尔格风格可视化 (模式: {MODE})")

    if not os.path.exists(REAL_DATA_PATH) or not os.path.exists(RESULT_NPZ_PATH):
        print("❌ 错误: 找不到数据文件，请检查路径。")
        return

    try:
        raw_data = np.load(REAL_DATA_PATH)['frames']
        enh_data = np.load(RESULT_NPZ_PATH)['frames']
    except KeyError:
        # 兼容旧版数据的 key
        loaded_raw = np.load(REAL_DATA_PATH)
        key_raw = 'frames' if 'frames' in loaded_raw else list(loaded_raw.keys())[0]
        raw_data = loaded_raw[key_raw]

        loaded_enh = np.load(RESULT_NPZ_PATH)
        key_enh = 'frames' if 'frames' in loaded_enh else list(loaded_enh.keys())[0]
        enh_data = loaded_enh[key_enh]

    print(f"   数据加载完毕. 原始: {raw_data.shape}, 增强: {enh_data.shape}")

    if MODE == 'image':
        idx = FRAME_IDX
        if idx >= len(raw_data): idx = len(raw_data) // 2
        raw_frame = normalize_raw_frame(raw_data[idx])
        enh_frame = enh_data[idx]
        make_image_plot(raw_frame, enh_frame, idx)

    elif MODE == 'video':
        make_video(raw_data, enh_data)

    else:
        print("❌ 未知模式")


if __name__ == "__main__":
    main()
