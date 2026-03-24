import scipy.io
import numpy as np
import cv2
import os
import matplotlib.cm as cm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ================= 配置 =================
MAT_FILE = PROJECT_ROOT / "data/processed/train_sim/demo/eit_demo_cases.mat"
OUTPUT_DIR = PROJECT_ROOT / "outputs/videos/demo_cases"
FPS = 15
VIDEO_SCALE = 4  # 放大4倍
CASE_NAMES = ['Normal', 'Right Collapse', 'Left Collapse', 'Severe COPD', 'Pleural Effusion']


# =======================================

def apply_colormap(data):
    """应用 Jet 伪彩 (0-1 范围)"""
    data = np.clip(data, 0, 1)
    img_u8 = (data * 255).astype(np.uint8)
    return cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)


def make_demo_videos():
    if not os.path.exists(MAT_FILE):
        print(f"❌ 找不到文件 {MAT_FILE}，请先运行 MATLAB 脚本。")
        return

    print(f"1. 加载数据: {MAT_FILE}")
    mat = scipy.io.loadmat(MAT_FILE)

    # 维度: [5, 50, 128, 128]
    inputs = mat['input_data']
    targets = mat['target_data']

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 遍历 5 个 Case
    for i, case_name in enumerate(CASE_NAMES):
        video_name = f"{OUTPUT_DIR}/Case{i}_{case_name.replace(' ', '_')}.mp4"
        print(f"2. 生成视频 [{i + 1}/5]: {video_name}")

        seq_in = inputs[i]
        seq_tar = targets[i]
        n_frames, h, w = seq_in.shape

        # 归一化 Input (因为 GREIT 重建值范围不固定)
        # 简单的动态范围归一化，模拟真实成像效果
        g_min, g_max = seq_in.min(), seq_in.max()
        seq_in_norm = (seq_in - g_min) / (g_max - g_min + 1e-6)

        # 归一化 Target (通常已经是 0-0.3，需拉伸到 0-1 显示)
        # 注意: Target 是电导率，0是背景，最大值约0.3
        seq_tar_norm = seq_tar / (seq_tar.max() + 1e-6)

        # 视频参数
        new_w, new_h = w * VIDEO_SCALE, h * VIDEO_SCALE
        vid_w = new_w * 2  # 左右并排
        vid_h = new_h

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_name, fourcc, FPS, (vid_w, vid_h))

        for f in range(n_frames):
            # 1. 准备图像
            img_tar = apply_colormap(seq_tar_norm[f])
            img_in = apply_colormap(seq_in_norm[f])

            # 2. 拼接
            combined = np.hstack((img_tar, img_in))

            # 3. 放大
            combined = cv2.resize(combined, (vid_w, vid_h), interpolation=cv2.INTER_NEAREST)

            # 4. 标注
            # 顶部大标题 (Case Name)
            cv2.rectangle(combined, (0, 0), (vid_w, 40), (0, 0, 0), -1)
            cv2.putText(combined, f"Case {i}: {case_name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 底部子标题
            cv2.putText(combined, "Ground Truth (Physiology)", (20, vid_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 2)
            cv2.putText(combined, "Input (Simulation)", (new_w + 20, vid_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 2)

            out.write(combined)

        out.release()

    print(f"✅ 所有视频生成完毕！请查看 {OUTPUT_DIR}")


if __name__ == "__main__":
    make_demo_videos()
