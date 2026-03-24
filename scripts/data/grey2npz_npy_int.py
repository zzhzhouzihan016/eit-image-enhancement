import struct
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_grey_final_uint16_w256(grey_path, out_folder_frames="frames_npy", out_npz="all_frames.npz"):
    print("--- 正在以 W=256, uint16 模式读取 grey 文件 ---")

    # 核心设定：根据数学分析结果
    bytes_per_pixel = 2  # uint16
    read_dtype = np.uint16

    with open(grey_path, "rb") as f:
        # 1) 读取并忽略文件头 (因为 W=252 是错的)
        _ = f.read(8)

        # 【干预点 1】：强制使用 C++ 代码提供的实际尺寸
        width = 256
        height = 176
        print(f"图像尺寸（强制干预）: width={width}, height={height}")

        # 2) 计算总帧数
        f.seek(0, 2)
        file_size = f.tell()
        data_size = file_size - 8  # 除去 8 字节头部 (按简单结构计算)

        pixels_per_frame = width * height
        bytes_per_frame = pixels_per_frame * bytes_per_pixel  # 256 * 176 * 2 = 88704 bytes

        # 验证整除性
        if data_size % bytes_per_frame != 0:
            raise ValueError(
                f"文件大小不匹配: 即使 W=256, uint16 也无法整除! 余数: {data_size % bytes_per_frame}"
            )

        num_frames = data_size // bytes_per_frame  # 应该是 2345
        print(f"检测到帧数: {num_frames} (数学上唯一能整除的结果)")

        # 回到数据开始
        f.seek(8)

        # 3) 创建输出文件夹
        os.makedirs(out_folder_frames, exist_ok=True)

        # 存放所有帧
        all_frames = np.zeros((num_frames, height, width), dtype=np.float32)

        # 4) 循环读取每一帧 (简单结构：不跳帧头)
        for i in range(num_frames):
            raw = f.read(bytes_per_frame)

            # 使用 uint16 读取
            frame_raw = np.frombuffer(raw, dtype=read_dtype)

            # 形状转换
            frame = frame_raw.reshape((height, width)).astype(np.float32)

            # 保存和存储
            npy_path = os.path.join(out_folder_frames, f"frame_{i:04d}.npy")
            np.save(npy_path, frame)
            all_frames[i] = frame

            if i % 100 == 0:
                print(f"已处理 {i}/{num_frames} 帧...")

        print("所有 .npy 帧已保存到目录:", out_folder_frames)

    # 5) 保存 npz
    np.savez_compressed(out_npz, frames=all_frames)
    print("已保存整序列 npz:", out_npz)
    print("最终 all_frames 形状:", all_frames.shape)


# =============== 示例调用 (运行修正后的函数) ===============
if __name__ == "__main__":
    grey_path = PROJECT_ROOT / "data/raw/grey/250kHzHR.grey"
    out_folder = PROJECT_ROOT / "data/raw/npy/250kHzHR"
    out_npz = PROJECT_ROOT / "data/raw/npz/250kHzHR/250kHzHR.npz"

    load_grey_final_uint16_w256(grey_path, out_folder, out_npz)
