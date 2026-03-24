import struct
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_grey_all_frames(grey_path, out_folder_frames="frames_npy", out_npz="all_frames.npz"):
    """
    读取整个二进制 grey 文件：
    - 前 8 字节：uint32 width, uint32 height
    - 后面按 double 排列的多帧数据

    功能：
    1. 将每一帧保存为 .npy 文件
    2. 最终将所有帧保存为一个 .npz 文件
    """

    print("正在读取 grey 文件:", grey_path)

    # 以二进制方式打开文件
    with open(grey_path, "rb") as f:
        # 1) 读取头部
        header = f.read(8)
        width = struct.unpack("<I", header[:4])[0]
        height = struct.unpack("<I", header[4:8])[0]
        print(f"图像尺寸: width={width}, height={height}")

        # 2) 计算总帧数
        f.seek(0, 2)  # 移动到文件末尾
        file_size = f.tell()
        data_size = file_size - 8  # 除去 8 字节头部

        pixels_per_frame = width * height
        bytes_per_frame = pixels_per_frame * 8  # double = 8 bytes

        if data_size % bytes_per_frame != 0:
            raise ValueError(
                f"文件大小不匹配: data_size={data_size}, 每帧={bytes_per_frame}"
            )

        num_frames = data_size // bytes_per_frame
        print(f"检测到帧数: {num_frames}")

        # 回到数据开始
        f.seek(8)

        # 3) 创建输出文件夹
        os.makedirs(out_folder_frames, exist_ok=True)

        # 存放所有帧
        all_frames = np.zeros((num_frames, height, width), dtype=np.float32)

        # 4) 循环读取每一帧
        for i in range(num_frames):
            raw = f.read(bytes_per_frame)
            frame = np.frombuffer(raw, dtype=np.float64)
            frame = frame.reshape((height, width)).astype(np.float32)

            # 保存单独帧
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


# =============== 示例调用 ===============
if __name__ == "__main__":
    grey_path = PROJECT_ROOT / "data/raw/grey/250kHzHR.grey"
    out_folder = PROJECT_ROOT / "data/raw/npy/250kHzHR"
    out_npz = PROJECT_ROOT / "data/raw/npz/250kHzHR/250kHzHR.npz"
    load_grey_all_frames(grey_path, out_folder, out_npz)
