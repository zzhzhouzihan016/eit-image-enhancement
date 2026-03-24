import struct
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def analyze_grey_file_structure(grey_path):
    print(f"--- 正在分析文件: {grey_path} ---")

    # 1. 读取文件总大小并确定数据区域大小
    try:
        with open(grey_path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()

            # 读取文件总头 (第一次的 8 字节: width, height)
            f.seek(0)
            header = f.read(8)
            width = struct.unpack("<I", header[:4])[0]
            height = struct.unpack("<I", header[4:8])[0]

    except FileNotFoundError:
        print(f"错误: 文件未找到，请检查路径: {grey_path}")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    pixels_per_frame = width * height
    initial_header_size = 8

    # 打印基础信息
    print(f"文件总大小 (Bytes): {file_size}")
    print(f"图像尺寸 (W x H): {width} x {height}")
    print(f"初始头大小: {initial_header_size} 字节")
    print(f"每帧像素数: {pixels_per_frame}")
    print("---------------------------------")

    # 假设文件结构只有 [Initial Header] + [Data]
    data_size_simple = file_size - initial_header_size
    print(f"【假设 A: 简单结构】纯数据大小 = {data_size_simple} 字节")

    # 假设文件结构是 [Initial Header] + N * ([Frame Header] + [Data])
    # 之前推断的重复头大小
    frame_header_size = 8

    # 2. 核心分析：测试各种数据类型和结构

    analysis_results = []

    # ----------------------------------------------------
    # 结构 1：简单结构 [Header] + [Data]
    # ----------------------------------------------------
    print("\n--- 结构 1 分析：[Header] + [Data] (忽略后续帧头) ---")

    data_types = {
        "float64 (double)": 8,
        "float32": 4,
        "uint16": 2,
    }

    for name, bytes_per_pixel in data_types.items():
        frame_data_bytes = pixels_per_frame * bytes_per_pixel

        # 帧数 = 纯数据大小 / 每帧数据大小
        num_frames = data_size_simple / frame_data_bytes

        is_integer = num_frames == int(num_frames)

        analysis_results.append({
            "Structure": "Simple",
            "Type": name,
            "Bytes/Pixel": bytes_per_pixel,
            "Total Data Bytes": data_size_simple,
            "Frame Data Bytes": frame_data_bytes,
            "Calculated Frames": num_frames,
            "Is Integer": is_integer,
        })

        status = "✅ 整数" if is_integer else "❌ 余数"
        print(f"  - 尝试 {name} ({bytes_per_pixel} 字节): 帧数 = {num_frames:.4f} ({status})")

    # ----------------------------------------------------
    # 结构 2：复杂结构 [H] + N * ([Frame Header] + [Data])
    # ----------------------------------------------------
    print("\n--- 结构 2 分析：[H] + N * ([Frame Header=8B] + [Data]) ---")

    data_after_initial_head = file_size - initial_header_size

    for name, bytes_per_pixel in data_types.items():
        frame_data_bytes = pixels_per_frame * bytes_per_pixel
        total_block_size = frame_data_bytes + frame_header_size  # 数据 + 8字节头

        # 帧数 = (初始头后的数据) / (块总大小)
        num_frames = data_after_initial_head / total_block_size

        is_integer = num_frames == int(num_frames)

        analysis_results.append({
            "Structure": "Complex",
            "Type": name,
            "Bytes/Pixel": bytes_per_pixel,
            "Total Data Bytes": data_after_initial_head,
            "Block Size": total_block_size,
            "Calculated Frames": num_frames,
            "Is Integer": is_integer,
        })

        status = "✅ 整数" if is_integer else "❌ 余数"
        print(
            f"  - 尝试 {name} ({bytes_per_pixel} 字节): 块大小 {total_block_size}, 帧数 = {num_frames:.4f} ({status})")

    print("---------------------------------")
    print("请根据 '✅ 整数' 的结果来确定正确的结构和数据类型。")


# =============== 示例调用 ===============
if __name__ == "__main__":
    # 请确保路径指向你的 grey 文件
    grey_path = PROJECT_ROOT / "data/raw/grey/250kHzHR.grey"
    analyze_grey_file_structure(grey_path)
