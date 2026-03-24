import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
file_path = PROJECT_ROOT / "data" / "processed" / "npz_norm" / "250kHzHR" / "250kHzHR_normalized_0p1_99p9.npz"

# 打印一下，方便你排查路径是否拼接正确
print(f"正在尝试加载文件: {file_path}")

# 检查文件是否存在，避免程序崩溃
if not file_path.exists():
    print(f"❌ 报错：在该路径下找不到文件！请检查数据目录是否位于 {PROJECT_ROOT} 中。")
else:
    # 2. 加载文件
    data = np.load(file_path)

    # 3. 查看里面有哪些变量名 (Keys)
    print("文件中的 Keys:", data.files)

    # 4. 获取第一个数组并查看数值
    array_name = data.files[0]
    frame_data = data[array_name]

    print(f"\n数组形状 (Shape): {frame_data.shape}")
    print(f"数据类型 (Dtype): {frame_data.dtype}")
    print("\n打印第 0 帧的前 20x20 个像素值:")
    print(frame_data[0, :50, :50])
