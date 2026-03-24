import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']       # 设置中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False         # 正常显示负号

# 修改为你的文件路径
data = np.load("data/npy/250kHz/frame_0002.npy")  # 或者一帧数据

# ROI 和背景分离
roi = data[data > 0]
bg = data[data <= 0]

plt.figure(figsize=(12,5))

# --- 1. 全局直方图 ---
plt.subplot(1,2,1)
plt.hist(data.flatten(), bins=100, color='gray')
plt.title("整体灰度分布")
plt.xlabel("灰度值")
plt.ylabel("像素数")

# --- 2. ROI 直方图 ---
plt.subplot(1,2,2)
plt.hist(roi, bins=100, color='blue')
plt.title("ROI（frame>0）灰度分布")
plt.xlabel("灰度值")

# 标注关键百分位点
percentiles = np.percentile(roi, [1, 5, 50, 95, 99])
for p, v in zip([1,5,50,95,99], percentiles):
    plt.axvline(v, color='red')
    plt.text(v, 0, f"{p}%={v:.1f}", rotation=90)

plt.tight_layout()
plt.show()
