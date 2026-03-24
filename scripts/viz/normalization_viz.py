import numpy as np
import matplotlib

# 【修复 1】强制使用对 Mac 友好的后端，防止窗口卡死
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
npz_path = PROJECT_ROOT / "data/processed/npz_norm/250kHzHR/250kHzHR_normalized_0p1_99p9.npz"


def interactive_viewer(file_path):
    # 1. 加载数据
    data = np.load(file_path)
    frames = data['frames']
    num_frames, height, width = frames.shape

    v_min, v_max = frames.min(), frames.max()
    print(f"数值范围: {v_min} ~ {v_max}")

    # 2. 创建画布
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)  # 底部留更多空间

    # 初始显示
    img_plot = ax.imshow(frames[0], cmap='gray', vmin=v_min, vmax=v_max)
    ax.set_title(f"Frame 0 / {num_frames - 1}")
    plt.colorbar(img_plot, ax=ax, label="Pixel Value")

    # 3. 添加滑块
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(
        ax=ax_slider,
        label='Frame Index',
        valmin=0,
        valmax=num_frames - 1,
        valinit=0,
        valstep=1
    )

    # 4. 更新函数
    def update(val):
        idx = int(slider.val)
        img_plot.set_data(frames[idx])
        ax.set_title(f"Frame {idx} / {num_frames - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    print("提示: 窗口已弹出。如果滑块仍无法拖动，请检查终端是否有报错。")

    # 【修复 2】必须显示调用 show()，且必须返回 slider 对象
    # 如果不返回，函数结束时 slider 会被当做垃圾回收，导致无法拖动
    return slider, fig


if __name__ == "__main__":
    # 【修复 3】用一个变量接住返回的 slider，保持它的生命周期
    # 这一步非常关键！不要写成直接调用 interactive_viewer(...)
    my_slider, my_fig = interactive_viewer(npz_path)

    # 最后的 show() 放在这里
    plt.show()
