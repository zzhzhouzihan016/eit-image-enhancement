import numpy as np
import plotly.graph_objects as go
import webbrowser
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ================= 配置 =================
FILE_PATH = PROJECT_ROOT / "data/raw/npz/250kHz/250kHz_all_frames.npz"


# =======================================

def interactive_view():
    print(f"1. 加载数据: {FILE_PATH}")
    try:
        data = np.load(FILE_PATH)
        key = data.files[0]
        frames = data[key]
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 计算曲线
    grayscale_sums = np.sum(frames, axis=(1, 2), dtype=np.float64)

    print("2. 生成交互式图表...")

    # 创建 Plotly 图表
    fig = go.Figure()

    # 添加曲线
    fig.add_trace(go.Scatter(
        y=grayscale_sums,
        mode='lines',
        name='Breathing Signal',
        line=dict(color='royalblue', width=1.5)
    ))

    # 设置布局
    fig.update_layout(
        title='<b>Interactive EIT Breathing Curve</b><br><sup>(Use mouse to zoom in/out, double-click to reset)</sup>',
        xaxis_title='Frame Index',
        yaxis_title='Global Intensity Sum',
        template='plotly_white',
        hovermode='x unified',  # 鼠标悬停显示数值
        height=600
    )

    # 添加一个滑动条 (Range Slider)
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="linear"
        )
    )

    # 保存并打开
    output_file = PROJECT_ROOT / "outputs/reports/curve_analysis/breathing_curve_interactive.html"
    fig.write_html(output_file)
    print(f"✅ 图表已保存为 {output_file}")

    # 自动在浏览器打开
    try:
        webbrowser.open('file://' + os.path.realpath(output_file))
    except:
        print("请手动打开生成的 html 文件。")


if __name__ == "__main__":
    interactive_view()
