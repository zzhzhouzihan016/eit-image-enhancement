import numpy as np
import csv
import os
from pathlib import Path
from scipy.ndimage import laplace
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==============================
# 配置区
# ==============================

NPZ_PATH = PROJECT_ROOT / "data/raw/npz/250kHz/250kHz_all_frames.npz"
CSV_PATH = PROJECT_ROOT / "outputs/reports/normalization_experiment/250kHz/new_bayesian.csv"

# 评估时使用多少帧（为了加速，不一定要全帧）
N_FRAMES_FOR_EVAL = 2345

# === [修正点] 参数范围微调 ===
# 下限稍微放宽 (0.1~4.0)，上限稍微拉高 (96.0~99.9) 以防止过早饱和
LOW_Q_RANGE = (0.1, 10.0)
HIGH_Q_RANGE = (90.0, 99.9)

# 贝叶斯优化参数
N_CALLS = 100  # 总评估次数
N_RANDOM_STARTS = 8  # 随机初始点数目


# ==============================
# 归一化函数：ROI + 分位数拉伸
# ==============================
def normalize_eit_frame(frame, low_q, high_q):
    """
    归一化到 [0, 1]。
    输入: frame (ROI区域 > 0, 背景 <= 0 或 -1)
    """
    # 1. 提取 ROI
    mask = frame > 0
    roi = frame[mask]

    if roi.size == 0:
        return np.zeros_like(frame, dtype=np.float32), mask

    # 2. 计算分位数 (只在 ROI 上计算)
    vmin = np.percentile(roi, low_q)
    vmax = np.percentile(roi, high_q)

    if vmax <= vmin:
        vmax = vmin + 1e-6

    # 3. 线性映射与截断
    norm_frame = np.zeros_like(frame, dtype=np.float32)

    # 只处理 ROI，背景保持为 0
    norm_roi = (roi - vmin) / (vmax - vmin)
    norm_roi = np.clip(norm_roi, 0, 1)  # 截断到 [0, 1]

    norm_frame[mask] = norm_roi

    return norm_frame, mask


# ==============================
# 评价指标计算
# ==============================
def total_variation(img):
    gx, gy = np.gradient(img)
    return np.mean(np.abs(gx) + np.abs(gy))


def laplacian_sharpness(img):
    lap = laplace(img)
    return np.var(lap)


def temporal_smoothness(img_t, img_prev):
    return np.mean(np.abs(img_t - img_prev))


# ==============================
# 核心评价函数 (含 Saturation 惩罚)
# ==============================
def evaluate_q(frames, low_q, high_q):
    tvs, sharps, contrasts, smooths, sats = [], [], [], [], []
    prev = None

    for f in frames:
        f_norm, mask = normalize_eit_frame(f, low_q, high_q)

        # 1. TV (平滑度)
        tvs.append(total_variation(f_norm))

        # 2. Sharpness (清晰度)
        sharps.append(laplacian_sharpness(f_norm))

        # 3. Contrast & Saturation (只在 ROI 区域统计)
        roi_pixels = f_norm[mask]
        if roi_pixels.size > 0:
            contrasts.append(np.std(roi_pixels))

            # Saturation: 统计纯黑(0)和纯白(1)的像素比例
            # 使用 0.001 和 0.999 作为浮点数容差边界
            n_sat = np.sum(roi_pixels <= 0.001) + np.sum(roi_pixels >= 0.999)
            sats.append(n_sat / roi_pixels.size)
        else:
            contrasts.append(0.0)
            sats.append(0.0)

        # 4. Temporal Smoothness
        if prev is None:
            smooths.append(0.0)
        else:
            smooths.append(temporal_smoothness(f_norm, prev))
        prev = f_norm

    TV = float(np.mean(tvs))
    SH = float(np.mean(sharps))
    CT = float(np.mean(contrasts))
    SM = float(np.mean(smooths))
    SAT = float(np.mean(sats))

    # === [修正点] 新 Score 公式 ===
    # 平衡权重 + 过饱和惩罚
    score = 1.0 * SH + 1.0 * CT - 3.0 * TV - 2.0 * SAT - 1.0 * SM

    return TV, SH, CT, SM, score


# ==============================
# CSV 读写 (修复路径和编码问题)
# ==============================
def load_existing_results(csv_path):
    """读取已有的结果，返回字典 {(low, high): score}"""
    existing = {}
    if not os.path.exists(csv_path):
        return existing

    # 使用 utf-8-sig 防止中文乱码
    with open(csv_path, "r", newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头
        for row in reader:
            if len(row) < 7: continue
            try:
                l_q = float(row[0])
                h_q = float(row[1])
                sc = float(row[6])  # 第7列是 Score
                existing[(l_q, h_q)] = sc
            except:
                continue
    return existing


def append_result_to_csv(csv_path, row):
    # === [修正点] 自动创建文件夹 ===
    directory = os.path.dirname(csv_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    file_exists = os.path.exists(csv_path)

    # === [修正点] 指定 utf-8-sig 编码 ===
    with open(csv_path, "a", newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["low_q", "high_q", "TV", "Sharpness", "Contrast", "Smoothness", "Score"])
        writer.writerow(row)


# ==============================
# 贝叶斯优化
# ==============================
def bayes_optimize_quantiles(frames):
    frames_eval = frames[:N_FRAMES_FOR_EVAL]
    print(f"用于评估的帧数: {len(frames_eval)}")

    space = [
        Real(LOW_Q_RANGE[0], LOW_Q_RANGE[1], name="low_q"),
        Real(HIGH_Q_RANGE[0], HIGH_Q_RANGE[1], name="high_q"),
    ]

    # 加载历史数据，避免重复计算 + 修正逻辑漏洞
    existing_data = load_existing_results(CSV_PATH)

    @use_named_args(space)
    def objective(**params):
        low_q = float(params["low_q"])
        high_q = float(params["high_q"])

        if low_q >= high_q:
            return 1e6

        # === [修正点] 检查历史记录 ===
        # 使用近似匹配 (保留4位小数)
        for (ex_l, ex_h), ex_score in existing_data.items():
            if abs(ex_l - low_q) < 1e-4 and abs(ex_h - high_q) < 1e-4:
                print(f"[跳过] 历史记录: low={low_q:.4f}, high={high_q:.4f}, score={ex_score:.4f}")
                # 返回历史真实分数的负值 (不要返回 0.0 !)
                return -ex_score

        print(f"[评估] low={low_q:.4f}, high={high_q:.4f}")

        TV, SH, CT, SM, score = evaluate_q(frames_eval, low_q, high_q)
        print(f"  -> Score={score:.4f} (TV={TV:.3f}, SH={SH:.3f}, CT={CT:.3f}, SAT={SM:.3f})")

        append_result_to_csv(CSV_PATH, [low_q, high_q, TV, SH, CT, SM, score])

        # 更新内存字典
        existing_data[(low_q, high_q)] = score

        return -score

    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=N_CALLS,
        n_initial_points=N_RANDOM_STARTS,
        acq_func="EI",
        random_state=42,  # 固定随机种子
    )

    best_low = float(result.x[0])
    best_high = float(result.x[1])
    best_score = -result.fun

    print("\n============================")
    print("贝叶斯优化完成")
    print(f"最佳 low_q  = {best_low:.4f}")
    print(f"最佳 high_q = {best_high:.4f}")
    print(f"对应 score  = {best_score:.4f}")
    print("============================")

    return best_low, best_high, best_score


# ==============================
# 入口
# ==============================
if __name__ == "__main__":
    if os.path.exists(NPZ_PATH):
        print("加载数据:", NPZ_PATH)
        data = np.load(NPZ_PATH)
        frames = data["frames"]
        print("frames.shape =", frames.shape)
        bayes_optimize_quantiles(frames)
    else:
        print(f"错误: 找不到文件 {NPZ_PATH}")
