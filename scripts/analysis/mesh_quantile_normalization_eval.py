import csv
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import laplace

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ==============================
# 归一化
# ==============================
def normalize_eit_frame(frame, low_q, high_q):
    mask = frame > 0
    roi = frame[mask]

    vmin = np.percentile(roi, low_q)
    vmax = np.percentile(roi, high_q)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = (frame - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    norm[~mask] = 0

    return norm


# ==============================
# 评价指标
# ==============================
def total_variation(img):
    gx, gy = np.gradient(img)
    return np.mean(np.abs(gx) + np.abs(gy))

def laplacian_sharpness(img):
    lap = laplace(img)
    return np.var(lap)

def local_contrast(img):
    roi = img[img > 0]
    return np.std(roi)

def temporal_smoothness(img_t, img_prev):
    return np.mean(np.abs(img_t - img_prev))


# ==============================
# 单组分位数评价
# ==============================
def evaluate_q(frames, low_q, high_q):
    tvs, sharps, contrasts, smooths = [], [], [], []
    prev = None

    for f in frames:
        f_norm = normalize_eit_frame(f, low_q, high_q)

        tvs.append(total_variation(f_norm))
        sharps.append(laplacian_sharpness(f_norm))
        contrasts.append(local_contrast(f_norm))

        if prev is None:
            smooths.append(0)
        else:
            smooths.append(temporal_smoothness(f_norm, prev))

        prev = f_norm

    TV = np.mean(tvs)
    SH = np.mean(sharps)
    CT = np.mean(contrasts)
    SM = np.mean(smooths)

    # 默认综合得分
    score = SH + 2 * CT - TV - 0.5 * SM

    return TV, SH, CT, SM, score


# ==============================
# 读取已存在 CSV，避免重复
# ==============================
def load_existing_results(csv_path):
    existing = set()

    if not os.path.exists(csv_path):
        return existing

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头

        for row in reader:
            if len(row) < 2:
                continue
            try:
                low_q = float(row[0])
                high_q = float(row[1])
                existing.add((low_q, high_q))
            except:
                continue

    return existing


# ==============================
# 网格搜索（追加写入，不覆盖，同参数跳过）
# ==============================
def run_grid_search(frames, csv_path):
    low_list = [5, 5.5, 6, 6.5, 7, 7.5, 8]
    high_list = [95, 94.5, 94, 93.5, 93, 92, 91, 90]

    # 已存在参数
    existing = load_existing_results(csv_path)

    # CSV 文件不存在 → 写表头
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline='') as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["low_q", "high_q", "TV", "Sharpness", "Contrast", "Smoothness", "Score"])

        for low_q in low_list:
            for high_q in high_list:
                if low_q >= high_q:
                    continue

                # 检查是否重复
                if (low_q, high_q) in existing:
                    print(f"跳过已存在记录：low={low_q}, high={high_q}")
                    continue

                print(f"评估：low={low_q}, high={high_q} ...")

                TV, SH, CT, SM, score = evaluate_q(frames, low_q, high_q)

                # 写入新记录
                writer.writerow([low_q, high_q, TV, SH, CT, SM, score])
                existing.add((low_q, high_q))

    print(f"\nCSV 写入完成：{csv_path}")

frames = np.load(PROJECT_ROOT / "data/raw/npz/250kHz/250kHz_all_frames.npz")["frames"]
run_grid_search(
    frames,
    PROJECT_ROOT / "outputs/reports/normalization_experiment/250kHz/quantile_grid_search_results.csv",
)
