# EIT Image Enhancement

本项目目前维护三条相关但不同的工作线：

- 当前主线：20 帧双输入 EIT 增强 baseline
- 历史主线：5 帧单输入 ST-UNet 系列训练与推理
- 扩展主线：CT slice -> MedSAM -> FEM mesh -> 呼吸周期仿真

如果你现在要继续做模型设计和训练，优先关注第一条主线，也就是：

- 数据集：`data/processed/train_sim/lctsc_cem_pathology_jac32_gt64`
- 模型：`dual_source_seq_unet`
- 配置：`configs/experiments/exp101_dual_source_seq_unet.yaml`
- 训练入口：`scripts/train.py`

## 当前状态

- `scripts/train.py` 已同时支持单源和双源训练
- `scripts/infer.py` 已整理为统一的单源推理入口，不再按分辨率拆成两个脚本
- 单源训练的 train/val 划分已改为“按原始序列切分”，避免滑窗泄漏
- 单源数据集默认不再对 target 做强制 blur
- 真实测量原始文件已统一归到 `data/raw/real_measurements/`
- WandB 的 `entity` 已改为可空，换机器或上服务器时不会默认绑定到旧账号

## 项目结构

```text
EITpic_improve-master/
├── src/eit/
│   ├── dataset.py                  # 单源数据集
│   ├── dataset_dual_source.py      # 双源 20 帧数据集
│   ├── models/                     # ST-UNet 系列与 dual-source baseline
│   ├── io/                         # DICOM 读取
│   ├── preprocess/                 # CT 预处理
│   ├── segmentation/               # MedSAM 分割
│   ├── mesh/                       # FEM 网格构建
│   ├── simulation/                 # pyEIT 呼吸周期仿真
│   └── utils/
├── scripts/
│   ├── train.py                    # 统一训练入口
│   ├── infer.py                    # 统一单源推理入口
│   ├── data/                       # 数据转换和数据整理脚本
│   ├── segmentation/               # 分割入口
│   ├── mesh/                       # 网格入口
│   ├── simulation/                 # 仿真入口
│   ├── analysis/                   # 研究型分析脚本
│   └── viz/                        # 可视化脚本
├── configs/experiments/
│   ├── exp001~exp007               # 历史单源实验
│   └── exp101_dual_source_seq_unet.yaml
├── data/
│   ├── raw/
│   │   ├── grey/
│   │   ├── npy/
│   │   ├── npz/
│   │   └── real_measurements/      # 真实测量 CSV / Excel
│   ├── processed/
│   │   ├── npz_norm/
│   │   └── train_sim/
│   └── interim/ct_cases/           # CT 单病例中间产物
├── docs/
├── outputs/
├── archives/
│   └── MedSAM-main/
└── pyproject.toml
```

## 主线说明

### 1. 双源 20 帧训练主线

这是现在最推荐继续推进的 baseline。

- 输入 1：20 帧重建图像 `recon`，形状 `[20, 32, 32]`
- 输入 2：20 帧电压序列 `voltage`，形状 `[20, 208]`
- 输出：20 帧增强目标 `target`，形状 `[20, 64, 64]`
- 默认噪声设置：固定 `noise_index = 2`，即 `30 dB`

对应文件：

- 数据集实现：`src/eit/dataset_dual_source.py`
- 模型实现：`src/eit/models/dual_source_seq_unet.py`
- 训练配置：`configs/experiments/exp101_dual_source_seq_unet.yaml`
- 训练入口：`scripts/train.py`

推荐先把这条线作为论文/实验 baseline 跑通，再继续迭代更复杂的融合结构。

### 2. 历史单源 ST-UNet 主线

这是旧的 5 帧输入预测中间帧的训练流程，仍然保留，适合做对照实验。

- 数据集实现：`src/eit/dataset.py`
- 历史模型配置：`configs/experiments/exp001_baseline_unet.yaml` 到 `exp007_early_sam_unet.yaml`
- 训练入口：`scripts/train.py`
- 推理入口：`scripts/infer.py`

注意：

- 现在只有一个 `infer.py`
- 推理脚本当前只针对单源模型
- 双源模型目前还没有单独整理好的推理 CLI

### 3. CT -> FEM -> 仿真链路

这条链路主要用于构建病例级中间产物和呼吸周期仿真数据。

核心入口：

- `scripts/data/build_ct_middle_slice.py`
- `scripts/segmentation/run_medsam_ct_slice.py`
- `scripts/mesh/build_fem_from_mask.py`
- `scripts/simulation/build_respiratory_cycle_dataset.py`

核心模块：

- `src/eit/io/`
- `src/eit/preprocess/`
- `src/eit/segmentation/`
- `src/eit/mesh/`
- `src/eit/simulation/`

## 数据目录约定

### 双源训练数据

当前主线数据根目录：

`data/processed/train_sim/lctsc_cem_pathology_jac32_gt64/`

典型结构：

```text
data/processed/train_sim/lctsc_cem_pathology_jac32_gt64/
├── cases/
├── global_samples_manifest.csv
├── samples_manifest.csv
├── summary.json
└── README.md
```

训练时真正使用的是：

- `global_samples_manifest.csv`
- `cases/<case_id>/slices/slice_<idx>/<sample_name>/sequence_data.npz`

### 历史单源数据

常见历史数据位置：

- 仿真训练数据：`data/processed/train_sim/npz/`
- 真实测量归一化数据：`data/processed/npz_norm/`

### 真实测量原始文件

统一放在：

`data/raw/real_measurements/`

这类文件是原始输入，不建议纳入 Git。

## 环境与安装

建议使用你的 `deepl_cv` 环境，或者创建新的 Python 环境后在项目根目录执行：

```bash
pip install -e .
```

说明：

- 依赖写在 `pyproject.toml`
- 当前项目不再依赖 `torchvision` 作为核心训练依赖
- 如果使用 MedSAM 相关脚本，仍需要 `segment_anything` 或 `archives/MedSAM-main`

## 常用命令

### 1. 双源 20 帧训练

```bash
python scripts/train.py \
  --config configs/experiments/exp101_dual_source_seq_unet.yaml
```

### 2. 历史单源训练

```bash
python scripts/train.py \
  --config configs/experiments/exp001_baseline_unet.yaml
```

### 3. 单源推理

```bash
python scripts/infer.py \
  --config configs/experiments/exp001_baseline_unet.yaml \
  --input data/processed/npz_norm/250kHzHR/250kHzHR_normalized_0p1_99p9.npz
```

推理规则：

- 默认从配置里的 `save_dir` 自动寻找 `best_ssim.pth`
- 输出默认保存到 `outputs/inference/`
- 输出尺寸自动跟随输入数据原始尺寸
- 如果输入已经是 `npz_norm`，通常保持默认 `--normalization none` 即可
- 若输入是未归一化原始帧，可显式使用 `--normalization global` 或 `--normalization per_frame`

### 4. CT slice 导出

```bash
python scripts/data/build_ct_middle_slice.py \
  --dicom-dir "/path/to/dicom_dir" \
  --case-id demo_case \
  --selector middle \
  --range-size 21
```

### 5. MedSAM 分割

```bash
python scripts/segmentation/run_medsam_ct_slice.py \
  --image data/interim/ct_cases/demo_case/slice/selected_slice_rgb.npy \
  --checkpoint archives/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth
```

### 6. FEM 网格构建

```bash
python scripts/mesh/build_fem_from_mask.py \
  --torso-mask data/interim/ct_cases/demo_case/segmentation/medsam/torso_mask.npy \
  --lung-mask data/interim/ct_cases/demo_case/segmentation/medsam/lung_mask.npy
```

### 7. 呼吸周期仿真

```bash
python scripts/simulation/build_respiratory_cycle_dataset.py \
  --fem-dir data/interim/ct_cases/demo_case/mesh/fem
```

## 训练与输出

训练输出默认放在：

- 权重与日志：`outputs/experiments/<experiment_name>/`
- 推理结果：`outputs/inference/`

典型训练产物：

- `best_ssim.pth`
- `last.pth`
- `results.csv`
- `case_splits.yaml`
- `vis/`

## WandB 说明

配置文件中的 `wandb.entity` 现在默认是 `null`。

如果你想在线同步：

- 在对应实验配置里填入正确的 team entity
- 或先执行 `wandb login`

如果你只想本地训练：

- 设置 `wandb.enable: False`
- 或设置 `wandb.mode: offline`

## 推荐阅读文档

如果你现在主要做双源 20 帧模型，建议先看这两个文档：

- `docs/dual_source_20frame_training_spec.md`
- `docs/dual_source_dataflow_architecture.md`

它们分别说明：

- 数据接口和训练样本约定
- 数据流向、融合方式和当前 baseline 结构

## Git 与服务器建议

建议纳入 Git 的内容：

- `src/`
- `scripts/`
- `configs/`
- `docs/`
- `README.md`
- `pyproject.toml`

建议不要纳入 Git 的内容：

- `data/raw/`
- `data/processed/`
- `data/interim/`
- `outputs/`
- `archives/MedSAM-main/`
- 各类权重、视频、`npy/npz/mat`

如果放到远程服务器训练，推荐做法是：

- 本地整理代码和配置
- 用 Git 或 `rsync` 同步代码到服务器
- 数据和训练放服务器
- 输出结果只按需回传

## 当前建议的工作顺序

如果你接下来继续推进论文或实验，推荐按下面顺序：

1. 以 `exp101_dual_source_seq_unet.yaml` 跑通双源 baseline
2. 确认训练/验证曲线和可视化样本正常
3. 在当前 dual-source 框架上迭代融合结构
4. 最后再回头做和历史单源模型的系统对比
