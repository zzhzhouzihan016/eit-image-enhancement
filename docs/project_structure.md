# 项目结构说明

项目按四层组织：

- `src/`：可复用核心逻辑
- `scripts/`：直接运行的入口
- `data/`：输入数据与中间产物
- `outputs/`：训练、推理与分析结果

## 1. 代码层

### `src/eit/`

- `io/`：DICOM 等输入读取
- `preprocess/`：窗宽窗位、切片选择等预处理
- `segmentation/`：MedSAM 分割与 bbox 逻辑
- `mesh/`：轮廓提取、二维 FEM 网格生成、矩阵插值映射
- `simulation/`：基于 pyEIT 的前向仿真、加噪、反演与数据集生成
- `physics/`：HU 到电导映射预留
- `models/`：现有 EIT 增强模型

### `scripts/`

- `data/`：数据转换、CT case 构建
- `segmentation/`：肺 / 胸廓分割入口
- `mesh/`：FEM 网格构建入口
- `simulation/`：20 帧呼吸周期 EIT 数据生成入口
- `analysis/`：曲线分析、实验检查
- `viz/`：数据与结果可视化

## 2. 数据层

### `data/raw/`

原始 grey / npy / npz 数据，以及真实测量 CSV 等未整理原始数据。

### `data/processed/`

训练或推理直接使用的整理后数据。

### `data/interim/ct_cases/`

按病例保存 CT 中间产物，每个 case 再按阶段拆分：

- `selection/`：选层过程的可视化
- `slice/`：当前保留的 2D slice
- `segmentation/medsam/`：MedSAM 分割结果
- `mesh/fem/`：清洗后的胸廓/双肺轮廓、节点、单元、电导率与可视化结果
- `simulation/respiratory_cycle/`：20 帧边界电压、64×64 输入、256×256 标签与质检图

这样的好处是：后续新增 `mesh/`、`conductivity/` 时可以直接平铺在 case 目录下，不会把所有文件堆在一层。

## 3. 输出层

### `outputs/`

- `experiments/`：训练过程与权重
- `inference/`：推理结果
- `figures/`：论文 / 分析图
- `reports/`：实验报告
- `videos/`：视频结果

`outputs/` 中的内容都是派生结果，建议默认不纳入 Git。

## 4. 归档层

### `archives/`

- `MedSAM-main/`：外部 MedSAM 源码与权重
- `legacy/`：历史代码与旧工作区

归档内容与主工程隔离，避免污染当前主线，同时保留回溯能力。

## 5. 当前主线流程

当前单病例工作流已经形成一条完整链路：

1. `scripts/data/build_ct_middle_slice.py`
2. `scripts/segmentation/run_medsam_ct_slice.py`
3. `scripts/mesh/build_fem_from_mask.py`
4. `scripts/simulation/build_respiratory_cycle_dataset.py`

也就是：

`DICOM -> 2D CT slice -> lung/thorax mask -> FEM mesh -> pyEIT simulation dataset`

## 6. Git 备份建议

建议提交到 Git 的内容：

- `src/`
- `scripts/`
- `configs/`
- `docs/`
- `README.md`
- `pyproject.toml`
- `.gitignore`

建议不要提交到 Git 的内容：

- `archives/`
- `data/raw/`
- `data/processed/`
- `data/interim/ct_cases/`
- `outputs/`
- 模型权重、视频、`.npy`、`.npz`、`.mat`
