# 双输入 20 帧模型的数据流向与网络结构

本文档解释当前双输入 baseline 的完整数据流向：

- 输入 1：20 帧重建图像 `recon`
- 输入 2：20 帧边界电压 `voltage`
- 输出：20 帧 `64x64` 增强目标图像

对应代码入口：

- 数据集读取：`src/eit/dataset_dual_source.py`
- 模型实现：`src/eit/models/dual_source_seq_unet.py`
- 训练入口：`scripts/train.py`
- 实验配置：`configs/experiments/exp101_dual_source_seq_unet.yaml`

## 1. 从磁盘到样本

单个样本在磁盘上的组织方式为：

```text
data/processed/train_sim/lctsc_cem_pathology_jac32_gt64/
└── cases/
    └── <case_id>/
        └── slices/
            └── slice_<idx>/
                └── <sample_name>/
                    ├── sequence_data.npz
                    └── metadata.json
```

其中一个 `sequence_data.npz` 内部包含：

- `input_recon`: `[5, 20, 32, 32]`
- `valid208_voltage_noisy`: `[5, 20, 208]`
- `target_delta_sigma`: `[20, 64, 64]`
- `target_pathology_mask`: `[64, 64]`
- `noise_levels_db`: `[5]`

这里的第一维 `5` 表示 5 个噪声级别：

- `0`: 无噪声
- `1`: 40 dB
- `2`: 30 dB
- `3`: 20 dB
- `4`: 10 dB

当前 baseline 配置默认使用：

- `noise_index = 2`
- 也就是 `30 dB`

## 2. Dataset 做了什么

`LCTSCDualSourceDataset` 的职责不是一次性把 5 个噪声都送进模型，而是：

1. 先根据 `global_samples_manifest.csv` 找到样本目录
2. 再根据 `case_id / slice_index / sample_name` 拼出相对路径
3. 从 `sequence_data.npz` 里取一个固定噪声级别 `k`
4. 返回训练真正需要的张量

也就是说，模型实际看到的是：

- `recon = input_recon[k]`，形状 `[20, 32, 32]`
- `voltage = valid208_voltage_noisy[k]`，形状 `[20, 208]`
- `target = target_delta_sigma`，形状 `[20, 64, 64]`
- `mask = target_pathology_mask`，形状 `[64, 64]`

第一版训练中，真正参与前向和主损失的是：

- `recon`
- `voltage`
- `target`

其他字段如 `mask`、`case_id`、`pathology_label` 主要用于：

- case 级划分
- 日志记录
- 可视化
- 后续扩展 ROI loss

## 3. DataLoader 后的 batch

如果 batch size 为 `B=8`，那么进入训练循环前的 batch 形状是：

- `recon`: `[8, 20, 32, 32]`
- `voltage`: `[8, 20, 208]`
- `target`: `[8, 20, 64, 64]`

注意这里已经不是旧任务里的：

- `5` 帧输入预测 `1` 帧

而是：

- `20` 帧输入预测 `20` 帧

## 4. 进入模型前的直观理解

你可以先把这两个输入这样理解：

- `recon` 提供空间结构信息
  - 哪些区域亮
  - 哪些区域暗
  - 每一帧重建的大致形状

- `voltage` 提供每一帧的全局观测信息
  - 边界测量告诉模型当前这帧整体该偏向什么状态
  - 但它本身不是一张二维图，所以需要先编码后再融合

## 5. 模型里的完整流向

### 5.1 电压分支编码

原始电压：

- `voltage`: `[B, 20, 208]`

每一帧 208 维电压向量先经过一个小 MLP：

```text
208 -> 64 -> 64
```

得到：

- `voltage_embed`: `[B, 20, 64]`

### 5.2 从电压特征里分出三条控制信号

从 `voltage_embed` 再分出三条头：

- `gate`: `[B, 20, 1, 1]`
- `bias`: `[B, 20, 1, 1]`
- `voltage_map`: `[B, 20, 1, 1]`，随后广播成 `[B, 20, 32, 32]`

它们的作用分别是：

- `gate`
  - 控制某一帧重建图整体放大或缩小多少

- `bias`
  - 控制某一帧重建图整体平移多少

- `voltage_map`
  - 生成一个由电压提供的辅助通道，和图像一起送进 U-Net

### 5.3 用电压调制重建图

融合公式是：

```python
recon_modulated = recon * (1.0 + gate) + bias
```

这里的意思是：

- 先按帧乘一个缩放系数
- 再按帧加一个偏置

因为 `gate` 和 `bias` 的形状是 `[B, 20, 1, 1]`，所以它们会对每一帧整张图做一致调节。

这就是“门控/偏置调制”的意思。

它不是在像素级决定空间位置，而是在每一帧层面做条件化控制。

### 5.4 把电压辅助图拼进去

随后构造：

- `recon_modulated`: `[B, 20, 32, 32]`
- `voltage_map`: `[B, 20, 32, 32]`

在通道维拼接：

```python
fused = torch.cat([recon_modulated, voltage_map], dim=1)
```

得到：

- `fused`: `[B, 40, 32, 32]`

你可以把它理解成：

- 前 20 个通道是“被电压调过的重建图序列”
- 后 20 个通道是“由电压生成的辅助提示图序列”

## 6. U-Net 主干做了什么

从这里开始，网络把 `40` 个通道当成一个多通道图像送进 U-Net：

- 输入：`[B, 40, 32, 32]`
- 编码器逐步下采样提特征
- 解码器逐步上采样恢复空间分辨率
- 最后 `1x1 conv` 输出 `20` 个通道

输出得到：

- `logits`: `[B, 20, 64, 64]`

如果解码后尺寸不是 `64x64`，模型最后还会补一个 `bilinear interpolate`，统一对齐到：

- `output_size = (64, 64)`

## 7. 从模型输出到训练目标

模型输出：

- `pred`: `[B, 20, 64, 64]`

监督标签：

- `target`: `[B, 20, 64, 64]`

两者是一一对齐的整段序列监督。

当前训练里使用了 3 类损失：

- `L1 loss`
  - 主回归损失

- `Edge loss`
  - 对空间边缘变化更敏感

- `TemporalDifferenceLoss`
  - 约束相邻帧之间的变化趋势

其中 `TemporalDifferenceLoss` 实际比较的是：

```text
pred[:, 1:] - pred[:, :-1]
target[:, 1:] - target[:, :-1]
```

也就是：

- 不是只看每帧像不像
- 还看相邻帧变化是不是合理

## 8. 可视化阶段做了什么

可视化时不会展示全部 20 帧，而是取中间帧做对比：

- 输入中间帧
- target 中间帧
- pred 中间帧

然后保存成：

```text
Input | Target | Pred
```

的伪彩图。

这只是为了快速看趋势，不代表训练只监督中间帧。

## 9. Mermaid 架构图

```mermaid
flowchart TD
    A[Disk Sample<br/>sequence_data.npz] --> B[Dataset<br/>select noise k]
    B --> C1[recon = input_recon[k]<br/>[B,20,32,32]]
    B --> C2[voltage = valid208_voltage_noisy[k]<br/>[B,20,208]]
    B --> C3[target = target_delta_sigma<br/>[B,20,64,64]]

    C2 --> D1[Voltage MLP<br/>208 -> 64 -> 64]
    D1 --> E1[gate<br/>[B,20,1,1]]
    D1 --> E2[bias<br/>[B,20,1,1]]
    D1 --> E3[voltage_map<br/>[B,20,1,1] -> [B,20,32,32]]

    C1 --> F1[recon * 1+gate + bias]
    E1 --> F1
    E2 --> F1

    F1 --> G[concat]
    E3 --> G

    G --> H[fused feature<br/>[B,40,32,32]]
    H --> I[U-Net Encoder]
    I --> J[Bottleneck]
    J --> K[U-Net Decoder]
    K --> L[1x1 Conv<br/>20 channels]
    L --> M[Interpolate to 64x64]
    M --> N[pred<br/>[B,20,64,64]]

    N --> O[L1 Loss]
    C3 --> O
    N --> P[Edge Loss]
    C3 --> P
    N --> Q[Temporal Difference Loss]
    C3 --> Q
```

## 10. 一句话总结

当前这版网络不是“让电压直接生成图像”，而是：

1. 先让电压告诉模型“每一帧该怎么调重建图”
2. 再把调好的图和电压辅助图一起送进 U-Net
3. 最后输出整段 `20` 帧 `64x64` 增强图像序列

## 11. 当前架构的优点和局限

优点：

- 结构简单，容易先跑通
- 双输入已经真正接入
- 输出和标签完全对齐，方便做整段序列监督

局限：

- `gate` 和 `bias` 是逐帧全局标量，不是空间自适应调制
- `voltage_map` 当前是广播得到的常数图，空间表达能力很弱
- 电压分支还没有显式做跨帧时序建模

所以这版更适合作为第一版 baseline，而不是最终结构。
