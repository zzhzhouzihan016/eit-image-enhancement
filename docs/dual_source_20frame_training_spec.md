# 双输入 20 帧训练接口规范

本文档定义新任务的第一版数据接口与训练约定，目标是支持：

- 输入 1：20 帧 EIT 重建图像序列
- 输入 2：20 帧对应的边界电压序列
- 输出：20 帧增强后的 64x64 EIT 目标图像序列

当前重点不是一次把模型做复杂，而是先把数据接口、训练样本定义和 baseline 约定固定下来，保证后续模型迭代不会反复返工数据层。

## 1. 任务定义

基于 `lctsc_cem_pathology_jac32_gt64` 数据集，第一版任务定义为：

- 图像分支输入：`input_recon[k]`，形状 `[20, 32, 32]`
- 电压分支输入：`valid208_voltage_noisy[k]`，形状 `[20, 208]`
- 监督目标：`target_delta_sigma`，形状 `[20, 64, 64]`

其中：

- `k` 为噪声级别索引
- 第一版 baseline 推荐固定 `k = 2`，即 `30 dB`
- 推荐先做整段序列到整段序列的监督，不再沿用“5 帧输入预测中间帧”的旧设定
- 如果你当前论文或实验里把网络输出称为“增强图像”，那么这里的 `target_delta_sigma` 就是第一版最合适的增强目标图像

## 2. 数据根目录建议

推荐将数据集放到：

`data/processed/train_sim/lctsc_cem_pathology_jac32_gt64/`

目录结构保持原样：

```text
data/processed/train_sim/lctsc_cem_pathology_jac32_gt64/
|- cases/
|- global_samples_manifest.csv
|- samples_manifest.csv
|- summary.json
\- README.md
```

说明：

- `global_samples_manifest.csv` 作为权威样本索引
- `samples_manifest.csv` 和 `summary.json` 仅作参考，不应作为全量训练索引
- manifest 中现有 Windows 绝对路径不应直接依赖，训练时应优先使用相对路径或由字段自行拼接路径

## 3. 单个训练样本定义

建议每个训练样本对应：

- 一个 `case_id`
- 一个 `slice_index`
- 一个 `sample_name`
- 一个 `noise_index`

也就是说，磁盘上仍然保留一个样本目录对应 5 个噪声级别的聚合存储，但训练时将其展开为多个可迭代样本。

一个逻辑训练样本包含：

```python
{
    "recon": Tensor[20, 32, 32],
    "voltage": Tensor[20, 208],
    "target": Tensor[20, 64, 64],
    "mask": Tensor[64, 64],
    "noise_db": float,
    "noise_index": int,
    "case_id": str,
    "slice_index": int,
    "sample_name": str,
    "pathology_label": str,
    "lung_side": str | None,
    "severity_level": int,
    "replicate_index": int,
}
```

推荐 `Dataset.__getitem__` 返回字典，而不是元组。这样在后续加分支、加辅助损失、加分类头时不需要频繁改调用接口。

其中可以把字段分成两类：

- 主训练字段：`recon`、`voltage`、`target`
- 辅助字段：`mask`、`noise_db`、`noise_index` 和各类元信息

第一版 baseline 中，辅助字段默认不直接参与模型前向，只用于数据划分、日志记录、可视化和可选损失扩展。

## 4. Batch 形状约定

设 batch size 为 `B`，则第一版 batch 约定为：

- `recon`: `[B, 20, 32, 32]`
- `voltage`: `[B, 20, 208]`
- `target`: `[B, 20, 64, 64]`
- `mask`: `[B, 64, 64]`

如果图像分支使用 2D CNN 编码每一帧，再做时间融合，常见内部变换为：

- 原始：`[B, T, H, W]`
- 加通道后：`[B, T, 1, H, W]`
- 合并 batch 和时间做逐帧编码：`[B*T, 1, H, W]`
- 编码后再还原时间维：`[B, T, C, H', W']`

如果电压分支使用 MLP 或 1D Temporal Encoder，输入可保持：

- `voltage`: `[B, T, 208]`

## 5. 推荐监督目标

第一版推荐优先使用：

- `target_delta_sigma`

原因：

- 当前输入本身来自 differential EIT reconstruction
- `frame 0` 为参考帧，和 `target_delta_sigma` 的物理定义更一致
- 训练初期更容易对齐输入和标签的动态变化

暂不推荐第一版直接预测：

- `target_sigma`

除非后续明确想做“由差分输入恢复绝对电导率”的更难任务。

从训练接口角度看：

- 如果你把 `target_delta_sigma` 当成监督标签，它就是网络要输出的 64x64 增强目标图像
- 如果后续你改成别的 64x64 标签定义，那么“增强图像”的具体物理含义也会随之变化

## 6. 噪声组织策略

当前 `input_recon` 的磁盘形状为 `[5, 20, 32, 32]`，这是合理的存储格式，不建议改成 5 份重复目录。

但训练时不建议把 5 个噪声级别同时送进网络作为输入主维度。第一版更推荐以下两种策略：

### 方案 A：固定噪声训练

- 仅使用 `k = 2`，即 `30 dB`
- 优点是变量最少，便于先验证模型结构

### 方案 B：展开为多条训练样本

- 将一个物理样本展开为 5 条样本，每条对应一个 `noise_index`
- 每次 forward 只输入一个噪声级别

第一阶段推荐先用方案 A，模型跑通后再切到方案 B。

不建议第一版直接把噪声维并入输入，例如：

- 图像变成 `[5, 20, 32, 32]`
- 电压变成 `[5, 20, 208]`

这样会把“主任务学习”和“噪声鲁棒性建模”混在一起，调试成本明显升高。

## 7. 数据划分规则

必须按 `case_id` 划分 train / val / test，不能按 sample 随机划分。

原因：

- 同一个 case 下不同 slice 共享解剖结构
- 同一个 slice 下不同 pathology sample 共享底层 mesh 和 anatomy
- 如果按 sample 随机切分，会产生明显的信息泄漏

推荐第一版划分原则：

- train：大多数 case
- val：少量 case
- test：保留若干完整 case，不参与调参

如果当前阶段只想快速验证流程，至少也应保证：

- val case 与 train case 不重叠

## 8. 归一化建议

第一版建议分支独立归一化：

### 图像分支

对 `input_recon[k]` 做样本内或全局统计归一化，二选一：

- 方案 1：按训练集统计全局 mean/std
- 方案 2：按样本做稳健缩放，例如按分位数裁剪后缩放

由于抽样可见 `input_recon` 数值幅度较小，且正负都有，建议保留符号信息，不要简单裁成 `[0, 1]`。

### 电压分支

若使用 `valid208_voltage_noisy[k]`：

- 建议按训练集统计对 208 通道做标准化

若使用 `delta_v_norm_noisy[k]`：

- 建议更谨慎，因为其数值范围波动更大
- 第一版更推荐先用 `valid208_voltage_noisy[k]`

## 9. 第一版 Baseline 约定

在模型结构尚未完全确定前，推荐先固定以下约定：

- 输入帧数：20
- 图像输入：`input_recon[2]`
- 电压输入：`valid208_voltage_noisy[2]`
- 输出目标：`target_delta_sigma`
- 输入分辨率：`32 x 32`
- 输出分辨率：`64 x 64`
- 单任务回归，不额外加分类头

第一版 baseline 目标不是追求最佳性能，而是回答以下问题：

1. 双分支数据是否能稳定读入
2. 模型是否能对齐整段 20 帧时序
3. loss 是否正常下降
4. 输出是否比单图像分支更稳定

## 10. 推荐的 Dataset 返回结构

建议新建专用数据集类，例如：

- `src/eit/dataset_dual_source.py`

推荐接口：

```python
sample = dataset[idx]

sample["recon"]      # FloatTensor [20, 32, 32]
sample["voltage"]    # FloatTensor [20, 208]
sample["target"]     # FloatTensor [20, 64, 64]
sample["mask"]       # FloatTensor [64, 64]
sample["noise_db"]   # float
sample["meta"]       # dict
```

其中：

- `meta` 可保留 `case_id`、`slice_index`、`sample_name`、`pathology_label` 等信息
- 训练主循环只取张量字段
- 验证和可视化阶段可使用 `meta`
- `mask` 在第一版可先不参与 loss；后续如果你想只强调肺区或病灶区，可以把它接入加权损失

## 11. 训练循环接口建议

训练循环建议从：

```python
for x, y in loader:
    pred = model(x)
```

改为：

```python
for batch in loader:
    recon = batch["recon"].to(device)
    voltage = batch["voltage"].to(device)
    target = batch["target"].to(device)
    pred = model(recon=recon, voltage=voltage)
```

推荐模型前向接口：

```python
pred = model(recon=recon, voltage=voltage)
```

而不是依赖位置参数，这样后面更容易扩展。

## 12. 第一版损失函数建议

第一版先用简单稳定的组合即可：

- 主损失：`L1` 或 `SmoothL1`
- 可选附加项：时序一致性损失

例如：

```python
loss_recon = l1(pred, target)
loss_temporal = l1(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
loss = loss_recon + 0.1 * loss_temporal
```

暂时不建议一开始就引入过多复杂损失。

## 13. 第一阶段最小目标

在真正设计复杂模型前，建议先完成以下最小闭环：

1. 新 `Dataset` 能按 `case_id` 正确切分
2. `DataLoader` 能产出 `recon / voltage / target`
3. 一个最小双分支模型能完成前向
4. 能在一个 batch 上过拟合
5. 能稳定训练 1 到 5 个 epoch

只要这个闭环通了，后续再尝试更复杂的时序模块、跨模态注意力或 Transformer 结构都会更稳。
