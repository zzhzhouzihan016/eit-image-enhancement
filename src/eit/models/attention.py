import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 平均池化和最大池化：压缩空间，提取通道特征
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享感知层 (Shared MLP)
        # 先降维 (in_planes // ratio) 再升维，减少参数量
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. AvgPool 分支
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 2. MaxPool 分支
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 3. 叠加 + Sigmoid 生成权重
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 输入通道是2 (一个来自AvgPool，一个来自MaxPool)
        # 使用 7x7 大卷积核感受更广的空间范围
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 在通道维度上做 AvgPool 和 MaxPool -> [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 2. 拼接 -> [B, 2, H, W]
        x = torch.cat([avg_out, max_out], dim=1)
        # 3. 卷积 + Sigmoid 生成权重
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 1. 先做通道注意力 refined = x * weight
        out = x * self.ca(x)
        # 2. 再做空间注意力
        result = out * self.sa(out)
        return result

class TemporalAttention(nn.Module):
    def __init__(self, num_frames=5, reduction=2):
        """
        时间注意力模块
        :param num_frames: 输入的时序帧数 (对应你的 n_frames)
        :param reduction: 降维系数，用于减少 MLP 的参数量和计算量
        """
        super(TemporalAttention, self).__init__()

        # 1. Squeeze: 空间维度的全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 2. Excitation: 学习帧与帧之间的时间依赖关系
        # 为了防止全连接层过拟合，中间加了一个降维瓶颈 (Bottleneck)
        mid_channels = max(1, num_frames // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(num_frames, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, num_frames, bias=False),
            nn.Sigmoid()  # 将权重归一化到 0~1 之间
        )

    def forward(self, x):
        """
        x shape: [Batch, T, H, W]
        """
        b, t, h, w = x.size()

        # Squeeze 阶段: [B, T, H, W] -> [B, T, 1, 1] -> [B, T]
        # 提取每一帧的全局特征
        y = self.avg_pool(x).view(b, t)

        # Excitation 阶段: [B, T] -> [B, T]
        # 计算每一帧的时间注意力权重
        weights = self.mlp(y)

        # Reweight 阶段: [B, T] -> [B, T, 1, 1]
        # 恢复维度以便与原特征图相乘
        weights = weights.view(b, t, 1, 1)

        # 将权重乘回原图
        return x * weights.expand_as(x)

class EarlySpatialAttention(nn.Module):
    """
    前置纯空间注意力模块 (Early Spatial Attention)
    只关注“哪里重要”，不区分“哪一帧重要”
    """

    def __init__(self, kernel_size=7):
        super(EarlySpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 输入是均值图(1通道) + 最大值图(1通道) = 2通道
        # 输出是 1通道 的空间权重图
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [Batch, 5(Frames), 176, 256]

        # 1. 沿着时间(通道)维度求平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [Batch, 1, 176, 256]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [Batch, 1, 176, 256]

        # 2. 拼接特征图
        concat = torch.cat([avg_out, max_out], dim=1)  # [Batch, 2, 176, 256]

        # 3. 经过卷积和 Sigmoid 激活，生成 0~1 的空间权重图
        # weights shape: [Batch, 1, 176, 256]
        weights = self.sigmoid(self.conv(concat))

        # 4. 将算出的空间权重，均匀地乘回原始的 5 帧图像上
        # 这样肺部区域在这 5 帧里都会被提亮，背景都会被抑制
        return x * weights