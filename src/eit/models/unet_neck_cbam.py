import torch.nn as nn

from .attention import CBAM
from .parts import DoubleConv, Down, Up, OutConv



class ST_UNet_NeckCBAM(nn.Module):
    def __init__(self, n_frames=5, n_classes=1, bilinear=True):
        super().__init__()
        self.n_frames = n_frames
        self.n_classes = n_classes
        self.bilinear = bilinear

        # ... (前面的 inc, down1, down2, down3 保持不变) ...
        self.inc = DoubleConv(n_frames, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  # 输出通道是 512 (如果 bilinear=True)

        # ==========================================
        # 🌟 新增：注意力模块
        # ==========================================
        # 瓶颈层的通道数是 1024 // factor。如果 bilinear=True，那就是 512。
        bottleneck_channels = 1024 // factor
        self.attention = CBAM(bottleneck_channels)

        # ... (后面的 up1, up2, up3, up4, outc 保持不变) ...
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # 1. 瓶颈层输出 (Features)
        x5 = self.down4(x4)

        # ==========================================
        # 🌟 插入：注意力增强
        # ==========================================
        # 输入: [B, 512, 11, 15] -> 输出: [B, 512, 11, 15]
        # 这一步，模型会重新审视特征图，把不重要的背景特征压低
        x5 = self.attention(x5)

        # 2. 解码 (使用增强后的 x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
