import torch
import torch.nn as nn
import torch.nn.functional as F

from .parts import DoubleConv, Down, Up, OutConv



class ST_UNet(nn.Module):
    def __init__(self, n_frames=5, n_classes=1, bilinear=True):
        super(ST_UNet, self).__init__()
        self.n_frames = n_frames
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_frames, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # [修改点]：减小了输出通道数，以适配下一层的拼接
        # 原来是 1024//factor (512), 改为 512//factor (256)
        self.up1 = Up(1024, 512 // factor, bilinear)

        # 原来是 512//factor (256), 改为 256//factor (128)
        self.up2 = Up(512, 256 // factor, bilinear)

        # 原来是 256//factor (128), 改为 128//factor (64)
        self.up3 = Up(256, 128 // factor, bilinear)

        # 最后一层保持不变，输出 64 给 outc
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # x shape: [Batch, n_frames, H, W]
        # 这里的输入 x 应该已经包含了 n_frames 个通道
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
