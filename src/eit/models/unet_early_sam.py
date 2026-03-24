import torch.nn as nn

from .attention import EarlySpatialAttention
from .parts import DoubleConv, Down, Up, OutConv



class ST_UNet_EarlySAM(nn.Module):
    def __init__(self, n_frames=5, n_classes=1, bilinear=True):
        super().__init__()
        self.n_frames = n_frames
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.early_sam = EarlySpatialAttention(kernel_size=7)

        self.inc = DoubleConv(n_frames, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  # 输出通道是 512 (如果 bilinear=True)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x = self.early_sam(x)

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
