import torch
import torch.nn as nn
import torch.nn.functional as F

from .parts import DoubleConv, Down, OutConv, Up


class ReconSeqUNet(nn.Module):
    """
    单源 20-frame baseline：
    - recon: [B, T, 32, 32]
    - pred: [B, T, 64, 64]

    设计上与双源 baseline 保持同级别容量，
    只移除电压分支，用于评估引入 voltage 是否真正带来收益。
    """

    def __init__(
        self,
        n_frames: int = 20,
        out_frames: int | None = None,
        bilinear: bool = True,
        base_channels: int = 32,
        output_size: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()

        self.n_frames = n_frames
        self.out_frames = out_frames if out_frames is not None else n_frames
        self.bilinear = bilinear
        self.base_channels = base_channels
        self.output_size = tuple(output_size)

        self.inc = DoubleConv(n_frames, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, (base_channels * 16) // factor)

        self.up1 = Up(base_channels * 16, (base_channels * 8) // factor, bilinear)
        self.up2 = Up(base_channels * 8, (base_channels * 4) // factor, bilinear)
        self.up3 = Up(base_channels * 4, (base_channels * 2) // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, self.out_frames)

    def forward(self, recon: torch.Tensor) -> torch.Tensor:
        if recon.ndim != 4:
            raise ValueError(f"recon 期望形状为 [B, T, H, W]，实际为 {tuple(recon.shape)}")

        _, num_frames, _, _ = recon.shape
        if num_frames != self.n_frames:
            raise ValueError(f"模型配置 n_frames={self.n_frames}，但收到 num_frames={num_frames}")

        x1 = self.inc(recon)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        if logits.shape[-2:] != self.output_size:
            logits = F.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)

        return logits
