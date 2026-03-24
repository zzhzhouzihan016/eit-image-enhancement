import torch
import torch.nn as nn
import torch.nn.functional as F

from .parts import DoubleConv, Down, OutConv, Up


class DualSourceSeqUNet(nn.Module):
    """
    最小双输入 baseline：
    - recon: [B, T, 32, 32]
    - voltage: [B, T, 208]
    - pred: [B, T, 64, 64]

    设计思路：
    1. 把 20 帧 recon 视为时序通道，联合建模整段序列。
    2. 用轻量 MLP 编码每一帧的电压，再生成对应的门控与偏置。
    3. 将电压信息广播回空间平面，与 recon 一起送入 U-Net。
    """

    def __init__(
        self,
        n_frames: int = 20,
        voltage_dim: int = 208,
        out_frames: int | None = None,
        bilinear: bool = True,
        base_channels: int = 32,
        voltage_hidden: int = 64,
        output_size: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()

        self.n_frames = n_frames
        self.voltage_dim = voltage_dim
        self.out_frames = out_frames if out_frames is not None else n_frames
        self.bilinear = bilinear
        self.base_channels = base_channels
        self.output_size = tuple(output_size)

        self.voltage_encoder = nn.Sequential(
            nn.Linear(voltage_dim, voltage_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(voltage_hidden, voltage_hidden),
            nn.ReLU(inplace=True),
        )
        self.voltage_gate = nn.Linear(voltage_hidden, 1)
        self.voltage_bias = nn.Linear(voltage_hidden, 1)
        self.voltage_map = nn.Linear(voltage_hidden, 1)

        in_channels = 2 * n_frames

        self.inc = DoubleConv(in_channels, base_channels)
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

    def forward(self, recon: torch.Tensor, voltage: torch.Tensor) -> torch.Tensor:
        if recon.ndim != 4:
            raise ValueError(f"recon 期望形状为 [B, T, H, W]，实际为 {tuple(recon.shape)}")
        if voltage.ndim != 3:
            raise ValueError(f"voltage 期望形状为 [B, T, D]，实际为 {tuple(voltage.shape)}")

        batch_size, num_frames, height, width = recon.shape
        if num_frames != self.n_frames:
            raise ValueError(f"模型配置 n_frames={self.n_frames}，但收到 num_frames={num_frames}")
        if voltage.shape[0] != batch_size or voltage.shape[1] != num_frames:
            raise ValueError("recon 与 voltage 的 batch/time 维不一致。")
        if voltage.shape[2] != self.voltage_dim:
            raise ValueError(f"模型配置 voltage_dim={self.voltage_dim}，但收到 {voltage.shape[2]}")

        voltage_embed = self.voltage_encoder(voltage.reshape(batch_size * num_frames, self.voltage_dim))
        voltage_embed = voltage_embed.view(batch_size, num_frames, -1)

        gate = torch.sigmoid(self.voltage_gate(voltage_embed)).view(batch_size, num_frames, 1, 1)
        bias = self.voltage_bias(voltage_embed).view(batch_size, num_frames, 1, 1)
        voltage_map = self.voltage_map(voltage_embed).view(batch_size, num_frames, 1, 1)
        voltage_map = voltage_map.expand(-1, -1, height, width)

        recon_modulated = recon * (1.0 + gate) + bias
        fused = torch.cat([recon_modulated, voltage_map], dim=1)

        x1 = self.inc(fused)
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
