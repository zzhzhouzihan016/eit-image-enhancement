import torch
import torch.nn as nn
import torch.nn.functional as F

from .parts import DoubleConv, Down, OutConv, Up


def _zero_init_linear(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class DualSourceTemporalFiLMUNet(nn.Module):
    """
    稳妥版 DSFNet-inspired 双源增强模型：
    1. 逐帧共享 2D U-Net encoder，避免把时间仅当作静态通道。
    2. 电压分支用 MLP 编码每一帧的边界电压。
    3. 图像 token 与电压 token 融合后送入 GRU，显式建模时间依赖。
    4. GRU 产生的时序上下文以 FiLM 方式调制深层特征，再逐帧解码。

    输入：
    - recon: [B, T, 32, 32]
    - voltage: [B, T, 208]

    输出：
    - pred: [B, T, 64, 64]
    """

    def __init__(
        self,
        n_frames: int = 20,
        voltage_dim: int = 208,
        out_frames: int | None = None,
        bilinear: bool = True,
        base_channels: int = 32,
        voltage_hidden: int = 64,
        temporal_hidden: int = 128,
        gru_layers: int = 1,
        output_size: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()

        self.n_frames = n_frames
        self.out_frames = out_frames if out_frames is not None else n_frames
        if self.out_frames != self.n_frames:
            raise ValueError("DualSourceTemporalFiLMUNet 当前仅支持输入帧数与输出帧数一致。")

        self.voltage_dim = voltage_dim
        self.bilinear = bilinear
        self.base_channels = base_channels
        self.output_size = tuple(output_size)

        self.inc = DoubleConv(1, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.bottleneck_channels = (base_channels * 16) // factor
        self.skip4_channels = base_channels * 8
        self.down4 = Down(base_channels * 8, self.bottleneck_channels)

        self.img_pool = nn.AdaptiveAvgPool2d(1)
        self.voltage_encoder = nn.Sequential(
            nn.Linear(voltage_dim, voltage_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(voltage_hidden, voltage_hidden),
            nn.ReLU(inplace=True),
        )
        self.fusion_proj = nn.Linear(self.bottleneck_channels + voltage_hidden, temporal_hidden)
        self.temporal_gru = nn.GRU(
            input_size=temporal_hidden,
            hidden_size=temporal_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(temporal_hidden)

        self.skip4_film = nn.Linear(temporal_hidden, self.skip4_channels * 2)
        self.bottleneck_film = nn.Linear(temporal_hidden, self.bottleneck_channels * 2)
        _zero_init_linear(self.skip4_film)
        _zero_init_linear(self.bottleneck_film)

        self.up1 = Up(base_channels * 16, (base_channels * 8) // factor, bilinear)
        self.up2 = Up(base_channels * 8, (base_channels * 4) // factor, bilinear)
        self.up3 = Up(base_channels * 4, (base_channels * 2) // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, 1)

    @staticmethod
    def _reshape_to_sequence(tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
        return tensor.reshape(batch_size, num_frames, *tensor.shape[1:])

    @staticmethod
    def _apply_film(features: torch.Tensor, gamma_beta: torch.Tensor) -> torch.Tensor:
        gamma, beta = torch.chunk(gamma_beta, 2, dim=2)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + gamma) + beta

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

        frame_inputs = recon.reshape(batch_size * num_frames, 1, height, width)

        x1 = self.inc(frame_inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        pooled_img = self.img_pool(x5).flatten(1)
        pooled_img = pooled_img.reshape(batch_size, num_frames, self.bottleneck_channels)

        voltage_embed = self.voltage_encoder(voltage.reshape(batch_size * num_frames, self.voltage_dim))
        voltage_embed = voltage_embed.reshape(batch_size, num_frames, -1)

        fused_tokens = torch.cat([pooled_img, voltage_embed], dim=-1)
        fused_tokens = self.fusion_proj(fused_tokens)

        temporal_features, _ = self.temporal_gru(fused_tokens)
        temporal_features = self.temporal_norm(temporal_features + fused_tokens)

        x4_seq = self._reshape_to_sequence(x4, batch_size, num_frames)
        x5_seq = self._reshape_to_sequence(x5, batch_size, num_frames)

        x4_seq = self._apply_film(x4_seq, self.skip4_film(temporal_features))
        x5_seq = self._apply_film(x5_seq, self.bottleneck_film(temporal_features))

        x4 = x4_seq.reshape(batch_size * num_frames, *x4.shape[1:])
        x5 = x5_seq.reshape(batch_size * num_frames, *x5.shape[1:])

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        if logits.shape[-2:] != self.output_size:
            logits = F.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)

        logits = logits.reshape(batch_size, num_frames, *self.output_size)
        return logits
