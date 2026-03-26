from __future__ import annotations

from typing import Callable

from .dual_source_seq_unet import DualSourceSeqUNet
from .recon_seq_unet import ReconSeqUNet
from .unet import ST_UNet
from .unet_early_cbam import ST_UNet_EarlyCBAM
from .unet_early_sam import ST_UNet_EarlySAM
from .unet_early_tam import ST_UNet_EarlyTAM
from .unet_neck_cbam import ST_UNet_NeckCBAM
from .unet_neck_sam import ST_UNet_NeckSAM
from .unet_neck_tam import ST_UNet_NeckTAM


def _build_st_unet(params: dict) -> ST_UNet:
    return ST_UNet(
        n_frames=params.get("n_frames", 5),
        n_classes=params.get("n_classes", 1),
        bilinear=params.get("bilinear", True),
    )


def _build_st_unet_neck_cbam(params: dict) -> ST_UNet_NeckCBAM:
    return ST_UNet_NeckCBAM(
        n_frames=params.get("n_frames", 5),
        n_classes=params.get("n_classes", 1),
        bilinear=params.get("bilinear", True),
    )


def _build_st_unet_early_cbam(params: dict) -> ST_UNet_EarlyCBAM:
    return ST_UNet_EarlyCBAM(
        n_frames=params.get("n_frames", 5),
        n_classes=params.get("n_classes", 1),
        bilinear=params.get("bilinear", True),
    )


def _build_st_unet_neck_sam(params: dict) -> ST_UNet_NeckSAM:
    return ST_UNet_NeckSAM(
        n_frames=params.get("n_frames", 5),
        n_classes=params.get("n_classes", 1),
        bilinear=params.get("bilinear", True),
    )


def _build_st_unet_early_sam(params: dict) -> ST_UNet_EarlySAM:
    return ST_UNet_EarlySAM(
        n_frames=params.get("n_frames", 5),
        n_classes=params.get("n_classes", 1),
        bilinear=params.get("bilinear", True),
    )


def _build_st_unet_neck_tam(params: dict) -> ST_UNet_NeckTAM:
    return ST_UNet_NeckTAM(
        n_frames=params.get("n_frames", 5),
        n_classes=params.get("n_classes", 1),
        bilinear=params.get("bilinear", True),
    )


def _build_st_unet_early_tam(params: dict) -> ST_UNet_EarlyTAM:
    return ST_UNet_EarlyTAM(
        n_frames=params.get("n_frames", 5),
        n_classes=params.get("n_classes", 1),
        bilinear=params.get("bilinear", True),
    )


def _build_dual_source_seq_unet(params: dict) -> DualSourceSeqUNet:
    return DualSourceSeqUNet(
        n_frames=params.get("n_frames", 20),
        voltage_dim=params.get("voltage_dim", 208),
        out_frames=params.get("out_frames", params.get("n_classes", params.get("n_frames", 20))),
        bilinear=params.get("bilinear", True),
        base_channels=params.get("base_channels", 32),
        voltage_hidden=params.get("voltage_hidden", 64),
        output_size=tuple(params.get("output_size", [64, 64])),
    )


def _build_recon_seq_unet(params: dict) -> ReconSeqUNet:
    return ReconSeqUNet(
        n_frames=params.get("n_frames", 20),
        out_frames=params.get("out_frames", params.get("n_classes", params.get("n_frames", 20))),
        bilinear=params.get("bilinear", True),
        base_channels=params.get("base_channels", 32),
        output_size=tuple(params.get("output_size", [64, 64])),
    )


MODEL_REGISTRY: dict[str, Callable[[dict], object]] = {
    "st_unet": _build_st_unet,
    "st_unet_neck_cbam": _build_st_unet_neck_cbam,
    "st_unet_early_cbam": _build_st_unet_early_cbam,
    "st_unet_neck_sam": _build_st_unet_neck_sam,
    "st_unet_early_sam": _build_st_unet_early_sam,
    "st_unet_neck_tam": _build_st_unet_neck_tam,
    "st_unet_early_tam": _build_st_unet_early_tam,
    "dual_source_seq_unet": _build_dual_source_seq_unet,
    "recon_seq_unet": _build_recon_seq_unet,
}


def get_model(config):
    model_name = config["name"]
    params = config.get("params", {})

    print(f"🔄 Initializing Model: {model_name}")
    try:
        builder = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"❌ Unknown model name: {model_name}. Available: {available}") from exc

    return builder(params)
