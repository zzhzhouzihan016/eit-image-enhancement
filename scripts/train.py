import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import wandb
except ImportError:
    wandb = None

from eit.dataset_dual_source import (
    LCTSCDualSourceDataset,
    LCTSCReconSequenceDataset,
    build_case_splits_from_manifest,
)
from eit.models import get_model
from eit.utils.seed import set_seed


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def as_spatial_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    if tensor.ndim == 4:
        if tensor.shape[1] == 1:
            return tensor
        return tensor.reshape(-1, 1, tensor.shape[-2], tensor.shape[-1])
    if tensor.ndim == 5:
        batch_size, time_steps, channels, height, width = tensor.shape
        return tensor.reshape(batch_size * time_steps, channels, height, width)
    raise ValueError(f"不支持的张量形状: {tuple(tensor.shape)}")


def select_middle_frame(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim == 4:
        if tensor.shape[1] == 1:
            return tensor[:, 0]
        return tensor[:, tensor.shape[1] // 2]
    if tensor.ndim == 5:
        return tensor[:, tensor.shape[1] // 2, 0]
    raise ValueError(f"不支持的张量形状: {tuple(tensor.shape)}")


def to_uint8_image(frame: torch.Tensor, value_range: tuple[float, float] | None = None) -> np.ndarray:
    array = frame.detach().cpu().numpy().astype(np.float32)
    if value_range is None:
        vmin = float(array.min())
        vmax = float(array.max())
    else:
        vmin, vmax = value_range

    if vmax <= vmin:
        vmax = vmin + 1e-6

    normalized = np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def forward_model(model: nn.Module, model_inputs):
    if isinstance(model_inputs, dict):
        return model(**model_inputs)
    return model(model_inputs)


def prepare_batch(batch, device: torch.device):
    if isinstance(batch, dict):
        recon = batch["recon"].to(device)
        target = batch["target"].to(device)
        model_inputs = {"recon": recon}
        if "voltage" in batch:
            model_inputs["voltage"] = batch["voltage"].to(device)
        return model_inputs, target, recon, target.size(0)

    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    return inputs, targets, inputs, targets.size(0)


class EITMetrics:
    @staticmethod
    def psnr(preds: torch.Tensor, targets: torch.Tensor, data_range: float | None = None) -> float:
        p = preds.reshape(preds.shape[0], -1)
        t = targets.reshape(targets.shape[0], -1)
        mse = torch.mean((p - t) ** 2, dim=1)

        if data_range is None:
            ranges = torch.clamp(t.max(dim=1).values - t.min(dim=1).values, min=1e-6)
        else:
            ranges = torch.full_like(mse, float(data_range))

        psnr_val = 10 * torch.log10((ranges ** 2) / (mse + 1e-8))
        return torch.mean(psnr_val).item()

    @staticmethod
    def cc(preds: torch.Tensor, targets: torch.Tensor) -> float:
        p = preds.reshape(preds.shape[0], -1)
        t = targets.reshape(targets.shape[0], -1)
        p_mean = p - torch.mean(p, dim=1, keepdim=True)
        t_mean = t - torch.mean(t, dim=1, keepdim=True)
        num = torch.sum(p_mean * t_mean, dim=1)
        den = torch.sqrt(torch.sum(p_mean ** 2, dim=1)) * torch.sqrt(torch.sum(t_mean ** 2, dim=1))
        return torch.mean(num / (den + 1e-8)).item()

    @staticmethod
    def rie(preds: torch.Tensor, targets: torch.Tensor) -> float:
        p = preds.reshape(preds.shape[0], -1)
        t = targets.reshape(targets.shape[0], -1)
        return torch.mean(torch.norm(p - t, p=2, dim=1) / (torch.norm(t, p=2, dim=1) + 1e-8)).item()

    @staticmethod
    def ssim(
        preds: torch.Tensor,
        targets: torch.Tensor,
        window_size: int = 11,
        window_sigma: float = 1.5,
        data_range: float | None = None,
    ) -> float:
        preds_4d = as_spatial_batch(preds)
        targets_4d = as_spatial_batch(targets)

        channel = preds_4d.size(1)
        gauss = torch.tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * window_sigma ** 2)) for x in range(window_size)],
            device=preds_4d.device,
            dtype=preds_4d.dtype,
        )
        gauss = gauss / gauss.sum()
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(channel, 1, window_size, window_size)

        mu1 = F.conv2d(preds_4d, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(targets_4d, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(preds_4d * preds_4d, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(targets_4d * targets_4d, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(preds_4d * targets_4d, window, padding=window_size // 2, groups=channel) - mu1_mu2

        if data_range is None:
            value_range = float((targets_4d.max() - targets_4d.min()).clamp_min(1e-6).item())
        else:
            value_range = float(data_range)

        c1 = (0.01 * value_range) ** 2
        c2 = (0.03 * value_range) ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
        )
        return ssim_map.mean().item()


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val_loss = 0.0
        self.psnr = 0.0
        self.ssim = 0.0
        self.cc = 0.0
        self.rie = 0.0
        self.count = 0

    def update(self, loss: float, p: float, s: float, c: float, r: float, n: int = 1) -> None:
        self.val_loss += loss * n
        self.psnr += p * n
        self.ssim += s * n
        self.cc += c * n
        self.rie += r * n
        self.count += n

    def avg(self) -> dict[str, float]:
        return {
            "val/loss": self.val_loss / self.count,
            "val/psnr": self.psnr / self.count,
            "val/ssim": self.ssim / self.count,
            "val/cc": self.cc / self.count,
            "val/rie": self.rie / self.count,
        }


class EdgeLoss(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        kernel = torch.tensor([[0.1, 0.3, 0.1], [0.3, -1.6, 0.3], [0.1, 0.3, 0.1]], device=device)
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_4d = as_spatial_batch(pred)
        target_4d = as_spatial_batch(target)
        pred_edges = F.conv2d(pred_4d, self.kernel, padding=1)
        target_edges = F.conv2d(target_4d, self.kernel, padding=1)
        return self.l1(pred_edges, target_edges)


class TemporalDifferenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 4 or target.ndim != 4 or pred.shape[1] <= 1:
            return pred.new_tensor(0.0)
        return self.l1(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])


def inspect_single_source_npz(npz_path: str | Path, frames_per_seq_hint: int | None = None) -> tuple[int, int]:
    with np.load(npz_path, allow_pickle=False) as raw:
        if "input_data" in raw:
            inputs = raw["input_data"]
        else:
            inputs = raw["input"]

        if inputs.ndim == 4:
            return int(inputs.shape[0]), int(inputs.shape[1])
        if inputs.ndim == 3:
            total_frames = int(inputs.shape[0])
            if frames_per_seq_hint is None:
                return 1, total_frames
            if total_frames % int(frames_per_seq_hint) != 0:
                raise ValueError(
                    "single-source 3D 输入无法按 frames_per_seq 正确分组: "
                    f"total_frames={total_frames}, frames_per_seq={frames_per_seq_hint}"
                )
            return total_frames // int(frames_per_seq_hint), int(frames_per_seq_hint)
        raise ValueError(f"不支持的 single-source 输入维度: {inputs.shape}")


def build_single_source_dataloaders(cfg: dict, seed: int):
    from eit.dataset import EITSequenceDataset

    data_cfg = cfg["data"]
    train_path = Path(data_cfg["train_path"])
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path
    frames_per_seq = data_cfg.get("frames_per_seq")
    num_samples, detected_frames_per_seq = inspect_single_source_npz(train_path, frames_per_seq_hint=frames_per_seq)

    if frames_per_seq is None:
        frames_per_seq = detected_frames_per_seq
    elif int(frames_per_seq) != int(detected_frames_per_seq):
        print(
            "⚠️ Single-source config 中的 frames_per_seq "
            f"({frames_per_seq}) 与数据实际值 ({detected_frames_per_seq}) 不一致，已自动修正。"
        )
        frames_per_seq = detected_frames_per_seq

    img_h = int(data_cfg.get("img_size_h", 176))
    img_w = int(data_cfg.get("img_size_w", 256))
    n_frames = int(cfg["model"]["params"]["n_frames"])
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))
    val_ratio = float(data_cfg.get("val_ratio", 0.05))

    target_blur_cfg = data_cfg.get("target_blur", {})
    apply_target_blur = bool(target_blur_cfg.get("enable", False))
    target_blur_kernel_size = int(target_blur_cfg.get("kernel_size", 5))
    target_blur_sigma = target_blur_cfg.get("sigma", 1.0)

    aug_cfg = data_cfg.get("augmentation", {})
    use_augmentation = bool(aug_cfg.get("enable", True))
    hflip_prob = float(aug_cfg.get("hflip_prob", 0.5))
    rotate_prob = float(aug_cfg.get("rotate_prob", 0.3))
    rotate_deg = float(aug_cfg.get("rotate_deg", 5.0))

    print(
        "🔧 Single-source Dataset Config: "
        f"samples={num_samples}, frames_per_seq={frames_per_seq}, size=({img_h}, {img_w}), "
        f"val_ratio={val_ratio:.2f}"
    )

    if num_samples < 2:
        raise ValueError("single-source 数据至少需要 2 个原始序列，才能进行 train/val 划分。")
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("data.val_ratio 建议设置在 (0, 0.5) 区间内。")

    generator = torch.Generator()
    generator.manual_seed(seed)
    sample_indices = torch.randperm(num_samples, generator=generator).tolist()
    val_sample_count = max(1, int(round(num_samples * val_ratio)))
    if val_sample_count >= num_samples:
        val_sample_count = num_samples - 1

    val_sample_ids = sorted(sample_indices[:val_sample_count])
    train_sample_ids = sorted(sample_indices[val_sample_count:])

    train_ds = EITSequenceDataset(
        train_path,
        n_frames=n_frames,
        frames_per_seq=frames_per_seq,
        target_size=(img_h, img_w),
        use_augmentation=use_augmentation,
        sample_ids=train_sample_ids,
        apply_target_blur=apply_target_blur,
        target_blur_kernel_size=target_blur_kernel_size,
        target_blur_sigma=target_blur_sigma,
        augment_hflip_prob=hflip_prob,
        augment_rotate_prob=rotate_prob,
        augment_rotate_deg=rotate_deg,
    )
    val_ds = EITSequenceDataset(
        train_path,
        n_frames=n_frames,
        frames_per_seq=frames_per_seq,
        target_size=(img_h, img_w),
        use_augmentation=False,
        sample_ids=val_sample_ids,
        apply_target_blur=False,
    )
    print(
        "📊 Single-source Split (by sequence): "
        f"TrainSeq={len(train_sample_ids)}, ValSeq={len(val_sample_ids)} | "
        f"TrainWindows={len(train_ds)}, ValWindows={len(val_ds)}"
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, None


def _resolve_lctsc_case_splits(data_cfg: dict, seed: int) -> tuple[str, dict[str, list[str]]]:
    manifest_name = data_cfg.get("manifest_name", "global_samples_manifest.csv")
    case_split_cfg = data_cfg.get("case_split", {})
    train_case_ids = case_split_cfg.get("train_case_ids")
    val_case_ids = case_split_cfg.get("val_case_ids")
    test_case_ids = case_split_cfg.get("test_case_ids")

    if train_case_ids is None or val_case_ids is None:
        splits = build_case_splits_from_manifest(
            dataset_root=data_cfg["dataset_root"],
            manifest_name=manifest_name,
            train_ratio=float(case_split_cfg.get("train_ratio", 0.7)),
            val_ratio=float(case_split_cfg.get("val_ratio", 0.15)),
            seed=seed,
        )
    else:
        splits = {
            "train": list(train_case_ids),
            "val": list(val_case_ids),
            "test": list(test_case_ids or []),
        }

    return manifest_name, splits


def _build_lctsc_sequence_dataloaders(cfg: dict, seed: int, dataset_cls, dataset_label: str):
    data_cfg = cfg["data"]
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))
    dataset_root = data_cfg["dataset_root"]
    manifest_name, splits = _resolve_lctsc_case_splits(data_cfg, seed)

    noise_cfg = data_cfg.get("noise", {})
    noise_mode = noise_cfg.get("mode", "fixed")
    fixed_noise_index = int(noise_cfg.get("fixed_index", 2))
    noise_indices = noise_cfg.get("indices")

    train_ds = dataset_cls(
        dataset_root=dataset_root,
        manifest_name=manifest_name,
        case_ids=splits["train"],
        noise_mode=noise_mode,
        fixed_noise_index=fixed_noise_index,
        noise_indices=noise_indices,
    )
    val_ds = dataset_cls(
        dataset_root=dataset_root,
        manifest_name=manifest_name,
        case_ids=splits["val"],
        noise_mode=noise_mode,
        fixed_noise_index=fixed_noise_index,
        noise_indices=noise_indices,
    )

    print(
        f"📊 {dataset_label} Split: "
        f"TrainCases={len(splits['train'])}, ValCases={len(splits['val'])}, TestCases={len(splits['test'])} | "
        f"TrainSamples={len(train_ds)}, ValSamples={len(val_ds)}"
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, splits


def build_dual_source_dataloaders(cfg: dict, seed: int):
    return _build_lctsc_sequence_dataloaders(
        cfg,
        seed,
        dataset_cls=LCTSCDualSourceDataset,
        dataset_label="Dual-source",
    )


def build_recon_sequence_dataloaders(cfg: dict, seed: int):
    return _build_lctsc_sequence_dataloaders(
        cfg,
        seed,
        dataset_cls=LCTSCReconSequenceDataset,
        dataset_label="Single-source LCTSC",
    )


def build_dataloaders(cfg: dict, seed: int):
    dataset_type = cfg["data"].get("dataset_type", "single_sequence")
    if dataset_type == "dual_source_lctsc":
        return build_dual_source_dataloaders(cfg, seed)
    if dataset_type == "single_source_lctsc_seq":
        return build_recon_sequence_dataloaders(cfg, seed)
    return build_single_source_dataloaders(cfg, seed)


def visualize_and_save(model, loader, device, save_dir: Path, epoch: int, num_samples: int = 3):
    model.eval()
    vis_dir = save_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    wandb_images = []

    try:
        batch = next(iter(loader))
        model_inputs, targets, vis_inputs, _ = prepare_batch(batch, device)
        with torch.no_grad():
            preds = forward_model(model, model_inputs)

        vis_input_frames = select_middle_frame(vis_inputs)
        vis_target_frames = select_middle_frame(targets)
        vis_pred_frames = select_middle_frame(preds)

        for index in range(min(num_samples, vis_input_frames.shape[0])):
            img_in = vis_input_frames[index]
            img_tar = vis_target_frames[index]
            img_pred = vis_pred_frames[index]

            target_range = (
                float(torch.min(torch.stack([img_tar.min(), img_pred.min()]))),
                float(torch.max(torch.stack([img_tar.max(), img_pred.max()]))),
            )

            input_uint8 = to_uint8_image(img_in)
            target_uint8 = to_uint8_image(img_tar, value_range=target_range)
            pred_uint8 = to_uint8_image(img_pred, value_range=target_range)

            target_height, target_width = target_uint8.shape
            if input_uint8.shape != (target_height, target_width):
                input_uint8 = cv2.resize(input_uint8, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            vis_in = cv2.applyColorMap(input_uint8, cv2.COLORMAP_JET)
            vis_tar = cv2.applyColorMap(target_uint8, cv2.COLORMAP_JET)
            vis_pred = cv2.applyColorMap(pred_uint8, cv2.COLORMAP_JET)

            combined = np.hstack((vis_in, vis_tar, vis_pred))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined, "Input", (10, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(combined, "Target", (10 + combined.shape[1] // 3, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(combined, "Pred", (10 + 2 * combined.shape[1] // 3, 20), font, 0.5, (255, 255, 255), 1)

            save_path = vis_dir / f"epoch_{epoch}_sample_{index}.jpg"
            cv2.imwrite(str(save_path), combined)

            if wandb is not None:
                combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
                wandb_images.append(wandb.Image(combined_rgb, caption=f"Ep{epoch}-S{index}"))

    except Exception as exc:
        print(f"⚠️ 可视化失败: {exc}")

    return wandb_images


def init_wandb(cfg: dict) -> bool:
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enable", False):
        return False

    if wandb is None:
        raise ImportError("❌ 当前环境未安装 wandb，请先安装或在配置中关闭 wandb。")

    init_kwargs = {
        "project": wandb_cfg["project"],
        "name": wandb_cfg["name"],
        "config": cfg,
        "tags": wandb_cfg.get("tags", []),
    }
    if wandb_cfg.get("entity"):
        init_kwargs["entity"] = wandb_cfg["entity"]
    if wandb_cfg.get("mode"):
        init_kwargs["mode"] = wandb_cfg["mode"]

    try:
        wandb.init(**init_kwargs)
        return True
    except Exception as exc:
        print(f"⚠️ WandB 初始化失败，已自动关闭本次 WandB 同步: {exc}")
        print("   如果要在线同步，请将 wandb.entity 改为你的 team entity。")
        print("   如果只想先本地跑通，也可以在配置里设置 wandb.enable: False 或 wandb.mode: offline。")
        wandb_cfg["enable"] = False
        return False


def save_case_splits(save_dir: Path, splits: dict[str, list[str]] | None) -> None:
    if splits is None:
        return

    split_path = save_dir / "case_splits.yaml"
    with open(split_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(splits, file, allow_unicode=True, sort_keys=False)


def train_pipeline(cfg: dict) -> None:
    save_dir = Path(cfg["save_dir"])
    if not save_dir.is_absolute():
        save_dir = PROJECT_ROOT / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    wandb_enabled = init_wandb(cfg)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = get_best_device()
    print(f"🚀 Device: {device} | Output: {save_dir}")

    train_loader, val_loader, case_splits = build_dataloaders(cfg, seed)
    save_case_splits(save_dir, case_splits)

    model = get_model(cfg["model"]).to(device)

    train_cfg = cfg["train"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    scheduler_name = str(train_cfg.get("scheduler", "CosineAnnealingLR"))
    if scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg["epochs"]),
            eta_min=1e-6,
        )
    else:
        scheduler = None

    l1_crit = nn.L1Loss()
    edge_crit = EdgeLoss(device)
    temporal_crit = TemporalDifferenceLoss()

    loss_weights = train_cfg.get("loss_weights", {})
    w_l1 = float(loss_weights.get("l1", 1.0))
    w_edge = float(loss_weights.get("edge", 0.0))
    w_temporal = float(loss_weights.get("temporal", 0.0))

    tracker = MetricTracker()
    best_ssim = -1.0

    csv_file = save_dir / "results.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as file:
        file.write("epoch,train_loss,val_loss,psnr,ssim,cc,rie\n")

    logging_cfg = cfg.get("logging", {})
    save_freq = int(logging_cfg.get("save_freq", 10))
    epochs = int(train_cfg["epochs"])

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{epochs}", unit="bt")

        for batch in pbar:
            model_inputs, targets, _, _ = prepare_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)
            preds = forward_model(model, model_inputs)

            loss = w_l1 * l1_crit(preds, targets)
            if w_edge > 0:
                loss = loss + w_edge * edge_crit(preds, targets)
            if w_temporal > 0:
                loss = loss + w_temporal * temporal_crit(preds, targets)

            loss.backward()
            optimizer.step()

            train_loss_acc += loss.item()
            pbar.set_postfix({"L": f"{loss.item():.4f}"})

        train_loss_avg = train_loss_acc / max(len(train_loader), 1)
        if scheduler is not None:
            scheduler.step()

        model.eval()
        tracker.reset()
        with torch.no_grad():
            for batch in val_loader:
                model_inputs, targets, _, batch_size = prepare_batch(batch, device)
                preds = forward_model(model, model_inputs)

                v_loss = w_l1 * l1_crit(preds, targets)
                if w_edge > 0:
                    v_loss = v_loss + w_edge * edge_crit(preds, targets)
                if w_temporal > 0:
                    v_loss = v_loss + w_temporal * temporal_crit(preds, targets)

                tracker.update(
                    v_loss.item(),
                    EITMetrics.psnr(preds, targets),
                    EITMetrics.ssim(preds, targets),
                    EITMetrics.cc(preds, targets),
                    EITMetrics.rie(preds, targets),
                    n=batch_size,
                )

        metrics = tracker.avg()
        print(
            f"  Valid -> "
            f"L:{metrics['val/loss']:.4f} | "
            f"PSNR:{metrics['val/psnr']:.2f} | "
            f"SSIM:{metrics['val/ssim']:.4f} | "
            f"CC:{metrics['val/cc']:.4f}"
        )

        with open(csv_file, "a", encoding="utf-8", newline="") as file:
            file.write(
                f"{epoch},{train_loss_avg},{metrics['val/loss']},"
                f"{metrics['val/psnr']},{metrics['val/ssim']},"
                f"{metrics['val/cc']},{metrics['val/rie']}\n"
            )

        wandb_images = None
        if epoch % save_freq == 0:
            wandb_images = visualize_and_save(model, val_loader, device, save_dir, epoch)

        if wandb_enabled:
            log_data = {
                "train/loss": train_loss_avg,
                "train/lr": optimizer.param_groups[0]["lr"],
                **metrics,
                "epoch": epoch,
            }
            if wandb_images:
                log_data["examples"] = wandb_images
            wandb.log(log_data)

        if metrics["val/ssim"] > best_ssim:
            best_ssim = metrics["val/ssim"]
            torch.save(model.state_dict(), save_dir / "best_ssim.pth")
            print("  🔥 Best SSIM Updated!")

        torch.save(model.state_dict(), save_dir / "last.pth")

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/exp003_early_cbam_unet.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    train_pipeline(cfg)
