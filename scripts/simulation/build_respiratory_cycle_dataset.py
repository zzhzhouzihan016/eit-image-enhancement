import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eit.simulation import load_case_fem_inputs, simulate_respiratory_cycle_from_fem


def resolve_default_output_dir(fem_dir: Path) -> Path:
    fem_dir = fem_dir.expanduser().resolve()
    if fem_dir.name == "fem" and fem_dir.parent.name == "mesh":
        case_dir = fem_dir.parent.parent
        return case_dir / "simulation" / "respiratory_cycle"
    return fem_dir / "respiratory_cycle"


def save_waveform_plot(waveform: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(7, 3))
    plt.plot(np.arange(len(waveform)), waveform, marker="o", linewidth=1.6)
    plt.title("Sinusoidal lung conductivity waveform")
    plt.xlabel("Frame")
    plt.ylabel("Conductivity")
    plt.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_electrode_layout(
    nodes_img: np.ndarray,
    boundary_cycle: np.ndarray,
    electrode_nodes: np.ndarray,
    torso_mask: np.ndarray,
    path: Path,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(torso_mask, cmap="gray")
    if boundary_cycle.size > 0:
        boundary_points = nodes_img[boundary_cycle]
        plt.plot(
            np.r_[boundary_points[:, 0], boundary_points[0, 0]],
            np.r_[boundary_points[:, 1], boundary_points[0, 1]],
            color="yellow",
            linewidth=1.5,
        )
    electrode_points = nodes_img[electrode_nodes]
    plt.scatter(electrode_points[:, 0], electrode_points[:, 1], c="red", s=22)
    for electrode_idx, point in enumerate(electrode_points):
        plt.text(point[0] + 2, point[1] + 2, str(electrode_idx), color="cyan", fontsize=7)
    plt.title("16-electrode layout on thorax boundary")
    plt.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_sequence_preview(low_res_inputs: np.ndarray, high_res_targets: np.ndarray, path: Path) -> None:
    preview_frames = [0, 5, 10, 15]
    plt.figure(figsize=(10, 5))
    for plot_idx, frame_idx in enumerate(preview_frames):
        ax_input = plt.subplot(2, 4, plot_idx + 1)
        ax_input.imshow(low_res_inputs[frame_idx], cmap="coolwarm")
        ax_input.set_title(f"Input f={frame_idx}")
        ax_input.axis("off")

        ax_target = plt.subplot(2, 4, plot_idx + 5)
        ax_target.imshow(high_res_targets[frame_idx], cmap="coolwarm")
        ax_target.set_title(f"Target f={frame_idx}")
        ax_target.axis("off")

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于单病例 FEM 网格生成 20 帧呼吸周期 EIT 仿真数据。")
    parser.add_argument(
        "--fem-dir",
        type=str,
        required=True,
        help="病例 FEM 目录，例如 data/interim/ct_cases/demo_case/mesh/fem",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录；默认保存到病例目录下的 simulation/respiratory_cycle/")
    parser.add_argument("--frames", type=int, default=20, help="呼吸周期帧数，默认 20")
    parser.add_argument("--snr-db", type=float, default=30.0, help="加性高斯白噪声 SNR，默认 30 dB")
    parser.add_argument("--c-exp", type=float, default=0.24, help="呼气末肺区电导率，默认 0.24")
    parser.add_argument("--c-insp", type=float, default=0.08, help="吸气末肺区电导率，默认 0.08")
    parser.add_argument("--background", type=float, default=0.38, help="背景/肌肉电导率，默认 0.38")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，默认 0")
    parser.add_argument("--jac-p", type=float, default=0.5, help="JAC 求解器 p 参数，默认 0.5")
    parser.add_argument("--jac-lambda", type=float, default=0.001, help="JAC 正则化系数，默认 0.001")
    parser.add_argument("--low-res", type=int, default=64, help="低分辨率输入边长，默认 64")
    parser.add_argument("--high-res", type=int, default=256, help="高分辨率标签边长，默认 256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fem_dir = Path(args.fem_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else resolve_default_output_dir(fem_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fem_inputs = load_case_fem_inputs(fem_dir)
    result = simulate_respiratory_cycle_from_fem(
        fem_inputs=fem_inputs,
        n_frames=args.frames,
        c_exp=args.c_exp,
        c_insp=args.c_insp,
        background_conductivity=args.background,
        snr_db=args.snr_db,
        low_res_shape=(args.low_res, args.low_res),
        high_res_shape=(args.high_res, args.high_res),
        jac_p=args.jac_p,
        jac_lambda=args.jac_lambda,
        seed=args.seed,
    )

    np.save(output_dir / "voltage_clean.npy", result.voltage_clean.astype(np.float32))
    np.save(output_dir / "voltage_noisy.npy", result.delta_voltage_noisy.astype(np.float32))
    np.save(output_dir / "delta_voltage_clean.npy", result.delta_voltage_clean.astype(np.float32))
    np.save(output_dir / "lung_waveform.npy", result.lung_conductivity_waveform.astype(np.float32))
    np.save(output_dir / "perm_sequence.npy", result.perm_sequence.astype(np.float32))
    np.save(output_dir / "reconstructed_ds.npy", result.reconstructed_ds.astype(np.float32))
    np.save(output_dir / "exact_ds.npy", result.exact_ds.astype(np.float32))
    np.save(output_dir / "input_low_res.npy", result.low_res_inputs.astype(np.float32))
    np.save(output_dir / "target_high_res.npy", result.high_res_targets.astype(np.float32))
    np.save(output_dir / "electrode_nodes.npy", result.mesh_bundle.electrode_node_indices.astype(np.int32))

    np.savez_compressed(
        output_dir / "respiratory_cycle_dataset.npz",
        voltage_noisy=result.delta_voltage_noisy.astype(np.float32),
        input_low_res=result.low_res_inputs.astype(np.float32),
        target_high_res=result.high_res_targets.astype(np.float32),
        lung_waveform=result.lung_conductivity_waveform.astype(np.float32),
        electrode_nodes=result.mesh_bundle.electrode_node_indices.astype(np.int32),
    )

    save_waveform_plot(result.lung_conductivity_waveform, output_dir / "waveform.png")
    save_electrode_layout(
        nodes_img=fem_inputs.nodes_img,
        boundary_cycle=result.mesh_bundle.boundary_cycle,
        electrode_nodes=result.mesh_bundle.electrode_node_indices,
        torso_mask=fem_inputs.torso_mask,
        path=output_dir / "electrode_layout.png",
    )
    save_sequence_preview(
        low_res_inputs=result.low_res_inputs,
        high_res_targets=result.high_res_targets,
        path=output_dir / "sequence_preview.png",
    )

    metadata: dict[str, Any] = {
        "fem_dir": str(fem_dir),
        "frames": int(args.frames),
        "snr_db": float(args.snr_db),
        "c_exp": float(args.c_exp),
        "c_insp": float(args.c_insp),
        "background_conductivity": float(args.background),
        "jac_p": float(args.jac_p),
        "jac_lambda": float(args.jac_lambda),
        "low_res_shape": [int(args.low_res), int(args.low_res)],
        "high_res_shape": [int(args.high_res), int(args.high_res)],
        "voltage_noisy_shape": list(result.delta_voltage_noisy.shape),
        "input_low_res_shape": list(result.low_res_inputs.shape),
        "target_high_res_shape": list(result.high_res_targets.shape),
        "electrode_count": int(result.mesh_bundle.electrode_node_indices.shape[0]),
        "reference_node": int(result.mesh_bundle.reference_node),
        "frame0_note": "frame 0 is the reference frame; its delta voltage is zero, so no AWGN is injected.",
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("✅ 呼吸周期 EIT 数据生成完成")
    print(f"   FEM 目录: {fem_dir}")
    print(f"   输出目录: {output_dir}")
    print(f"   noisy voltage: {tuple(result.delta_voltage_noisy.shape)}")
    print(f"   low-res input: {tuple(result.low_res_inputs.shape)}")
    print(f"   high-res target: {tuple(result.high_res_targets.shape)}")


if __name__ == "__main__":
    main()
