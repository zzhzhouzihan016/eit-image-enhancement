from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from eit.mesh import compute_triangle_centroids


ArrayPath = Union[str, Path]


@dataclass
class CaseFemInputs:
    nodes_img: np.ndarray
    elements: np.ndarray
    torso_mask: np.ndarray
    lung_element_mask: np.ndarray
    outer_contour_img: Optional[np.ndarray] = None


@dataclass
class PyEITMeshBundle:
    mesh_obj: Any
    protocol_obj: Any
    nodes_cart: np.ndarray
    electrode_node_indices: np.ndarray
    boundary_cycle: np.ndarray
    reference_node: int


@dataclass
class RespiratoryCycleSimulationResult:
    mesh_bundle: PyEITMeshBundle
    lung_conductivity_waveform: np.ndarray
    perm_sequence: np.ndarray
    voltage_clean: np.ndarray
    delta_voltage_clean: np.ndarray
    delta_voltage_noisy: np.ndarray
    reconstructed_ds: np.ndarray
    exact_ds: np.ndarray
    low_res_inputs: np.ndarray
    high_res_targets: np.ndarray


def _import_pyeit() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from pyeit.eit import JAC
        from pyeit.eit.fem import EITForward
        from pyeit.eit.protocol import create as create_protocol
        from pyeit.mesh.wrapper import PyEITMesh, check_order
    except ImportError as exc:
        raise ImportError(
            "未检测到 `pyeit`。请在 `deepl_cv` 环境中安装：`pip install pyeit`。"
        ) from exc

    return PyEITMesh, check_order, create_protocol, EITForward, JAC


def _load_array(path: ArrayPath) -> np.ndarray:
    array_path = Path(path).expanduser().resolve()
    if not array_path.exists():
        raise FileNotFoundError(f"找不到文件：{array_path}")
    return np.load(array_path)


def load_case_fem_inputs(fem_dir: ArrayPath) -> CaseFemInputs:
    fem_root = Path(fem_dir).expanduser().resolve()
    if not fem_root.exists():
        raise FileNotFoundError(f"找不到 FEM 目录：{fem_root}")

    nodes_img = _load_array(fem_root / "nodes.npy").astype(np.float64)
    elements = _load_array(fem_root / "elements.npy").astype(np.int32)
    torso_mask = _load_array(fem_root / "cleaned_torso_mask.npy").astype(np.uint8)

    region_labels_path = fem_root / "region_labels.npy"
    if region_labels_path.exists():
        lung_element_mask = _load_array(region_labels_path).astype(np.int32) > 0
    else:
        conductivity = _load_array(fem_root / "conductivity.npy").astype(np.float32)
        lung_element_mask = conductivity < 0.31

    if lung_element_mask.shape[0] != elements.shape[0]:
        raise ValueError(
            f"lung_element_mask 长度 ({lung_element_mask.shape[0]}) 与单元数 ({elements.shape[0]}) 不一致。"
        )

    outer_contour_path = fem_root / "outer_contour.npy"
    outer_contour_img = _load_array(outer_contour_path).astype(np.float64) if outer_contour_path.exists() else None

    return CaseFemInputs(
        nodes_img=nodes_img,
        elements=elements,
        torso_mask=torso_mask,
        lung_element_mask=lung_element_mask.astype(bool),
        outer_contour_img=outer_contour_img,
    )


def _image_to_cartesian(points_img: np.ndarray, image_height: int) -> np.ndarray:
    points_cart = points_img.astype(np.float64).copy()
    points_cart[:, 1] = (image_height - 1) - points_cart[:, 1]
    return points_cart


def _polygon_signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _extract_outer_boundary_cycle(elements: np.ndarray) -> np.ndarray:
    edge_counter: Counter[tuple[int, int]] = Counter()
    for tri in elements:
        a, b, c = map(int, tri)
        for edge in ((a, b), (b, c), (c, a)):
            edge_counter[tuple(sorted(edge))] += 1

    boundary_edges = [edge for edge, count in edge_counter.items() if count == 1]
    if not boundary_edges:
        raise RuntimeError("未能从三角网格中提取外边界。")

    adjacency: dict[int, list[int]] = defaultdict(list)
    for node_a, node_b in boundary_edges:
        adjacency[node_a].append(node_b)
        adjacency[node_b].append(node_a)

    degrees = {node_idx: len(neighbors) for node_idx, neighbors in adjacency.items()}
    if any(degree != 2 for degree in degrees.values()):
        raise RuntimeError("外边界节点度数异常，无法构造闭合边界。")

    start_node = min(adjacency.keys())
    ordered_nodes = [start_node]
    previous_node = -1
    current_node = start_node

    while True:
        neighbors = adjacency[current_node]
        next_node = neighbors[0] if neighbors[0] != previous_node else neighbors[1]
        if next_node == start_node:
            break
        ordered_nodes.append(next_node)
        previous_node, current_node = current_node, next_node
        if len(ordered_nodes) > len(adjacency) + 2:
            raise RuntimeError("外边界遍历失败：边界链长度异常。")

    return np.asarray(ordered_nodes, dtype=np.int32)


def _orient_boundary_cycle(nodes_cart: np.ndarray, boundary_cycle: np.ndarray) -> np.ndarray:
    boundary_points = nodes_cart[boundary_cycle]
    if _polygon_signed_area(boundary_points) < 0:
        boundary_cycle = boundary_cycle[::-1].copy()
        boundary_points = nodes_cart[boundary_cycle]

    top_node_idx = int(np.argmax(boundary_points[:, 1]))
    return np.roll(boundary_cycle, -top_node_idx)


def _sample_electrode_nodes(
    nodes_cart: np.ndarray,
    boundary_cycle: np.ndarray,
    n_electrodes: int,
) -> np.ndarray:
    boundary_points = nodes_cart[boundary_cycle]
    closed_points = np.vstack([boundary_points, boundary_points[0]])
    segment_lengths = np.linalg.norm(np.diff(closed_points, axis=0), axis=1)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    perimeter = float(cumulative_lengths[-1])

    if perimeter <= 0.0:
        raise RuntimeError("边界周长异常，无法放置电极。")

    boundary_node_arclength = cumulative_lengths[:-1]
    target_arclengths = np.linspace(0.0, perimeter, n_electrodes, endpoint=False)

    selected_nodes: list[int] = []
    for target in target_arclengths:
        nearest_idx = int(np.argmin(np.abs(boundary_node_arclength - target)))
        if int(boundary_cycle[nearest_idx]) in selected_nodes:
            found = False
            for offset in range(1, len(boundary_cycle)):
                candidate_indices = [
                    (nearest_idx - offset) % len(boundary_cycle),
                    (nearest_idx + offset) % len(boundary_cycle),
                ]
                for candidate_idx in candidate_indices:
                    candidate_node = int(boundary_cycle[candidate_idx])
                    if candidate_node not in selected_nodes:
                        nearest_idx = candidate_idx
                        found = True
                        break
                if found:
                    break

        selected_nodes.append(int(boundary_cycle[nearest_idx]))

    if len(set(selected_nodes)) != n_electrodes:
        raise RuntimeError("电极采样失败：边界节点不足以生成唯一电极位置。")

    return np.asarray(selected_nodes, dtype=np.int32)


def _choose_reference_node(nodes_cart: np.ndarray, electrode_nodes: np.ndarray) -> int:
    electrode_set = set(int(node_idx) for node_idx in electrode_nodes.tolist())
    centroid = nodes_cart.mean(axis=0)
    distances = np.linalg.norm(nodes_cart - centroid[None, :], axis=1)
    for node_idx in np.argsort(distances):
        if int(node_idx) not in electrode_set:
            return int(node_idx)
    raise RuntimeError("未能找到可用的参考节点。")


def build_pyeit_mesh_from_fem(
    fem_inputs: CaseFemInputs,
    n_electrodes: int = 16,
    dist_exc: int = 1,
    step_meas: int = 1,
    background_conductivity: float = 0.38,
    lung_exp_conductivity: float = 0.24,
) -> PyEITMeshBundle:
    PyEITMesh, check_order, create_protocol, _, _ = _import_pyeit()

    image_height = int(fem_inputs.torso_mask.shape[0])
    nodes_cart = _image_to_cartesian(fem_inputs.nodes_img, image_height=image_height)
    elements_ccw = check_order(nodes_cart.copy(), fem_inputs.elements.copy()).astype(np.int32)

    boundary_cycle = _extract_outer_boundary_cycle(elements_ccw)
    boundary_cycle = _orient_boundary_cycle(nodes_cart, boundary_cycle)
    electrode_nodes = _sample_electrode_nodes(
        nodes_cart=nodes_cart,
        boundary_cycle=boundary_cycle,
        n_electrodes=n_electrodes,
    )
    reference_node = _choose_reference_node(nodes_cart, electrode_nodes)

    perm0 = np.full(elements_ccw.shape[0], background_conductivity, dtype=np.float64)
    perm0[fem_inputs.lung_element_mask] = float(lung_exp_conductivity)

    mesh_obj = PyEITMesh(
        node=nodes_cart.astype(np.float64),
        element=elements_ccw,
        perm=perm0,
        el_pos=electrode_nodes,
        ref_node=reference_node,
    )
    protocol_obj = create_protocol(
        n_el=n_electrodes,
        dist_exc=dist_exc,
        step_meas=step_meas,
        parser_meas="std",
    )

    return PyEITMeshBundle(
        mesh_obj=mesh_obj,
        protocol_obj=protocol_obj,
        nodes_cart=nodes_cart,
        electrode_node_indices=electrode_nodes,
        boundary_cycle=boundary_cycle,
        reference_node=reference_node,
    )


def generate_sinusoidal_permittivity_sequence(
    lung_element_mask: np.ndarray,
    n_elements: int,
    n_frames: int = 20,
    c_exp: float = 0.24,
    c_insp: float = 0.08,
    background_conductivity: float = 0.38,
) -> tuple[np.ndarray, np.ndarray]:
    frame_indices = np.arange(n_frames, dtype=np.float64)
    lung_waveform = ((c_exp + c_insp) / 2.0) + ((c_exp - c_insp) / 2.0) * np.cos(
        2.0 * np.pi * frame_indices / float(n_frames)
    )

    perm_sequence = np.full((n_frames, n_elements), background_conductivity, dtype=np.float64)
    perm_sequence[:, lung_element_mask.astype(bool)] = lung_waveform[:, None]
    return lung_waveform.astype(np.float32), perm_sequence.astype(np.float32)


def simulate_voltage_sequence(
    mesh_obj: Any,
    protocol_obj: Any,
    perm_sequence: np.ndarray,
) -> np.ndarray:
    _, _, _, EITForward, _ = _import_pyeit()
    forward_solver = EITForward(mesh_obj, protocol_obj)

    voltage_frames = []
    for perm in perm_sequence:
        voltage = forward_solver.solve_eit(perm=perm.astype(np.float64))
        voltage_frames.append(voltage.reshape(protocol_obj.n_exc, protocol_obj.n_meas))

    return np.stack(voltage_frames, axis=0).astype(np.float32)


def add_awgn(
    signal: np.ndarray,
    snr_db: float,
    rng: Optional[np.random.Generator] = None,
    zero_power_eps: float = 1e-12,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    signal = np.asarray(signal, dtype=np.float64)
    signal_power = float(np.mean(np.square(np.abs(signal))))
    if signal_power <= zero_power_eps:
        return signal.astype(np.float32)

    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=signal.shape)
    return (signal + noise).astype(np.float32)


def add_awgn_to_voltage_differences(
    delta_voltage_sequence: np.ndarray,
    snr_db: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noisy_frames = [
        add_awgn(frame, snr_db=snr_db, rng=rng) for frame in np.asarray(delta_voltage_sequence, dtype=np.float32)
    ]
    return np.stack(noisy_frames, axis=0).astype(np.float32)


def reconstruct_sequence_with_jac(
    mesh_obj: Any,
    protocol_obj: Any,
    reference_voltage: np.ndarray,
    delta_voltage_noisy: np.ndarray,
    perm0: np.ndarray,
    jac_p: float = 0.5,
    jac_lambda: float = 0.001,
    jac_method: str = "kotre",
) -> np.ndarray:
    _, _, _, _, JAC = _import_pyeit()

    solver = JAC(mesh_obj, protocol_obj)
    solver.setup(
        p=jac_p,
        lamb=jac_lambda,
        method=jac_method,
        perm=perm0.astype(np.float64),
        jac_normalized=False,
    )

    reference_voltage_flat = reference_voltage.reshape(-1).astype(np.float64)
    reconstructed_frames = []
    for delta_voltage in delta_voltage_noisy:
        current_voltage = reference_voltage_flat + delta_voltage.reshape(-1).astype(np.float64)
        ds = solver.solve(current_voltage, reference_voltage_flat, normalize=False)
        reconstructed_frames.append(np.real(ds).astype(np.float32))

    return np.stack(reconstructed_frames, axis=0)


def _resize_mask(mask: np.ndarray, grid_shape: Tuple[int, int]) -> np.ndarray:
    grid_height, grid_width = grid_shape
    resized_mask = cv2.resize(
        mask.astype(np.uint8),
        (grid_width, grid_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized_mask > 0).astype(np.uint8)


def _precompute_triangle_pixel_indices(
    nodes_img: np.ndarray,
    elements: np.ndarray,
    src_shape: Tuple[int, int],
    dst_shape: Tuple[int, int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    src_height, src_width = src_shape
    dst_height, dst_width = dst_shape

    scaled_nodes = nodes_img.astype(np.float32).copy()
    scaled_nodes[:, 0] *= float(dst_width - 1) / float(max(src_width - 1, 1))
    scaled_nodes[:, 1] *= float(dst_height - 1) / float(max(src_height - 1, 1))

    triangle_pixel_indices: list[tuple[np.ndarray, np.ndarray]] = []
    for tri in elements:
        tri_points = scaled_nodes[tri]
        tri_points_int = np.rint(tri_points).astype(np.int32)

        x_min = int(np.clip(np.min(tri_points_int[:, 0]), 0, dst_width - 1))
        x_max = int(np.clip(np.max(tri_points_int[:, 0]), 0, dst_width - 1))
        y_min = int(np.clip(np.min(tri_points_int[:, 1]), 0, dst_height - 1))
        y_max = int(np.clip(np.max(tri_points_int[:, 1]), 0, dst_height - 1))

        if x_max < x_min or y_max < y_min:
            triangle_pixel_indices.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)))
            continue

        local_mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        local_points = tri_points_int.copy()
        local_points[:, 0] -= x_min
        local_points[:, 1] -= y_min
        cv2.fillConvexPoly(local_mask, local_points, color=1)

        ys_local, xs_local = np.where(local_mask > 0)
        triangle_pixel_indices.append((ys_local.astype(np.int32) + y_min, xs_local.astype(np.int32) + x_min))

    return triangle_pixel_indices


def rasterize_element_sequence_to_grid(
    element_value_sequence: np.ndarray,
    nodes_img: np.ndarray,
    elements: np.ndarray,
    src_shape: Tuple[int, int],
    dst_shape: Tuple[int, int],
    torso_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    value_sequence = np.asarray(element_value_sequence, dtype=np.float32)
    if value_sequence.ndim == 1:
        value_sequence = value_sequence[None, :]

    triangle_pixel_indices = _precompute_triangle_pixel_indices(
        nodes_img=nodes_img,
        elements=elements,
        src_shape=src_shape,
        dst_shape=dst_shape,
    )

    dst_height, dst_width = dst_shape
    output_sequence = np.zeros((value_sequence.shape[0], dst_height, dst_width), dtype=np.float32)
    torso_mask_resized = _resize_mask(torso_mask, dst_shape) if torso_mask is not None else None

    for frame_idx, frame_values in enumerate(value_sequence):
        value_sum = np.zeros((dst_height, dst_width), dtype=np.float32)
        value_count = np.zeros((dst_height, dst_width), dtype=np.float32)

        for element_idx, (ys, xs) in enumerate(triangle_pixel_indices):
            if ys.size == 0:
                continue
            value = float(frame_values[element_idx])
            value_sum[ys, xs] += value
            value_count[ys, xs] += 1.0

        frame_output = np.zeros((dst_height, dst_width), dtype=np.float32)
        valid_pixels = value_count > 0.0
        frame_output[valid_pixels] = value_sum[valid_pixels] / value_count[valid_pixels]

        if torso_mask_resized is not None:
            frame_output[torso_mask_resized == 0] = 0.0

        output_sequence[frame_idx] = frame_output

    return output_sequence.astype(np.float32)


def simulate_respiratory_cycle_dataset(
    mesh_obj: Any,
    lung_element_mask: np.ndarray,
    nodes_img: np.ndarray,
    elements: np.ndarray,
    torso_mask: np.ndarray,
    protocol_obj: Optional[Any] = None,
    n_frames: int = 20,
    c_exp: float = 0.24,
    c_insp: float = 0.08,
    background_conductivity: float = 0.38,
    snr_db: float = 30.0,
    low_res_shape: Tuple[int, int] = (64, 64),
    high_res_shape: Tuple[int, int] = (256, 256),
    jac_p: float = 0.5,
    jac_lambda: float = 0.001,
    jac_method: str = "kotre",
    seed: int = 0,
    mesh_bundle: Optional[PyEITMeshBundle] = None,
) -> RespiratoryCycleSimulationResult:
    _, _, create_protocol, _, _ = _import_pyeit()

    lung_element_mask = np.asarray(lung_element_mask, dtype=bool)
    if lung_element_mask.shape[0] != elements.shape[0]:
        raise ValueError("lung_element_mask 与 elements 数量不匹配。")

    if protocol_obj is None:
        protocol_obj = create_protocol(n_el=16, dist_exc=1, step_meas=1, parser_meas="std")

    if mesh_bundle is None:
        mesh_bundle = PyEITMeshBundle(
            mesh_obj=mesh_obj,
            protocol_obj=protocol_obj,
            nodes_cart=np.asarray(mesh_obj.node, dtype=np.float64),
            electrode_node_indices=np.asarray(mesh_obj.el_pos, dtype=np.int32),
            boundary_cycle=np.empty(0, dtype=np.int32),
            reference_node=int(mesh_obj.ref_node),
        )

    lung_waveform, perm_sequence = generate_sinusoidal_permittivity_sequence(
        lung_element_mask=lung_element_mask,
        n_elements=elements.shape[0],
        n_frames=n_frames,
        c_exp=c_exp,
        c_insp=c_insp,
        background_conductivity=background_conductivity,
    )

    voltage_clean = simulate_voltage_sequence(
        mesh_obj=mesh_obj,
        protocol_obj=protocol_obj,
        perm_sequence=perm_sequence,
    )
    reference_voltage = voltage_clean[0]
    delta_voltage_clean = voltage_clean - reference_voltage[None, :, :]

    rng = np.random.default_rng(seed)
    delta_voltage_noisy = add_awgn_to_voltage_differences(
        delta_voltage_sequence=delta_voltage_clean,
        snr_db=snr_db,
        rng=rng,
    )

    reconstructed_ds = reconstruct_sequence_with_jac(
        mesh_obj=mesh_obj,
        protocol_obj=protocol_obj,
        reference_voltage=reference_voltage,
        delta_voltage_noisy=delta_voltage_noisy,
        perm0=perm_sequence[0],
        jac_p=jac_p,
        jac_lambda=jac_lambda,
        jac_method=jac_method,
    )

    exact_ds = (perm_sequence - perm_sequence[0][None, :]).astype(np.float32)
    src_shape = tuple(int(v) for v in torso_mask.shape[:2])
    low_res_inputs = rasterize_element_sequence_to_grid(
        element_value_sequence=reconstructed_ds,
        nodes_img=nodes_img,
        elements=elements,
        src_shape=src_shape,
        dst_shape=low_res_shape,
        torso_mask=torso_mask,
    )
    high_res_targets = rasterize_element_sequence_to_grid(
        element_value_sequence=exact_ds,
        nodes_img=nodes_img,
        elements=elements,
        src_shape=src_shape,
        dst_shape=high_res_shape,
        torso_mask=torso_mask,
    )

    return RespiratoryCycleSimulationResult(
        mesh_bundle=mesh_bundle,
        lung_conductivity_waveform=lung_waveform.astype(np.float32),
        perm_sequence=perm_sequence.astype(np.float32),
        voltage_clean=voltage_clean.astype(np.float32),
        delta_voltage_clean=delta_voltage_clean.astype(np.float32),
        delta_voltage_noisy=delta_voltage_noisy.astype(np.float32),
        reconstructed_ds=reconstructed_ds.astype(np.float32),
        exact_ds=exact_ds.astype(np.float32),
        low_res_inputs=low_res_inputs.astype(np.float32),
        high_res_targets=high_res_targets.astype(np.float32),
    )


def simulate_respiratory_cycle_from_fem(
    fem_inputs: CaseFemInputs,
    n_electrodes: int = 16,
    dist_exc: int = 1,
    step_meas: int = 1,
    n_frames: int = 20,
    c_exp: float = 0.24,
    c_insp: float = 0.08,
    background_conductivity: float = 0.38,
    snr_db: float = 30.0,
    low_res_shape: Tuple[int, int] = (64, 64),
    high_res_shape: Tuple[int, int] = (256, 256),
    jac_p: float = 0.5,
    jac_lambda: float = 0.001,
    jac_method: str = "kotre",
    seed: int = 0,
) -> RespiratoryCycleSimulationResult:
    mesh_bundle = build_pyeit_mesh_from_fem(
        fem_inputs=fem_inputs,
        n_electrodes=n_electrodes,
        dist_exc=dist_exc,
        step_meas=step_meas,
        background_conductivity=background_conductivity,
        lung_exp_conductivity=c_exp,
    )

    return simulate_respiratory_cycle_dataset(
        mesh_obj=mesh_bundle.mesh_obj,
        lung_element_mask=fem_inputs.lung_element_mask,
        nodes_img=fem_inputs.nodes_img,
        elements=fem_inputs.elements,
        torso_mask=fem_inputs.torso_mask,
        protocol_obj=mesh_bundle.protocol_obj,
        n_frames=n_frames,
        c_exp=c_exp,
        c_insp=c_insp,
        background_conductivity=background_conductivity,
        snr_db=snr_db,
        low_res_shape=low_res_shape,
        high_res_shape=high_res_shape,
        jac_p=jac_p,
        jac_lambda=jac_lambda,
        jac_method=jac_method,
        seed=seed,
        mesh_bundle=mesh_bundle,
    )


def derive_lung_element_mask_from_pixel_mask(
    nodes_img: np.ndarray,
    elements: np.ndarray,
    lung_mask: np.ndarray,
) -> np.ndarray:
    centroids = compute_triangle_centroids(nodes_img, elements)
    lung_binary = (lung_mask > 0).astype(np.uint8)
    height, width = lung_binary.shape

    pixel_x = np.clip(np.rint(centroids[:, 0]).astype(np.int32), 0, width - 1)
    pixel_y = np.clip(np.rint(centroids[:, 1]).astype(np.int32), 0, height - 1)
    return (lung_binary[pixel_y, pixel_x] > 0).astype(bool)
