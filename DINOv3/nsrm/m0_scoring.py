from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
DINOV2_ROOT = ROOT if (ROOT / "nvs" / "conditional_nvs").is_dir() else ROOT / "DINOv2"
if str(DINOV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DINOV2_ROOT))

from nvs.conditional_nvs.metrics import pixel_aupr, pixel_aupro, safe_auroc  # noqa: E402


_CUDA_RANKING_LOCK = threading.Lock()
_GIB = 1024**3
_CUDA_NATIVE_MAP_HEADROOM_BYTES = 4 * _GIB
_HOST_HISTOGRAM_MIN_HEADROOM_BYTES = 1 * _GIB
_HOST_HISTOGRAM_HEADROOM_BYTES = 8 * _GIB


def _cuda_device_index(device: torch.device) -> int:
    """Resolve an index for PyTorch versions that reject bare ``cuda`` here."""
    if device.type != "cuda":
        raise ValueError(f"expected a CUDA device, got {device}")
    return int(torch.cuda.current_device() if device.index is None else device.index)


def _read_integer_file(path: str) -> int | None:
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
        if not value or value == "max":
            return None
        return int(value)
    except (OSError, ValueError):
        return None


def _host_available_memory_bytes() -> int | None:
    """Best-effort host availability, including the active cgroup limit."""
    candidates: list[int] = []
    try:
        import psutil

        candidates.append(int(psutil.virtual_memory().available))
    except (ImportError, AttributeError, OSError, ValueError):
        pass
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        candidates.append(pages * page_size)
    except (AttributeError, OSError, ValueError):
        pass

    # AutoDL and most contemporary containers use cgroup v2.  Retain the v1
    # fallback so the preflight remains meaningful on older images.
    cgroup_limit = _read_integer_file("/sys/fs/cgroup/memory.max")
    cgroup_current = _read_integer_file("/sys/fs/cgroup/memory.current")
    if cgroup_limit is None:
        cgroup_limit = _read_integer_file(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        )
        cgroup_current = _read_integer_file(
            "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        )
    # cgroup v1 represents an unlimited hierarchy with a near-int64 sentinel.
    if (
        cgroup_limit is not None
        and cgroup_current is not None
        and 0 < cgroup_limit < (1 << 60)
    ):
        candidates.append(max(0, int(cgroup_limit) - int(cgroup_current)))
    return min(candidates) if candidates else None


@dataclass(frozen=True)
class MethodCalibration:
    median: float
    mad: float
    tau_image: float
    tau_pixel: float

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(raw, dtype=np.float32) - float(self.median))
            / float(self.mad)
        ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class NativeMaskContext:
    masks: tuple[np.ndarray, ...]
    labels: np.ndarray
    flat_masks: np.ndarray
    # Each component is represented by compact flat pixel indices.  Never keep
    # one full-resolution boolean image per component: on PCB that representation
    # multiplies a 2048/2560 mask by the number of defect instances and can exceed
    # the complete 120 GiB job limit before metrics start.
    component_indices_by_image: tuple[tuple[np.ndarray, ...], ...]
    resolutions: tuple[tuple[int, int], ...]


def reshape_patch_scores(scores, grid: tuple[int, int] = (32, 32)) -> np.ndarray:
    """Enforce the scorer contract: [image, grid_y, grid_x]."""
    values = np.asarray(scores, dtype=np.float32)
    expected = int(grid[0]) * int(grid[1])
    if values.ndim == 2 and values.shape[1] == expected:
        values = values.reshape(values.shape[0], int(grid[0]), int(grid[1]))
    if values.ndim != 3 or tuple(values.shape[1:]) != tuple(int(v) for v in grid):
        raise ValueError(
            f"patch score contract requires [image,{grid[0]},{grid[1]}], got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("patch scores must be finite")
    return values


@torch.inference_mode()
def cosine_nn_scores(
    query_features: torch.Tensor,
    memory_bank: torch.Tensor,
    *,
    device: torch.device,
    query_chunk_size: int = 65_536,
    bank_chunk_size: int = 131_072,
) -> torch.Tensor:
    query = F.normalize(query_features.reshape(-1, query_features.shape[-1]).float(), dim=-1)
    bank_cpu = F.normalize(memory_bank.reshape(-1, memory_bank.shape[-1]).float(), dim=-1).cpu()
    output = torch.empty((query.shape[0],), dtype=torch.float32)
    for query_start in range(0, int(query.shape[0]), int(query_chunk_size)):
        query_stop = min(query_start + int(query_chunk_size), int(query.shape[0]))
        q = query[query_start:query_stop].to(device, non_blocking=True)
        best = torch.full((q.shape[0],), -float("inf"), device=device)
        for bank_start in range(0, int(bank_cpu.shape[0]), int(bank_chunk_size)):
            bank_stop = min(bank_start + int(bank_chunk_size), int(bank_cpu.shape[0]))
            bank = bank_cpu[bank_start:bank_stop].to(device, non_blocking=True)
            best = torch.maximum(best, torch.matmul(q, bank.T).max(dim=1).values)
            del bank
        output[query_start:query_stop] = (1.0 - best).cpu()
        del q, best
    return output.reshape(query_features.shape[:-1])


@torch.inference_mode()
def prepare_normalized_memory_banks(
    memory_banks: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Normalize with the frozen CPU path once, then keep all small banks on GPU."""
    return {
        method: F.normalize(bank.reshape(-1, bank.shape[-1]).float(), dim=-1)
        .contiguous()
        .to(device, non_blocking=device.type == "cuda")
        for method, bank in memory_banks.items()
    }


@torch.inference_mode()
def cosine_nn_scores_multi(
    query_features: torch.Tensor,
    normalized_memory_banks: dict[str, torch.Tensor],
    *,
    device: torch.device,
    query_chunk_size: int = 65_536,
    bank_chunk_size: int = 131_072,
) -> dict[str, torch.Tensor]:
    """Normalize each query once and score all resident method banks per GPU chunk."""
    if not normalized_memory_banks:
        raise ValueError("multi-bank cosine scoring requires at least one memory bank")
    query = F.normalize(
        query_features.reshape(-1, query_features.shape[-1]).float(), dim=-1
    ).cpu()
    outputs = {
        method: torch.empty((query.shape[0],), dtype=torch.float32)
        for method in normalized_memory_banks
    }
    for query_start in range(0, int(query.shape[0]), int(query_chunk_size)):
        query_stop = min(query_start + int(query_chunk_size), int(query.shape[0]))
        q = query[query_start:query_stop].to(device, non_blocking=device.type == "cuda")
        for method, bank in normalized_memory_banks.items():
            if bank.device.type != device.type or (
                device.index is not None and bank.device.index != device.index
            ):
                raise ValueError(f"normalized memory bank {method} is not resident on {device}")
            best = torch.full((q.shape[0],), -float("inf"), device=device)
            for bank_start in range(0, int(bank.shape[0]), int(bank_chunk_size)):
                bank_stop = min(bank_start + int(bank_chunk_size), int(bank.shape[0]))
                best = torch.maximum(
                    best,
                    torch.matmul(q, bank[bank_start:bank_stop].T).max(dim=1).values,
                )
            outputs[method][query_start:query_stop] = (1.0 - best).cpu()
            del best
        del q
    shape = query_features.shape[:-1]
    return {method: values.reshape(shape) for method, values in outputs.items()}


def fit_calibration(raw_scores: np.ndarray, epsilon: float = 1e-6) -> MethodCalibration:
    raw = reshape_patch_scores(raw_scores).astype(np.float64, copy=False)
    median = float(np.median(raw))
    mad = float(np.median(np.abs(raw - median)) + float(epsilon))
    normalized = (raw - median) / mad
    return MethodCalibration(
        median=median,
        mad=mad,
        tau_image=float(np.quantile(normalized.reshape(normalized.shape[0], -1).max(axis=1), 0.95)),
        tau_pixel=float(np.quantile(normalized, 0.995)),
    )


def patch_maps(
    scores: np.ndarray,
    input_size: int | tuple[int, int] = 512,
    *,
    device: torch.device | None = None,
) -> np.ndarray:
    score_grid = reshape_patch_scores(scores)
    values = torch.from_numpy(score_grid)[:, None]
    target_device = device or torch.device("cpu")
    values = values.to(target_device, non_blocking=target_device.type == "cuda")
    size = (int(input_size), int(input_size)) if isinstance(input_size, int) else tuple(int(v) for v in input_size)
    output = F.interpolate(
        values,
        size=size,
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return output.float().cpu().numpy()


@torch.inference_mode()
def native_patch_maps(
    scores: np.ndarray,
    resolutions: Sequence[tuple[int, int]],
    *,
    device: torch.device,
    max_output_pixels_per_batch: int = 32_000_000,
) -> list[np.ndarray]:
    """Interpolate ragged native maps on GPU with one compact backing array."""
    _, maps = native_patch_maps_flat(
        scores,
        resolutions,
        device=device,
        max_output_pixels_per_batch=max_output_pixels_per_batch,
    )
    return maps


@torch.inference_mode()
def native_patch_maps_flat(
    scores: np.ndarray,
    resolutions: Sequence[tuple[int, int]],
    *,
    device: torch.device,
    max_output_pixels_per_batch: int = 32_000_000,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Write bounded interpolation batches directly into one CPU flat array."""
    score_grid = reshape_patch_scores(scores)
    if len(resolutions) != int(score_grid.shape[0]):
        raise ValueError("native resolutions must align with patch-score images")
    if int(max_output_pixels_per_batch) < 1:
        raise ValueError("max_output_pixels_per_batch must be positive")
    grouped: dict[tuple[int, int], list[int]] = {}
    for index, resolution in enumerate(resolutions):
        size = tuple(int(value) for value in resolution)
        grouped.setdefault(size, []).append(int(index))
    offsets = [0]
    for resolution in resolutions:
        offsets.append(offsets[-1] + int(resolution[0]) * int(resolution[1]))
    flat_output = np.empty(offsets[-1], dtype=np.float32)
    for size, image_indices in grouped.items():
        pixels_per_image = int(size[0]) * int(size[1])
        chunk_images = max(1, int(max_output_pixels_per_batch) // pixels_per_image)
        for start in range(0, len(image_indices), chunk_images):
            selected = image_indices[start : start + chunk_images]
            values = torch.from_numpy(score_grid[selected])[:, None].to(
                device, non_blocking=device.type == "cuda"
            )
            batch_maps = F.interpolate(
                values,
                size=size,
                mode="bilinear",
                align_corners=False,
            )[:, 0].float().cpu().numpy()
            for local_index, image_index in enumerate(selected):
                start_offset = offsets[image_index]
                stop_offset = offsets[image_index + 1]
                flat_output[start_offset:stop_offset] = batch_maps[local_index].reshape(-1)
            del values, batch_maps
    maps = [
        flat_output[offsets[index] : offsets[index + 1]].reshape(resolutions[index])
        for index in range(len(resolutions))
    ]
    return flat_output, maps


@torch.inference_mode()
def native_patch_maps_flat_tensor(
    scores: np.ndarray,
    resolutions: Sequence[tuple[int, int]],
    *,
    device: torch.device,
    max_output_pixels_per_batch: int = 32_000_000,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Interpolate directly into one resident device tensor for formal GPU metrics."""
    score_grid = reshape_patch_scores(scores)
    if len(resolutions) != int(score_grid.shape[0]):
        raise ValueError("native resolutions must align with patch-score images")
    grouped: dict[tuple[int, int], list[int]] = {}
    offsets = [0]
    for index, resolution in enumerate(resolutions):
        size = tuple(int(value) for value in resolution)
        grouped.setdefault(size, []).append(int(index))
        offsets.append(offsets[-1] + int(size[0]) * int(size[1]))
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(_cuda_device_index(device))
        resident_bytes = int(offsets[-1]) * np.dtype(np.float32).itemsize
        interpolation_pixels = min(
            int(offsets[-1]), int(max_output_pixels_per_batch)
        )
        interpolation_workspace_bytes = interpolation_pixels * 4
        fixed_headroom = max(
            int(_CUDA_NATIVE_MAP_HEADROOM_BYTES), int(float(total_bytes) * 0.15)
        )
        required_bytes = (
            int(resident_bytes)
            + int(interpolation_workspace_bytes)
            + int(fixed_headroom)
        )
        print(
            "[NSRM-M0] native resident-map CUDA preflight "
            f"pixels={offsets[-1]} resident_GiB={resident_bytes / _GIB:.3f} "
            f"interpolation_workspace_GiB={interpolation_workspace_bytes / _GIB:.3f} "
            f"headroom_GiB={fixed_headroom / _GIB:.3f} "
            f"free_GiB={free_bytes / _GIB:.3f} required_GiB={required_bytes / _GIB:.3f}",
            flush=True,
        )
        if int(free_bytes) < int(required_bytes):
            raise RuntimeError(
                "insufficient CUDA headroom for resident native score maps: "
                f"free={free_bytes / _GIB:.3f} GiB required={required_bytes / _GIB:.3f} GiB"
            )
    flat_output = torch.empty(offsets[-1], dtype=torch.float32, device=device)
    for size, image_indices in grouped.items():
        pixels_per_image = int(size[0]) * int(size[1])
        chunk_images = max(1, int(max_output_pixels_per_batch) // pixels_per_image)
        for start in range(0, len(image_indices), chunk_images):
            selected = image_indices[start : start + chunk_images]
            values = torch.from_numpy(score_grid[selected])[:, None].to(
                device, non_blocking=device.type == "cuda"
            )
            batch_maps = F.interpolate(
                values,
                size=size,
                mode="bilinear",
                align_corners=False,
            )[:, 0].float()
            for local_index, image_index in enumerate(selected):
                flat_output[offsets[image_index] : offsets[image_index + 1]].copy_(
                    batch_maps[local_index].reshape(-1)
                )
            del values, batch_maps
    maps = [
        flat_output[offsets[index] : offsets[index + 1]].reshape(resolutions[index])
        for index in range(len(resolutions))
    ]
    return flat_output, maps


def _image_aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _torch_binary_ranking_metrics_and_thresholds(
    labels: np.ndarray,
    scores: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    max_thresholds: int = 256,
    endpoint_chunk_size: int = 8_000_000,
) -> tuple[float, float, np.ndarray]:
    """Exact AUROC/AP and NumPy-linear quantiles from one tie-grouped sort."""
    label_values = np.asarray(labels, dtype=bool).reshape(-1)
    if isinstance(scores, torch.Tensor):
        score_tensor = scores.reshape(-1).to(device=device, dtype=torch.float32)
        score_count = int(score_tensor.numel())
        if not bool(torch.isfinite(score_tensor).all().item()):
            raise ValueError("binary ranking scores must be finite")
    else:
        score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
        score_count = int(score_values.size)
        score_tensor = torch.from_numpy(score_values).to(
            device, non_blocking=device.type == "cuda"
        )
    if label_values.size != score_count or label_values.size == 0:
        raise ValueError("binary ranking labels/scores must be aligned and non-empty")
    if int(max_thresholds) < 1:
        raise ValueError("max_thresholds must be positive")
    positive_count = int(label_values.sum())
    negative_count = int(label_values.size - positive_count)
    sorted_scores, order = torch.sort(score_tensor, descending=True)
    del score_tensor
    quantile_count = min(int(max_thresholds), int(sorted_scores.numel()))
    probabilities = torch.from_numpy(
        np.linspace(0.0, 1.0, quantile_count, dtype=np.float64)
    ).to(device)
    positions = probabilities * float(int(sorted_scores.numel()) - 1)
    lower = torch.floor(positions).to(torch.int64)
    upper = torch.ceil(positions).to(torch.int64)
    fraction = positions - lower.to(torch.float64)
    # sorted_scores is descending, whereas NumPy quantile indexes ascending.
    last = int(sorted_scores.numel()) - 1
    lower_value = sorted_scores[last - lower].to(torch.float64)
    upper_value = sorted_scores[last - upper].to(torch.float64)
    quantiles = lower_value + (upper_value - lower_value) * fraction
    thresholds = np.unique(quantiles.cpu().numpy())[::-1]
    if positive_count == 0 or negative_count == 0:
        del sorted_scores, order
        return float("nan"), float("nan"), thresholds
    label_tensor = torch.from_numpy(label_values).to(
        device, non_blocking=device.type == "cuda"
    )
    sorted_positive = label_tensor[order].to(torch.int64)
    del label_tensor, order
    true_positive = torch.cumsum(sorted_positive, dim=0)
    del sorted_positive
    distinct_end = torch.ones_like(sorted_scores, dtype=torch.bool)
    distinct_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    endpoint_indices = torch.nonzero(distinct_end, as_tuple=False).reshape(-1)
    del distinct_end, sorted_scores
    average_precision = torch.zeros((), dtype=torch.float64, device=device)
    auroc = torch.zeros((), dtype=torch.float64, device=device)
    previous_tp = torch.zeros((), dtype=torch.float64, device=device)
    previous_fp = torch.zeros((), dtype=torch.float64, device=device)
    for start in range(0, int(endpoint_indices.numel()), int(endpoint_chunk_size)):
        indices = endpoint_indices[start : start + int(endpoint_chunk_size)]
        tp = true_positive[indices].to(torch.float64)
        positions = (indices + 1).to(torch.float64)
        fp = positions - tp
        prior_tp = torch.cat((previous_tp.reshape(1), tp[:-1]))
        prior_fp = torch.cat((previous_fp.reshape(1), fp[:-1]))
        average_precision += torch.sum(
            ((tp - prior_tp) / float(positive_count)) * (tp / positions)
        )
        auroc += torch.sum(
            ((fp - prior_fp) / float(negative_count))
            * ((tp + prior_tp) / (2.0 * float(positive_count)))
        )
        previous_tp = tp[-1]
        previous_fp = fp[-1]
    del endpoint_indices, true_positive
    return float(auroc.item()), float(average_precision.item()), thresholds


def _torch_binary_ranking_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    device: torch.device,
) -> tuple[float, float]:
    auroc, aupr, _ = _torch_binary_ranking_metrics_and_thresholds(
        labels, scores, device=device
    )
    return auroc, aupr


def _float32_order_keys(values: np.ndarray) -> np.ndarray:
    """Monotonic IEEE-754 keys; -0 and +0 are canonicalized as one tie."""
    scores = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError("float32 ranking keys require finite scores")
    canonical = scores.copy()
    canonical[canonical == 0] = np.float32(0.0)
    bits = canonical.view(np.uint32)
    negative = (bits & np.uint32(0x80000000)) != 0
    return np.where(
        negative,
        np.bitwise_not(bits),
        np.bitwise_xor(bits, np.uint32(0x80000000)),
    ).astype(np.uint32, copy=False)


def _float32_from_order_keys(keys: np.ndarray) -> np.ndarray:
    key_values = np.asarray(keys, dtype=np.uint32)
    nonnegative = (key_values & np.uint32(0x80000000)) != 0
    bits = np.where(
        nonnegative,
        np.bitwise_xor(key_values, np.uint32(0x80000000)),
        np.bitwise_not(key_values),
    ).astype(np.uint32, copy=False)
    return bits.view(np.float32)


def _torch_float32_order_keys(values: torch.Tensor) -> torch.Tensor:
    """CUDA/torch equivalent of ``_float32_order_keys`` returning int64 keys."""
    scores = values.to(torch.float32)
    canonical = torch.where(scores == 0, torch.zeros_like(scores), scores)
    bits = torch.bitwise_and(canonical.view(torch.int32).to(torch.int64), 0xFFFFFFFF)
    negative = torch.bitwise_and(bits, 0x80000000) != 0
    return torch.where(
        negative,
        torch.bitwise_and(torch.bitwise_not(bits), 0xFFFFFFFF),
        torch.bitwise_xor(bits, 0x80000000),
    )


def _ranking_from_dense_float32_histograms(
    total_histogram: np.ndarray,
    positive_histogram: np.ndarray,
    *,
    key_min: int,
    total_count: int,
    positive_count: int,
    max_thresholds: int,
    scan_block_size: int = 8_000_000,
) -> tuple[float, float, np.ndarray]:
    """Exact sklearn-compatible ranking/quantiles from dense float32 tie counts."""
    total_histogram = np.asarray(total_histogram, dtype=np.uint32)
    positive_histogram = np.asarray(positive_histogram, dtype=np.uint32)
    if total_histogram.shape != positive_histogram.shape or total_histogram.ndim != 1:
        raise ValueError("float32 ranking histograms must be aligned 1-D arrays")
    if int(total_histogram.sum(dtype=np.uint64)) != int(total_count):
        raise RuntimeError("float32 total histogram count mismatch")
    if int(positive_histogram.sum(dtype=np.uint64)) != int(positive_count):
        raise RuntimeError("float32 positive histogram count mismatch")
    quantile_count = min(int(max_thresholds), int(total_count))
    probabilities = np.linspace(0.0, 1.0, quantile_count, dtype=np.float64)
    positions = probabilities * float(int(total_count) - 1)
    lower_rank = np.floor(positions).astype(np.int64)
    upper_rank = np.ceil(positions).astype(np.int64)
    required_ranks = np.unique(np.concatenate((lower_rank, upper_rank)))
    rank_keys = np.empty(required_ranks.size, dtype=np.uint32)
    cumulative_before = 0
    rank_cursor = 0
    for start in range(0, int(total_histogram.size), int(scan_block_size)):
        stop = min(start + int(scan_block_size), int(total_histogram.size))
        counts = total_histogram[start:stop].astype(np.uint64, copy=False)
        block_total = int(counts.sum(dtype=np.uint64))
        if block_total == 0:
            continue
        cumulative = np.cumsum(counts, dtype=np.uint64)
        block_end = cumulative_before + block_total
        while rank_cursor < required_ranks.size and int(required_ranks[rank_cursor]) < block_end:
            rank = int(required_ranks[rank_cursor])
            local = int(np.searchsorted(cumulative, rank - cumulative_before, side="right"))
            rank_keys[rank_cursor] = np.uint32(int(key_min) + start + local)
            rank_cursor += 1
        cumulative_before = block_end
    if rank_cursor != required_ranks.size:
        raise RuntimeError("failed to resolve every float32 quantile rank")
    rank_values = _float32_from_order_keys(rank_keys).astype(np.float64)
    lower_index = np.searchsorted(required_ranks, lower_rank)
    upper_index = np.searchsorted(required_ranks, upper_rank)
    fraction = positions - lower_rank.astype(np.float64)
    quantiles = rank_values[lower_index] + (
        rank_values[upper_index] - rank_values[lower_index]
    ) * fraction
    thresholds = np.unique(quantiles)[::-1]
    negative_count = int(total_count) - int(positive_count)
    if int(positive_count) == 0 or negative_count == 0:
        return float("nan"), float("nan"), thresholds
    cumulative_tp_before = 0
    cumulative_total_before = 0
    average_precision = 0.0
    auroc = 0.0
    for stop in range(int(total_histogram.size), 0, -int(scan_block_size)):
        start = max(0, stop - int(scan_block_size))
        totals = total_histogram[start:stop]
        nonzero = np.flatnonzero(totals)[::-1]
        if not nonzero.size:
            continue
        group_total = totals[nonzero].astype(np.int64, copy=False)
        group_positive = positive_histogram[start:stop][nonzero].astype(
            np.int64, copy=False
        )
        cumulative_tp = cumulative_tp_before + np.cumsum(group_positive, dtype=np.int64)
        cumulative_total = cumulative_total_before + np.cumsum(group_total, dtype=np.int64)
        cumulative_fp = cumulative_total - cumulative_tp
        prior_tp = np.concatenate((
            np.asarray([cumulative_tp_before], dtype=np.int64), cumulative_tp[:-1]
        ))
        prior_fp = np.concatenate((
            np.asarray([cumulative_total_before - cumulative_tp_before], dtype=np.int64),
            cumulative_fp[:-1],
        ))
        average_precision += float(np.sum(
            ((cumulative_tp - prior_tp) / float(positive_count))
            * (cumulative_tp / cumulative_total),
            dtype=np.float64,
        ))
        auroc += float(np.sum(
            ((cumulative_fp - prior_fp) / float(negative_count))
            * ((cumulative_tp + prior_tp) / (2.0 * float(positive_count))),
            dtype=np.float64,
        ))
        cumulative_tp_before = int(cumulative_tp[-1])
        cumulative_total_before = int(cumulative_total[-1])
    return float(auroc), float(average_precision), thresholds


@torch.inference_mode()
def _cuda_float32_histogram_ranking_metrics_and_thresholds(
    labels: np.ndarray,
    scores: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    max_thresholds: int = 256,
    chunk_size: int = 8_000_000,
    progress_label: str | None = None,
) -> tuple[float, float, np.ndarray]:
    """Exact billion-pixel ranking via GPU tie aggregation and bounded host histograms."""
    histogram_started = perf_counter()
    label_values = np.asarray(labels, dtype=bool).reshape(-1)
    if isinstance(scores, torch.Tensor):
        if scores.device.type != device.type:
            raise ValueError("resident ranking scores are on the wrong device")
        score_tensor_all = scores.reshape(-1).to(torch.float32)
        if not bool(torch.isfinite(score_tensor_all).all().item()):
            raise ValueError("CUDA histogram ranking scores must be finite")
        score_count = int(score_tensor_all.numel())
        score_values = None
    else:
        score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
        if not np.isfinite(score_values).all():
            raise ValueError("CUDA histogram ranking scores must be finite")
        score_tensor_all = None
        score_count = int(score_values.size)
    if label_values.size != score_count or label_values.size == 0:
        raise ValueError("CUDA histogram ranking requires aligned non-empty inputs")
    if score_count >= np.iinfo(np.uint32).max:
        raise ValueError("CUDA float32 histogram count exceeds uint32 protocol capacity")
    key_min = np.iinfo(np.uint32).max
    key_max = 0
    chunk_count = (score_count + int(chunk_size) - 1) // int(chunk_size)
    if progress_label:
        print(
            f"[NSRM-M0] ranking histogram key-range {progress_label} "
            f"pixels={score_count} chunks={chunk_count}",
            flush=True,
        )
    key_range_started = perf_counter()
    for start in range(0, score_count, int(chunk_size)):
        stop = min(start + int(chunk_size), score_count)
        score_chunk = (
            score_tensor_all[start:stop]
            if score_tensor_all is not None
            else torch.from_numpy(score_values[start:stop]).to(device, non_blocking=True)
        )
        keys = _torch_float32_order_keys(score_chunk)
        key_min = min(int(key_min), int(keys.min().item()))
        key_max = max(int(key_max), int(keys.max().item()))
        del score_chunk, keys
    key_range_seconds = perf_counter() - key_range_started
    key_span = int(key_max) - int(key_min) + 1
    histogram_bytes = int(key_span) * 2 * np.dtype(np.uint32).itemsize
    host_available_bytes = _host_available_memory_bytes()
    host_headroom_bytes = min(
        int(_HOST_HISTOGRAM_HEADROOM_BYTES),
        max(int(_HOST_HISTOGRAM_MIN_HEADROOM_BYTES), int(histogram_bytes) // 2),
    )
    host_required_bytes = int(histogram_bytes) + int(host_headroom_bytes)
    if progress_label:
        available_text = (
            f"{host_available_bytes / _GIB:.3f}"
            if host_available_bytes is not None
            else "unknown"
        )
        print(
            f"[NSRM-M0] ranking histogram allocate {progress_label} "
            f"key_span={key_span} host_GiB={histogram_bytes / _GIB:.3f} "
            f"headroom_GiB={host_headroom_bytes / _GIB:.3f} "
            f"available_GiB={available_text} required_GiB={host_required_bytes / _GIB:.3f}",
            flush=True,
        )
    if (
        host_available_bytes is not None
        and int(host_available_bytes) < int(host_required_bytes)
    ):
        raise MemoryError(
            "insufficient host headroom for exact float32 ranking histograms: "
            f"available={host_available_bytes / _GIB:.3f} GiB "
            f"required={host_required_bytes / _GIB:.3f} GiB "
            f"key_span={key_span}"
        )
    allocation_started = perf_counter()
    total_histogram = np.zeros(key_span, dtype=np.uint32)
    positive_histogram = np.zeros(key_span, dtype=np.uint32)
    allocation_seconds = perf_counter() - allocation_started
    gpu_accumulation_seconds = 0.0
    host_histogram_update_seconds = 0.0
    for chunk_index, start in enumerate(
        range(0, score_count, int(chunk_size)), start=1
    ):
        gpu_chunk_started = perf_counter()
        stop = min(start + int(chunk_size), score_count)
        labels_np = label_values[start:stop]
        score_chunk = (
            score_tensor_all[start:stop]
            if score_tensor_all is not None
            else torch.from_numpy(score_values[start:stop]).to(device, non_blocking=True)
        )
        keys = _torch_float32_order_keys(score_chunk)
        positives = torch.from_numpy(labels_np).to(device, non_blocking=True)
        sorted_keys, order = torch.sort(keys)
        sorted_positive = positives[order].to(torch.int64)
        del score_chunk, keys, positives, order
        endpoint = torch.ones_like(sorted_keys, dtype=torch.bool)
        endpoint[:-1] = sorted_keys[:-1] != sorted_keys[1:]
        endpoint_indices = torch.nonzero(endpoint, as_tuple=False).reshape(-1)
        unique_keys = sorted_keys[endpoint_indices]
        cumulative_positive = torch.cumsum(sorted_positive, dim=0)
        positive_at_endpoint = cumulative_positive[endpoint_indices]
        prior_endpoint = torch.cat((
            torch.full((1,), -1, dtype=torch.int64, device=device),
            endpoint_indices[:-1],
        ))
        prior_positive = torch.cat((
            torch.zeros((1,), dtype=torch.int64, device=device),
            positive_at_endpoint[:-1],
        ))
        total_counts = endpoint_indices - prior_endpoint
        positive_counts = positive_at_endpoint - prior_positive
        unique_host = unique_keys.cpu().numpy()
        total_host = total_counts.cpu().numpy()
        positive_host = positive_counts.cpu().numpy()
        gpu_accumulation_seconds += perf_counter() - gpu_chunk_started
        host_update_started = perf_counter()
        unique_np = unique_host.astype(np.int64, copy=False) - int(key_min)
        total_np = total_host.astype(np.uint32, copy=False)
        positive_np = positive_host.astype(np.uint32, copy=False)
        total_histogram[unique_np] += total_np
        positive_histogram[unique_np] += positive_np
        host_histogram_update_seconds += perf_counter() - host_update_started
        del unique_host, total_host, positive_host, unique_np, total_np, positive_np
        del (
            sorted_keys,
            sorted_positive,
            endpoint,
            endpoint_indices,
            unique_keys,
            cumulative_positive,
            positive_at_endpoint,
            prior_endpoint,
            prior_positive,
            total_counts,
            positive_counts,
        )
        if progress_label and (
            chunk_index == chunk_count
            or chunk_index % max(1, chunk_count // 10) == 0
        ):
            print(
                f"[NSRM-M0] ranking histogram GPU {progress_label} "
                f"{chunk_index}/{chunk_count}",
                flush=True,
            )
    cpu_scan_started = perf_counter()
    result = _ranking_from_dense_float32_histograms(
        total_histogram,
        positive_histogram,
        key_min=int(key_min),
        total_count=score_count,
        positive_count=int(label_values.sum()),
        max_thresholds=int(max_thresholds),
    )
    cpu_scan_seconds = perf_counter() - cpu_scan_started
    if progress_label:
        print(
            f"[NSRM-M0] ranking histogram timings {progress_label} "
            f"key_range_s={key_range_seconds:.3f} "
            f"host_allocate_s={allocation_seconds:.3f} "
            f"gpu_tie_aggregation_s={gpu_accumulation_seconds:.3f} "
            f"host_histogram_update_s={host_histogram_update_seconds:.3f} "
            f"cpu_scan_s={cpu_scan_seconds:.3f} "
            f"total_s={perf_counter() - histogram_started:.3f}",
            flush=True,
        )
    return result


def pixel_ranking_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    device: torch.device,
    cuda_memory_fraction: float = 0.60,
) -> tuple[float, float, str]:
    """Use exact CUDA sorting when it fits, otherwise preserve the CPU contract."""
    auroc, aupr, _, backend = pixel_ranking_metrics_and_thresholds(
        labels,
        scores,
        device=device,
        cuda_memory_fraction=cuda_memory_fraction,
    )
    return auroc, aupr, backend


def pixel_ranking_metrics_and_thresholds(
    labels: np.ndarray,
    scores: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    cuda_memory_fraction: float = 0.60,
    max_thresholds: int = 256,
    progress_label: str | None = None,
    allow_cpu_fallback: bool = True,
) -> tuple[float, float, np.ndarray, str]:
    """Compute exact ranking metrics and frozen AUPRO thresholds in one sort."""
    label_values = np.asarray(labels, dtype=bool).reshape(-1)
    if isinstance(scores, torch.Tensor):
        score_tensor = scores.reshape(-1).to(torch.float32)
        score_count = int(score_tensor.numel())
        if not bool(torch.isfinite(score_tensor).all().item()):
            raise ValueError("pixel ranking scores must be finite")
        score_values = None
    else:
        score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
        score_tensor = None
        score_count = int(score_values.size)
        if not np.isfinite(score_values).all():
            raise ValueError("pixel ranking scores must be finite")
    if label_values.size != score_count or label_values.size == 0:
        raise ValueError("pixel ranking labels/scores must be aligned and non-empty")
    if int(max_thresholds) < 1:
        raise ValueError("max_thresholds must be positive")
    cuda_available = device.type == "cuda" and torch.cuda.is_available()
    use_cuda = cuda_available
    if use_cuda:
        free_bytes, _ = torch.cuda.mem_get_info(_cuda_device_index(device))
        # One global score sort plus an int64 cumulative count and endpoint index
        # requires about 40 bytes/pixel; endpoint arithmetic is separately chunked.
        estimated_bytes = score_count * 40
        use_cuda = estimated_bytes <= int(float(free_bytes) * float(cuda_memory_fraction))
    if use_cuda:
        try:
            # One exact sort is enough to saturate the GPU.  Serialize it across
            # method workers so two threads cannot both reserve a near-limit
            # tie-grouped-sort workspace after observing the same free-memory value.
            with _CUDA_RANKING_LOCK:
                free_bytes, _ = torch.cuda.mem_get_info(_cuda_device_index(device))
                estimated_bytes = score_count * 40
                if estimated_bytes > int(float(free_bytes) * float(cuda_memory_fraction)):
                    raise torch.cuda.OutOfMemoryError("bounded CUDA ranking workspace unavailable")
                auroc, aupr, thresholds = _torch_binary_ranking_metrics_and_thresholds(
                    label_values,
                    score_tensor if score_tensor is not None else score_values,
                    device=device,
                    max_thresholds=max_thresholds,
                )
            return auroc, aupr, thresholds, "cuda_exact_tie_grouped_sort"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    histogram_error: BaseException | None = None
    if cuda_available:
        try:
            with _CUDA_RANKING_LOCK:
                auroc, aupr, thresholds = (
                    _cuda_float32_histogram_ranking_metrics_and_thresholds(
                        label_values,
                        score_tensor if score_tensor is not None else score_values,
                        device=device,
                        max_thresholds=max_thresholds,
                        progress_label=progress_label,
                    )
                )
            return auroc, aupr, thresholds, "cuda_exact_float32_histogram"
        except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
            histogram_error = exc
            torch.cuda.empty_cache()
    if cuda_available and not bool(allow_cpu_fallback):
        raise RuntimeError(
            "exact CUDA pixel ranking could not allocate its bounded global-sort "
            "or float32-histogram workspace; CPU fallback is disabled by protocol"
        ) from histogram_error
    if score_values is None:
        score_values = score_tensor.cpu().numpy()
    thresholds = np.unique(
        np.quantile(
            score_values,
            np.linspace(0.0, 1.0, min(int(max_thresholds), score_values.size)),
        )
    )[::-1]
    return (
        safe_auroc(label_values, score_values),
        pixel_aupr(label_values, score_values),
        thresholds,
        "cpu_sklearn_exact_fallback",
    )


def _component_indices(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return compact flat indices for every connected component.

    The previous implementation returned ``labels == component_id`` for every
    component.  That is O(component_count * H * W) memory.  Bounding-box-local
    extraction below is O(number_of_defect_pixels) while selecting exactly the
    same pixels as SciPy's default connected components.
    """
    from scipy import ndimage

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"native component mask must be 2-D, got {mask.shape}")
    labels, count = ndimage.label(mask)
    height, width = mask.shape
    components = []
    for component_id, bounds in enumerate(ndimage.find_objects(labels), start=1):
        if bounds is None:
            continue
        local_y, local_x = np.nonzero(labels[bounds] == int(component_id))
        flat = (
            (local_y.astype(np.int64) + int(bounds[0].start)) * int(width)
            + local_x.astype(np.int64)
            + int(bounds[1].start)
        )
        if flat.size:
            index_dtype = np.int32 if int(height) * int(width) <= np.iinfo(np.int32).max else np.int64
            components.append(flat.astype(index_dtype, copy=False))
    if len(components) != int(count):
        raise RuntimeError(
            f"connected-component extraction mismatch: expected {count}, got {len(components)}"
        )
    return tuple(components)


def prepare_native_mask_context(
    masks: Sequence[np.ndarray], labels: np.ndarray
) -> NativeMaskContext:
    mask_tuple = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    label_values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not mask_tuple or len(mask_tuple) != int(label_values.size):
        raise ValueError("native masks and labels must be aligned and non-empty")
    if not np.isin(label_values, (0, 1)).all():
        raise ValueError("native labels must be binary")
    inconsistent_normal = [
        index
        for index, (mask, label) in enumerate(zip(mask_tuple, label_values))
        if int(label) == 0 and bool(mask.any())
    ]
    if inconsistent_normal:
        raise ValueError(
            "normal native images must have empty masks; inconsistent indices="
            f"{inconsistent_normal[:8]}"
        )
    components = tuple(
        _component_indices(mask) if int(label_values[index]) == 1 else tuple()
        for index, mask in enumerate(mask_tuple)
    )
    return NativeMaskContext(
        masks=mask_tuple,
        labels=label_values,
        flat_masks=np.concatenate([mask.reshape(-1) for mask in mask_tuple]),
        component_indices_by_image=components,
        resolutions=tuple(tuple(int(value) for value in mask.shape) for mask in mask_tuple),
    )


def _pixel_aupro_native_lists(
    masks: Sequence[np.ndarray],
    maps: Sequence[np.ndarray],
    *,
    max_fpr: float = 0.30,
    max_thresholds: int = 256,
    component_indices_by_image: Sequence[Sequence[np.ndarray]] | None = None,
    flat_scores: np.ndarray | None = None,
    thresholds: np.ndarray | None = None,
) -> float:
    """Memory-bounded CPU AUPRO matching the frozen NumPy implementation."""
    if len(masks) != len(maps) or not masks:
        raise ValueError("native masks/maps must be aligned and non-empty")
    mask_list = [np.asarray(mask, dtype=bool) for mask in masks]
    # Interpolated inputs are float32.  Keeping them float32 preserves their exact
    # ordering while avoiding an unnecessary 2x full-pixel copy to float64.
    map_list = [np.asarray(score, dtype=np.float32) for score in maps]
    if any(mask.shape != score.shape for mask, score in zip(mask_list, map_list)):
        raise ValueError("each native mask/map pair must have matching shape")
    component_lists = (
        component_indices_by_image
        if component_indices_by_image is not None
        else [_component_indices(mask) for mask in mask_list]
    )
    if len(component_lists) != len(mask_list):
        raise ValueError("native component lists must align with masks")
    if flat_scores is None:
        flat_values = np.concatenate([score.reshape(-1) for score in map_list])
    else:
        flat_values = np.asarray(flat_scores, dtype=np.float32).reshape(-1)
        expected = sum(int(score.size) for score in map_list)
        if int(flat_values.size) != int(expected):
            raise ValueError("flat native scores do not match native maps")
    if thresholds is None:
        threshold_values = np.unique(
            np.quantile(
                flat_values,
                np.linspace(0.0, 1.0, min(int(max_thresholds), flat_values.size)),
            )
        )[::-1]
    else:
        threshold_values = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    background_positive = np.zeros(threshold_values.size, dtype=np.int64)
    background_count = 0
    overlap_sum = np.zeros(threshold_values.size, dtype=np.float64)
    region_count = 0
    for mask, score, components in zip(mask_list, map_list, component_lists):
        if np.any(~mask):
            background = np.asarray(score[~mask], dtype=np.float32)
            background.sort()
            background_positive += int(background.size) - np.searchsorted(
                background, threshold_values, side="left"
            )
            background_count += int(background.size)
        flat_score = score.reshape(-1)
        for indices in components:
            if not indices.size:
                continue
            region = np.asarray(flat_score[indices], dtype=np.float32)
            region.sort()
            overlap_sum += (
                int(region.size) - np.searchsorted(region, threshold_values, side="left")
            ) / float(region.size)
            region_count += 1
    if background_count == 0 or region_count == 0:
        return float("nan")
    fprs = background_positive.astype(np.float64) / float(background_count)
    pros = overlap_sum / float(region_count)
    return _integrate_aupro_curve(fprs, pros, max_fpr=float(max_fpr))


def _integrate_aupro_curve(
    fprs: np.ndarray,
    pros: np.ndarray,
    *,
    max_fpr: float,
) -> float:
    """Frozen AUPRO curve integration shared by CPU and CUDA count backends."""
    points = [(0.0, 0.0), *zip(fprs.tolist(), pros.tolist()), (1.0, 1.0)]
    points.sort(key=lambda pair: pair[0])
    fpr = np.asarray([point[0] for point in points])
    pro = np.asarray([point[1] for point in points])
    unique_fpr = np.unique(fpr)
    max_pro = np.asarray([pro[fpr == value].max() for value in unique_fpr])
    if unique_fpr[-1] < float(max_fpr):
        unique_fpr = np.append(unique_fpr, float(max_fpr))
        max_pro = np.append(max_pro, max_pro[-1])
    elif float(max_fpr) not in unique_fpr:
        interpolated = np.interp(float(max_fpr), unique_fpr, max_pro)
        keep = unique_fpr < float(max_fpr)
        unique_fpr = np.append(unique_fpr[keep], float(max_fpr))
        max_pro = np.append(max_pro[keep], interpolated)
    else:
        keep = unique_fpr <= float(max_fpr)
        unique_fpr, max_pro = unique_fpr[keep], max_pro[keep]
    return float(np.trapz(max_pro, unique_fpr) / float(max_fpr))


def _float32_ceiling_thresholds(thresholds: np.ndarray) -> np.ndarray:
    """Map float64 thresholds to the smallest float32 value >= each threshold."""
    values64 = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    values32 = values64.astype(np.float32)
    rounded_down = values32.astype(np.float64) < values64
    values32[rounded_down] = np.nextafter(
        values32[rounded_down], np.float32(np.inf), dtype=np.float32
    )
    return values32


@torch.inference_mode()
def _pixel_aupro_native_torch_lists(
    masks: Sequence[np.ndarray],
    maps: Sequence[np.ndarray | torch.Tensor],
    *,
    thresholds: np.ndarray,
    component_indices_by_image: Sequence[Sequence[np.ndarray]],
    device: torch.device,
    max_fpr: float = 0.30,
) -> float:
    """Exact threshold counts on CUDA/torch with one native image resident at a time."""
    if len(masks) != len(maps) or len(masks) != len(component_indices_by_image):
        raise ValueError("torch AUPRO inputs must be aligned")
    threshold_values = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    search_thresholds = torch.from_numpy(
        _float32_ceiling_thresholds(threshold_values)
    ).to(device)
    ascending_thresholds = torch.flip(search_thresholds, dims=(0,)).contiguous()
    threshold_count = int(search_thresholds.numel())
    background_positive = torch.zeros(
        search_thresholds.numel(), dtype=torch.int64, device=device
    )
    overlap_sum = torch.zeros(
        search_thresholds.numel(), dtype=torch.float64, device=device
    )
    background_count = 0
    region_count = 0
    for mask, score, components in zip(masks, maps, component_indices_by_image):
        mask_array = np.asarray(mask, dtype=bool)
        if isinstance(score, torch.Tensor):
            score_shape = tuple(int(value) for value in score.shape)
            if score.device.type != device.type:
                raise ValueError("resident AUPRO score map is on the wrong device")
            score_tensor = score.reshape(-1).to(torch.float32)
        else:
            score_array = np.asarray(score, dtype=np.float32)
            score_shape = score_array.shape
            score_tensor = torch.from_numpy(score_array.reshape(-1)).to(
                device, non_blocking=device.type == "cuda"
            )
        if mask_array.shape != score_shape:
            raise ValueError("torch AUPRO native mask/map shapes must match")
        mask_tensor = torch.from_numpy(mask_array.reshape(-1)).to(
            device, non_blocking=device.type == "cuda"
        )
        background = score_tensor[~mask_tensor]
        if background.numel():
            background_start = threshold_count - torch.searchsorted(
                ascending_thresholds, background, right=True
            )
            background_positive += torch.cumsum(
                torch.bincount(
                    background_start,
                    minlength=threshold_count + 1,
                )[:threshold_count],
                dim=0,
            )
            background_count += int(background.numel())
        nonempty_components = [indices for indices in components if indices.size]
        if nonempty_components:
            component_lengths_np = np.asarray(
                [indices.size for indices in nonempty_components], dtype=np.int64
            )
            concatenated_indices = np.concatenate(nonempty_components).astype(
                np.int64, copy=False
            )
            index_tensor = torch.from_numpy(concatenated_indices).to(
                device, non_blocking=device.type == "cuda"
            )
            component_lengths = torch.from_numpy(component_lengths_np).to(
                device, non_blocking=device.type == "cuda"
            )
            component_ids = torch.repeat_interleave(
                torch.arange(
                    len(nonempty_components), device=device, dtype=torch.int64
                ),
                component_lengths,
            )
            region_values = score_tensor[index_tensor]
            region_start = threshold_count - torch.searchsorted(
                ascending_thresholds, region_values, right=True
            )
            combined_bins = component_ids * (threshold_count + 1) + region_start
            counts = torch.bincount(
                combined_bins,
                minlength=len(nonempty_components) * (threshold_count + 1),
            ).reshape(len(nonempty_components), threshold_count + 1)
            counts = torch.cumsum(counts[:, :threshold_count], dim=1)
            overlap_sum += torch.sum(
                counts.to(torch.float64) / component_lengths[:, None], dim=0
            )
            region_count += len(nonempty_components)
            del (
                index_tensor,
                component_lengths,
                component_ids,
                region_values,
                region_start,
                combined_bins,
                counts,
            )
        del score_tensor, mask_tensor, background
    if background_count == 0 or region_count == 0:
        return float("nan")
    fprs = (background_positive.to(torch.float64) / float(background_count)).cpu().numpy()
    pros = (overlap_sum / float(region_count)).cpu().numpy()
    return _integrate_aupro_curve(fprs, pros, max_fpr=float(max_fpr))


def pixel_aupro_native_backend(
    masks: Sequence[np.ndarray],
    maps: Sequence[np.ndarray | torch.Tensor],
    *,
    thresholds: np.ndarray,
    component_indices_by_image: Sequence[Sequence[np.ndarray]],
    flat_scores: np.ndarray | None,
    device: torch.device,
    max_fpr: float = 0.30,
    allow_cpu_fallback: bool = True,
) -> tuple[float, str]:
    """Prefer bounded CUDA AUPRO counts; retain exact streaming CPU fallback."""
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            with _CUDA_RANKING_LOCK:
                value = _pixel_aupro_native_torch_lists(
                    masks,
                    maps,
                    thresholds=thresholds,
                    component_indices_by_image=component_indices_by_image,
                    device=device,
                    max_fpr=max_fpr,
                )
            return value, "cuda_exact_streaming_counts"
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            if not bool(allow_cpu_fallback):
                raise RuntimeError(
                    "exact CUDA AUPRO workspace unavailable and CPU fallback is disabled"
                ) from exc
    value = _pixel_aupro_native_lists(
        masks,
        maps,
        max_fpr=max_fpr,
        component_indices_by_image=component_indices_by_image,
        flat_scores=flat_scores,
        thresholds=thresholds,
    )
    return value, "cpu_exact_streaming_fallback"


def evaluate_method_group(
    *,
    method: str,
    category: str,
    domain: str,
    shift: str,
    normalized_patch_scores: np.ndarray,
    calibration: MethodCalibration,
    masks: Sequence[np.ndarray],
    labels: np.ndarray,
    input_size: int = 512,
    native_context: NativeMaskContext | None = None,
    metric_device: torch.device | None = None,
) -> tuple[dict, list[dict]]:
    patch = reshape_patch_scores(normalized_patch_scores)
    context = native_context or prepare_native_mask_context(masks, labels)
    mask_list = list(context.masks)
    if len(mask_list) != int(patch.shape[0]):
        raise ValueError("native masks must align with patch-score images")
    target_device = metric_device or torch.device("cpu")
    if target_device.type == "cuda":
        flat_maps, maps = native_patch_maps_flat_tensor(
            patch,
            context.resolutions,
            device=target_device,
        )
        flat_score_bytes = int(flat_maps.numel()) * int(flat_maps.element_size())
    else:
        flat_maps, maps = native_patch_maps_flat(
            patch,
            context.resolutions,
            device=target_device,
        )
        flat_score_bytes = int(flat_maps.nbytes)
    labels = context.labels
    flat_masks = context.flat_masks
    pixel_auroc_value, pixel_aupr_value, aupro_thresholds, pixel_ranking_backend = (
        pixel_ranking_metrics_and_thresholds(
            flat_masks,
            flat_maps,
            device=target_device,
            progress_label=f"{category}:{domain}:{shift}:{method}",
            allow_cpu_fallback=False,
        )
    )
    pixel_aupro_value, pixel_aupro_backend = pixel_aupro_native_backend(
        mask_list,
        maps,
        thresholds=aupro_thresholds,
        component_indices_by_image=context.component_indices_by_image,
        flat_scores=flat_maps if isinstance(flat_maps, np.ndarray) else None,
        device=target_device,
        max_fpr=0.30,
        allow_cpu_fallback=False,
    )
    image_scores = patch.reshape(patch.shape[0], -1).max(axis=1)
    image_positive = image_scores >= float(calibration.tau_image)
    normal = labels == 0
    anomaly = labels == 1
    instances = []
    overseg_sum = 0.0
    overseg_count = 0
    for image_index in np.flatnonzero(anomaly):
        image_mask = mask_list[image_index]
        image_map = maps[image_index]
        if isinstance(image_map, torch.Tensor):
            image_map = image_map.float().cpu().numpy()
        flat_image_map = image_map.reshape(-1)
        for ordinal, component_indices in enumerate(
            context.component_indices_by_image[image_index]
        ):
            response = float(np.quantile(flat_image_map[component_indices], 0.95))
            instances.append({
                "method": method,
                "category": category,
                "domain": domain,
                "shift": shift,
                "image_index": int(image_index),
                "instance_index": int(ordinal),
                "response_q95": response,
                "margin": float(response - calibration.tau_pixel),
                "detected": bool(response >= calibration.tau_pixel),
            })
        overseg_values = image_map[~image_mask]
        overseg_sum += float(overseg_values.sum(dtype=np.float64))
        overseg_count += int(overseg_values.size)
    instance_response = np.asarray([item["response_q95"] for item in instances], dtype=np.float64)
    instance_margin = np.asarray([item["margin"] for item in instances], dtype=np.float64)
    overseg_mean = float(overseg_sum / overseg_count) if overseg_count else float("nan")
    metric = {
        "method": method,
        "category": category,
        "domain": domain,
        "shift": shift,
        "image_count": int(labels.size),
        "normal_image_count": int(normal.sum()),
        "anomaly_image_count": int(anomaly.sum()),
        "image_AUROC": safe_auroc(labels, image_scores),
        "image_AUPR": _image_aupr(labels, image_scores),
        "pixel_AUROC": pixel_auroc_value,
        "pixel_AUPR": pixel_aupr_value,
        "pixel_AUPRO": pixel_aupro_value,
        "pixel_ranking_item_count": int(flat_masks.size),
        "pixel_native_score_storage_GiB": float(flat_score_bytes / 1024**3),
        "pixel_global_sort_estimated_GiB": float(flat_masks.size * 40 / 1024**3),
        "normal_image_FPR": float(image_positive[normal].mean()) if np.any(normal) else float("nan"),
        "image_TPR": float(image_positive[anomaly].mean()) if np.any(anomaly) else float("nan"),
        "defect_instance_count": int(len(instances)),
        "defect_instance_TPR": float(np.mean([item["detected"] for item in instances]))
        if instances
        else float("nan"),
        "defect_instance_q95_mean": float(np.mean(instance_response))
        if instance_response.size
        else float("nan"),
        "defect_instance_margin_mean": float(np.mean(instance_margin))
        if instance_margin.size
        else float("nan"),
        "overseg_mean_score": overseg_mean,
        "defect_overseg_gap": float(np.mean(instance_response) - overseg_mean)
        if instance_response.size and np.isfinite(overseg_mean)
        else float("nan"),
        "tau_image": float(calibration.tau_image),
        "tau_pixel": float(calibration.tau_pixel),
        "pixel_map_height": (
            int(mask_list[0].shape[0])
            if len({mask.shape for mask in mask_list}) == 1 else None
        ),
        "pixel_map_width": (
            int(mask_list[0].shape[1])
            if len({mask.shape for mask in mask_list}) == 1 else None
        ),
        "native_resolution_count": int(len({mask.shape for mask in mask_list})),
        "native_resolutions": ";".join(
            f"{height}x{width}" for height, width in sorted({mask.shape for mask in mask_list})
        ),
        "pixel_map_resolution": "native_per_image_resolution",
        "native_interpolation_backend": (
            "cuda_bounded_batches" if target_device.type == "cuda" else "cpu_torch"
        ),
        "pixel_ranking_backend": pixel_ranking_backend,
        "pixel_aupro_backend": pixel_aupro_backend,
    }
    image_rows = [
        {
            "method": method,
            "category": category,
            "domain": domain,
            "shift": shift,
            "image_index": int(index),
            "label": int(labels[index]),
            "image_score": float(image_scores[index]),
            "image_positive": bool(image_positive[index]),
            "pixel_map_height": int(mask_list[index].shape[0]),
            "pixel_map_width": int(mask_list[index].shape[1]),
        }
        for index in range(len(labels))
    ]
    return metric, image_rows + instances


def paired_fpr_bootstrap(
    baseline_positive: np.ndarray,
    candidate_positive: np.ndarray,
    *,
    seed: int,
    samples: int = 2000,
) -> dict:
    baseline = np.asarray(baseline_positive, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_positive, dtype=np.float64).reshape(-1)
    if baseline.shape != candidate.shape or baseline.size == 0:
        raise ValueError("paired FPR bootstrap requires aligned non-empty normal images")
    rng = np.random.default_rng(int(seed))
    deltas = np.empty((int(samples),), dtype=np.float64)
    for index in range(int(samples)):
        selection = rng.integers(0, baseline.size, size=baseline.size)
        deltas[index] = float(np.mean(candidate[selection] - baseline[selection]))
    return {
        "paired_image_count": int(baseline.size),
        "bootstrap_samples": int(samples),
        "delta_fpr": float(np.mean(candidate - baseline)),
        "ci_lower": float(np.quantile(deltas, 0.025)),
        "ci_upper": float(np.quantile(deltas, 0.975)),
    }
