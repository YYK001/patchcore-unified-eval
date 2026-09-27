"""Memory-only variants for the pinned SuperADD RobustAD transfer.

The official SuperADD detector remains the host: this module only changes how
the prototype database is selected.  It deliberately does not modify the
upstream checkout, score definition, threshold rule, or post-processing.
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DINOv3.nsrm.backbone import MultiAttentionCapture  # noqa: E402
from DINOv3.nsrm.protocol import TRANSFORMS  # noqa: E402
from DINOv3.nsrm.relation import relation_signature  # noqa: E402
from DINOv3.nsrm.transforms import apply_transform  # noqa: E402


MEMORY_VARIANT_PROTOCOL = "SuperADD_memory_variant_v4_ram_first_disk_fallback_exact"
RELATION_AGGREGATION = "H1"
RELATION_TRANSFORM_COUNT = len(TRANSFORMS)
RELATION_MODES = {"nsrm_id", "nsrm"}
CUSTOM_MODES = {"featstrat", "nsrm_id", "nsrm"}
SAMPLER_DISTANCE_MATRIX_MAX_BYTES = 1 << 30
RAM_BACKEND_HEADROOM_BYTES = 16 << 30


@dataclass(frozen=True)
class VariantBuild:
    banks: dict[int, torch.Tensor]
    metadata: dict[str, Any]
    selected_candidate_indices: dict[int, np.ndarray] | None = None


@dataclass
class _MatrixSpool:
    """Append-only float32 matrix stored on the data disk."""

    path: Path
    rows: int = 0
    columns: int = 0

    def append(self, values: np.ndarray) -> None:
        array = np.ascontiguousarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("spooled values must be a non-empty rank-2 matrix")
        if self.columns == 0:
            self.columns = int(array.shape[1])
        elif self.columns != int(array.shape[1]):
            raise ValueError("spooled matrix column count changed")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            array.tofile(handle)
        self.rows += int(array.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.rows * self.columns * np.dtype(np.float32).itemsize)

    def open(self) -> np.memmap:
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("cannot open an empty matrix spool")
        if not self.path.is_file() or self.path.stat().st_size != self.nbytes:
            raise RuntimeError(f"matrix spool size mismatch: {self.path}")
        return np.memmap(
            self.path,
            mode="r+",
            dtype=np.float32,
            shape=(self.rows, self.columns),
            order="C",
        )


@dataclass
class _RamMatrix:
    """Append-only float32 matrix preallocated in process memory."""

    expected_blocks: int
    expected_rows: int | None = None
    rows_per_block: int = 0
    rows: int = 0
    columns: int = 0
    blocks: int = 0
    data: np.ndarray | None = None

    def append(self, values: np.ndarray) -> None:
        array = np.ascontiguousarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("RAM matrix values must be a non-empty rank-2 matrix")
        if self.data is None:
            self.rows_per_block = int(array.shape[0])
            self.columns = int(array.shape[1])
            capacity_rows = (
                int(self.expected_rows)
                if self.expected_rows is not None
                else self.expected_blocks * self.rows_per_block
            )
            if capacity_rows <= 0:
                raise ValueError("RAM matrix expected row count must be positive")
            self.data = np.empty(
                (capacity_rows, self.columns),
                dtype=np.float32,
                order="C",
            )
        elif int(array.shape[1]) != self.columns:
            raise ValueError("RAM matrix column count changed")
        elif self.expected_rows is None and int(array.shape[0]) != self.rows_per_block:
            raise ValueError("RAM matrix block shape changed")
        if self.blocks >= self.expected_blocks:
            raise ValueError("too many RAM matrix blocks")
        stop = self.rows + int(array.shape[0])
        if stop > int(self.data.shape[0]):
            raise ValueError("RAM matrix rows exceed the preallocated capacity")
        self.data[self.rows : stop] = array
        self.rows = stop
        self.blocks += 1

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes) if self.data is not None else 0

    def open(self) -> np.ndarray:
        if (
            self.data is None
            or self.blocks != self.expected_blocks
            or self.rows != int(self.data.shape[0])
        ):
            raise RuntimeError("RAM matrix is incomplete")
        return self.data

    def release(self) -> None:
        self.data = None


def _memory_available_for_candidates() -> tuple[int | None, str]:
    """Return reclaimable-aware bytes available inside the current cgroup."""
    current_path = Path("/sys/fs/cgroup/memory.current")
    maximum_path = Path("/sys/fs/cgroup/memory.max")
    try:
        if current_path.is_file() and maximum_path.is_file():
            current = int(current_path.read_text().strip())
            maximum_text = maximum_path.read_text().strip()
            if maximum_text != "max":
                maximum = int(maximum_text)
                stats: dict[str, int] = {}
                stats_path = Path("/sys/fs/cgroup/memory.stat")
                if stats_path.is_file():
                    for line in stats_path.read_text().splitlines():
                        key, _, value = line.partition(" ")
                        if value.isdigit():
                            stats[key] = int(value)
                reclaimable = (
                    stats.get("inactive_file", 0)
                    + stats.get("active_file", 0)
                    + stats.get("slab_reclaimable", 0)
                )
                non_reclaimable = max(0, current - reclaimable)
                return max(0, maximum - non_reclaimable), "cgroup_reclaimable_aware"
    except (OSError, ValueError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024, "host_memavailable"
    except (OSError, ValueError, IndexError):
        pass
    return None, "unavailable"


def choose_candidate_storage_backend(
    estimated_peak_bytes: int,
    *,
    headroom_bytes: int = RAM_BACKEND_HEADROOM_BYTES,
) -> dict[str, Any]:
    """Choose RAM when the estimated peak plus headroom fits the memory budget."""
    required = int(estimated_peak_bytes)
    if required <= 0:
        raise ValueError("estimated_peak_bytes must be positive")
    available, source = _memory_available_for_candidates()
    if available is None:
        return {
            "backend": "disk",
            "estimated_peak_bytes": required,
            "ram_available_bytes": None,
            "ram_headroom_bytes": int(headroom_bytes),
            "ram_budget_source": source,
            "reason": "memory_budget_unavailable",
        }
    passes = int(available) >= required + int(headroom_bytes)
    return {
        "backend": "ram" if passes else "disk",
        "estimated_peak_bytes": required,
        "ram_available_bytes": int(available),
        "ram_headroom_bytes": int(headroom_bytes),
        "ram_budget_source": source,
        "reason": (
            "estimated_peak_with_headroom_fits"
            if passes
            else "estimated_peak_exceeds_memory_budget"
        ),
    }


def _normalized_feature_array(
    source: np.ndarray,
    *,
    chunk_rows: int = 8192,
) -> np.ndarray:
    """L2-normalize rows into an in-memory float32 matrix."""
    values = np.asarray(source, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("feature normalization requires a non-empty matrix")
    target = np.empty(values.shape, dtype=np.float32, order="C")
    rows_per_chunk = max(1, int(chunk_rows))
    for start in range(0, int(values.shape[0]), rows_per_chunk):
        stop = min(start + rows_per_chunk, int(values.shape[0]))
        chunk = torch.from_numpy(np.asarray(values[start:stop], dtype=np.float32))
        target[start:stop] = F.normalize(chunk, dim=-1).numpy()
    return target


def _close_memmap(value: np.ndarray) -> None:
    if isinstance(value, np.memmap):
        value.flush()
        value._mmap.close()


def _normalized_feature_spool(
    source: np.ndarray,
    path: Path,
    *,
    chunk_rows: int = 8192,
) -> np.memmap:
    """L2-normalize rows into a second memmap with the original Torch rule."""
    values = np.asarray(source, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("feature normalization requires a non-empty matrix")
    rows_per_chunk = max(1, int(chunk_rows))
    target = np.memmap(path, mode="w+", dtype=np.float32, shape=values.shape, order="C")
    for start in range(0, int(values.shape[0]), rows_per_chunk):
        stop = min(start + rows_per_chunk, int(values.shape[0]))
        chunk = torch.from_numpy(np.asarray(values[start:stop], dtype=np.float32))
        target[start:stop] = F.normalize(chunk, dim=-1).numpy()
    target.flush()
    del target
    return np.memmap(path, mode="r+", dtype=np.float32, shape=values.shape, order="C")


def waterfill_budgets(counts: np.ndarray, total_budget: int, minimum: int = 32) -> np.ndarray:
    """Deterministic proportional integer allocation with capacity caps."""
    capacity = np.asarray(counts, dtype=np.int64).reshape(-1)
    if capacity.size == 0 or np.any(capacity < 0):
        raise ValueError("cluster counts must be a non-empty non-negative vector")
    target = int(total_budget)
    if target <= 0 or target > int(capacity.sum()):
        raise ValueError("total budget must be in [1, sum(cluster counts)]")
    minimum = int(minimum)
    if minimum < 0:
        raise ValueError("minimum must be non-negative")
    budget = np.minimum(capacity, minimum)
    if int(budget.sum()) > target:
        budget[:] = 0
    remaining = target - int(budget.sum())
    while remaining:
        room = capacity - budget
        active = room > 0
        if not np.any(active):
            raise AssertionError("water filling exhausted cluster capacity")
        weights = capacity.astype(np.float64)
        weights[~active] = 0.0
        ideal = remaining * weights / weights.sum()
        addition = np.minimum(room, np.floor(ideal).astype(np.int64))
        if int(addition.sum()) == 0:
            fractions = ideal - np.floor(ideal)
            order = sorted(
                np.flatnonzero(active).tolist(),
                key=lambda index: (-fractions[index], -capacity[index], index),
            )
            for index in order[:remaining]:
                addition[index] += 1
        budget += addition
        remaining = target - int(budget.sum())
    if int(budget.sum()) != target or np.any(budget > capacity):
        raise AssertionError("invalid water-filled budget")
    return budget


def fit_feature_labels(features: np.ndarray, seed: int, clusters: int = 64) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("feature labels require a finite [candidate, dim] matrix")
    count = min(int(clusters), int(values.shape[0]))
    if count < 2:
        raise ValueError("feature stratification requires at least two candidates")
    return MiniBatchKMeans(
        n_clusters=count,
        batch_size=min(4096, values.shape[0]),
        n_init=10,
        max_iter=300,
        random_state=int(seed),
    ).fit_predict(values).astype(np.int64, copy=False)


@torch.inference_mode()
def _chunked_exact_nearest_neighbors(
    features_query: torch.Tensor,
    features_key: torch.Tensor,
    knn_neighbors: int,
    normalize: bool = False,
    *,
    max_distance_matrix_bytes: int = SAMPLER_DISTANCE_MATRIX_MAX_BYTES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact upstream kNN without materializing the full Nq x Nk matrix."""
    if features_query.ndim != 2 or features_key.ndim != 2:
        raise ValueError("nearest-neighbor features must be rank-2 tensors")
    if features_query.shape[1] != features_key.shape[1]:
        raise ValueError("query and key feature dimensions must match")
    if features_query.device != features_key.device:
        raise ValueError("query and key features must be on the same device")
    if features_query.dtype != features_key.dtype:
        raise ValueError("query and key features must have the same dtype")

    if normalize:
        features_query = F.normalize(features_query, dim=-1)
        features_key = F.normalize(features_key, dim=-1)

    query_count = int(features_query.shape[0])
    key_count = int(features_key.shape[0])
    neighbors = int(knn_neighbors)
    if query_count <= 0 or key_count <= 0:
        raise ValueError("nearest-neighbor inputs must be non-empty")
    if neighbors <= 0 or neighbors > key_count:
        raise ValueError("knn_neighbors must be in [1, key_count]")

    byte_budget = int(max_distance_matrix_bytes)
    if byte_budget <= 0:
        raise ValueError("max_distance_matrix_bytes must be positive")
    distance_bytes_per_query = key_count * features_query.element_size()
    query_chunk = max(1, min(query_count, byte_budget // max(1, distance_bytes_per_query)))
    chunk_count = (query_count + query_chunk - 1) // query_chunk

    values_cpu = torch.empty((query_count, neighbors), dtype=features_query.dtype, device="cpu")
    indices_cpu = torch.empty((query_count, neighbors), dtype=torch.int64, device="cpu")
    if chunk_count > 1:
        distance_gib = query_chunk * distance_bytes_per_query / float(1 << 30)
        print(
            "[SuperADD-memory] exact chunked kNN "
            f"queries={query_count} keys={key_count} k={neighbors} "
            f"query_chunk={query_chunk} chunks={chunk_count} "
            f"distance_chunk_GiB={distance_gib:.3f}",
            flush=True,
        )

    progress_interval = max(1, chunk_count // 10)
    for chunk_index, start in enumerate(range(0, query_count, query_chunk), start=1):
        stop = min(start + query_chunk, query_count)
        distances = torch.cdist(
            features_query[start:stop],
            features_key,
            compute_mode="use_mm_for_euclid_dist",
        )
        topk_values, topk_indices = torch.topk(
            distances,
            k=neighbors,
            dim=-1,
            largest=False,
            sorted=False,
        )
        values_cpu[start:stop].copy_(topk_values.cpu())
        indices_cpu[start:stop].copy_(topk_indices.cpu())
        del distances, topk_values, topk_indices
        if chunk_count > 1 and (chunk_index % progress_interval == 0 or chunk_index == chunk_count):
            print(
                f"[SuperADD-memory] exact chunked kNN completed {chunk_index}/{chunk_count}",
                flush=True,
            )
    return values_cpu, indices_cpu


def _official_subsample_values(
    features: np.ndarray,
    quota: int,
    device: torch.device,
    *,
    seed: int | None,
    iterations: int,
) -> np.ndarray:
    """Run the upstream sampler with an exact CUDA-memory-bounded kNN backend."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must be [candidate, dim]")
    quota = int(quota)
    count = int(values.shape[0])
    if quota <= 0:
        return np.empty((0, values.shape[1]), dtype=np.float32)
    if quota >= count:
        return np.asarray(values, dtype=np.float32)
    official_src = PROJECT_ROOT / "external_baselines" / "SuperADD" / "tracks" / "industrial" / "src"
    if str(official_src) not in sys.path:
        sys.path.insert(0, str(official_src))
    from industrial import nearest_neighbor as upstream_nearest_neighbor

    if seed is not None:
        np.random.seed(int(seed))
    original_nearest_neighbors = upstream_nearest_neighbor.nearest_neighbors

    def chunked_nearest_neighbors(
        features_query: torch.Tensor,
        features_key: torch.Tensor,
        knn_neighbors: int,
        normalize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _chunked_exact_nearest_neighbors(
            features_query,
            features_key,
            knn_neighbors,
            normalize,
            max_distance_matrix_bytes=SAMPLER_DISTANCE_MATRIX_MAX_BYTES,
        )

    upstream_nearest_neighbor.nearest_neighbors = chunked_nearest_neighbors
    try:
        selected_values = upstream_nearest_neighbor.subsampling_distance_based_fast(
            values,
            quota,
            str(device),
            iterations=int(iterations),
            normalize=False,
            knn_neighbors=100,
        )
    finally:
        upstream_nearest_neighbor.nearest_neighbors = original_nearest_neighbors

    selected_values = np.asarray(selected_values, dtype=np.float32)
    if selected_values.ndim != 2 or selected_values.shape != (quota, values.shape[1]):
        raise RuntimeError("upstream sampler returned an invalid feature matrix")
    return selected_values


def _official_subsample(features: np.ndarray, quota: int, device: torch.device, seed: int) -> np.ndarray:
    """Return physical indices selected by the upstream one-stratum sampler."""
    values = np.asarray(features, dtype=np.float32)
    selected_values = _official_subsample_values(
        values,
        int(quota),
        device,
        seed=int(seed),
        iterations=1,
    )
    buckets: dict[bytes, list[int]] = {}
    for index, row in enumerate(values):
        buckets.setdefault(row.tobytes(), []).append(index)
    selected: list[int] = []
    for row in selected_values:
        queue = buckets.get(row.tobytes())
        if not queue:
            raise RuntimeError("upstream sampler returned a row absent from its input")
        selected.append(queue.pop(0))
    result = np.asarray(selected, dtype=np.int64)
    if result.size != int(quota) or np.unique(result).size != result.size:
        raise RuntimeError("upstream sampler did not return the requested unique budget")
    return result


def _official_subsample_values_with_indices(
    features: np.ndarray,
    quota: int,
    device: torch.device,
    *,
    seed: int | None,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact upstream sampler plus keep-mask indices for the opt-in sidecar."""
    values = np.asarray(features, dtype=np.float32)
    quota, count = int(quota), int(values.shape[0])
    if quota <= 0:
        return np.empty((0, values.shape[1]), np.float32), np.empty(0, np.int64)
    if quota >= count:
        return np.asarray(values, np.float32), np.arange(count, dtype=np.int64)
    if seed is not None:
        np.random.seed(int(seed))
    keep_mask_total = np.full(count, False)
    size_of_subsets = int(1 / int(iterations) * count)
    target_to_keep_subset = quota // int(iterations)
    for _ in range(int(iterations)):
        candidate_indices = np.where(np.invert(keep_mask_total))[0]
        indices = np.random.choice(
            candidate_indices, size=min(size_of_subsets, len(candidate_indices)), replace=False,
        )
        subset = torch.from_numpy(np.asarray(values[indices], dtype=np.float32)).to(device)
        dists, _ = _chunked_exact_nearest_neighbors(
            subset, subset, 100, False,
            max_distance_matrix_bytes=SAMPLER_DISTANCE_MATRIX_MAX_BYTES,
        )
        dists_np = dists.numpy()
        target_distance = np.mean(np.float64(dists_np)) / 10
        number_of_samples = target_to_keep_subset + 1
        random_numbers = np.random.rand(len(indices))
        while number_of_samples > target_to_keep_subset:
            subsampling_factor = np.sum(dists_np < target_distance, axis=-1) + 1
            keep_mask_subset = random_numbers < (1 / subsampling_factor)
            number_of_samples = np.sum(keep_mask_subset)
            target_distance *= 1.1
        keep_mask_total[indices] = keep_mask_subset
        del subset, dists, dists_np
    difference = quota - int(np.sum(keep_mask_total))
    if difference > 0:
        indices_to_add_back = np.where(np.invert(keep_mask_total))[0]
        np.random.shuffle(indices_to_add_back)
        keep_mask_total[indices_to_add_back[:difference]] = True
    selected_indices = np.flatnonzero(keep_mask_total).astype(np.int64)
    selected_values = np.asarray(values[selected_indices], dtype=np.float32)
    if selected_indices.size != quota or selected_values.shape != (quota, values.shape[1]):
        raise RuntimeError("indexed Official sampler returned an invalid budget")
    return selected_values, selected_indices


def _match_selected_candidate_indices(features: np.ndarray, selected_values: np.ndarray) -> np.ndarray:
    """Map selected rows to candidates; exact equality guards hash collisions.

    This provenance helper is never called by the default Official builder.
    """
    values = np.asarray(features, dtype=np.float32)
    selected = np.asarray(selected_values, dtype=np.float32)
    if values.ndim != 2 or selected.ndim != 2 or values.shape[1] != selected.shape[1]:
        raise ValueError("candidate and selected matrices must share their feature dimension")
    buckets: dict[int, list[int]] = {}
    selected_bytes: list[bytes] = []
    for selected_index, row in enumerate(selected):
        raw = row.tobytes()
        selected_bytes.append(raw)
        buckets.setdefault(hash(raw), []).append(selected_index)
    result = np.full(selected.shape[0], -1, dtype=np.int64)
    for candidate_index, row in enumerate(values):
        raw = row.tobytes()
        row_hash = hash(raw)
        queue = buckets.get(row_hash)
        if not queue:
            continue
        for queue_offset, selected_index in enumerate(queue):
            if raw == selected_bytes[selected_index]:
                result[selected_index] = candidate_index
                queue.pop(queue_offset)
                if not queue:
                    del buckets[row_hash]
                break
        if not buckets:
            break
    if (result < 0).any() or np.unique(result).size != result.size:
        raise RuntimeError("could not map every Official memory entry to one unique candidate")
    return result


def stratified_official_indices(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    budget: int,
    minimum: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(features, dtype=np.float32)
    groups = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or groups.shape != (values.shape[0],):
        raise ValueError("features and labels must align")
    unique, counts = np.unique(groups, return_counts=True)
    quotas = waterfill_budgets(counts, int(budget), int(minimum))
    selected_parts: list[np.ndarray] = []
    quota_map: dict[str, int] = {}
    for offset, (cluster, quota) in enumerate(zip(unique.tolist(), quotas.tolist())):
        members = np.flatnonzero(groups == int(cluster))
        local = _official_subsample(values[members], int(quota), device, int(seed) + offset)
        selected_parts.append(members[local])
        quota_map[str(int(cluster))] = int(quota)
    selected = np.concatenate(selected_parts).astype(np.int64, copy=False)
    if selected.size != int(budget) or np.unique(selected).size != selected.size:
        raise AssertionError("stratified SuperADD memory does not have the exact budget")
    return selected, {"cluster_count": int(unique.size), "cluster_budgets": quota_map}


def _stitch(
    prediction: torch.Tensor,
    *,
    batch: int,
    patch_exec: Any,
    input_height: int,
    input_width: int,
) -> torch.Tensor:
    """Reconstruct a global patch map using the official overlap geometry."""
    input_rois_y, prediction_rois_y, result_rois_y = patch_exec.axis_patch_split(input_height)
    input_rois_x, prediction_rois_x, result_rois_x = patch_exec.axis_patch_split(input_width)
    patch_count = len(input_rois_y) * len(input_rois_x)
    local_h = patch_exec.patch_size // patch_exec.model_patch_size
    local_w = local_h
    channels = int(prediction.shape[-1])
    prediction = prediction.reshape(batch, patch_count, local_h, local_w, channels)
    result = torch.zeros(
        (batch, input_height // patch_exec.model_patch_size, input_width // patch_exec.model_patch_size, channels),
        device=prediction.device,
        dtype=prediction.dtype,
    )
    result_pairs = product(result_rois_y, result_rois_x)
    prediction_pairs = product(prediction_rois_y, prediction_rois_x)
    for index, ((py0, py1), (px0, px1)) in enumerate(result_pairs):
        pred_y, pred_x = prediction_pairs[index]
        result[:, py0:py1, px0:px1] = prediction[
            :, index, pred_y[0] : pred_y[1], pred_x[0] : pred_x[1]
        ]
    return result


def product(left, right):
    # Local tiny equivalent avoids importing itertools in hot paths.
    return [(a, b) for a in left for b in right]


@torch.inference_mode()
def patched_features_and_relations(
    model: Any,
    image: torch.Tensor,
    *,
    relation_layers: tuple[int, ...],
    aggregation: str = RELATION_AGGREGATION,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Run official patched execution once and return feature/relation maps."""
    if image.ndim != 4:
        raise ValueError("image must be [B,C,H,W]")
    b, _, height, width = image.shape
    patch_exec = model.patch_exec
    input_rois_y, _, _ = patch_exec.axis_patch_split(height)
    input_rois_x, _, _ = patch_exec.axis_patch_split(width)
    crops = torch.empty(
        (b, len(input_rois_y) * len(input_rois_x), image.shape[1], patch_exec.patch_size, patch_exec.patch_size),
        device=image.device,
        dtype=image.dtype,
    )
    for index, ((y0, y1), (x0, x1)) in enumerate(product(input_rois_y, input_rois_x)):
        crops[:, index] = image[:, :, y0:y1, x0:x1]
    crops = crops.reshape(-1, image.shape[1], patch_exec.patch_size, patch_exec.patch_size)
    dino = model.backbone.dino
    layers = tuple(int(layer) for layer in model.layers)
    local_grid = (patch_exec.patch_size // patch_exec.model_patch_size,) * 2
    # Four ViT-H attention tensors at a 40x40 grid are much larger than the
    # feature maps.  Capture them in small crop batches, then stitch the same
    # way as the official PatchedExecution.  Feature-only mode keeps the full
    # crop batch for speed.
    crop_batch_size = 2 if relation_layers else int(crops.shape[0])
    feature_chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    relation_chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in relation_layers}
    for start in range(0, int(crops.shape[0]), crop_batch_size):
        crop_batch = crops[start : start + crop_batch_size]
        with MultiAttentionCapture(dino, relation_layers) as capture:
            outputs = dino.get_intermediate_layers(crop_batch, n=layers, norm=False)
            attention = {layer: value.float() for layer, value in capture.weights.items()}
        for layer, output in zip(layers, outputs):
            feature_chunks[layer].append(output)
        for layer in relation_layers:
            relation_chunks[layer].append(
                relation_signature(attention[layer], local_grid, aggregation=aggregation)
            )
        del outputs, attention, crop_batch
    features: dict[int, torch.Tensor] = {}
    relations: dict[int, torch.Tensor] = {}
    for layer in layers:
        features[layer] = _stitch(
            torch.cat(feature_chunks[layer], dim=0),
            batch=b,
            patch_exec=patch_exec,
            input_height=height,
            input_width=width,
        )
    for layer in relation_layers:
        relations[layer] = _stitch(
            torch.cat(relation_chunks[layer], dim=0),
            batch=b,
            patch_exec=patch_exec,
            input_height=height,
            input_width=width,
        )
    return features, relations


def _raw_image(path: str | Path) -> torch.Tensor:
    with Image.open(path) as handle:
        return ToTensor()(handle.convert("RGB"))


def _transformed_image(path: str | Path, transform_id: int) -> torch.Tensor:
    with Image.open(path) as handle:
        image = apply_transform(handle.convert("RGB"), int(transform_id))
    return ToTensor()(image)

def _deterministic_official_preprocessing(model: Any, image: torch.Tensor) -> torch.Tensor:
    """Official resize/normalize with fixed factor 1 and no RNG consumption."""
    resize_factor = float(model.preprocessing.resize_factor)
    new_shape = (int(image.shape[-2] * resize_factor), int(image.shape[-1] * resize_factor))
    resized = F.interpolate(image, size=new_shape, mode="bicubic", align_corners=False, antialias=True)
    return (resized - model.preprocessing.mean) / model.preprocessing.std


def _median_relation_trajectory(trajectory: list[np.ndarray]) -> np.ndarray:
    """Reduce one image's transform trajectory before retaining it in RAM."""
    values = [np.asarray(item, dtype=np.float32) for item in trajectory]
    if not values or len({item.shape for item in values}) != 1:
        raise ValueError("relation trajectory must contain aligned transform arrays")
    stacked = np.stack(values, axis=0)
    if not np.isfinite(stacked).all():
        raise ValueError("relation trajectory contains NaN or Inf")
    return np.median(stacked, axis=0).astype(np.float32, copy=False)

def _reduce_relation_trajectory(
    mode: str,
    trajectory: list[np.ndarray],
) -> np.ndarray:
    if mode == "nsrm_id":
        if len(trajectory) != 1:
            raise ValueError("NSRM-Id requires exactly one identity relation")
        return np.asarray(trajectory[0], dtype=np.float32)
    if mode == "nsrm":
        return _median_relation_trajectory(trajectory)
    raise ValueError(f"Unsupported relation mode: {mode}")


@torch.inference_mode()
def build_official_memory_spooled(
    model: Any,
    train_tensors: list[torch.Tensor],
    *,
    device: torch.device,
    scratch_dir: str | Path,
    return_selected_indices: bool = False,
    storage_backend: str = "disk",
    expected_candidate_rows: int | None = None,
    estimated_peak_bytes: int | None = None,
) -> VariantBuild:
    """Build exact upstream Official memory with RAM-first capable candidates.

    The default remains the historical disk spool. Opt-in ``auto`` chooses
    RAM only when the exact candidate allocation plus frozen headroom fits;
    candidate order and the Official sampler are identical across backends.
    """
    prototype_tensors = [
        tensor
        for index, tensor in enumerate(train_tensors)
        if index % int(model.threshold_fraction) != 0
    ]
    if not prototype_tensors:
        raise ValueError("No Official prototype images after the threshold split")
    layers = tuple(int(layer) for layer in model.layers)
    if storage_backend not in {"auto", "ram", "disk"}:
        raise ValueError(f"Unsupported Official candidate storage backend: {storage_backend}")
    if storage_backend == "auto":
        if estimated_peak_bytes is None:
            raise ValueError("Official auto storage requires estimated_peak_bytes")
        storage_decision = choose_candidate_storage_backend(int(estimated_peak_bytes))
        selected_backend = str(storage_decision["backend"])
    else:
        selected_backend = storage_backend
        storage_decision = {
            "backend": selected_backend,
            "estimated_peak_bytes": (
                int(estimated_peak_bytes) if estimated_peak_bytes is not None else None
            ),
            "ram_available_bytes": None,
            "ram_headroom_bytes": RAM_BACKEND_HEADROOM_BYTES,
            "ram_budget_source": "explicit",
            "reason": "explicit_backend",
        }
    available_bytes = storage_decision.get("ram_available_bytes")
    print(
        "[SuperADD-memory] candidate storage preflight "
        f"mode=official backend={selected_backend} "
        f"candidate_GiB={int(estimated_peak_bytes or 0) / float(1 << 30):.3f} "
        f"ram_available_GiB="
        f"{'unknown' if available_bytes is None else f'{int(available_bytes) / float(1 << 30):.3f}'} "
        f"headroom_GiB={int(storage_decision['ram_headroom_bytes']) / float(1 << 30):.3f} "
        f"reason={storage_decision['reason']}",
        flush=True,
    )

    scratch_context: tempfile.TemporaryDirectory[str] | None = None
    if selected_backend == "ram":
        if expected_candidate_rows is None or int(expected_candidate_rows) <= 0:
            raise ValueError("Official RAM storage requires expected_candidate_rows")
        feature_spools: dict[int, _RamMatrix | _MatrixSpool] = {
            layer: _RamMatrix(
                len(prototype_tensors),
                expected_rows=int(expected_candidate_rows),
            )
            for layer in layers
        }
    else:
        scratch_parent = Path(scratch_dir)
        scratch_parent.mkdir(parents=True, exist_ok=True)
        scratch_context = tempfile.TemporaryDirectory(
            prefix="official_", dir=scratch_parent,
        )
        scratch_root = Path(scratch_context.name)
        feature_spools = {
            layer: _MatrixSpool(scratch_root / f"features_layer_{layer}.f32")
            for layer in layers
        }

    for tensor in prototype_tensors:
        image = tensor.to(device)[None]
        processed = model.augmented_preprocessing(image)
        predictions = model.patch_exec(processed, model.backbone)
        if len(predictions) != len(layers):
            raise RuntimeError("Official patch execution layer count mismatch")
        for layer, embedding in zip(layers, predictions):
            values = np.asarray(embedding, dtype=np.float32)
            feature_spools[layer].append(values.reshape(-1, values.shape[-1]))
        del image, processed, predictions
        if device.type == "cuda":
            torch.cuda.empty_cache()

    candidate_bytes = sum(spool.nbytes for spool in feature_spools.values())
    print(
        f"[SuperADD-memory] candidate storage mode=official "
        f"backend={selected_backend} "
        f"candidate_GiB={candidate_bytes / float(1 << 30):.3f}",
        flush=True,
    )

    banks: dict[int, torch.Tensor] = {}
    selected_candidate_indices: dict[int, np.ndarray] | None = (
        {} if return_selected_indices else None
    )
    audits: dict[str, Any] = {}
    for layer in layers:
        features_np = feature_spools[layer].open()
        quota = min(int(model.max_database_size), int(features_np.shape[0]))
        if selected_candidate_indices is None:
            selected_values = _official_subsample_values(
                features_np, quota, device, seed=None, iterations=100,
            )
        else:
            selected_values, selected_candidate_indices[layer] = (
                _official_subsample_values_with_indices(
                    features_np, quota, device, seed=None, iterations=100,
                )
            )
        selected_features = np.array(selected_values, dtype=np.float32, order="C", copy=True)
        banks[layer] = torch.from_numpy(selected_features).to(device=device).contiguous()
        audits[str(layer)] = {
            "candidate_count": int(features_np.shape[0]),
            "memory_entries": int(selected_features.shape[0]),
            "label_basis": "unstratified_official_patch_features",
        }
        if isinstance(features_np, np.memmap):
            features_np._mmap.close()
        del selected_values, selected_features, features_np

    result = VariantBuild(
        banks=banks,
        metadata={
            "protocol": MEMORY_VARIANT_PROTOCOL,
            "mode": "official",
            "layers": list(layers),
            "memory_budget_per_layer": int(model.max_database_size),
            "candidate_feature_protocol": "upstream_patch_exec_and_augmented_preprocessing",
            "candidate_storage_backend": (
                "preallocated_float32_ram_matrix_per_layer"
                if selected_backend == "ram"
                else "append_only_float32_disk_memmap_per_layer"
            ),
            "candidate_storage_decision": storage_decision,
            "candidate_resident_bytes": int(candidate_bytes) if selected_backend == "ram" else 0,
            "candidate_scratch_bytes": int(candidate_bytes) if selected_backend == "disk" else 0,
            "candidate_scratch_peak_bytes": (
                int(candidate_bytes) if selected_backend == "disk" else 0
            ),
            "sampler": "upstream_subsampling_distance_based_fast",
            "sampler_iterations": 100,
            "sampler_distance_backend": "exact_query_chunked_torch_cdist_topk",
            "sampler_distance_matrix_max_bytes": int(SAMPLER_DISTANCE_MATRIX_MAX_BYTES),
            "layer_audits": audits,
        },
        selected_candidate_indices=selected_candidate_indices,
    )
    for spool in feature_spools.values():
        if isinstance(spool, _RamMatrix):
            spool.release()
    if scratch_context is not None:
        scratch_context.cleanup()
    return result

def build_memory_variant(
    model: Any,
    train_records: list[Any],
    *,
    config: dict[str, Any],
    mode: str,
    seed: int,
    device: torch.device,
    relation_aggregation: str = RELATION_AGGREGATION,
    cluster_count: int = 64,
    cluster_minimum: int = 32,
    scratch_dir: str | Path | None = None,
    storage_backend: str = "disk",
    estimated_peak_bytes: int | None = None,
    candidate_storage_decision: dict[str, Any] | None = None,
    expected_candidate_rows: int | None = None,
) -> VariantBuild:
    """Build a SuperADD-compatible bank for one category.

    ``mode=featstrat`` clusters the same official prototype embeddings;
    ``mode=nsrm_id`` clusters identity-only H1 attention relations;
    ``mode=nsrm`` clusters the coordinatewise median H1 attention relation
    over nine transforms. All modes retain the official database size and
    official distance-based sampler inside each stratum.
    """
    if mode not in CUSTOM_MODES:
        raise ValueError(f"Unsupported custom SuperADD memory mode: {mode}")
    train_prototypes = [record for index, record in enumerate(train_records) if index % int(config["threshold_fraction"]) != 0]
    if not train_prototypes:
        raise ValueError("No prototype images after the official threshold split")
    layers = tuple(int(layer) for layer in config["layers"])
    if storage_backend not in {"auto", "ram", "disk"}:
        raise ValueError(f"Unsupported candidate storage backend: {storage_backend}")
    if candidate_storage_decision is not None:
        storage_decision = dict(candidate_storage_decision)
        selected_backend = str(storage_decision.get("backend"))
        if selected_backend not in {"ram", "disk"}:
            raise ValueError("candidate_storage_decision has an invalid backend")
        if storage_backend != "auto" and storage_backend != selected_backend:
            raise ValueError("candidate storage backend/decision mismatch")
    elif storage_backend == "auto":
        storage_decision = (
            choose_candidate_storage_backend(int(estimated_peak_bytes))
            if estimated_peak_bytes is not None
            else {
                "backend": "disk",
                "estimated_peak_bytes": None,
                "ram_available_bytes": None,
                "ram_headroom_bytes": RAM_BACKEND_HEADROOM_BYTES,
                "ram_budget_source": "not_requested",
                "reason": "estimated_peak_unavailable",
            }
        )
        selected_backend = str(storage_decision["backend"])
    else:
        selected_backend = storage_backend
        storage_decision = {
            "backend": selected_backend,
            "estimated_peak_bytes": (
                int(estimated_peak_bytes) if estimated_peak_bytes is not None else None
            ),
            "ram_available_bytes": None,
            "ram_headroom_bytes": RAM_BACKEND_HEADROOM_BYTES,
            "ram_budget_source": "explicit",
            "reason": "explicit_backend",
        }
    scratch_context: tempfile.TemporaryDirectory[str] | None = None
    scratch_root: Path | None = None
    if selected_backend == "ram":
        feature_spools = {
            layer: _RamMatrix(
                len(train_prototypes), expected_rows=expected_candidate_rows
            )
            for layer in layers
        }
        relation_spools = {
            layer: _RamMatrix(
                len(train_prototypes), expected_rows=expected_candidate_rows
            )
            for layer in layers
        }
    else:
        scratch_parent = Path(scratch_dir) if scratch_dir is not None else Path.cwd() / ".superadd_memory_scratch"
        scratch_parent.mkdir(parents=True, exist_ok=True)
        scratch_context = tempfile.TemporaryDirectory(prefix=f"{mode}_", dir=scratch_parent)
        scratch_root = Path(scratch_context.name)
        feature_spools = {
            layer: _MatrixSpool(scratch_root / f"features_layer_{layer}.f32")
            for layer in layers
        }
        relation_spools = {
            layer: _MatrixSpool(scratch_root / f"relations_layer_{layer}.f32")
            for layer in layers
        }
    for image_index, record in enumerate(train_prototypes):
        raw = _raw_image(record.path).to(device=device, dtype=next(model.backbone.dino.parameters()).dtype)[None]
        augmented = model.augmented_preprocessing(raw)
        features, _ = patched_features_and_relations(
            model,
            augmented,
            relation_layers=(),
            aggregation=relation_aggregation,
        )
        for layer in layers:
            values = features[layer][0].detach().float().cpu().numpy()
            feature_spools[layer].append(values.reshape(-1, values.shape[-1]))
        if mode in RELATION_MODES:
            transform_ids = (
                (0,) if mode == "nsrm_id"
                else range(RELATION_TRANSFORM_COUNT)
            )
            trajectory: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
            for transform_id in transform_ids:
                transformed = _transformed_image(record.path, transform_id).to(
                    device=device, dtype=next(model.backbone.dino.parameters()).dtype
                )[None]
                processed = _deterministic_official_preprocessing(model, transformed)
                _, relations = patched_features_and_relations(
                    model,
                    processed,
                    relation_layers=layers,
                    aggregation=relation_aggregation,
                )
                for layer in layers:
                    values = relations[layer][0].detach().float().cpu().numpy()
                    trajectory[layer].append(values.reshape(-1, values.shape[-1]))
            for layer in layers:
                relation_spools[layer].append(
                    _reduce_relation_trajectory(mode, trajectory[layer])
                )
            del trajectory, relations, transformed, processed
        del raw, augmented, features
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scratch_bytes = sum(spool.nbytes for spool in feature_spools.values())
    scratch_bytes += sum(spool.nbytes for spool in relation_spools.values())
    normalization_peak_bytes = max(spool.nbytes for spool in feature_spools.values())
    scratch_peak_bytes = scratch_bytes + (normalization_peak_bytes if mode == "featstrat" else 0)
    print(
        f"[SuperADD-memory] candidate storage mode={mode} "
        f"backend={selected_backend} "
        f"candidate_GiB={scratch_bytes / float(1 << 30):.3f} "
        f"peak_candidate_GiB={scratch_peak_bytes / float(1 << 30):.3f}",
        flush=True,
    )

    banks: dict[int, torch.Tensor] = {}
    audits: dict[str, Any] = {}
    for layer in layers:
        features_np = feature_spools[layer].open()
        if mode == "featstrat":
            if selected_backend == "ram":
                normalized_features = _normalized_feature_array(features_np)
                labels = fit_feature_labels(normalized_features, seed, cluster_count)
                del normalized_features
            else:
                assert scratch_root is not None
                normalized_path = scratch_root / f"normalized_features_layer_{layer}.f32"
                normalized_features = _normalized_feature_spool(
                    features_np,
                    normalized_path,
                )
                labels = fit_feature_labels(normalized_features, seed, cluster_count)
                _close_memmap(normalized_features)
                del normalized_features
                normalized_path.unlink()
            label_basis = "official_patch_features"
        else:
            median_values = relation_spools[layer].open()
            if median_values.shape[0] != features_np.shape[0]:
                raise RuntimeError(f"NSRM relation/feature physical rows differ at layer {layer}")
            center = np.median(median_values, axis=0)
            q1 = np.percentile(median_values, 25, axis=0)
            q3 = np.percentile(median_values, 75, axis=0)
            scale = np.maximum(q3 - q1, 1e-8)
            normalized = np.clip((median_values - center) / scale, -5.0, 5.0)
            norms = np.linalg.norm(normalized, axis=-1, keepdims=True)
            if not np.isfinite(normalized).all() or np.any(norms < 1e-8):
                raise ValueError(f"NSRM relation normalization invalid at layer {layer}")
            relation_values = normalized / norms
            labels = fit_feature_labels(relation_values, seed, cluster_count)
            label_basis = (
                f"attention_relation_{relation_aggregation}_identity_transform_0"
                if mode == "nsrm_id"
                else f"attention_relation_{relation_aggregation}_median_over_{RELATION_TRANSFORM_COUNT}_transforms"
            )
        selected, audit = stratified_official_indices(
            features_np,
            labels,
            budget=int(config["max_database_size"]),
            minimum=int(cluster_minimum),
            seed=int(seed),
            device=device,
        )
        selected_features = np.ascontiguousarray(features_np[selected], dtype=np.float32)
        banks[layer] = torch.from_numpy(selected_features).to(device=device).contiguous()
        audits[str(layer)] = {
            "candidate_count": int(features_np.shape[0]),
            "memory_entries": int(selected.size),
            "label_basis": label_basis,
            **audit,
        }
        _close_memmap(features_np)
        if isinstance(feature_spools[layer], _RamMatrix):
            feature_spools[layer].release()
        if mode in RELATION_MODES:
            _close_memmap(median_values)
            if isinstance(relation_spools[layer], _RamMatrix):
                relation_spools[layer].release()
        del selected_features, selected, labels, features_np
        if mode in RELATION_MODES:
            del median_values, center, q1, q3, scale, normalized, norms, relation_values

    result = VariantBuild(
        banks=banks,
        metadata={
            "protocol": MEMORY_VARIANT_PROTOCOL,
            "mode": mode,
            "relation_aggregation": relation_aggregation if mode in RELATION_MODES else None,
            "relation_transform_ids": (
                [0]
                if mode == "nsrm_id"
                else list(range(RELATION_TRANSFORM_COUNT))
                if mode == "nsrm"
                else None
            ),
            "relation_reduction": (
                "identity_only"
                if mode == "nsrm_id"
                else "coordinatewise_median"
                if mode == "nsrm"
                else None
            ),
            "layers": list(layers),
            "memory_budget_per_layer": int(config["max_database_size"]),
            "cluster_count": int(cluster_count),
            "cluster_minimum": int(cluster_minimum),
            "official_threshold_split": f"index % {int(config['threshold_fraction'])} != 0",
            "candidate_feature_protocol": "official_augmented_preprocessing_shared_across_memory_modes",
            "transform_order": "raw_pil_transform_then_official_resize_and_normalize",
            "relation_trajectory_storage": (
                "per_image_identity_relation_streamed"
                if mode == "nsrm_id"
                else "per_image_median_streamed_semantics_identical"
                if mode == "nsrm"
                else None
            ),
            "candidate_storage_backend": (
                "preallocated_float32_ram_matrix_per_layer"
                if selected_backend == "ram"
                else "append_only_float32_disk_memmap_per_layer"
            ),
            "candidate_storage_decision": storage_decision,
            "candidate_resident_bytes": int(scratch_bytes) if selected_backend == "ram" else 0,
            "candidate_scratch_bytes": int(scratch_bytes) if selected_backend == "disk" else 0,
            "candidate_scratch_peak_bytes": int(scratch_peak_bytes) if selected_backend == "disk" else 0,
            "feature_normalization_backend": (
                "chunked_torch_F_normalize_to_float32_ram"
                if selected_backend == "ram"
                else "chunked_torch_F_normalize_to_float32_memmap"
            ),
            "sampler": "upstream_subsampling_distance_based_fast_per_stratum",
            "sampler_iterations_per_stratum": 1,
            "sampler_distance_backend": "exact_query_chunked_torch_cdist_topk",
            "sampler_distance_matrix_max_bytes": int(SAMPLER_DISTANCE_MATRIX_MAX_BYTES),
            "layer_audits": audits,
        },
    )
    if scratch_context is not None:
        scratch_context.cleanup()
    return result




