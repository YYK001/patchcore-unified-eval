from __future__ import annotations

import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from external_baselines.superadd_external.memory_variants import (
    VariantBuild,
    build_official_memory_spooled,
    patched_features_and_relations,
)

from .datasets import Record
from .protocol import (
    JOINT_RELATION_DIM,
    LAYERS,
    RELATION_AGGREGATION,
    file_sha256,
    validate_superadd_config,
)


_ORIGINAL_TORCH_HUB_LOAD = torch.hub.load
_LOCAL_DINOV3_ROOT: Path | None = None
_EXPECTED_CHECKPOINT: Path | None = None


def _local_dinov3_load(repo_or_dir: str, model: str, *args: Any, **kwargs: Any) -> Any:
    if repo_or_dir != "facebookresearch/dinov3":
        raise RuntimeError(f"unexpected torch.hub repository: {repo_or_dir}")
    if _LOCAL_DINOV3_ROOT is None or _EXPECTED_CHECKPOINT is None:
        raise RuntimeError("local DINOv3 loader was not configured")
    requested_weights = kwargs.get("weights")
    if requested_weights is None:
        raise RuntimeError("SuperADD did not provide a DINOv3 checkpoint path")
    actual_checkpoint = Path(requested_weights).resolve()
    if actual_checkpoint != _EXPECTED_CHECKPOINT:
        raise RuntimeError(
            "SuperADD resolved a different DINOv3 checkpoint: "
            f"expected {_EXPECTED_CHECKPOINT}, got {actual_checkpoint}"
        )
    kwargs["source"] = "local"
    return _ORIGINAL_TORCH_HUB_LOAD(
        str(_LOCAL_DINOV3_ROOT), model, *args, **kwargs,
    )


@dataclass(frozen=True)
class PatchObservation:
    raw_distances: np.ndarray
    top1_indices: np.ndarray
    grid: tuple[int, int]
    relation_signatures: np.ndarray | None = None
    layer23_features: np.ndarray | None = None


@dataclass
class _PendingObservation:
    raw_distances: np.ndarray
    top1_indices: np.ndarray
    grid: tuple[int, int]
    positions: np.ndarray
    queries: dict[int, torch.Tensor]
    relation_signatures: np.ndarray | None = None
    layer23_features: np.ndarray | None = None


def preprocessed_spatial_shape(
    model: Any,
    input_height: int,
    input_width: int,
) -> tuple[int, int]:
    """Return Official size, upscaling only when a side would be below 640."""
    resize_factor = float(model.preprocessing.resize_factor)
    base_height = int(int(input_height) * resize_factor)
    base_width = int(int(input_width) * resize_factor)
    minimum = int(model.patch_exec.patch_size)
    patch = int(model.patch_exec.model_patch_size)
    if min(base_height, base_width) >= minimum:
        return base_height, base_width
    scale = minimum / float(min(base_height, base_width))
    output_height = max(minimum, math.ceil(base_height * scale / patch) * patch)
    output_width = max(minimum, math.ceil(base_width * scale / patch) * patch)
    return int(output_height), int(output_width)


class _MinimumSpatialPreprocessing:
    def __init__(self, delegate: Any, model: Any) -> None:
        self._delegate = delegate
        self._model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("SuperADD preprocessing expects [B,C,H,W]")
        brightness = np.random.uniform(
            self._delegate.min_brightness_factor,
            self._delegate.max_brightness_factor,
        )
        x = torch.clip(x * brightness, 0, 1)
        shape = preprocessed_spatial_shape(
            self._model, int(x.shape[-2]), int(x.shape[-1]),
        )
        x = F.interpolate(
            x, size=shape, mode="bicubic", align_corners=False, antialias=True,
        )
        return (x - self._delegate.mean) / self._delegate.std


def install_minimum_spatial_preprocessing(model: Any) -> None:
    model.preprocessing = _MinimumSpatialPreprocessing(model.preprocessing, model)
    model.augmented_preprocessing = _MinimumSpatialPreprocessing(
        model.augmented_preprocessing, model,
    )


def deterministic_official_preprocessing(
    model: Any,
    image: torch.Tensor,
) -> torch.Tensor:
    """Official normalization without brightness RNG, using the shared size rule."""
    shape = preprocessed_spatial_shape(
        model, int(image.shape[-2]), int(image.shape[-1]),
    )
    resized = F.interpolate(
        image, size=shape, mode="bicubic", align_corners=False, antialias=True,
    )
    return (resized - model.preprocessing.mean) / model.preprocessing.std


def install_import_paths(project_root: Path, official_root: Path, dinov3_root: Path) -> None:
    paths = (
        project_root,
        official_root / "tracks" / "industrial" / "src",
        official_root / "utils",
        dinov3_root,
    )
    for path in reversed(paths):
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


def install_local_dinov3_loader(dinov3_root: Path, checkpoint: Path) -> None:
    """Install one process-wide, idempotent local DINOv3 loader."""
    global _LOCAL_DINOV3_ROOT, _EXPECTED_CHECKPOINT
    _LOCAL_DINOV3_ROOT = dinov3_root.resolve()
    _EXPECTED_CHECKPOINT = checkpoint.resolve()
    if torch.hub.load is not _local_dinov3_load:
        torch.hub.load = _local_dinov3_load


def construct_exact_host(
    official_root: Path,
    dinov3_root: Path,
    checkpoint: Path,
    approved_checkpoint_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    checkpoint = checkpoint.resolve()
    if file_sha256(checkpoint) != approved_checkpoint_sha256.lower():
        raise RuntimeError("DINOv3 ViT-H+/16 checkpoint SHA-256 mismatch")
    config = json.loads((official_root / "config.json").read_text(encoding="utf-8"))
    validate_superadd_config(config)
    install_local_dinov3_loader(dinov3_root, checkpoint)
    from external_baselines.superadd_external.run import construct_model

    model = construct_model(config)
    install_minimum_spatial_preprocessing(model)
    return model, config


def image_tensor(image: Image.Image) -> torch.Tensor:
    return ToTensor()(image.convert("RGB"))


def record_tensors(records: Sequence[Record]) -> list[torch.Tensor]:
    result = []
    for record in records:
        with Image.open(record.path) as image:
            result.append(image_tensor(image))
    return result


def _official_candidate_resource_estimate(
    model: Any,
    train_tensors: Sequence[torch.Tensor],
) -> tuple[int, int]:
    """Return exact stitched candidate rows and four-layer float32 bytes."""
    rows = 0
    prototype_count = 0
    patch = int(model.patch_exec.model_patch_size)
    for index, tensor in enumerate(train_tensors):
        if index % int(model.threshold_fraction) == 0:
            continue
        if tensor.ndim != 3:
            raise ValueError("Official train tensors must be [C,H,W]")
        height, width = preprocessed_spatial_shape(
            model, int(tensor.shape[-2]), int(tensor.shape[-1]),
        )
        rows += (int(height) // patch) * (int(width) // patch)
        prototype_count += 1
    if rows <= 0 or prototype_count <= 0:
        raise RuntimeError("Official candidate resource estimate is empty")
    embedding_dim = int(model.backbone.dino.embed_dim)
    candidate_bytes = rows * embedding_dim * np.dtype(np.float32).itemsize * len(model.layers)
    return int(rows), int(candidate_bytes)


def build_exact_official_memory(
    model: Any,
    train_records: Sequence[Record],
    *,
    device: torch.device,
    scratch_dir: Path,
) -> VariantBuild:
    tensors = record_tensors(train_records)
    expected_rows, candidate_bytes = _official_candidate_resource_estimate(
        model, tensors,
    )
    build = build_official_memory_spooled(
        model,
        tensors,
        device=device,
        scratch_dir=scratch_dir,
        return_selected_indices=True,
        storage_backend="auto",
        expected_candidate_rows=expected_rows,
        estimated_peak_bytes=candidate_bytes,
    )
    if build.selected_candidate_indices is None:
        raise RuntimeError("Exact Official memory builder did not return the requested sidecar")
    model.prototype_embeddings = build.banks
    del tensors
    gc.collect()
    return build


def _exact_top1_chunked(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    max_matrix_bytes: int = 1 << 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    if queries.ndim != 2 or keys.ndim != 2 or queries.shape[1] != keys.shape[1]:
        raise ValueError("1-NN inputs must be compatible rank-2 matrices")
    bytes_per_row = max(1, int(keys.shape[0]) * queries.element_size())
    chunk = max(1, min(int(queries.shape[0]), max_matrix_bytes // bytes_per_row))
    distances = torch.empty(queries.shape[0], device="cpu", dtype=torch.float32)
    indices = torch.empty(queries.shape[0], device="cpu", dtype=torch.int64)
    for start in range(0, int(queries.shape[0]), chunk):
        stop = min(start + chunk, int(queries.shape[0]))
        matrix = torch.cdist(
            queries[start:stop].float(), keys.float(),
            compute_mode="use_mm_for_euclid_dist",
        )
        values, positions = matrix.min(dim=1)
        distances[start:stop] = values.cpu()
        indices[start:stop] = positions.cpu()
        del matrix, values, positions
    return distances, indices


def _crop_count_for_image(model: Any, image: Image.Image) -> int:
    height, width = preprocessed_spatial_shape(model, image.height, image.width)
    input_rois_y, _, _ = model.patch_exec.axis_patch_split(height)
    input_rois_x, _, _ = model.patch_exec.axis_patch_split(width)
    return len(input_rois_y) * len(input_rois_x)


@torch.inference_mode()
def _extract_processed_batch(
    model: Any,
    processed: torch.Tensor,
    *,
    device: torch.device,
    query_masks: Sequence[np.ndarray | None],
    include_identity_descriptors: bool,
) -> list[_PendingObservation]:
    """Extract GPU-resident features without launching per-microbatch 1-NN."""
    batch = int(processed.shape[0])
    if len(query_masks) != batch:
        raise ValueError("query-mask count does not match observation batch")
    relation_layers = LAYERS if include_identity_descriptors else ()
    features, relations = patched_features_and_relations(
        model,
        processed,
        relation_layers=relation_layers,
        aggregation=RELATION_AGGREGATION,
    )
    if tuple(sorted(features)) != tuple(sorted(LAYERS)):
        raise RuntimeError("GPU-resident SuperADD feature layers are incomplete")

    first = features[LAYERS[0]]
    if first.ndim != 4 or int(first.shape[0]) != batch:
        raise RuntimeError(f"unexpected GPU-resident feature shape: {tuple(first.shape)}")
    _, height, width, _ = first.shape
    grid = (int(height), int(width))
    patch_count = grid[0] * grid[1]

    positions: list[np.ndarray] = []
    for mask in query_masks:
        if mask is None:
            value = np.ones(grid, dtype=bool)
        else:
            value = np.asarray(mask, dtype=bool)
            if value.shape != grid:
                raise ValueError(
                    f"1-NN query mask shape {value.shape} does not match patch grid {grid}"
                )
        flat_positions = np.flatnonzero(value.reshape(-1)).astype(np.int64, copy=False)
        if flat_positions.size == 0:
            raise ValueError("1-NN query mask selects no physical patches")
        positions.append(flat_positions)

    raw_output = np.zeros((batch, len(LAYERS), *grid), dtype=np.float32)
    top1_output = np.full((batch, len(LAYERS), *grid), -1, dtype=np.int64)
    pending: list[_PendingObservation] = []
    for image_index in range(batch):
        relation_array: np.ndarray | None = None
        feature_array: np.ndarray | None = None
        if include_identity_descriptors:
            relation_array = np.concatenate([
                relations[layer][image_index]
                .detach().float().cpu().numpy().reshape(-1, 11)
                for layer in LAYERS
            ], axis=1).astype(np.float32, copy=False)
            feature_array = (
                features[23][image_index]
                .detach().float().cpu().numpy().reshape(patch_count, -1)
            )
            if relation_array.shape != (patch_count, JOINT_RELATION_DIM):
                raise RuntimeError("identity relation descriptor shape mismatch")
        pending.append(_PendingObservation(
            raw_distances=raw_output[image_index],
            top1_indices=top1_output[image_index],
            grid=grid,
            positions=positions[image_index],
            queries={},
            relation_signatures=relation_array,
            layer23_features=feature_array,
        ))

    for layer in LAYERS:
        value = features[layer]
        if value.ndim != 4 or tuple(value.shape[:3]) != (batch, *grid):
            raise RuntimeError(f"unexpected layer {layer} feature shape: {tuple(value.shape)}")
        channels = int(value.shape[-1])
        flat = value.reshape(batch, patch_count, channels).to(dtype=torch.float32)
        for image_index, index in enumerate(positions):
            pending[image_index].queries[layer] = flat[image_index].index_select(
                0, torch.as_tensor(index, dtype=torch.long, device=flat.device),
            )
        del flat
    del features, relations
    return pending


def _resolve_pending_observations(
    model: Any,
    pending: Sequence[_PendingObservation],
) -> list[PatchObservation]:
    """Resolve one bounded observation group with one exact chunked 1-NN call per layer."""
    if not pending:
        return []
    for layer_offset, layer in enumerate(LAYERS):
        queries = [item.queries[layer] for item in pending]
        combined = torch.cat(queries, dim=0)
        channels = int(combined.shape[1])
        distances, indices = _exact_top1_chunked(
            combined, model.prototype_embeddings[layer],
        )
        cursor = 0
        for item, query in zip(pending, queries):
            stop = cursor + int(query.shape[0])
            item.raw_distances[layer_offset].reshape(-1)[item.positions] = (
                distances[cursor:stop].numpy() / float(channels)
            )
            item.top1_indices[layer_offset].reshape(-1)[item.positions] = (
                indices[cursor:stop].numpy()
            )
            cursor = stop
        del combined, distances, indices
    result = [
        PatchObservation(
            raw_distances=item.raw_distances,
            top1_indices=item.top1_indices,
            grid=item.grid,
            relation_signatures=item.relation_signatures,
            layer23_features=item.layer23_features,
        )
        for item in pending
    ]
    for item in pending:
        item.queries.clear()
    return result

@torch.inference_mode()
def observe_pils(
    model: Any,
    images: Sequence[Image.Image],
    *,
    device: torch.device,
    query_masks: Sequence[np.ndarray | None] | None = None,
    include_identity_descriptors: bool = False,
    max_single_crop_batch: int = 1,
) -> list[PatchObservation]:
    """Observe images with Official-equivalent per-image feature extraction.

    Exact 1-NN queries may still be resolved as one ordered transform group,
    but the DINO forward batch is frozen to one image because real ViT-H
    outputs are not numerically invariant to a change in batch size.
    """
    if not images:
        raise ValueError("at least one image is required")
    masks = list(query_masks) if query_masks is not None else [None] * len(images)
    if len(masks) != len(images):
        raise ValueError("query-mask count does not match image count")
    if max_single_crop_batch != 1:
        raise ValueError(
            "Exact SuperADD observation freezes DINO forward batch size to one"
        )

    pending: list[_PendingObservation] = []
    start = 0
    while start < len(images):
        image = images[start]
        crop_count = _crop_count_for_image(model, image)
        batch_limit = 1
        stop = start + 1
        while stop < min(len(images), start + batch_limit):
            candidate = images[stop]
            if candidate.size != image.size or _crop_count_for_image(model, candidate) != 1:
                break
            stop += 1
        tensor = torch.stack(
            [image_tensor(value) for value in images[start:stop]], dim=0,
        ).to(device)
        processed = model.preprocessing(tensor)
        pending.extend(_extract_processed_batch(
            model,
            processed,
            device=device,
            query_masks=masks[start:stop],
            include_identity_descriptors=include_identity_descriptors,
        ))
        del tensor, processed
        start = stop
    return _resolve_pending_observations(model, pending)


@torch.inference_mode()
def observe_pil(
    model: Any,
    image: Image.Image,
    *,
    device: torch.device,
    include_identity_descriptors: bool = False,
    query_mask: np.ndarray | None = None,
) -> PatchObservation:
    return observe_pils(
        model,
        [image],
        device=device,
        query_masks=[query_mask],
        include_identity_descriptors=include_identity_descriptors,
        max_single_crop_batch=1,
    )[0]

def interpolate_score(score: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    value = torch.from_numpy(np.asarray(score, dtype=np.float32))[None, None]
    return F.interpolate(value, size=output_shape, mode="bilinear", align_corners=False)[0, 0].numpy()


def mask_at_patch_grid(mask: Image.Image, grid: tuple[int, int]) -> np.ndarray:
    values = torch.from_numpy(
        (np.asarray(mask.convert("L"), dtype=np.uint8) > 0).astype(np.float32)
    )[None, None]
    pooled = F.adaptive_max_pool2d(values, output_size=grid)[0, 0]
    result = pooled.numpy() > 0
    if not result.any():
        raise RuntimeError("pseudo anomaly does not overlap a physical patch")
    return result
