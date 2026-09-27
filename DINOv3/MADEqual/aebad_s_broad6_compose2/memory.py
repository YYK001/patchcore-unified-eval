from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from DINOv3.relation_reliability.datasets import Record
from DINOv3.relation_reliability.protocol import LAYERS
from DINOv3.relation_reliability.superadd_host import (
    _official_candidate_resource_estimate,
    deterministic_official_preprocessing,
    image_tensor,
)
from external_baselines.superadd_external.memory_variants import build_official_memory_spooled

from .augmentations import SEED, broad6_compose2


PROTOCOL_ID = "mad_equal_aebad_s_broad6_compose2_seed42_v1"


class _Broad6Preprocessing:
    """Preserve the Official sampler RNG stream without applying brightness."""

    def __init__(self, model: Any, official_preprocessing: Any) -> None:
        self.model = model
        self.minimum = float(official_preprocessing.min_brightness_factor)
        self.maximum = float(official_preprocessing.max_brightness_factor)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # Upstream consumes one draw per memory-fit image before sampling. The
        # draw is deliberately discarded: Broad6 has already transformed RGB.
        np.random.uniform(self.minimum, self.maximum)
        return deterministic_official_preprocessing(self.model, image)


def build_broad6_memory(
    model: Any,
    train_records: Sequence[Record],
    *,
    dataset_root: Path,
    device: torch.device,
    scratch_dir: Path,
) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    """Build one four-layer 100K bank from one Compose2 view per memory-fit image."""
    tensors: list[torch.Tensor] = []
    manifest: list[dict[str, Any]] = []
    for index, record in enumerate(train_records):
        if index % int(model.threshold_fraction) == 0:
            tensors.append(torch.empty(0))  # filtered by the existing Official builder
            continue
        path = Path(record.path).resolve().relative_to(dataset_root).as_posix()
        with Image.open(record.path) as source:
            result = broad6_compose2(source.convert("RGB"), path, seed=SEED)
        tensors.append(image_tensor(result.image))
        manifest.append(result.manifest_row(path))

    if len(manifest) != 455:
        raise RuntimeError(f"Broad6 memory-fit count must be 455, got {len(manifest)}")
    expected_rows, estimated_bytes = _official_candidate_resource_estimate(model, tensors)
    original = model.augmented_preprocessing
    try:
        model.augmented_preprocessing = _Broad6Preprocessing(model, original)
        np.random.seed(SEED)
        build = build_official_memory_spooled(
            model,
            tensors,
            device=device,
            scratch_dir=scratch_dir,
            storage_backend="auto",
            expected_candidate_rows=expected_rows,
            estimated_peak_bytes=estimated_bytes,
        )
    finally:
        model.augmented_preprocessing = original
    for layer in LAYERS:
        bank = build.banks[layer]
        if tuple(bank.shape) != (100_000, 1280):
            raise RuntimeError(f"Broad6 L{layer} bank shape is {tuple(bank.shape)}, expected (100000, 1280)")
    model.prototype_embeddings = build.banks
    return build.banks, manifest, build.metadata


def save_memory(path: Path, banks: dict[int, torch.Tensor], *, source_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"protocol_id": PROTOCOL_ID, "source_count": int(source_count)}
    for layer in LAYERS:
        state[f"bank_{layer}"] = banks[layer].detach().cpu()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_memory(path: Path, model: Any, *, source_count: int, device: torch.device) -> bool:
    if not path.is_file():
        return False
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("protocol_id") != PROTOCOL_ID or int(state.get("source_count", -1)) != source_count:
        raise RuntimeError(f"incompatible Broad6 memory cache: {path}")
    banks: dict[int, torch.Tensor] = {}
    for layer in LAYERS:
        value = state.get(f"bank_{layer}")
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (100_000, 1280):
            raise RuntimeError(f"invalid Broad6 memory cache L{layer}: {path}")
        banks[layer] = value.to(device=device, dtype=torch.float32)
    model.prototype_embeddings = banks
    return True
