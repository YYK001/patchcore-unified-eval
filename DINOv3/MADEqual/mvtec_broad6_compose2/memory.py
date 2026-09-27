from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from DINOv3.MADEqual.aebad_s_broad6_compose2.augmentations import (
    SEED,
    broad6_compose2,
    compose2_manifest_row,
)
from DINOv3.MADEqual.aebad_s_broad6_compose2.memory import _Broad6Preprocessing
from DINOv3.relation_reliability.datasets import Record, split_source_normal
from DINOv3.relation_reliability.protocol import LAYERS
from DINOv3.relation_reliability.superadd_host import (
    _official_candidate_resource_estimate,
    image_tensor,
)
from external_baselines.superadd_external.memory_variants import build_official_memory_spooled


PROTOCOL_ID = "mad_equal_mvtec_broad6_compose2_seed42_v1"
METHOD = "BROAD6_COMPOSE2_MAD_EQUAL"


def broad6_plan(
    train_records: Sequence[Record],
    *,
    dataset_root: Path,
    threshold_fraction: int = 8,
) -> list[dict[str, Any]]:
    memory_records, _calibration_records = split_source_normal(
        train_records, threshold_fraction=threshold_fraction,
    )
    root = Path(dataset_root).resolve()
    return [
        {
            "category": record.category,
            **compose2_manifest_row(
                Path(record.path).resolve().relative_to(root).as_posix(), seed=SEED,
            ),
        }
        for record in memory_records
    ]


def _validate_bank_shapes(
    banks: dict[int, torch.Tensor],
    *,
    expected_entries: int | None = None,
) -> dict[int, int]:
    entries: dict[int, int] = {}
    for layer in LAYERS:
        value = banks.get(layer)
        shape = None if value is None else tuple(value.shape)
        valid = (
            isinstance(value, torch.Tensor)
            and value.ndim == 2
            and int(value.shape[1]) == 1280
            and 0 < int(value.shape[0]) <= 100_000
            and (
                expected_entries is None
                or int(value.shape[0]) == int(expected_entries)
            )
        )
        if not valid:
            shape = None if value is None else tuple(value.shape)
            raise RuntimeError(
                f"Broad6 L{layer} bank shape is {shape}; expected "
                f"({expected_entries if expected_entries is not None else '1..100000'}, 1280)"
            )
        entries[layer] = int(value.shape[0])
    return entries


def build_broad6_memory(
    model: Any,
    train_records: Sequence[Record],
    *,
    dataset_root: Path,
    device: torch.device,
    scratch_dir: Path,
) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    """Use the existing Broad6 transform and Official sampler for one MVTec category."""
    threshold_fraction = int(model.threshold_fraction)
    memory_records, calibration_records = split_source_normal(
        train_records, threshold_fraction=threshold_fraction,
    )
    root = Path(dataset_root).resolve()
    tensors: list[torch.Tensor] = []
    manifest: list[dict[str, Any]] = []
    memory_paths = {Path(record.path).resolve() for record in memory_records}
    calibration_paths = {Path(record.path).resolve() for record in calibration_records}
    if memory_paths & calibration_paths:
        raise RuntimeError("MVTec memory-fit and calibration records overlap")

    for index, record in enumerate(train_records):
        if index % threshold_fraction == 0:
            # The Official builder owns the frozen index%threshold_fraction split.
            tensors.append(torch.empty(0))
            continue
        relative_path = Path(record.path).resolve().relative_to(root).as_posix()
        with Image.open(record.path) as source:
            result = broad6_compose2(source.convert("RGB"), relative_path, seed=SEED)
        tensors.append(image_tensor(result.image))
        manifest.append({"category": record.category, **result.manifest_row(relative_path)})

    if len(manifest) != len(memory_records):
        raise RuntimeError(
            f"Broad6 view count mismatch: {len(manifest)} != {len(memory_records)}"
        )
    expected_rows, estimated_bytes = _official_candidate_resource_estimate(model, tensors)
    original = model.augmented_preprocessing
    try:
        # This consumes and discards the upstream brightness draw, then applies
        # deterministic Official resize/normalization to the already transformed RGB.
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
    expected_entries = min(int(model.max_database_size), int(expected_rows))
    _validate_bank_shapes(build.banks, expected_entries=expected_entries)
    model.prototype_embeddings = build.banks
    return build.banks, manifest, build.metadata


def save_memory(
    path: Path,
    banks: dict[int, torch.Tensor],
    *,
    category: str,
    train_count: int,
) -> None:
    entries = _validate_bank_shapes(banks)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "category": category,
        "method": METHOD,
        "train_count": int(train_count),
        "entries_per_layer": {str(layer): entries[layer] for layer in LAYERS},
    }
    for layer in LAYERS:
        state[f"bank_{layer}"] = banks[layer].detach().float().cpu()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_memory(
    path: Path,
    model: Any,
    *,
    category: str,
    train_count: int,
    device: torch.device,
) -> bool:
    if not path.is_file():
        return False
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "category": category,
        "method": METHOD,
        "train_count": int(train_count),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(f"incompatible Broad6 memory cache {key}: {path}")
    banks = {layer: state.get(f"bank_{layer}") for layer in LAYERS}
    declared = state.get("entries_per_layer", {})
    actual = _validate_bank_shapes(banks)
    if declared and {
        str(layer): int(actual[layer]) for layer in LAYERS
    } != {str(key): int(value) for key, value in declared.items()}:
        raise RuntimeError(f"Broad6 memory entry metadata mismatch: {path}")
    model.prototype_embeddings = {
        layer: banks[layer].to(device=device, dtype=torch.float32) for layer in LAYERS
    }
    return True
