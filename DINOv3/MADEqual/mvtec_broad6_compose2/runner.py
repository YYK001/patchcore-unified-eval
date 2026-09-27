from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image

from DINOv3.MADEqual.aebad_s_broad6_compose2.augmentations import SEED
from DINOv3.nsrm.m0_scoring import prepare_native_mask_context
from DINOv3.relation_oracle_cache.extract import mad_equal
from DINOv3.relation_reliability.calibration import MADCalibrator, fit_calibrators
from DINOv3.relation_reliability.datasets import (
    Record,
    mvtec_ad_records,
    split_source_normal,
)
from DINOv3.relation_reliability.protocol import LAYERS, MVTEC_AD_CATEGORIES
from DINOv3.relation_reliability.superadd_host import (
    build_exact_official_memory,
    construct_exact_host,
    install_import_paths,
    interpolate_score,
    observe_pil,
    preprocessed_spatial_shape,
)

from .memory import (
    METHOD as BROAD6_METHOD,
    PROTOCOL_ID,
    broad6_plan,
    build_broad6_memory,
    load_memory as load_broad6_memory,
    save_memory as save_broad6_memory,
)
from .metrics import FAST_AUPRO_BACKEND, METRIC_NAMES, category_macro, evaluate_fast


OFFICIAL_METHOD = "OFFICIAL_MAD_EQUAL"
METHODS = (OFFICIAL_METHOD, BROAD6_METHOD)
CHECKPOINT_SHA256 = "7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488"
CACHE_REVISION = "mvtec_broad6_raw_v1"
PHASE_REVISION = "mvtec_broad6_global_barrier_range200_v2"


@dataclass(frozen=True)
class BaselineAssets:
    category_root: Path
    bank_path: Path | None
    medians: np.ndarray | None
    scales: np.ndarray | None
    patch_counts: tuple[int, ...] | None
    raw_by_image: dict[str, Path]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames: list[str] = []
    for row in values:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def _fit_method_mad(samples: dict[int, np.ndarray]) -> MADCalibrator:
    _ecdf, mad = fit_calibrators(samples)
    return mad


def _score_raw(raw: np.ndarray, mad: MADCalibrator) -> np.ndarray:
    score = mad_equal(raw, mad.medians, mad.scales)
    if score.shape != raw.shape[1:] or not np.isfinite(score).all():
        raise RuntimeError("invalid MAD-Equal patch map")
    return score.astype(np.float32, copy=False)


def _relative(record: Record, dataset_root: Path) -> str:
    return Path(record.path).resolve().relative_to(dataset_root.resolve()).as_posix()


def _expected_memory_entries(model: Any, train_records: Sequence[Record]) -> int:
    memory_records, _calibration_records = split_source_normal(
        train_records, int(model.threshold_fraction),
    )
    patch = int(model.patch_exec.model_patch_size)
    candidate_count = 0
    for record in memory_records:
        with Image.open(record.path) as image:
            height, width = preprocessed_spatial_shape(model, image.height, image.width)
        candidate_count += (int(height) // patch) * (int(width) // patch)
    if candidate_count <= 0:
        raise RuntimeError("Official candidate count is empty")
    return min(int(model.max_database_size), int(candidate_count))


def _load_mask(record: Record, output_shape: tuple[int, int]) -> np.ndarray:
    if record.label == 0:
        return np.zeros(output_shape, dtype=bool)
    if not record.mask_path:
        raise FileNotFoundError(f"MVTec anomaly mask is missing: {record.path}")
    with Image.open(record.mask_path) as source:
        mask = source.convert("L").resize(
            (output_shape[1], output_shape[0]), Image.Resampling.NEAREST,
        )
    return np.asarray(mask, dtype=np.uint8) > 0


def _validate_raw(
    raw: np.ndarray,
    output_shape: tuple[int, int],
    *,
    image_path: str,
    model: Any,
) -> None:
    with Image.open(image_path) as image:
        expected_output = (
            image.height // int(model.evaluation_downscale),
            image.width // int(model.evaluation_downscale),
        )
        spatial = preprocessed_spatial_shape(model, image.height, image.width)
    patch = int(model.patch_exec.model_patch_size)
    expected_grid = (spatial[0] // patch, spatial[1] // patch)
    if (
        raw.shape != (len(LAYERS), *expected_grid)
        or tuple(output_shape) != expected_output
        or not np.isfinite(raw).all()
    ):
        raise RuntimeError(f"invalid raw-distance cache for {image_path}")


def _read_external_raw(path: Path, image_path: str) -> tuple[np.ndarray, tuple[int, int]]:
    with np.load(path, allow_pickle=False) as payload:
        raw = np.asarray(payload["raw_distances"], dtype=np.float32)
        shape_key = (
            "evaluation_output_shape"
            if "evaluation_output_shape" in payload.files
            else "output_shape"
        )
        output_shape = tuple(int(value) for value in payload[shape_key].tolist())
        if "image_path" in payload.files:
            cached_image = str(payload["image_path"].item())
            if Path(cached_image).resolve() != Path(image_path).resolve():
                raise RuntimeError(f"baseline raw cache image mismatch: {path}")
    return raw, output_shape


def _observe_or_load(
    path: Path,
    *,
    category: str,
    method: str,
    relative_path: str,
    image_path: str,
    model: Any,
    device: torch.device,
    external_raw: Path | None = None,
    allow_observe: bool = True,
) -> tuple[np.ndarray, tuple[int, int]]:
    if path.is_file():
        with np.load(path, allow_pickle=False) as payload:
            expected = {
                "protocol_id": PROTOCOL_ID,
                "cache_revision": CACHE_REVISION,
                "category": category,
                "method": method,
                "relative_path": relative_path,
            }
            for key, value in expected.items():
                if str(payload[key].item()) != str(value):
                    raise RuntimeError(f"raw cache {key} mismatch: {path}")
            raw = np.asarray(payload["raw_distances"], dtype=np.float32)
            output_shape = tuple(int(value) for value in payload["output_shape"].tolist())
        _validate_raw(raw, output_shape, image_path=image_path, model=model)
        return raw, output_shape
    if external_raw is not None:
        raw, output_shape = _read_external_raw(external_raw, image_path)
    elif allow_observe:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            output_shape = (
                image.height // int(model.evaluation_downscale),
                image.width // int(model.evaluation_downscale),
            )
            observation = observe_pil(model, image, device=device)
        raw = np.asarray(observation.raw_distances, dtype=np.float32)
    else:
        raise RuntimeError(f"prediction cache is missing after category inference: {path}")
    _validate_raw(raw, output_shape, image_path=image_path, model=model)
    _atomic_npz(
        path,
        protocol_id=np.asarray(PROTOCOL_ID),
        cache_revision=np.asarray(CACHE_REVISION),
        category=np.asarray(category),
        method=np.asarray(method),
        relative_path=np.asarray(relative_path),
        raw_distances=raw,
        output_shape=np.asarray(output_shape, dtype=np.int32),
    )
    return raw, output_shape


def _baseline_raw_index(category_root: Path) -> dict[str, Path]:
    images = category_root / "images"
    if not images.is_dir():
        return {}
    output: dict[str, Path] = {}
    for path in images.rglob("*.npz"):
        try:
            with np.load(path, allow_pickle=False) as payload:
                if "image_path" not in payload.files or "raw_distances" not in payload.files:
                    continue
                image_path = str(payload["image_path"].item())
            output[str(Path(image_path).resolve())] = path
        except (OSError, ValueError, KeyError):
            continue
    return output


def load_baseline_assets(root: Path, category: str) -> BaselineAssets:
    category_root = root / "categories" / "mvtec_ad" / category
    complete_path = category_root / "complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(complete_path)
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("status") != "complete" or complete.get("category") != category:
        raise RuntimeError(f"incompatible MVTec baseline category: {category}")

    bank = category_root / "memory" / "official_banks.npz"
    bank_path = bank if bank.is_file() else None
    calibration = category_root / "calibration" / "calibration_state.npz"
    mad_state = category_root / "mad_state.npz"
    state_path = calibration if calibration.is_file() else mad_state
    medians: np.ndarray | None = None
    scales: np.ndarray | None = None
    patch_counts: tuple[int, ...] | None = None
    if state_path.is_file():
        with np.load(state_path, allow_pickle=False) as payload:
            medians = np.asarray(payload["mad_medians"], dtype=np.float32)
            scales = np.asarray(payload["mad_scales"], dtype=np.float32)
            if all(f"ecdf_{layer}" in payload.files for layer in LAYERS):
                patch_counts = tuple(int(payload[f"ecdf_{layer}"].size) for layer in LAYERS)
        if medians.shape != (4,) or scales.shape != (4,) or not np.isfinite(scales).all():
            raise RuntimeError(f"invalid baseline MAD state: {state_path}")
    return BaselineAssets(
        category_root=category_root,
        bank_path=bank_path,
        medians=medians,
        scales=scales,
        patch_counts=patch_counts,
        raw_by_image=_baseline_raw_index(category_root),
    )


def _install_npz_bank(
    path: Path,
    model: Any,
    device: torch.device,
    *,
    expected_entries: int,
) -> None:
    banks: dict[int, torch.Tensor] = {}
    with np.load(path, allow_pickle=False) as payload:
        for layer in LAYERS:
            value = np.asarray(payload[str(layer)], dtype=np.float32)
            if value.shape != (int(expected_entries), 1280) or not np.isfinite(value).all():
                raise RuntimeError(f"invalid baseline Official bank L{layer}: {path}")
            banks[layer] = torch.from_numpy(value).to(device=device)
    model.prototype_embeddings = banks


def _save_official_memory(
    path: Path, banks: dict[int, torch.Tensor], *, category: str, train_count: int,
    expected_entries: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "category": category,
        "method": OFFICIAL_METHOD,
        "train_count": int(train_count),
    }
    for layer in LAYERS:
        value = banks[layer]
        if tuple(value.shape) != (int(expected_entries), 1280):
            raise RuntimeError(
                f"Official L{layer} bank is not ({expected_entries},1280)"
            )
        state[f"bank_{layer}"] = value.detach().float().cpu()
    state["entries_per_layer"] = int(expected_entries)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _load_official_memory(
    path: Path, model: Any, *, category: str, train_count: int,
    expected_entries: int, device: torch.device,
) -> bool:
    if not path.is_file():
        return False
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (
        state.get("protocol_id") != PROTOCOL_ID
        or state.get("category") != category
        or state.get("method") != OFFICIAL_METHOD
        or int(state.get("train_count", -1)) != int(train_count)
        or int(state.get("entries_per_layer", -1)) != int(expected_entries)
    ):
        raise RuntimeError(f"incompatible rebuilt Official memory: {path}")
    banks: dict[int, torch.Tensor] = {}
    for layer in LAYERS:
        value = state.get(f"bank_{layer}")
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (int(expected_entries), 1280):
            raise RuntimeError(f"invalid rebuilt Official bank L{layer}: {path}")
        banks[layer] = value.to(device=device, dtype=torch.float32)
    model.prototype_embeddings = banks
    return True


def _calibrate(
    *,
    category: str,
    method: str,
    records: Sequence[Record],
    dataset_root: Path,
    cache_root: Path,
    model: Any,
    device: torch.device,
) -> tuple[MADCalibrator, list[dict[str, Any]]]:
    values: dict[int, list[np.ndarray]] = {layer: [] for layer in LAYERS}
    for index, record in enumerate(records, start=1):
        raw, _shape = _observe_or_load(
            cache_root / f"calibration_{index - 1:04d}.npz",
            category=category,
            method=method,
            relative_path=_relative(record, dataset_root),
            image_path=record.path,
            model=model,
            device=device,
        )
        for offset, layer in enumerate(LAYERS):
            values[layer].append(raw[offset].reshape(-1))
        print(
            f"[MVTEC-BROAD6] calibration {category} {method} {index}/{len(records)}",
            flush=True,
        )
    pooled = {layer: np.concatenate(values[layer]) for layer in LAYERS}
    mad = _fit_method_mad(pooled)
    rows = [
        {
            "category": category,
            "method": method,
            "layer": layer,
            "patch_count": int(pooled[layer].size),
            "median": float(mad.medians[offset]),
            "mad_scale_1p4826": float(mad.scales[offset]),
            "source": "current_unaugmented_calibration_raw",
        }
        for offset, layer in enumerate(LAYERS)
    ]
    return mad, rows


def _baseline_mad_rows(
    category: str, assets: BaselineAssets,
) -> tuple[MADCalibrator, list[dict[str, Any]]]:
    if assets.medians is None or assets.scales is None:
        raise RuntimeError(f"baseline MAD state is unavailable for {category}")
    mad = MADCalibrator(assets.medians, assets.scales)
    counts = assets.patch_counts or (0, 0, 0, 0)
    rows = [
        {
            "category": category,
            "method": OFFICIAL_METHOD,
            "layer": layer,
            "patch_count": int(counts[offset]),
            "median": float(mad.medians[offset]),
            "mad_scale_1p4826": float(mad.scales[offset]),
            "source": "completed_relation_reliability_baseline",
        }
        for offset, layer in enumerate(LAYERS)
    ]
    return mad, rows


def _mad_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    category: str,
    method: str,
) -> MADCalibrator:
    selected = sorted(
        (
            row for row in rows
            if str(row["category"]) == category and str(row["method"]) == method
        ),
        key=lambda row: int(row["layer"]),
    )
    by_layer = {int(row["layer"]): row for row in selected}
    if set(by_layer) != set(LAYERS):
        raise RuntimeError(f"calibration rows are incomplete: {category}/{method}")
    return MADCalibrator(
        np.asarray([float(by_layer[layer]["median"]) for layer in LAYERS], dtype=np.float32),
        np.asarray(
            [float(by_layer[layer]["mad_scale_1p4826"]) for layer in LAYERS],
            dtype=np.float32,
        ),
    )


def _phase_a_category(
    category: str,
    train_records: Sequence[Record],
    test_records: Sequence[Record],
    *,
    dataset_root: Path,
    output: Path,
    model: Any,
    device: torch.device,
    baseline_root: Path | None,
    rebuild_baseline: bool,
) -> None:
    started = time.perf_counter()
    category_root = output / "categories" / category
    barrier_path = category_root / "prediction_barrier.json"
    if barrier_path.is_file():
        barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
        if (
            barrier.get("status") != "complete"
            or barrier.get("protocol_id") != PROTOCOL_ID
            or barrier.get("phase_revision") != PHASE_REVISION
            or barrier.get("category") != category
            or int(barrier.get("official_prediction_count", -1)) != len(test_records)
            or int(barrier.get("broad6_prediction_count", -1)) != len(test_records)
        ):
            raise RuntimeError(f"incompatible category Phase-A barrier: {category}")
        for required in ("calibration_stats.csv", "broad6_manifest.csv", "runtime.csv"):
            if not (category_root / required).is_file():
                raise RuntimeError(f"Phase-A file is missing: {category_root / required}")
        print(f"[MVTEC-BROAD6] resume Phase A {category}", flush=True)
        return

    category_root.mkdir(parents=True, exist_ok=True)
    memory_records, calibration_records = split_source_normal(train_records, 8)
    expected_entries = _expected_memory_entries(model, train_records)
    runtime: list[dict[str, Any]] = []
    assets = load_baseline_assets(baseline_root, category) if baseline_root else None

    official_memory_path = category_root / "official" / "official_memory.pt"
    official_installed = False
    if assets and assets.bank_path:
        _install_npz_bank(
            assets.bank_path, model, device, expected_entries=expected_entries,
        )
        official_installed = True
    elif rebuild_baseline:
        if not _load_official_memory(
            official_memory_path, model, category=category,
            train_count=len(train_records), expected_entries=expected_entries,
            device=device,
        ):
            stage = time.perf_counter()
            np.random.seed(SEED)
            build = build_exact_official_memory(
                model,
                train_records,
                device=device,
                scratch_dir=category_root / "official" / "memory_scratch",
            )
            _save_official_memory(
                official_memory_path, build.banks,
                category=category, train_count=len(train_records),
                expected_entries=expected_entries,
            )
            runtime.append({
                "category": category, "stage": "official_memory_build",
                "seconds": time.perf_counter() - stage,
            })
        official_installed = True

    if assets and assets.medians is not None:
        official_mad, official_calibration_rows = _baseline_mad_rows(category, assets)
    else:
        if not official_installed:
            raise RuntimeError(
                f"Official calibration for {category} needs a baseline bank or --rebuild-baseline"
            )
        stage = time.perf_counter()
        official_mad, official_calibration_rows = _calibrate(
            category=category,
            method=OFFICIAL_METHOD,
            records=calibration_records,
            dataset_root=dataset_root,
            cache_root=category_root / "official" / "raw_cache" / "calibration",
            model=model,
            device=device,
        )
        runtime.append({
            "category": category, "stage": "official_calibration",
            "seconds": time.perf_counter() - stage,
        })

    # Official predictions are completed before any mask file is opened.
    stage = time.perf_counter()
    official_test_root = category_root / "official" / "raw_cache" / "test"
    for index, record in enumerate(test_records, start=1):
        external = (
            assets.raw_by_image.get(str(Path(record.path).resolve())) if assets else None
        )
        if external is None and not official_installed and not (
            official_test_root / f"test_{index - 1:04d}.npz"
        ).is_file():
            raise RuntimeError(
                f"Official raw map is unavailable for {category}/{Path(record.path).name}"
            )
        _observe_or_load(
            official_test_root / f"test_{index - 1:04d}.npz",
            category=category,
            method=OFFICIAL_METHOD,
            relative_path=_relative(record, dataset_root),
            image_path=record.path,
            model=model,
            device=device,
            external_raw=external,
            allow_observe=official_installed,
        )
        print(f"[MVTEC-BROAD6] test raw {category} Official {index}/{len(test_records)}", flush=True)
    runtime.append({
        "category": category, "stage": "official_test_raw",
        "seconds": time.perf_counter() - stage,
    })

    # Broad6 uses one transformed view per memory-fit image and its own bank/MAD.
    stage = time.perf_counter()
    broad6_memory_path = category_root / "broad6" / "broad6_memory.pt"
    plan = broad6_plan(train_records, dataset_root=dataset_root)
    metadata: dict[str, Any] = {"resumed": True}
    if not load_broad6_memory(
        broad6_memory_path,
        model,
        category=category,
        train_count=len(train_records),
        device=device,
    ):
        banks, built_manifest, metadata = build_broad6_memory(
            model,
            train_records,
            dataset_root=dataset_root,
            device=device,
            scratch_dir=category_root / "broad6" / "memory_scratch",
        )
        if built_manifest != plan:
            raise RuntimeError(f"Broad6 plan changed during construction: {category}")
        save_broad6_memory(
            broad6_memory_path,
            banks,
            category=category,
            train_count=len(train_records),
        )
    broad6_entries = {
        layer: int(model.prototype_embeddings[layer].shape[0]) for layer in LAYERS
    }
    if any(value != expected_entries for value in broad6_entries.values()):
        raise RuntimeError(
            f"Broad6 bank entries do not match min(candidate_count,100000): "
            f"{category} expected={expected_entries} actual={broad6_entries}"
        )
    runtime.append({
        "category": category, "stage": "broad6_memory_load_or_build",
        "seconds": time.perf_counter() - stage,
    })

    stage = time.perf_counter()
    broad6_mad, broad6_calibration_rows = _calibrate(
        category=category,
        method=BROAD6_METHOD,
        records=calibration_records,
        dataset_root=dataset_root,
        cache_root=category_root / "broad6" / "raw_cache" / "calibration",
        model=model,
        device=device,
    )
    runtime.append({
        "category": category, "stage": "broad6_calibration",
        "seconds": time.perf_counter() - stage,
    })

    stage = time.perf_counter()
    broad6_test_root = category_root / "broad6" / "raw_cache" / "test"
    for index, record in enumerate(test_records, start=1):
        _observe_or_load(
            broad6_test_root / f"test_{index - 1:04d}.npz",
            category=category,
            method=BROAD6_METHOD,
            relative_path=_relative(record, dataset_root),
            image_path=record.path,
            model=model,
            device=device,
        )
        print(f"[MVTEC-BROAD6] test raw {category} Broad6 {index}/{len(test_records)}", flush=True)
    runtime.append({
        "category": category, "stage": "broad6_test_raw",
        "seconds": time.perf_counter() - stage,
    })
    calibration_rows = [*official_calibration_rows, *broad6_calibration_rows]
    runtime.append({
        "category": category, "stage": "phase_a_total",
        "seconds": time.perf_counter() - started,
    })
    _write_csv(category_root / "calibration_stats.csv", calibration_rows)
    _write_csv(category_root / "broad6_manifest.csv", plan)
    _write_csv(category_root / "runtime.csv", runtime)
    _atomic_json(category_root / "memory_manifest.json", {
        "protocol_id": PROTOCOL_ID,
        "phase_revision": PHASE_REVISION,
        "category": category,
        "train_count": len(train_records),
        "memory_fit_count": len(memory_records),
        "calibration_count": len(calibration_records),
        "candidate_entry_cap": 100_000,
        "expected_entries_per_layer": expected_entries,
        "actual_broad6_entries_per_layer": {
            str(layer): broad6_entries[layer] for layer in LAYERS
        },
        "official_source": "baseline" if assets else "rebuilt_with_existing_builder",
        "broad6_view_count": len(plan),
        "broad6_official_brightness_applied": False,
        "broad6_clean_aug_union": False,
        "broad6_builder_metadata": metadata,
    })
    _atomic_json(barrier_path, {
        "protocol_id": PROTOCOL_ID,
        "phase_revision": PHASE_REVISION,
        "status": "complete",
        "category": category,
        "official_prediction_count": len(test_records),
        "broad6_prediction_count": len(test_records),
        "ground_truth_masks_opened": False,
    })


def _phase_b_category(
    category: str,
    test_records: Sequence[Record],
    *,
    dataset_root: Path,
    output: Path,
    model: Any,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    started = time.perf_counter()
    root_barrier_path = output / "prediction_barrier.json"
    if not root_barrier_path.is_file():
        raise RuntimeError("root prediction barrier is missing before Phase B")
    root_barrier = json.loads(root_barrier_path.read_text(encoding="utf-8"))
    if (
        root_barrier.get("status") != "complete"
        or root_barrier.get("protocol_id") != PROTOCOL_ID
        or root_barrier.get("phase_revision") != PHASE_REVISION
        or int(root_barrier.get("category_count", -1)) != 15
        or bool(root_barrier.get("ground_truth_masks_opened", True))
    ):
        raise RuntimeError("root prediction barrier is incompatible")

    category_root = output / "categories" / category
    complete_path = category_root / "complete.json"
    required = (
        "category_metrics.csv", "sample_scores.csv", "calibration_stats.csv",
        "broad6_manifest.csv", "runtime.csv",
    )
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if (
            complete.get("status") != "complete"
            or complete.get("protocol_id") != PROTOCOL_ID
            or complete.get("phase_revision") != PHASE_REVISION
            or complete.get("category") != category
        ):
            raise RuntimeError(f"incompatible completed category: {category}")
        if any(not (category_root / name).is_file() for name in required):
            raise RuntimeError(f"completed category files are incomplete: {category}")
        print(f"[MVTEC-BROAD6] resume Phase B {category}", flush=True)
        return tuple(_read_csv(category_root / name) for name in required)  # type: ignore[return-value]

    category_barrier = json.loads(
        (category_root / "prediction_barrier.json").read_text(encoding="utf-8")
    )
    if category_barrier.get("phase_revision") != PHASE_REVISION:
        raise RuntimeError(f"category Phase-A barrier is incompatible: {category}")
    calibration_rows = _read_csv(category_root / "calibration_stats.csv")
    official_mad = _mad_from_rows(
        calibration_rows, category=category, method=OFFICIAL_METHOD,
    )
    broad6_mad = _mad_from_rows(
        calibration_rows, category=category, method=BROAD6_METHOD,
    )
    plan = _read_csv(category_root / "broad6_manifest.csv")
    runtime: list[dict[str, Any]] = list(_read_csv(category_root / "runtime.csv"))

    # The root barrier proves that all 15 categories completed both prediction branches.
    stage = time.perf_counter()
    official_test_root = category_root / "official" / "raw_cache" / "test"
    broad6_test_root = category_root / "broad6" / "raw_cache" / "test"
    labels = [int(record.label) for record in test_records]
    masks: list[np.ndarray] = []
    maps = {method: [] for method in METHODS}
    sample_rows: list[dict[str, Any]] = []
    for index, record in enumerate(test_records):
        official_raw, official_shape = _observe_or_load(
            official_test_root / f"test_{index:04d}.npz",
            category=category,
            method=OFFICIAL_METHOD,
            relative_path=_relative(record, dataset_root),
            image_path=record.path,
            model=model,
            device=device,
            allow_observe=False,
        )
        broad6_raw, broad6_shape = _observe_or_load(
            broad6_test_root / f"test_{index:04d}.npz",
            category=category,
            method=BROAD6_METHOD,
            relative_path=_relative(record, dataset_root),
            image_path=record.path,
            model=model,
            device=device,
            allow_observe=False,
        )
        if official_shape != broad6_shape:
            raise RuntimeError(f"Official/Broad6 output shape mismatch: {record.path}")
        masks.append(_load_mask(record, official_shape))
        method_raw = {
            OFFICIAL_METHOD: (official_raw, official_mad),
            BROAD6_METHOD: (broad6_raw, broad6_mad),
        }
        for method, (raw, mad) in method_raw.items():
            final_map = interpolate_score(_score_raw(raw, mad), official_shape).astype(np.float32)
            maps[method].append(final_map)
            sample_rows.append({
                "category": category,
                "sample_index": index,
                "relative_path": _relative(record, dataset_root),
                "defect_type": record.shift,
                "label": record.label,
                "method": method,
                "image_score": float(final_map.max()),
            })
    mask_context = prepare_native_mask_context(masks, np.asarray(labels, dtype=np.int64))
    metric_rows: list[dict[str, Any]] = []
    for method in METHODS:
        metric_rows.append({
            "method": method,
            "category": category,
            "image_count": len(test_records),
            "normal_count": sum(label == 0 for label in labels),
            "anomaly_count": sum(label == 1 for label in labels),
            **evaluate_fast(
                labels,
                masks,
                maps[method],
                device=device,
                allow_cpu_fallback=False,
                mask_context=mask_context,
            ),
        })
    official_row = next(row for row in metric_rows if row["method"] == OFFICIAL_METHOD)
    for row in metric_rows:
        for metric in METRIC_NAMES:
            row[f"delta_vs_official_{metric}"] = float(row[metric]) - float(official_row[metric])
    runtime.append({
        "category": category, "stage": "fast_metrics",
        "seconds": time.perf_counter() - stage,
    })
    runtime.append({
        "category": category, "stage": "category_total",
        "seconds": time.perf_counter() - started,
    })

    _write_csv(category_root / "category_metrics.csv", metric_rows)
    _write_csv(category_root / "sample_scores.csv", sample_rows)
    _write_csv(category_root / "runtime.csv", runtime)
    _atomic_json(complete_path, {
        "protocol_id": PROTOCOL_ID,
        "phase_revision": PHASE_REVISION,
        "status": "complete",
        "category": category,
        "methods": list(METHODS),
        "prediction_count_per_method": len(test_records),
        "aupro_backend": FAST_AUPRO_BACKEND,
        "aupro_threshold_sampling": "linear_global_score_range_max_to_min_200",
        "component_connectivity": "4-connected_scipy_ndimage_default",
    })
    return metric_rows, sample_rows, calibration_rows, plan, runtime


def _report(category_rows: Sequence[dict[str, Any]], macro_rows: Sequence[dict[str, Any]]) -> str:
    official = next(row for row in macro_rows if row["method"] == OFFICIAL_METHOD)
    broad6 = next(row for row in macro_rows if row["method"] == BROAD6_METHOD)
    lines = [
        "# MVTec AD 15类 Official brightness 与 Broad6 Compose2 MAD-Equal 客观结果",
        "",
        f"Protocol: `{PROTOCOL_ID}`。每类独立 memory、calibration 和评价；macro 为15类非加权平均。",
        "",
        f"AUPRO backend: `{FAST_AUPRO_BACKEND}`；每个 method×category 在完整全局分数 "
        "[min,max] 范围生成固定200个线性阈值，使用严格 score > threshold；背景像素 "
        "pooled FPR、缺陷区域等权 PRO、FPR 上限0.3，连通域为 scipy.ndimage.label "
        "默认的4连通。",
        "",
        "## 15类 macro",
        "",
        "| Method | I-AUROC | I-AUPR | P-AUROC | P-AUPR | AUPRO@0.3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in (
        ("Official + MAD-Equal", official),
        ("Broad6 Compose2 + MAD-Equal", broad6),
    ):
        lines.append(
            f"| {label} | {row['image_AUROC']:.6f} | {row['image_AUPR']:.6f} | "
            f"{row['pixel_AUROC']:.6f} | {row['pixel_AUPR']:.6f} | "
            f"{row['AUPRO_at_0p3']:.6f} |"
        )
    lines.append(
        "| Delta | "
        + " | ".join(
            f"{float(broad6[metric]) - float(official[metric]):+.6f}"
            for metric in METRIC_NAMES
        )
        + " |"
    )
    lines += [
        "", "## 逐类别差值（Broad6−Official）", "",
        "| Category | Δ I-AUROC | Δ I-AUPR | Δ P-AUROC | Δ P-AUPR | Δ AUPRO@0.3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category in MVTEC_AD_CATEGORIES:
        row = next(
            item for item in category_rows
            if item["category"] == category and item["method"] == BROAD6_METHOD
        )
        lines.append(
            f"| {category} | "
            + " | ".join(f"{float(row[f'delta_vs_official_{metric}']):+.6f}" for metric in METRIC_NAMES)
            + " |"
        )
    lines += [
        "",
        "本报告仅陈述冻结协议下的数值结果，不作统计显著性或SOTA判断。",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal MVTec Broad6 Compose2 experiment requires CUDA")
    if args.baseline_output:
        baseline_root = Path(args.baseline_output).resolve()
        if not baseline_root.is_dir():
            raise FileNotFoundError(baseline_root)
    else:
        baseline_root = None
    if baseline_root is None and not args.rebuild_baseline:
        raise RuntimeError(
            "set --baseline-output or explicitly pass --rebuild-baseline; "
            "Official baseline is never rebuilt implicitly"
        )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    install_import_paths(
        Path(args.project_root), Path(args.official_root), Path(args.dinov3_root),
    )
    model, host_config = construct_exact_host(
        Path(args.official_root),
        Path(args.dinov3_root),
        Path(args.checkpoint),
        CHECKPOINT_SHA256,
    )
    train, evaluation = mvtec_ad_records(Path(args.mvtec_root))
    first = Path(train[MVTEC_AD_CATEGORIES[0]][0].path).resolve()
    dataset_root = first.parents[3]

    # Phase A is global: complete both prediction branches for all 15 categories
    # without opening any evaluation mask or consuming GT-derived metrics.
    for category in MVTEC_AD_CATEGORIES:
        print(f"[MVTEC-BROAD6] Phase A category {category}", flush=True)
        _phase_a_category(
            category,
            train[category],
            evaluation[category],
            dataset_root=dataset_root,
            output=output,
            model=model,
            device=device,
            baseline_root=baseline_root,
            rebuild_baseline=bool(args.rebuild_baseline),
        )
        model.prototype_embeddings = {}
        gc.collect()
        torch.cuda.empty_cache()

    total_predictions = sum(len(evaluation[category]) for category in MVTEC_AD_CATEGORIES)
    _atomic_json(output / "prediction_barrier.json", {
        "protocol_id": PROTOCOL_ID,
        "phase_revision": PHASE_REVISION,
        "status": "complete",
        "category_count": len(MVTEC_AD_CATEGORIES),
        "categories": list(MVTEC_AD_CATEGORIES),
        "official_prediction_count": total_predictions,
        "broad6_prediction_count": total_predictions,
        "ground_truth_masks_opened": False,
    })

    # Phase B starts only after the root barrier above exists.
    all_metrics: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    all_calibration: list[dict[str, Any]] = []
    all_manifest: list[dict[str, Any]] = []
    all_runtime: list[dict[str, Any]] = []
    for category in MVTEC_AD_CATEGORIES:
        print(f"[MVTEC-BROAD6] Phase B category {category}", flush=True)
        metrics, samples, calibration, manifest, runtime = _phase_b_category(
            category,
            evaluation[category],
            dataset_root=dataset_root,
            output=output,
            model=model,
            device=device,
        )
        all_metrics.extend(metrics)
        all_samples.extend(samples)
        all_calibration.extend(calibration)
        all_manifest.extend(manifest)
        all_runtime.extend(runtime)
        model.prototype_embeddings = {}
        gc.collect()
        torch.cuda.empty_cache()

    macro_rows = category_macro(all_metrics)
    official_macro = next(row for row in macro_rows if row["method"] == OFFICIAL_METHOD)
    for row in macro_rows:
        for metric in METRIC_NAMES:
            row[f"delta_vs_official_{metric}"] = (
                float(row[metric]) - float(official_macro[metric])
            )
    _write_csv(output / "broad6_manifest.csv", all_manifest)
    _write_csv(output / "calibration_stats.csv", all_calibration)
    _write_csv(output / "category_metrics.csv", all_metrics)
    _write_csv(output / "macro_metrics.csv", macro_rows)
    _write_csv(output / "sample_scores.csv", all_samples)
    _write_csv(output / "runtime.csv", all_runtime)
    (output / "objective_report.md").write_text(
        _report(all_metrics, macro_rows), encoding="utf-8",
    )
    _atomic_json(output / "complete.json", {
        "protocol_id": PROTOCOL_ID,
        "phase_revision": PHASE_REVISION,
        "status": "complete",
        "run_type": "formal",
        "dataset": "MVTec AD",
        "seed": SEED,
        "category_count": 15,
        "methods": list(METHODS),
        "baseline_output": str(baseline_root) if baseline_root else None,
        "rebuild_baseline": bool(args.rebuild_baseline),
        "aupro_backend": FAST_AUPRO_BACKEND,
        "aupro_threshold_sampling": "linear_global_score_range_max_to_min_200",
        "aupro_threshold_comparison": "score_strictly_greater_than_threshold",
        "aupro_component_connectivity": "4-connected_scipy_ndimage_default",
        "global_prediction_barrier": True,
        "host_config": host_config,
    })
    print(f"MVTEC_BROAD6_COMPOSE2_COMPLETE={output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MVTec AD Official brightness versus Broad6 Compose2 MAD-Equal",
    )
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--dinov3-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-output", default="")
    parser.add_argument("--rebuild-baseline", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
