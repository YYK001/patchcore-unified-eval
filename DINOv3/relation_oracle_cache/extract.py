from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from DINOv3.relation_reliability.calibration import ECDFCalibrator
from DINOv3.relation_reliability.datasets import (
    Record,
    mvtec_ad_records,
    robustad_full_records,
)
from DINOv3.relation_reliability.protocol import (
    LAYERS,
    MVTEC_AD_CATEGORIES,
    ROBUSTAD_CATEGORIES,
    SEED,
    file_sha256,
    stable_json_hash,
)
from DINOv3.relation_reliability.scoring import resolve_anchor_labels
from DINOv3.relation_reliability.superadd_host import (
    construct_exact_host,
    install_import_paths,
    mask_at_patch_grid,
    observe_pil,
)
from external_baselines.superadd_external.metrics import write_csv, write_json


PROTOCOL_ID = "relation_oracle_four_layer_cache_seed42_v2"
SOURCE_PROTOCOL_ID = "relation_reliability_exact_superadd_seed42_v6"
MEMORY_ENTRIES_PER_LAYER = 100_000


@dataclass(frozen=True)
class CategoryState:
    source_identity: str
    source_complete: dict[str, Any]
    bank_path: Path
    memory_entries: dict[int, int]
    relation_labels: dict[int, np.ndarray]
    shuffled_labels: dict[int, np.ndarray]
    ecdf: ECDFCalibrator
    mad_medians: np.ndarray
    mad_scales: np.ndarray


def mad_calibrated(
    raw_distances: np.ndarray,
    medians: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(raw_distances, dtype=np.float32)
    medians = np.asarray(medians, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != len(LAYERS):
        raise ValueError(f"raw distances must be [4,H,W], got {raw.shape}")
    if medians.shape != (len(LAYERS),) or scales.shape != (len(LAYERS),):
        raise ValueError("MAD parameters must contain one value per frozen layer")
    if not np.isfinite(raw).all() or not np.isfinite(medians).all():
        raise FloatingPointError("raw distances or MAD medians contain NaN/Inf")
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise FloatingPointError("MAD scales must be finite and positive")
    return (
        (raw - medians[:, None, None]) / scales[:, None, None]
    ).astype(np.float32, copy=False)


def mad_equal(
    raw_distances: np.ndarray,
    medians: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    return mad_calibrated(raw_distances, medians, scales).mean(axis=0)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, payload)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        # Deliberately uncompressed: GPU inference, not single-core compression,
        # remains the dominant stage. The cache is still only patch resolution.
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _atomic_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(
        temporary, format="PNG",
    )
    os.replace(temporary, path)


def _load_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _source_paths(
    source_output: Path,
    dataset: str,
    category: str,
) -> dict[str, Path]:
    root = source_output / "categories" / dataset / category
    return {
        "root": root,
        "complete": root / "complete.json",
        "banks": root / "memory" / "official_banks.npz",
        "sidecars": root / "memory" / "anchor_sidecars.npz",
        "calibration": root / "calibration" / "calibration_state.npz",
        "memory_provenance": root / "memory" / "memory_provenance.json",
    }


def load_category_state(
    source_output: Path,
    dataset: str,
    category: str,
) -> CategoryState:
    paths = _source_paths(source_output, dataset, category)
    for name, path in paths.items():
        if name != "root" and not path.is_file():
            raise FileNotFoundError(path)
    complete = json.loads(paths["complete"].read_text(encoding="utf-8"))
    if (
        complete.get("status") != "complete"
        or complete.get("protocol") != SOURCE_PROTOCOL_ID
        or complete.get("dataset") != dataset
        or complete.get("category") != category
        or int(complete.get("seed", -1)) != SEED
    ):
        raise RuntimeError(f"source v6 category identity mismatch: {category}")
    provenance = json.loads(paths["memory_provenance"].read_text(encoding="utf-8"))
    audits = provenance.get("official_builder", {}).get("layer_audits", {})
    memory_entries: dict[int, int] = {}
    for layer in LAYERS:
        audit = audits.get(str(layer), {})
        candidate_count = int(audit.get("candidate_count", -1))
        entries = int(audit.get("memory_entries", -1))
        expected_entries = min(candidate_count, MEMORY_ENTRIES_PER_LAYER)
        if candidate_count <= 0 or entries != expected_entries:
            raise RuntimeError(f"source memory budget mismatch: {category}/L{layer}")
        memory_entries[layer] = entries

    with np.load(paths["sidecars"], allow_pickle=False) as payload:
        relation = {
            layer: np.asarray(payload[f"relation_{layer}"], dtype=np.int64)
            for layer in LAYERS
        }
        shuffled = {
            layer: np.asarray(payload[f"shuffled_{layer}"], dtype=np.int64)
            for layer in LAYERS
        }
    for layer in LAYERS:
        if (
            relation[layer].shape != (memory_entries[layer],)
            or shuffled[layer].shape != (memory_entries[layer],)
        ):
            raise RuntimeError(f"source sidecar shape mismatch: {category}/L{layer}")

    with np.load(paths["calibration"], allow_pickle=False) as payload:
        ecdf = ECDFCalibrator({
            layer: np.asarray(payload[f"ecdf_{layer}"], dtype=np.float32)
            for layer in LAYERS
        })
        medians = np.asarray(payload["mad_medians"], dtype=np.float32)
        scales = np.asarray(payload["mad_scales"], dtype=np.float32)
    # Exercise the exact offline contract before starting expensive inference.
    mad_calibrated(
        np.zeros((len(LAYERS), 1, 1), dtype=np.float32),
        medians,
        scales,
    )
    source_identity = stable_json_hash({
        "protocol": SOURCE_PROTOCOL_ID,
        "complete": complete,
        "bank_size": paths["banks"].stat().st_size,
        "sidecars_sha256": file_sha256(paths["sidecars"]),
        "calibration_sha256": file_sha256(paths["calibration"]),
        "memory_provenance_sha256": file_sha256(paths["memory_provenance"]),
    })
    return CategoryState(
        source_identity=source_identity,
        source_complete=complete,
        bank_path=paths["banks"],
        memory_entries=memory_entries,
        relation_labels=relation,
        shuffled_labels=shuffled,
        ecdf=ecdf,
        mad_medians=medians,
        mad_scales=scales,
    )


def install_category_banks(
    model: Any,
    bank_path: Path,
    *,
    expected_entries: dict[int, int],
    device: torch.device,
) -> None:
    banks: dict[int, torch.Tensor] = {}
    with np.load(bank_path, allow_pickle=False) as payload:
        for layer in LAYERS:
            array = np.asarray(payload[str(layer)], dtype=np.float32)
            if (
                array.ndim != 2
                or array.shape[0] != expected_entries[layer]
                or not np.isfinite(array).all()
            ):
                raise RuntimeError(
                    f"invalid Official bank L{layer}: shape={array.shape}"
                )
            banks[layer] = torch.from_numpy(array).to(
                device=device, dtype=torch.float32,
            )
    model.prototype_embeddings = banks


def _record_identity(record: Record, source_identity: str) -> str:
    image = Path(record.path).resolve()
    mask_hash = None
    if record.mask_path:
        mask = Path(record.mask_path).resolve()
        if not mask.is_file():
            raise FileNotFoundError(mask)
        mask_hash = file_sha256(mask)
    return stable_json_hash({
        "protocol": PROTOCOL_ID,
        "source_identity": source_identity,
        "record": record.as_row(),
        "image_sha256": file_sha256(image),
        "mask_sha256": mask_hash,
    })


def _evaluation_mask(
    record: Record,
    output_shape: tuple[int, int],
) -> np.ndarray | None:
    if record.mask_capability != "pixel":
        return None
    if record.label == 0:
        return np.zeros(output_shape, dtype=np.uint8)
    if not record.mask_path:
        raise FileNotFoundError(f"pixel-evaluable anomaly lacks GT mask: {record.path}")
    with Image.open(record.mask_path) as mask:
        resized = mask.convert("L").resize(
            (output_shape[1], output_shape[0]),
            Image.Resampling.NEAREST,
        )
    return (np.asarray(resized, dtype=np.uint8) > 0).astype(np.uint8)


def _patch_mask(
    record: Record,
    grid: tuple[int, int],
) -> np.ndarray | None:
    if record.mask_capability != "pixel":
        return None
    if record.label == 0:
        return np.zeros(grid, dtype=np.uint8)
    if not record.mask_path:
        raise FileNotFoundError(f"pixel-evaluable anomaly lacks GT mask: {record.path}")
    with Image.open(record.mask_path) as mask:
        return mask_at_patch_grid(mask, grid).astype(np.uint8)


def _cache_paths(
    category_dir: Path,
    record: Record,
) -> tuple[Path, Path]:
    defect = f"{record.domain}__{record.shift}"
    stem = Path(record.path).stem
    cache = category_dir / "images" / defect / f"{stem}.npz"
    mask = category_dir / "masks_eval" / defect / f"{stem}.png"
    return cache, mask


def extract_record(
    model: Any,
    record: Record,
    state: CategoryState,
    *,
    category_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    cache_path, mask_path = _cache_paths(category_dir, record)
    identity = _record_identity(record, state.source_identity)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as payload:
            actual = str(payload["cache_identity_sha256"].item())
            raw = np.asarray(payload["raw_distances"], dtype=np.float32)
            top1 = np.asarray(payload["top1_indices"], dtype=np.int64)
            relation = np.asarray(payload["relation_labels"], dtype=np.int64)
            shuffled = np.asarray(payload["shuffled_labels"], dtype=np.int64)
            pixel_gt_available = bool(int(payload["pixel_gt_available"].item()))
            patch_truth = (
                np.asarray(payload["gt_mask_patch"], dtype=np.uint8)
                if pixel_gt_available else None
            )
            output_shape = tuple(
                int(value) for value in payload["evaluation_output_shape"].tolist()
            )
        if actual != identity:
            raise RuntimeError(f"stale cache identity: {cache_path}")
        status = "reused"
    else:
        image = _load_rgb(record.path)
        observation = observe_pil(model, image, device=device)
        raw = np.asarray(observation.raw_distances, dtype=np.float32)
        top1 = np.asarray(observation.top1_indices, dtype=np.int64)
        flat_ecdf = state.ecdf.transform(raw).reshape(len(LAYERS), -1)
        flat_top1 = top1.reshape(len(LAYERS), -1)
        relation = resolve_anchor_labels(
            state.relation_labels,
            flat_top1,
            flat_ecdf,
        ).reshape(observation.grid)
        shuffled = resolve_anchor_labels(
            state.shuffled_labels,
            flat_top1,
            flat_ecdf,
        ).reshape(observation.grid)
        patch_truth = _patch_mask(record, observation.grid)
        output_shape = (
            image.height // int(model.evaluation_downscale),
            image.width // int(model.evaluation_downscale),
        )
        _atomic_npz(
            cache_path,
            protocol=np.asarray(PROTOCOL_ID),
            source_protocol=np.asarray(SOURCE_PROTOCOL_ID),
            source_identity_sha256=np.asarray(state.source_identity),
            cache_identity_sha256=np.asarray(identity),
            image_id=np.asarray(
                f"{record.dataset}/{record.category}/{record.domain}/"
                f"{record.shift}/{Path(record.path).name}"
            ),
            image_path=np.asarray(str(Path(record.path).resolve())),
            dataset=np.asarray(record.dataset),
            category=np.asarray(record.category),
            domain=np.asarray(record.domain),
            defect_type=np.asarray(record.shift),
            label=np.asarray(record.label, dtype=np.int8),
            layers=np.asarray(LAYERS, dtype=np.int16),
            raw_distances=raw.astype(np.float32),
            top1_indices=top1.astype(np.int32),
            relation_labels=relation.astype(np.int16),
            shuffled_labels=shuffled.astype(np.int16),
            pixel_gt_available=np.asarray(
                int(patch_truth is not None), dtype=np.uint8,
            ),
            gt_mask_patch=(
                patch_truth.astype(np.uint8)
                if patch_truth is not None
                else np.empty((0, 0), dtype=np.uint8)
            ),
            original_image_shape=np.asarray(
                (image.height, image.width), dtype=np.int32,
            ),
            evaluation_output_shape=np.asarray(output_shape, dtype=np.int32),
        )
        evaluation_mask = _evaluation_mask(record, output_shape)
        if record.label == 1 and evaluation_mask is not None:
            _atomic_mask_png(mask_path, evaluation_mask)
        status = "created"

    if record.label == 1 and record.mask_capability == "pixel" and not mask_path.is_file():
        evaluation_mask = _evaluation_mask(record, output_shape)
        assert evaluation_mask is not None
        _atomic_mask_png(mask_path, evaluation_mask)

    if (
        raw.ndim != 3
        or raw.shape[0] != len(LAYERS)
        or top1.shape != raw.shape
        or relation.shape != raw.shape[1:]
        or shuffled.shape != relation.shape
        or not np.isfinite(raw).all()
    ):
        raise RuntimeError(f"invalid four-layer cache contract: {cache_path}")
    if patch_truth is not None and patch_truth.shape != relation.shape:
        raise RuntimeError(f"patch GT shape mismatch: {cache_path}")
    if top1.size and (int(top1.min()) < 0 or int(top1.max()) >= 100_000):
        raise RuntimeError(f"top-1 memory index is outside the frozen bank: {cache_path}")
    # Verify that the saved raw maps reconstruct finite MAD maps exactly.
    reconstructed = mad_calibrated(
        raw, state.mad_medians, state.mad_scales,
    )
    return {
        "category": record.category,
        "dataset": record.dataset,
        "domain": record.domain,
        "defect_type": record.shift,
        "label": int(record.label),
        "image_id": (
            f"{record.dataset}/{record.category}/{record.domain}/"
            f"{record.shift}/{Path(record.path).name}"
        ),
        "image_path": str(Path(record.path).resolve()),
        "cache_path": str(cache_path.resolve()),
        "cache_relative_path": cache_path.relative_to(category_dir).as_posix(),
        "cache_status": status,
        "cache_identity_sha256": identity,
        "cache_bytes": cache_path.stat().st_size,
        "patch_height": int(raw.shape[1]),
        "patch_width": int(raw.shape[2]),
        "evaluation_height": int(output_shape[0]),
        "evaluation_width": int(output_shape[1]),
        "gt_mask_eval_path": (
            str(mask_path.resolve())
            if record.label == 1 and record.mask_capability == "pixel" else ""
        ),
        "gt_mask_eval_relative_path": (
            mask_path.relative_to(category_dir).as_posix()
            if record.label == 1 and record.mask_capability == "pixel" else ""
        ),
        "gt_mask_eval_semantics": (
            "saved_png"
            if record.label == 1 and record.mask_capability == "pixel"
            else (
                "implicit_all_zero"
                if record.label == 0 and record.mask_capability == "pixel"
                else "pixel_ground_truth_unavailable"
            )
        ),
        "pixel_gt_available": patch_truth is not None,
        "mad_equal_min": float(reconstructed.mean(axis=0).min()),
        "mad_equal_max": float(reconstructed.mean(axis=0).max()),
    }


def _category_identity(
    dataset: str,
    category: str,
    records: Sequence[Record],
    state: CategoryState,
) -> str:
    return stable_json_hash({
        "protocol": PROTOCOL_ID,
        "dataset": dataset,
        "category": category,
        "source_identity": state.source_identity,
        "records": [
            {
                "row": record.as_row(),
                "image_size": Path(record.path).stat().st_size,
                "mask_size": (
                    Path(record.mask_path).stat().st_size
                    if record.mask_path else None
                ),
            }
            for record in records
        ],
    })


def run_category(
    model: Any,
    dataset: str,
    category: str,
    records: Sequence[Record],
    *,
    source_output: Path,
    output_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    category_dir = output_dir / "categories" / dataset / category
    complete_path = category_dir / "complete.json"
    manifest_path = category_dir / "manifest.csv"
    state = load_category_state(source_output, dataset, category)
    identity = _category_identity(dataset, category, records, state)
    if complete_path.is_file() or manifest_path.is_file():
        if not complete_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"partial category output requires inspection: {category}")
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("category_identity_sha256") != identity:
            raise RuntimeError(f"category resume identity mismatch: {category}")
        import csv

        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(records):
            raise RuntimeError(f"category manifest is incomplete: {category}")
        print(f"[relation-cache] resume complete {category}", flush=True)
        return rows

    category_dir.mkdir(parents=True, exist_ok=True)
    install_category_banks(
        model,
        state.bank_path,
        expected_entries=state.memory_entries,
        device=device,
    )
    _atomic_npz(
        category_dir / "mad_state.npz",
        protocol=np.asarray(PROTOCOL_ID),
        source_protocol=np.asarray(SOURCE_PROTOCOL_ID),
        source_identity_sha256=np.asarray(state.source_identity),
        dataset=np.asarray(dataset),
        category=np.asarray(category),
        layers=np.asarray(LAYERS, dtype=np.int16),
        mad_medians=state.mad_medians.astype(np.float32),
        mad_scales=state.mad_scales.astype(np.float32),
    )
    rows = []
    for index, record in enumerate(records, start=1):
        rows.append(extract_record(
            model,
            record,
            state,
            category_dir=category_dir,
            device=device,
        ))
        if index % max(1, len(records) // 10) == 0 or index == len(records):
            print(
                f"[relation-cache] {category} {index}/{len(records)}",
                flush=True,
            )
    write_csv(manifest_path, rows)
    _atomic_json(complete_path, {
        "status": "complete",
        "protocol": PROTOCOL_ID,
        "source_protocol": SOURCE_PROTOCOL_ID,
        "seed": SEED,
        "dataset": dataset,
        "category": category,
        "image_count": len(records),
        "category_identity_sha256": identity,
        "source_identity_sha256": state.source_identity,
        "layers": list(LAYERS),
        "saved_arrays": [
            "raw_distances",
            "top1_indices",
            "relation_labels",
            "shuffled_labels",
            "pixel_gt_available",
            "gt_mask_patch",
        ],
        "mad_reconstruction": (
            "(raw_distances - mad_medians[:,None,None]) / "
            "mad_scales[:,None,None]"
        ),
        "mad_equal_reconstruction": "mad_calibrated.mean(axis=0)",
    })
    del state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != SEED:
        raise ValueError(f"cache extraction freezes seed={SEED}")
    if not torch.cuda.is_available():
        raise RuntimeError("four-layer SuperADD extraction requires CUDA")
    source_complete = args.source_output_dir / "complete.json"
    if not source_complete.is_file():
        raise FileNotFoundError(source_complete)
    source = json.loads(source_complete.read_text(encoding="utf-8"))
    if (
        source.get("status") != "complete"
        or source.get("protocol") != SOURCE_PROTOCOL_ID
        or int(source.get("category_count", -1)) != 18
    ):
        raise RuntimeError("source output is not the completed frozen v6 experiment")
    mvtec_categories = tuple(args.mvtec_categories or MVTEC_AD_CATEGORIES)
    robustad_categories = tuple(args.robustad_categories or ROBUSTAD_CATEGORIES)
    unknown_mvtec = sorted(set(mvtec_categories) - set(MVTEC_AD_CATEGORIES))
    unknown_robustad = sorted(set(robustad_categories) - set(ROBUSTAD_CATEGORIES))
    if unknown_mvtec:
        raise ValueError(f"unknown MVTec AD categories: {unknown_mvtec}")
    if unknown_robustad:
        raise ValueError(f"unknown RobustAD categories: {unknown_robustad}")

    project_root = Path(__file__).resolve().parents[2]
    install_import_paths(project_root, args.official_root, args.dinov3_root)
    _seed_everything(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol": PROTOCOL_ID,
        "source_protocol": SOURCE_PROTOCOL_ID,
        "seed": SEED,
        "datasets": {
            "mvtec_ad": list(mvtec_categories),
            "robustad": list(robustad_categories),
        },
        "layers": list(LAYERS),
        "checkpoint_sha256": args.weights_sha256.lower(),
        "source_complete_sha256": file_sha256(source_complete),
        "cache_contract": (
            "per_image_patch_grid_raw_four_layer_1NN_distances_plus_"
            "top1_relation_shuffled_and_GT"
        ),
    }
    protocol["sha256"] = stable_json_hash(protocol)
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("output directory contains another extraction protocol")
    else:
        _atomic_json(protocol_path, protocol)
    _atomic_json(args.output_dir / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    })

    _mvtec_train, mvtec_evaluation = mvtec_ad_records(
        args.mvtec_root, mvtec_categories,
    )
    _robust_train, robust_evaluation = robustad_full_records(
        args.robustad_root, robustad_categories,
    )
    datasets = (
        ("mvtec_ad", mvtec_categories, mvtec_evaluation),
        ("robustad", robustad_categories, robust_evaluation),
    )
    model, _config = construct_exact_host(
        args.official_root,
        args.dinov3_root,
        args.checkpoint,
        args.weights_sha256,
    )
    device = torch.device("cuda")
    all_rows = []
    for dataset, categories, evaluation in datasets:
        for category in categories:
            print(f"[relation-cache] category {dataset}/{category}", flush=True)
            all_rows.extend(run_category(
                model,
                dataset,
                category,
                evaluation[category],
                source_output=args.source_output_dir,
                output_dir=args.output_dir,
                device=device,
            ))
    write_csv(args.output_dir / "manifest.csv", all_rows)
    result = {
        "status": "complete",
        "protocol": PROTOCOL_ID,
        "source_protocol": SOURCE_PROTOCOL_ID,
        "seed": SEED,
        "datasets": {
            "mvtec_ad": list(mvtec_categories),
            "robustad": list(robustad_categories),
        },
        "category_count": len(mvtec_categories) + len(robustad_categories),
        "image_count": len(all_rows),
        "layers": list(LAYERS),
        "manifest": str((args.output_dir / "manifest.csv").resolve()),
    }
    _atomic_json(args.output_dir / "complete.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract portable MVTec four-layer maps from frozen v6 banks",
    )
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--mvtec-root", type=Path, required=True)
    parser.add_argument("--robustad-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--dinov3-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mvtec-categories", nargs="*", default=None)
    parser.add_argument("--robustad-categories", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
