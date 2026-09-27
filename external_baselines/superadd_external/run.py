"""Run the custom SuperADD detector and memory variants on RobustAD."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from .dataset_adapter import Record, robustad_records
    from .metrics import CORE_METRICS, aggregate, file_hash, image_metrics, pixel_metrics, write_csv, write_json
    from .memory_variants import (
        MEMORY_VARIANT_PROTOCOL,
        build_memory_variant,
        choose_candidate_storage_backend,
    )
except ImportError:
    from dataset_adapter import Record, robustad_records
    from metrics import CORE_METRICS, aggregate, file_hash, image_metrics, pixel_metrics, write_csv, write_json
    from memory_variants import (
        MEMORY_VARIANT_PROTOCOL,
        build_memory_variant,
        choose_candidate_storage_backend,
    )


EXPECTED_OFFICIAL_COMMIT = "44cf25144442fbbc1334ea59d1632327a4376d1a"
EXPECTED_DINOV3_COMMIT = "346f38fee679c56a6888f91c51670fae61d364e0"
DEFAULT_SEED = 42
SUPPORTED_SENSITIVITY_SEEDS = (41, 42, 43)
IMPLEMENTATION = "custom_superadd_44cf251_robustad_memory_variants_v5_multiseed"
EVALUATION_ONLY_PROTOCOL = "SuperADD_RobustAD_existing_memory_evaluation_v1"
DINO_PATCH_SIZE = 16
DINO_VITH16PLUS_EMBED_DIM = 1280
H1_RELATION_DIM = 11
DISK_HEADROOM_BYTES = 10 << 30


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def read_git_head_without_git(repo: Path) -> str:
    head = (repo / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = repo / ".git" / head.removeprefix("ref: ")
        if ref.is_file(): return ref.read_text(encoding="utf-8").strip()
        packed = repo / ".git" / "packed-refs"
        if packed.is_file():
            reference = head.removeprefix("ref: ")
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith("#") and line.split()[-1] == reference: return line.split()[0]
        raise RuntimeError(f"Cannot resolve official repository HEAD: {head}")
    return head


def verify_clean_tracked_repo(repo: Path, expected_commit: str, name: str, allowed_modified_paths: tuple[str, ...] = ()) -> dict[str, Any]:
    commit = read_git_head_without_git(repo)
    if commit != expected_commit: raise RuntimeError(f"{name} commit mismatch: expected {expected_commit}, found {commit}")
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], check=True, capture_output=True, text=True)
    dirty_lines = [line for line in status.stdout.splitlines() if line.strip()]
    unexpected = [line for line in dirty_lines if line[3:].strip().replace("\\", "/") not in allowed_modified_paths]
    if unexpected: raise RuntimeError(f"{name} has unapproved tracked-file changes:\n" + "\n".join(unexpected))
    tracked = subprocess.run(["git", "-C", str(repo), "ls-files"], check=True, capture_output=True, text=True).stdout.splitlines()
    hashes = {relative: file_hash(repo / relative) for relative in sorted(tracked) if (repo / relative).is_file()}
    return {"commit": commit, "approved_modified_paths": list(allowed_modified_paths), "tracked_source_tree_hash": stable_hash(hashes), "tracked_file_hashes": hashes}


def verify_weight_hash(path: Path, approved_sha256: str) -> str:
    if len(approved_sha256) != 64 or any(character not in "0123456789abcdefABCDEF" for character in approved_sha256): raise ValueError("--weights-sha256 must be a 64-character hexadecimal hash")
    actual = file_hash(path)
    if actual.lower() != approved_sha256.lower(): raise RuntimeError(f"DINOv3 weight hash mismatch: expected {approved_sha256}, found {actual}")
    return actual


def load_records(data_root: Path, categories: list[str] | None):
    return robustad_records(data_root, categories)


def estimate_custom_candidate_resources(
    train_records: list[Record],
    config: dict[str, Any],
    mode: str,
) -> dict[str, int]:
    """Estimate the resident candidate peak for the frozen custom builders."""
    if mode not in {"featstrat", "nsrm_id", "nsrm"}:
        raise ValueError(f"Resource estimation does not support mode={mode}")
    if config["backbone"] != "dinov3_vith16plus":
        raise RuntimeError("Candidate estimator is pinned to DINOv3 ViT-H/16plus")
    resize_factor = float(config["patch_size"]) / 1024.0
    threshold_fraction = int(config["threshold_fraction"])
    prototype_patch_count = 0
    prototype_image_count = 0
    for index, record in enumerate(train_records):
        if index % threshold_fraction == 0:
            continue
        with Image.open(record.path) as image:
            resized_height = int(image.height * resize_factor)
            resized_width = int(image.width * resize_factor)
        if (
            resized_height < int(config["patch_size"])
            or resized_width < int(config["patch_size"])
        ):
            raise RuntimeError(
                f"Official patch execution would receive an undersized image: {record.path}"
            )
        prototype_patch_count += (
            resized_height // DINO_PATCH_SIZE
        ) * (
            resized_width // DINO_PATCH_SIZE
        )
        prototype_image_count += 1
    if prototype_patch_count <= 0:
        raise RuntimeError("No prototype patches available for candidate estimation")

    layer_count = len(config["layers"])
    float_bytes = np.dtype(np.float32).itemsize
    feature_layer_bytes = (
        prototype_patch_count * DINO_VITH16PLUS_EMBED_DIM * float_bytes
    )
    feature_bytes = feature_layer_bytes * layer_count
    relation_bytes = (
        prototype_patch_count * H1_RELATION_DIM * float_bytes * layer_count
        if mode in {"nsrm_id", "nsrm"}
        else 0
    )
    normalization_bytes = feature_layer_bytes if mode == "featstrat" else 0
    candidate_peak_bytes = feature_bytes + relation_bytes + normalization_bytes
    final_bank_bytes = (
        min(prototype_patch_count, int(config["max_database_size"]))
        * DINO_VITH16PLUS_EMBED_DIM
        * float_bytes
        * layer_count
    )
    return {
        "prototype_image_count": int(prototype_image_count),
        "prototype_patch_count": int(prototype_patch_count),
        "feature_candidate_bytes": int(feature_bytes),
        "relation_candidate_bytes": int(relation_bytes),
        "normalization_peak_bytes": int(normalization_bytes),
        "candidate_peak_bytes": int(candidate_peak_bytes),
        "final_bank_bytes": int(final_bank_bytes),
    }


def make_manifest(train: dict[str, list[Record]], evaluation: dict[str, list[Record]]) -> list[dict[str, Any]]:
    rows = []
    for category in train:
        for role, records in (("official_train_good", train[category]), ("evaluation", evaluation[category])):
            for record in records:
                row = record.__dict__.copy(); row["protocol_role"] = role; rows.append(row)
    return rows


def audit_rows(train: dict[str, list[Record]], evaluation: dict[str, list[Record]], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    resize_factor = config["patch_size"] / 1024
    for category in train:
        train_records = train[category]
        if not train_records: raise RuntimeError(f"No normal training images for {category}")
        threshold_count = sum(index % config["threshold_fraction"] == 0 for index in range(len(train_records)))
        prototype_count = len(train_records) - threshold_count
        if not threshold_count or not prototype_count: raise RuntimeError(f"Invalid official train/threshold split for {category}")
        train_shapes = []
        for record in train_records:
            with Image.open(record.path) as image: train_shapes.append((image.height, image.width))
        if any(int(height * resize_factor) < config["patch_size"] or int(width * resize_factor) < config["patch_size"] for height, width in train_shapes):
            raise RuntimeError(f"Official patch execution requires resized dimensions >= {config['patch_size']} for {category}")
        groups = sorted({(record.domain, record.shift) for record in evaluation[category]})
        for domain, shift in groups:
            group = [record for record in evaluation[category] if (record.domain, record.shift) == (domain, shift)]
            bad = [record for record in group if record.label == 1]
            capability = {record.mask_capability for record in group}
            if len(capability) != 1: raise RuntimeError(f"Mixed mask capability in {category}/{domain}/{shift}")
            capability_value = next(iter(capability))
            missing_masks = empty_raw = empty_project = empty_official = unreadable_masks = 0
            normal_mask_count = sum(record.mask_path is not None for record in group if record.label == 0)
            evaluation_too_small = 0
            for record in group:
                try:
                    with Image.open(record.path) as image: height, width = image.height, image.width
                except Exception as error:
                    raise RuntimeError(f"Unreadable evaluation image: {record.path}") from error
                evaluation_too_small += int(int(height * resize_factor) < config["patch_size"] or int(width * resize_factor) < config["patch_size"])
                if record.label != 1 or capability_value != "pixel": continue
                if not record.mask_path or not Path(record.mask_path).is_file(): missing_masks += 1; continue
                try:
                    raw = np.asarray(Image.open(record.mask_path).convert("L"), dtype=np.uint8) > 0
                    project = resized_mask(record.mask_path, (height // config["evaluation_downscale"], width // config["evaluation_downscale"]), 0)
                    official = resized_mask(record.mask_path, (height // config["evaluation_downscale"], width // config["evaluation_downscale"]), 1)
                    empty_raw += int(not raw.any()); empty_project += int(not project.any()); empty_official += int(not official.any())
                except Exception:
                    unreadable_masks += 1
            rows.append({
                "category": category, "domain": domain, "shift": shift,
                "official_train_good_images": len(train_records), "official_prototype_images": prototype_count,
                "official_threshold_images": threshold_count, "good_images": sum(record.label == 0 for record in group),
                "bad_images": len(bad), "mask_capability": capability_value, "missing_bad_masks": missing_masks,
                "empty_bad_masks_raw": empty_raw, "empty_bad_masks_project_resized": empty_project, "empty_bad_masks_official_resized": empty_official,
                "unreadable_bad_masks": unreadable_masks, "normal_mask_count": normal_mask_count, "evaluation_images_too_small": evaluation_too_small,
                "image_metrics_evaluable": len({record.label for record in group}) == 2,
                "pixel_metrics_evaluable": capability_value == "pixel" and bool(bad) and missing_masks == 0,
                "min_train_height": min(shape[0] for shape in train_shapes), "min_train_width": min(shape[1] for shape in train_shapes),
            })
    return rows


def add_official_import_paths(repo: Path) -> None:
    sys.path.insert(0, str(repo / "tracks" / "industrial" / "src"))
    sys.path.insert(1, str(repo / "utils"))


def package_versions() -> dict[str, str]:
    names = ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "scikit-image", "opencv-python", "tifffile", "Pillow", "dinov3")
    distribution_aliases = {
        "opencv-python": ("opencv-python", "opencv-python-headless"),
    }
    versions: dict[str, str] = {}
    for name in names:
        candidates = distribution_aliases.get(name, (name,))
        for distribution_name in candidates:
            try:
                versions[name] = importlib.metadata.version(distribution_name)
                if distribution_name != name:
                    versions[f"{name}_distribution"] = distribution_name
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        else:
            versions[name] = "source_checkout" if name == "dinov3" else "distribution_metadata_unavailable"
    return versions


def construct_model(config: dict[str, Any]):
    from industrial.model import SuperADD
    return SuperADD(
        backbone=config["backbone"], layers=config["layers"], resize_factor=config["patch_size"] / 1024,
        patch_size=config["patch_size"], patch_overlap=config["patch_overlap"], max_database_size=config["max_database_size"],
        threshold_fraction=config["threshold_fraction"], subsampling_iterations=config["subsampling_iterations"],
        threshold_percentile=config["threshold_percentile"], threshold_factor=config["threshold_factor"],
        evaluation_downscale=config["evaluation_downscale"], closing_radius=config["closing_radius"],
        closing_angles=config["closing_angles"], closing_lower_threshold=config["closing_lower_threshold"],
        binary_erosion=config["binary_erosion"], brightness_augmentation=config["brightness_augmentation"], device="cuda",
    )


def train_with_variable_shape_threshold_compat(model: Any, train_tensors: list[Any]) -> None:
    """Run official training with pooled percentiles for unequal map shapes.

    Official SuperADD passes its list of threshold anomaly maps directly to
    ``np.percentile``. NumPy accepts that only when every map has the same
    shape. RobustAD contains variable-resolution images, so normalize only the
    heterogeneous-list case to the equivalent pooled-pixel representation.
    """
    official_percentile = np.percentile

    def percentile_compat(values: Any, percentile: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(values, list) and values and all(isinstance(value, np.ndarray) for value in values):
            if len({value.shape for value in values}) > 1:
                values = np.concatenate([np.asarray(value).reshape(-1) for value in values])
        return official_percentile(values, percentile, *args, **kwargs)

    np.percentile = percentile_compat
    try:
        model.train(train_tensors)
    finally:
        np.percentile = official_percentile


def estimate_official_threshold(model: Any, threshold_tensors: list[Any], config: dict[str, Any]) -> float:
    """Estimate the official threshold after a memory-only replacement."""
    anomaly_maps = []
    for tensor in threshold_tensors:
        anomaly_map, _ = model.predict(tensor)
        finite_gate("threshold anomaly map", anomaly_map)
        anomaly_maps.append(np.asarray(anomaly_map).reshape(-1))
    if not anomaly_maps:
        raise RuntimeError("Official threshold split is empty")
    pooled = np.concatenate(anomaly_maps)
    threshold = float(np.percentile(pooled, config["threshold_percentile"]) * config["threshold_factor"])
    if not np.isfinite(threshold):
        raise RuntimeError("Memory variant threshold is non-finite")
    model.threshold = threshold
    return threshold


def install_pinned_local_dinov3_loader(dinov3_root: Path) -> None:
    """Keep upstream model.py unchanged while making its torch.hub call revision-locked."""
    import torch
    original = torch.hub.load
    def pinned_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir != "facebookresearch/dinov3": raise RuntimeError(f"Unexpected torch.hub repository: {repo_or_dir}")
        kwargs["source"] = "local"
        return original(str(dinov3_root), model, *args, **kwargs)
    torch.hub.load = pinned_load


def hash_tensor_state(module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous(); digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def verify_actual_dinov3_source(model) -> dict[str, str]:
    """Ensure torch.hub did not silently use code different from uv.lock's DINOv3."""
    import dinov3
    actual = Path(inspect.getsourcefile(type(model.backbone.dino))).resolve()
    parts = actual.parts; indices = [index for index, part in enumerate(parts) if part == "dinov3"]
    if not indices: raise RuntimeError(f"Cannot identify torch.hub DINOv3 source path: {actual}")
    relative = Path(*parts[indices[-1] + 1:]); installed = Path(dinov3.__file__).resolve().parent / relative
    if not installed.is_file(): raise RuntimeError(f"Locked installed DINOv3 source is missing counterpart for {actual}")
    actual_hash, installed_hash = file_hash(actual), file_hash(installed)
    if actual_hash != installed_hash: raise RuntimeError("torch.hub DINOv3 implementation differs from declared custom DINOv3 source")
    return {"torch_hub_source_logical_path": str(Path("dinov3") / relative), "torch_hub_source_sha256": actual_hash, "locked_installed_source_sha256": installed_hash}


def load_tensor(path: str):
    from torchvision.transforms import ToTensor
    return ToTensor()(Image.open(path).convert("RGB"))


def resized_mask(path: str | None, shape: tuple[int, int], interpolation: int) -> np.ndarray:
    import cv2
    if not path: return np.zeros(shape, dtype=bool)
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return cv2.resize(mask, shape[::-1], interpolation=interpolation) > 0


def save_prediction(output: Path, record: Record, anomaly_map: np.ndarray, binary: np.ndarray) -> None:
    import cv2
    import tifffile
    label_directory = "anomaly" if record.label == 1 else "normal"
    directory = output / "predictions" / record.category / record.domain / record.shift / label_directory
    directory.mkdir(parents=True, exist_ok=True); stem = Path(record.path).stem
    tifffile.imwrite(directory / f"{stem}.tiff", anomaly_map.astype(np.float16))
    cv2.imwrite(str(directory / f"{stem}_binary.png"), binary.astype(np.uint8))


def finite_gate(name: str, value: Any) -> None:
    array = np.asarray(value)
    if array.size == 0 or not np.isfinite(array).all(): raise RuntimeError(f"Completion gate failed: {name} is empty or non-finite")


def validate_model_source_dir(
    source_dir: Path,
    *,
    categories: list[str],
    memory_mode: str,
    config: dict[str, Any],
    weight_sha256: str,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Bind an evaluation-only run to completed, intact SuperADD memories."""
    source_dir = source_dir.resolve()
    complete_path = source_dir / "complete.json"
    resolved_path = source_dir / "resolved_config.json"
    if not complete_path.is_file() or not resolved_path.is_file():
        raise RuntimeError(f"Model source is missing complete.json or resolved_config.json: {source_dir}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    if complete.get("status") != "complete":
        raise RuntimeError(f"Model source is not complete: {source_dir}")
    if complete.get("memory_mode") != memory_mode:
        raise RuntimeError(
            f"Model source mode mismatch: expected {memory_mode}, found {complete.get('memory_mode')}"
        )
    if complete.get("dino_weight_sha256", "").lower() != weight_sha256.lower():
        raise RuntimeError("Model source DINOv3 checkpoint hash does not match the current run")
    if resolved.get("official_commit") != EXPECTED_OFFICIAL_COMMIT:
        raise RuntimeError("Model source official SuperADD commit does not match the frozen protocol")
    if resolved.get("seed") != seed:
        raise RuntimeError("Model source seed does not match the current experiment seed")
    if resolved.get("official_config") != config:
        raise RuntimeError("Model source official SuperADD configuration differs from the current configuration")

    models = {}
    for category in categories:
        stem = source_dir / "models" / category
        json_path = stem.with_suffix(".json")
        npz_path = stem.with_suffix(".npz")
        metadata_path = stem.with_suffix(".memory_variant.json")
        if not json_path.is_file() or not npz_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"Model source files are incomplete for {category}: {stem}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("category") != category or metadata.get("mode") != memory_mode:
            raise RuntimeError(f"Model source metadata identity mismatch for {category}")
        if metadata.get("model_json_sha256") != file_hash(json_path):
            raise RuntimeError(f"Model source JSON checksum mismatch for {category}")
        if metadata.get("prototype_npz_sha256") != file_hash(npz_path):
            raise RuntimeError(f"Model source NPZ checksum mismatch for {category}")
        if memory_mode in {"nsrm_id", "nsrm"} and metadata.get("relation_aggregation") != "H1":
            raise RuntimeError(f"NSRM model source is not the frozen H1 variant for {category}")
        configuration = json.loads(json_path.read_text(encoding="utf-8"))
        constructor = configuration.get("constructor", {})
        expected = {
            "backbone": config["backbone"],
            "layers": config["layers"],
            "threshold_fraction": config["threshold_fraction"],
            "resize_factor": config["patch_size"] / 1024,
            "patch_size": config["patch_size"],
            "patch_overlap": config["patch_overlap"],
            "max_database_size": config["max_database_size"],
            "subsampling_iterations": config["subsampling_iterations"],
            "threshold_percentile": config["threshold_percentile"],
            "threshold_factor": config["threshold_factor"],
            "evaluation_downscale": config["evaluation_downscale"],
            "closing_radius": config["closing_radius"],
            "closing_angles": config["closing_angles"],
            "closing_lower_threshold": config["closing_lower_threshold"],
            "binary_erosion": config["binary_erosion"],
            "brightness_augmentation": config["brightness_augmentation"],
        }
        normalized_constructor = dict(constructor)
        normalized_constructor.pop("device", None)
        if normalized_constructor != expected:
            raise RuntimeError(f"Model source constructor configuration mismatch for {category}")
        trained_threshold = float(configuration.get("trained", {}).get("threshold"))
        if not np.isfinite(trained_threshold) or not np.isclose(
            trained_threshold, float(metadata.get("official_threshold")), rtol=0.0, atol=0.0
        ):
            raise RuntimeError(f"Model source threshold identity mismatch for {category}")
        models[category] = {
            "model_json": str(json_path),
            "model_json_sha256": metadata["model_json_sha256"],
            "prototype_npz": str(npz_path),
            "prototype_npz_sha256": metadata["prototype_npz_sha256"],
            "memory_variant_metadata": str(metadata_path),
            "memory_variant_metadata_sha256": file_hash(metadata_path),
            "official_threshold": trained_threshold,
        }
    return {
        "protocol": EVALUATION_ONLY_PROTOCOL,
        "source_dir": str(source_dir),
        "complete_json_sha256": file_hash(complete_path),
        "resolved_config_sha256": file_hash(resolved_path),
        "source_full_fingerprint": complete.get("full_fingerprint"),
        "source_loaded_dinov3_tensor_state_sha256": complete.get("loaded_dinov3_tensor_state_sha256"),
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom SuperADD RobustAD transfer evaluation")
    parser.add_argument("command", choices=("audit", "run")); parser.add_argument("--dataset", required=True, choices=("robustad",))
    parser.add_argument("--data-root", required=True, type=Path); parser.add_argument("--official-root", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dinov3-root", type=Path, help="local DINOv3 checkout pinned to the custom experiment commit")
    parser.add_argument("--weights-sha256", help="approved SHA-256 of the DINOv3 checkpoint file")
    parser.add_argument("--categories", nargs="*"); parser.add_argument("--shifts", nargs="*")
    parser.add_argument("--max-eval-per-label-per-group", type=int, help="smoke-only deterministic cap applied after group construction")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED); parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--memory-mode", choices=("official", "featstrat", "nsrm_id", "nsrm"), default="official", help="memory-only variant; official keeps the upstream builder")
    parser.add_argument("--model-source-dir", type=Path, help="completed same-mode output whose models/*.json and *.npz are reused without rebuilding memory")
    args = parser.parse_args()
    if args.seed not in SUPPORTED_SENSITIVITY_SEEDS:
        raise ValueError(f"Seed must be one of the preregistered sensitivity seeds {SUPPORTED_SENSITIVITY_SEEDS}")
    official_root = args.official_root.resolve(); official_repo = verify_clean_tracked_repo(official_root, EXPECTED_OFFICIAL_COMMIT, "official SuperADD", ("config.json",))
    commit = official_repo["commit"]
    config = json.loads((official_root / "config.json").read_text(encoding="utf-8"))
    train, evaluation = load_records(args.data_root, args.categories)
    if args.shifts:
        requested = set(args.shifts); evaluation = {category: [record for record in records if record.shift in requested] for category, records in evaluation.items()}
        if any(not records for records in evaluation.values()): raise ValueError("Requested smoke shifts produced an empty category")
    if args.max_eval_per_label_per_group is not None:
        if args.max_eval_per_label_per_group <= 0: raise ValueError("--max-eval-per-label-per-group must be positive")
        limited = {}
        for category, records in evaluation.items():
            kept = []
            for domain, shift in sorted({(record.domain, record.shift) for record in records}):
                group = [record for record in records if (record.domain, record.shift) == (domain, shift)]
                for label in (0, 1): kept.extend([record for record in group if record.label == label][:args.max_eval_per_label_per_group])
            limited[category] = kept
        evaluation = limited
    manifest = make_manifest(train, evaluation)
    external_dir = Path(__file__).resolve().parent
    external_source_hashes = {path.name: file_hash(path) for path in (Path(__file__).resolve(), external_dir / "dataset_adapter.py", external_dir / "metrics.py", external_dir / "memory_variants.py")}
    protocol = {"implementation": IMPLEMENTATION, "official_commit": commit, "official_tracked_source_tree_hash": official_repo["tracked_source_tree_hash"], "external_source_hashes": external_source_hashes, "dataset": args.dataset, "seed": args.seed, "seed_protocol": "preregistered_sensitivity_seeds_41_42_43_seed42_is_primary", "official_config": config, "data_root": str(args.data_root.resolve()), "evaluation_protocol": "robustad_transfer_external_project_groups", "selected_shifts": args.shifts, "max_eval_per_label_per_group": args.max_eval_per_label_per_group, "model_source_dir": str(args.model_source_dir.resolve()) if args.model_source_dir is not None else None}
    fingerprint = {"manifest_hash": stable_hash(manifest), "config_hash": stable_hash(protocol)}; fingerprint["protocol_hash"] = stable_hash(fingerprint)
    rows = audit_rows(train, evaluation, config)
    if args.command == "audit":
        if args.output_dir.exists() and any(args.output_dir.iterdir()): raise RuntimeError("Audit output directory must be empty")
        args.output_dir.mkdir(parents=True, exist_ok=True); write_json(args.output_dir / "resolved_config.json", protocol); write_csv(args.output_dir / "dataset_manifest.csv", manifest); write_csv(args.output_dir / "dataset_audit.csv", rows); write_json(args.output_dir / "protocol_fingerprint.json", fingerprint); write_json(args.output_dir / "audit_complete.json", {"status": "audit_complete", **fingerprint}); return
    audit_marker = args.output_dir / "audit_complete.json"
    if not audit_marker.is_file() or json.loads(audit_marker.read_text(encoding="utf-8")).get("protocol_hash") != fingerprint["protocol_hash"]: raise RuntimeError("Run requires a matching completed audit")
    if (args.output_dir / "complete.json").exists(): raise RuntimeError("Completed output already exists")
    support_fields = ("missing_bad_masks", "empty_bad_masks_raw", "empty_bad_masks_project_resized", "empty_bad_masks_official_resized", "unreadable_bad_masks", "normal_mask_count", "evaluation_images_too_small")
    if any(not row["image_metrics_evaluable"] or any(row[field] for field in support_fields) for row in rows): raise RuntimeError("Dataset support gate failed")
    if args.dinov3_root is None or not args.weights_sha256: raise RuntimeError("Run requires --dinov3-root and --weights-sha256")
    dinov3_root = args.dinov3_root.resolve(); dinov3_repo = verify_clean_tracked_repo(dinov3_root, EXPECTED_DINOV3_COMMIT, "DINOv3")
    add_official_import_paths(official_root)
    import cv2
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("Official SuperADD run requires CUDA")
    weights_dir = Path(config["dino_weights_dir"]); weights_dir = weights_dir if weights_dir.is_absolute() else official_root / weights_dir
    weights = list(weights_dir.glob(f"{config['backbone']}_pretrain_*.pth"))
    if len(weights) != 1: raise RuntimeError(f"Expected exactly one official DINOv3 weight file, found {weights}")
    weight_hash = verify_weight_hash(weights[0], args.weights_sha256)
    install_pinned_local_dinov3_loader(dinov3_root)
    model_source = None
    if args.model_source_dir is not None:
        model_source = validate_model_source_dir(
            args.model_source_dir,
            categories=list(train),
            memory_mode=args.memory_mode,
            config=config,
            weight_sha256=weight_hash,
            seed=args.seed,
        )
    provenance = {**fingerprint, "memory_mode": args.memory_mode, "memory_variant_protocol": MEMORY_VARIANT_PROTOCOL, "execution_mode": "evaluation_only_existing_memory" if model_source else "build_and_evaluate", "model_source_identity": model_source, "official_superadd_repository": official_repo, "dinov3_repository": dinov3_repo, "dino_weight_path": str(weights[0].resolve()), "approved_dino_weight_sha256": args.weights_sha256.lower(), "dino_weight_sha256": weight_hash, "packages": package_versions(), "python": sys.version, "platform": platform.platform(), "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0)}
    provenance["full_fingerprint"] = stable_hash(provenance); write_json(args.output_dir / "environment.json", provenance)
    group_rows, category_rows, sample_rows, timing_rows = [], [], [], []
    for category in train:
        np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        candidate_backend = None
        candidate_storage_decision = None
        candidate_resources = None
        if not model_source and args.memory_mode != "official":
            candidate_resources = estimate_custom_candidate_resources(
                train[category], config, args.memory_mode
            )
            candidate_storage_decision = choose_candidate_storage_backend(
                candidate_resources["candidate_peak_bytes"]
            )
            candidate_backend = str(candidate_storage_decision["backend"])
            disk_free_bytes = int(shutil.disk_usage(args.output_dir).free)
            disk_working_bytes = (
                candidate_resources["candidate_peak_bytes"]
                if candidate_backend == "disk"
                else 0
            )
            disk_required_bytes = (
                max(disk_working_bytes, candidate_resources["final_bank_bytes"])
                + DISK_HEADROOM_BYTES
            )
            ram_available = candidate_storage_decision.get("ram_available_bytes")
            ram_available_text = (
                f"{int(ram_available) / float(1 << 30):.3f}"
                if ram_available is not None
                else "unknown"
            )
            print(
                f"[SuperADD-RobustAD] resource preflight {category} "
                f"mode={args.memory_mode} candidate_backend={candidate_backend} "
                f"candidate_peak_GiB={candidate_resources['candidate_peak_bytes'] / float(1 << 30):.3f} "
                f"ram_available_GiB={ram_available_text} "
                f"ram_headroom_GiB={int(candidate_storage_decision['ram_headroom_bytes']) / float(1 << 30):.3f} "
                f"final_bank_GiB={candidate_resources['final_bank_bytes'] / float(1 << 30):.3f} "
                f"disk_free_GiB={disk_free_bytes / float(1 << 30):.3f}",
                flush=True,
            )
            if disk_free_bytes < disk_required_bytes:
                raise RuntimeError(
                    f"Insufficient disk for {category} {args.memory_mode} with "
                    f"candidate_backend={candidate_backend}: "
                    f"required_free_GiB={disk_required_bytes / float(1 << 30):.3f}, "
                    f"available_free_GiB={disk_free_bytes / float(1 << 30):.3f}"
                )
        started = time.perf_counter()
        if model_source:
            from industrial.model import SuperADD
            model = SuperADD.from_disk(Path(model_source["models"][category]["model_json"]).with_suffix(""))
            model_load_seconds = time.perf_counter() - started
            build_seconds = 0.0
        else:
            model = construct_model(config)
            model_load_seconds = time.perf_counter() - started
        if "actual_dinov3_source" not in provenance:
            provenance["actual_dinov3_source"] = verify_actual_dinov3_source(model); provenance["loaded_dinov3_tensor_state_sha256"] = hash_tensor_state(model.backbone.dino); provenance["full_fingerprint"] = stable_hash({key: value for key, value in provenance.items() if key != "full_fingerprint"}); write_json(args.output_dir / "environment.json", provenance)
        if model_source:
            source_state = model_source.get("source_loaded_dinov3_tensor_state_sha256")
            if source_state and provenance["loaded_dinov3_tensor_state_sha256"] != source_state:
                raise RuntimeError("Loaded DINOv3 tensor state differs from the reused model source")
            expected_layers = {int(layer) for layer in config["layers"]}
            if set(model.prototype_embeddings) != expected_layers:
                raise RuntimeError(f"Reused memory layers mismatch for {category}")
            for layer, bank in model.prototype_embeddings.items():
                if int(bank.shape[0]) != int(config["max_database_size"]):
                    raise RuntimeError(f"Reused memory budget mismatch for {category}/layer{layer}")
        else:
            started = time.perf_counter()
            train_tensors = []
            variant_metadata = {"protocol": "official_superadd_memory", "mode": "official"}
            if args.memory_mode == "official":
                train_tensors = [load_tensor(record.path) for record in train[category]]
                train_with_variable_shape_threshold_compat(model, train_tensors)
            else:
                variant = build_memory_variant(
                    model,
                    train[category],
                    config=config,
                    mode=args.memory_mode,
                    seed=args.seed,
                    device=torch.device("cuda"),
                    relation_aggregation="H1",
                    cluster_count=64,
                    cluster_minimum=32,
                    scratch_dir=args.output_dir / "builder_scratch" / category,
                    storage_backend=candidate_backend,
                    estimated_peak_bytes=candidate_resources["candidate_peak_bytes"],
                    candidate_storage_decision=candidate_storage_decision,
                    expected_candidate_rows=candidate_resources["prototype_patch_count"],
                )
                model.prototype_embeddings = variant.banks
                threshold_records = [
                    load_tensor(record.path)
                    for index, record in enumerate(train[category])
                    if index % int(config["threshold_fraction"]) == 0
                ]
                estimate_official_threshold(model, threshold_records, config)
                del threshold_records
                variant_metadata = variant.metadata
            build_seconds = time.perf_counter() - started
            model_path = args.output_dir / "models" / category; model_path.parent.mkdir(parents=True, exist_ok=True); model.to_disk(model_path)
            write_json(model_path.with_suffix(".memory_variant.json"), {"category": category, "mode": args.memory_mode, "threshold_calibration_rule": "official_threshold_split_percentile_and_factor_reestimated_for_this_memory", "official_threshold": float(model.threshold), "prototype_npz_sha256": file_hash(model_path.with_suffix(".npz")), "model_json_sha256": file_hash(model_path.with_suffix(".json")), **variant_metadata})
        predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}; scoring_seconds = 0.0
        for index, record in enumerate(evaluation[category]):
            torch.cuda.synchronize(); started = time.perf_counter(); anomaly_map, binary = model.predict(load_tensor(record.path)); torch.cuda.synchronize(); scoring_seconds += time.perf_counter() - started
            finite_gate("anomaly map", anomaly_map); predictions[index] = (anomaly_map, binary)
            sample_rows.append({"category": category, "domain": record.domain, "shift": record.shift, "path": record.path, "label": record.label, "image_score_max_raw_map": float(np.max(anomaly_map)), "official_binary_has_foreground": bool(np.any(binary)), "official_threshold": float(model.threshold)})
            if not args.no_save_predictions: save_prediction(args.output_dir, record, anomaly_map, binary)
        category_groups = []
        for domain, shift in sorted({(record.domain, record.shift) for record in evaluation[category]}):
            indices = [index for index, record in enumerate(evaluation[category]) if (record.domain, record.shift) == (domain, shift)]
            labels = np.asarray([evaluation[category][index].label for index in indices]); scores = np.asarray([np.max(predictions[index][0]) for index in indices]); flags = np.asarray([np.any(predictions[index][1]) for index in indices])
            row = {"category": category, "domain": domain, "shift": shift, "official_seed": args.seed, "experiment_seed": args.seed, "official_threshold": float(model.threshold), **image_metrics(labels, scores, flags)}
            if evaluation[category][indices[0]].mask_capability == "pixel":
                maps = [predictions[index][0] for index in indices]; binaries = [predictions[index][1] for index in indices]
                masks = [resized_mask(evaluation[category][index].mask_path, maps[position].shape, cv2.INTER_NEAREST) for position, index in enumerate(indices)]
                official_masks = [resized_mask(evaluation[category][index].mask_path, maps[position].shape, cv2.INTER_LINEAR) for position, index in enumerate(indices)]
                flat_truth = np.concatenate([mask.ravel() for mask in masks])
                if not flat_truth.any() or flat_truth.all(): raise RuntimeError(f"Invalid pixel support in {category}/{domain}/{shift}")
                row.update(pixel_metrics(masks, maps, binaries, official_masks))
            for key, value in row.items():
                if isinstance(value, (int, float, np.floating)) and key not in {"official_seed"}: finite_gate(f"{category}/{domain}/{shift}/{key}", value)
            category_groups.append(row)
        group_rows.extend(category_groups); category_row = {"category": category, **aggregate(category_groups, "external_evaluation_group_macro_within_category", CORE_METRICS)}; category_rows.append(category_row)
        timing_rows.append({"category": category, "official_seed": args.seed, "experiment_seed": args.seed, "execution_mode": "evaluation_only_existing_memory" if model_source else "build_and_evaluate", "model_load_seconds": model_load_seconds, "memory_and_threshold_build_seconds": build_seconds, "gpu_scoring_ms_per_image": 1000 * scoring_seconds / len(evaluation[category]), "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "official_threshold": float(model.threshold), "memory_mode": args.memory_mode, **{f"memory_entries_layer_{layer}": int(model.prototype_embeddings[layer].shape[0]) for layer in config["layers"]}})
        del model, predictions
        if not model_source:
            del train_tensors
        torch.cuda.empty_cache()
    expected_groups = sum(len({(record.domain, record.shift) for record in records}) for records in evaluation.values())
    if len(group_rows) != expected_groups: raise RuntimeError("Completion gate failed: group count mismatch")
    write_csv(args.output_dir / "group_metrics.csv", group_rows); write_csv(args.output_dir / "category_metrics.csv", category_rows); write_csv(args.output_dir / "sample_scores.csv", sample_rows); write_csv(args.output_dir / "timing.csv", timing_rows)
    pixel_groups = [row for row in group_rows if "pixel_AUROC" in row]
    summary = {"memory_mode": args.memory_mode, "memory_variant_protocol": MEMORY_VARIANT_PROTOCOL, "execution_mode": "evaluation_only_existing_memory" if model_source else "build_and_evaluate", "model_source_identity": model_source, "external_evaluation_group_macro": aggregate(group_rows, "external_evaluation_group_macro", CORE_METRICS), "external_category_macro": aggregate(category_rows, "external_category_macro", CORE_METRICS), "evaluation_group_count": len(group_rows), "pixel_evaluable_group_count": len(pixel_groups), "core_comparison_metrics": list(CORE_METRICS), "external_image_score_definition": "maximum of the official raw SuperADD anomaly map", "normal_FP_definition": "fraction of normal images whose official postprocessed binary map contains foreground", "protocol_boundary": "RobustAD transfer evaluation with unchanged official SuperADD model and hyperparameters; not an upstream official benchmark"}
    write_json(args.output_dir / "metrics_summary.json", summary); write_json(args.output_dir / "complete.json", {"status": "complete", **provenance})


if __name__ == "__main__": main()
