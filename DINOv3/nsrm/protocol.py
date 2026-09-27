from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

NSRM_PROTOCOL = "DINOv3_NSRM_P0_source_normal_v5"
FROZEN_CATEGORIES = ("MetalParts", "PCB", "PiledBags")
FROZEN_SEED = 42
FROZEN_CLUSTER_COUNT = 64
FROZEN_BOOTSTRAP_SAMPLES = 2000
FROZEN_BOOTSTRAP_SEED = 20260715
FROZEN_USE_HALF = True
FROZEN_DINOV3_COMMIT = "346f38fee679c56a6888f91c51670fae61d364e0"
FROZEN_DINOV3_VITB16_SHA256 = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"
INPUT_SIZE = 512
PATCH_SIZE = 16
PATCH_GRID = 32
PATCH_COUNT = PATCH_GRID * PATCH_GRID
MAX_CANDIDATES_PER_CATEGORY = 50_000
MEMORY_BUDGET_PER_CATEGORY = 10_000
CALIBRATION_FRACTION = 0.20
TRANSFORM_VERSION = "nsrm_aligned_transforms_v1"
P0_BOOTSTRAP_TYPE = "conditional_fixed_natural_scale"
P0_MIN_MEMORY_FIT_IMAGES_PER_CATEGORY = 5
P0_MIN_VALID_IMAGES_PER_TRANSFORM = 5
P0_MIN_VALID_PATCH_PAIRS_PER_TRANSFORM = 256
P0_MIN_VALID_FRACTION_PER_TRANSFORM = 0.90
P0_MIN_NATURAL_BASELINE_IMAGES = 5
P0_MIN_NATURAL_BASELINE_DISTANCES = 256
P0_MIN_CLUSTER_CONSISTENCY_PAIRS = 256


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def record_fileset_sha256(records, root: str | Path, *, include_masks: bool = False) -> str:
    """Content and protocol-semantic identity for an ordered record set."""
    base = Path(root).resolve()
    hasher = hashlib.sha256()
    semantic_fields = ("category", "domain", "role", "shift", "label")

    def semantic_payload(record) -> dict[str, object]:
        return {
            field: getattr(record, field, None)
            for field in semantic_fields
        }

    ordered = sorted(
        records,
        key=lambda item: (
            str(Path(item.path).resolve()),
            json.dumps(semantic_payload(item), sort_keys=True, default=str),
        ),
    )
    for record in ordered:
        path = Path(record.path).resolve()
        relative = str(path.relative_to(base)).replace("\\", "/")
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(file_sha256(path).encode("ascii"))
        hasher.update(b"\0record_semantics\0")
        hasher.update(
            json.dumps(
                semantic_payload(record),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        if include_masks and getattr(record, "mask_path", None) is not None:
            mask_path = Path(record.mask_path).resolve()
            mask_relative = str(mask_path.relative_to(base)).replace("\\", "/")
            hasher.update(b"\0mask\0")
            hasher.update(mask_relative.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(file_sha256(mask_path).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def path_set_sha256(paths: Sequence[str | Path], root: str | Path) -> str:
    base = Path(root).resolve()
    hasher = hashlib.sha256()
    for path_value in sorted((Path(path).resolve() for path in paths), key=str):
        relative = str(path_value.relative_to(base)).replace("\\", "/")
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(file_sha256(path_value).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _dinov2_root(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    return root if (root / "nvs" / "conditional_nvs").is_dir() else root / "DINOv2"


def p0_builder_source_sha256(project_root: str | Path) -> str:
    root = Path(project_root).resolve()
    dinov2 = _dinov2_root(root)
    nsrm = root / "DINOv3" / "nsrm"
    paths = [
        nsrm / name
        for name in (
            "backbone.py",
            "p0_probe.py",
            "protocol.py",
            "relation.py",
            "transforms.py",
        )
    ]
    paths.append(dinov2 / "nvs" / "conditional_nvs" / "robustad_category_protocol.py")
    return path_set_sha256(paths, root)


def m0_execution_dependency_sha256(project_root: str | Path) -> str:
    root = Path(project_root).resolve()
    dinov2 = _dinov2_root(root)
    paths = list((root / "DINOv3" / "nsrm").glob("*.py"))
    paths.extend(
        dinov2 / "nvs" / "conditional_nvs" / name
        for name in ("memory.py", "metrics.py", "robustad_category_protocol.py")
    )
    return path_set_sha256(paths, root)


def _json_safe(value):
    """Return a strict-JSON-compatible value, mapping non-finite numbers to null."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def json_identity_sha256(payload) -> str:
    canonical = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_commit(repo: str | Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(repo)), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()

TRANSFORMS: tuple[dict[str, object], ...] = (
    {"name": "identity"},
    {"name": "brightness-low", "value": 0.8},
    {"name": "brightness-high", "value": 1.2},
    {"name": "gamma-low", "value": 0.8},
    {"name": "gamma-high", "value": 1.2},
    {"name": "white-balance-warm", "value": (1.10, 1.00, 0.90)},
    {"name": "white-balance-cool", "value": (0.90, 1.00, 1.10)},
    {"name": "gaussian-blur", "value": 1.0},
    {"name": "jpeg", "value": 70},
)


@dataclass(frozen=True)
class PhysicalPatch:
    category: str
    split: str
    image_relative_path: str
    patch_x: int
    patch_y: int
    transform_id: int
    seed: int
    transform_version: str = TRANSFORM_VERSION

    @property
    def physical_id(self) -> str:
        return (
            f"{self.category}|{self.split}|{self.image_relative_path}|"
            f"{self.patch_x}|{self.patch_y}"
        )

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        row["physical_id"] = self.physical_id
        return row


def stable_u64(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def transform_id_for(
    category: str,
    split: str,
    image_relative_path: str,
    patch_x: int,
    patch_y: int,
    seed: int,
) -> int:
    key = f"{category}|{split}|{image_relative_path}|{patch_x}|{patch_y}|{seed}"
    return stable_u64(key) % len(TRANSFORMS)


def _balanced_coordinates(
    category: str,
    split: str,
    seed: int,
    limit: int,
) -> list[tuple[int, int]]:
    coordinates = [(y, x) for y in range(PATCH_GRID) for x in range(PATCH_GRID)]
    if PATCH_COUNT <= limit:
        return coordinates
    # Stable coordinate-only ordering avoids a spatial left-strip bias while
    # keeping the selected physical coordinates identical across images.
    # The natural P0 baseline compares the same physical patch coordinate
    # across memory images, so the ordering must not depend on image_path.
    return sorted(
        coordinates,
        key=lambda item: stable_u64(f"{category}|{split}|{item[1]}|{item[0]}|{seed}"),
    )[:limit]


def build_transform_manifest(
    category: str,
    split: str,
    image_relative_paths: Sequence[str],
    seed: int,
    max_candidates: int = MAX_CANDIDATES_PER_CATEGORY,
) -> list[PhysicalPatch]:
    if not image_relative_paths:
        raise ValueError("Cannot build a manifest from zero images")
    if int(max_candidates) <= 0 or int(max_candidates) > MAX_CANDIDATES_PER_CATEGORY:
        raise ValueError(f"max_candidates must be in [1, {MAX_CANDIDATES_PER_CATEGORY}]")
    if int(max_candidates) < len(image_relative_paths):
        raise ValueError("max_candidates must allow at least one patch per image")
    total = len(image_relative_paths) * PATCH_COUNT
    rows: list[PhysicalPatch] = []
    image_paths = sorted(str(path) for path in image_relative_paths)
    if total <= max_candidates:
        budgets = [PATCH_COUNT] * len(image_paths)
    else:
        base = max_candidates // len(image_paths)
        remainder = max_candidates % len(image_paths)
        budgets = [base + (1 if index < remainder else 0) for index in range(len(image_paths))]
    for image_path, budget in zip(image_paths, budgets):
        for y, x in _balanced_coordinates(category, split, seed, budget):
            rows.append(
                PhysicalPatch(
                    category=str(category),
                    split=str(split),
                    image_relative_path=image_path,
                    patch_x=int(x),
                    patch_y=int(y),
                    transform_id=transform_id_for(
                        category, split, image_path, x, y, seed
                    ),
                    seed=int(seed),
                )
            )
    if len(rows) > max_candidates:
        rows = sorted(
            rows,
            key=lambda item: stable_u64(f"{item.physical_id}|{seed}"),
        )[:max_candidates]
        rows.sort(key=lambda item: (item.image_relative_path, item.patch_y, item.patch_x))
    ids = [row.physical_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Physical patch manifest contains duplicate IDs")
    return rows


def write_manifest(rows: Iterable[PhysicalPatch], path: str | Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty physical patch manifest")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].as_row().keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row.as_row() for row in rows)


def read_manifest(
    path: str | Path,
    expected_category: str | None = None,
    expected_split: str | None = None,
    expected_seed: int | None = None,
    expected_image_paths: set[str] | None = None,
) -> list[PhysicalPatch]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "category", "split", "image_relative_path", "patch_x", "patch_y",
            "transform_id", "seed", "transform_version", "physical_id",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Manifest missing fields: {missing}")
        rows = []
        for row in reader:
            item = PhysicalPatch(
                category=str(row["category"]),
                split=str(row["split"]),
                image_relative_path=str(row["image_relative_path"]),
                patch_x=int(row["patch_x"]),
                patch_y=int(row["patch_y"]),
                transform_id=int(row["transform_id"]),
                seed=int(row["seed"]),
                transform_version=str(row["transform_version"]),
            )
            if item.physical_id != str(row["physical_id"]):
                raise ValueError("Manifest physical_id does not match its coordinate fields")
            if not 0 <= item.transform_id < len(TRANSFORMS):
                raise ValueError("Manifest contains an invalid transform_id")
            if not 0 <= item.patch_x < PATCH_GRID or not 0 <= item.patch_y < PATCH_GRID:
                raise ValueError("Manifest contains an invalid patch coordinate")
            if item.transform_version != TRANSFORM_VERSION:
                raise ValueError("Manifest transform_version does not match the frozen protocol")
            if expected_category is not None and item.category != str(expected_category):
                raise ValueError("Manifest category does not match the requested category")
            if expected_split is not None and item.split != str(expected_split):
                raise ValueError("Manifest split does not match the requested split")
            if expected_seed is not None and item.seed != int(expected_seed):
                raise ValueError("Manifest seed does not match the requested seed")
            expected_transform = transform_id_for(
                item.category, item.split, item.image_relative_path,
                item.patch_x, item.patch_y, item.seed,
            )
            if item.transform_id != expected_transform:
                raise ValueError("Manifest transform_id does not match the frozen BLAKE2b assignment")
            if expected_image_paths is not None and item.image_relative_path not in expected_image_paths:
                raise ValueError("Manifest contains an image outside the current memory-fit split")
            rows.append(item)
    if not rows:
        raise ValueError("Manifest is empty")
    if len(rows) > MAX_CANDIDATES_PER_CATEGORY:
        raise ValueError("Manifest exceeds the frozen candidate budget")
    if len({item.physical_id for item in rows}) != len(rows):
        raise ValueError("Manifest contains duplicate physical IDs")
    return rows


def validate_manifest_matches_frozen_rule(
    rows: Sequence[PhysicalPatch],
    *,
    category: str,
    split: str,
    image_relative_paths: Sequence[str],
    seed: int,
    max_candidates: int = MAX_CANDIDATES_PER_CATEGORY,
) -> None:
    expected = build_transform_manifest(
        category,
        split,
        image_relative_paths,
        seed,
        max_candidates=max_candidates,
    )
    actual_rows = [row.as_row() for row in rows]
    expected_rows = [row.as_row() for row in expected]
    if actual_rows != expected_rows:
        raise ValueError(
            "Manifest does not exactly match the frozen balanced-coordinate/hash rule"
        )


def write_json(payload, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(output)
