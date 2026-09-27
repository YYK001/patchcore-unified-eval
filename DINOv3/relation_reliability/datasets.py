from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from external_baselines.superadd_external.dataset_adapter import (
    Record as ExternalRecord,
    robustad_records,
)

from .protocol import MVTEC_AD_CATEGORIES, ROBUSTAD_CATEGORIES, THRESHOLD_FRACTION


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ROBUSTAD_OFFICIAL_GROUPS = {
    "MetalParts": {
        ("source", "source"),
        ("target", "background_1"), ("target", "background_2"),
        ("target", "lighting"), ("target", "position"),
        ("target", "rotation"), ("target", "scale"),
    },
    "PCB": {
        ("source", "source"), ("target", "lighting"),
        ("target", "white_balancing"), ("target", "rotation"),
        ("target", "position"), ("target", "shadow"),
    },
    "PiledBags": {
        ("source", "source"), ("target", "background_box_color"),
        ("target", "lighting"), ("target", "position_rotation"),
        ("target", "scale"), ("target", "shadow"),
    },
}


@dataclass(frozen=True)
class Record:
    path: str
    label: int
    mask_path: str | None
    category: str
    dataset: str
    domain: str
    shift: str
    role: str
    mask_capability: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _mvtec_root(root: Path) -> Path:
    for candidate in (root, root / "mvtec", root / "mvtec_ad"):
        if all((candidate / category).is_dir() for category in MVTEC_AD_CATEGORIES):
            return candidate
    return root


def mvtec_ad_records(
    root: Path,
    categories: Sequence[str] | None = None,
) -> tuple[dict[str, list[Record]], dict[str, list[Record]]]:
    root = _mvtec_root(Path(root))
    selected = tuple(categories or MVTEC_AD_CATEGORIES)
    unknown = sorted(set(selected) - set(MVTEC_AD_CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown MVTec AD categories: {unknown}")
    train: dict[str, list[Record]] = {}
    evaluation: dict[str, list[Record]] = {}
    for category in selected:
        category_root = root / category
        good_train = _images(category_root / "train" / "good")
        if not good_train:
            raise RuntimeError(f"MVTec AD train/good is empty for {category}")
        train[category] = [
            Record(
                path=str(path), label=0, mask_path=None, category=category,
                dataset="mvtec_ad", domain="source", shift="train_good",
                role="train", mask_capability="not_evaluation",
            )
            for path in good_train
        ]
        test_root = category_root / "test"
        if not test_root.is_dir():
            raise FileNotFoundError(test_root)
        records: list[Record] = []
        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            defect = defect_dir.name
            for image in _images(defect_dir):
                if defect == "good":
                    label, mask = 0, None
                else:
                    label = 1
                    mask_candidate = category_root / "ground_truth" / defect / f"{image.stem}_mask.png"
                    if not mask_candidate.is_file():
                        raise FileNotFoundError(mask_candidate)
                    mask = str(mask_candidate)
                records.append(
                    Record(
                        path=str(image), label=label, mask_path=mask,
                        category=category, dataset="mvtec_ad", domain="test",
                        shift=defect, role="evaluation", mask_capability="pixel",
                    )
                )
        if not records or {record.label for record in records} != {0, 1}:
            raise RuntimeError(f"MVTec AD evaluation support is incomplete for {category}")
        evaluation[category] = records
    return train, evaluation


def _convert_robust(record: ExternalRecord) -> Record:
    return Record(
        path=record.path,
        label=record.label,
        mask_path=record.mask_path,
        category=record.category,
        dataset="robustad",
        domain=record.domain,
        shift=record.shift,
        role=record.role,
        mask_capability=record.mask_capability,
    )


def robustad_full_records(
    root: Path,
    categories: Sequence[str] | None = None,
) -> tuple[dict[str, list[Record]], dict[str, list[Record]]]:
    selected = tuple(categories or ROBUSTAD_CATEGORIES)
    train, evaluation = robustad_records(Path(root), selected)
    converted_train = {
        category: [_convert_robust(record) for record in records]
        for category, records in train.items()
    }
    converted_evaluation = {
        category: [_convert_robust(record) for record in records]
        for category, records in evaluation.items()
    }
    for category, records in converted_evaluation.items():
        groups = {(record.domain, record.shift) for record in records}
        expected_groups = ROBUSTAD_OFFICIAL_GROUPS[category]
        if groups != expected_groups:
            raise RuntimeError(
                f"RobustAD/{category} official group mismatch: "
                f"missing={sorted(expected_groups - groups)}, "
                f"extra={sorted(groups - expected_groups)}"
            )
        for group in expected_groups:
            labels = {
                record.label for record in records
                if (record.domain, record.shift) == group
            }
            if labels != {0, 1}:
                raise RuntimeError(f"RobustAD/{category}/{group} lacks normal/anomaly image support")
    return converted_train, converted_evaluation


def split_source_normal(
    records: Sequence[Record],
    threshold_fraction: int = THRESHOLD_FRACTION,
) -> tuple[list[Record], list[Record]]:
    if threshold_fraction <= 1:
        raise ValueError("threshold_fraction must exceed one")
    calibration = [record for index, record in enumerate(records) if index % threshold_fraction == 0]
    memory = [record for index, record in enumerate(records) if index % threshold_fraction != 0]
    if not memory or not calibration:
        raise RuntimeError("source-normal split must contain memory and calibration images")
    if any(record.label != 0 for record in (*memory, *calibration)):
        raise RuntimeError("memory/calibration splits must be source-normal only")
    return memory, calibration


def manifest_rows(
    datasets: Iterable[
        tuple[str, dict[str, list[Record]], dict[str, list[Record]]]
    ],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_name, train, evaluation in datasets:
        for category in train:
            memory, calibration = split_source_normal(train[category])
            for protocol_role, records in (
                ("memory_fit_source_normal", memory),
                ("reliability_calibration_source_normal", calibration),
                ("evaluation", evaluation[category]),
            ):
                for record in records:
                    row = record.as_row()
                    row["dataset"] = dataset_name
                    row["protocol_role"] = protocol_role
                    rows.append(row)
    return rows


def audit_dataset(
    dataset_name: str,
    train: dict[str, list[Record]],
    evaluation: dict[str, list[Record]],
) -> list[dict[str, object]]:
    if set(train) != set(evaluation):
        raise RuntimeError(f"{dataset_name} train/evaluation category mismatch")
    rows: list[dict[str, object]] = []
    for category in train:
        memory, calibration = split_source_normal(train[category])
        groups = sorted({(record.domain, record.shift) for record in evaluation[category]})
        if not groups:
            raise RuntimeError(f"{dataset_name}/{category} has no evaluation groups")
        for domain, shift in groups:
            group = [
                record for record in evaluation[category]
                if (record.domain, record.shift) == (domain, shift)
            ]
            labels = {record.label for record in group}
            bad = [record for record in group if record.label == 1]
            missing_masks = sum(
                not record.mask_path or not Path(record.mask_path).is_file()
                for record in bad if record.mask_capability == "pixel"
            )
            rows.append({
                "dataset": dataset_name,
                "category": category,
                "domain": domain,
                "shift": shift,
                "memory_fit_normal_images": len(memory),
                "calibration_normal_images": len(calibration),
                "evaluation_images": len(group),
                "normal_images": sum(record.label == 0 for record in group),
                "anomaly_images": len(bad),
                "mask_capability": group[0].mask_capability,
                "missing_anomaly_masks": missing_masks,
                "image_metric_support": labels == {0, 1} if dataset_name == "robustad" else True,
            })
            if missing_masks:
                raise RuntimeError(f"Missing masks in {dataset_name}/{category}/{domain}/{shift}")
    return rows
