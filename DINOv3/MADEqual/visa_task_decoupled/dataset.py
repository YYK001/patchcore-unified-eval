from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath

from DINOv3.relation_reliability.datasets import Record


VISA_CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)
VISA_FIELDS = ("object", "split", "label", "image", "mask")


def find_visa_root(root: str | Path) -> Path:
    supplied = Path(root).expanduser().resolve()
    for candidate in (
        supplied,
        supplied / "Visa",
        supplied / "VisA",
        supplied / "VisA_20220922",
    ):
        if (candidate / "split_csv" / "1cls.csv").is_file():
            return candidate
    raise FileNotFoundError(f"VisA split_csv/1cls.csv not found under {supplied}")


def _relative_path(value: str, *, field: str) -> PurePosixPath:
    path = PurePosixPath(value.strip())
    if not value.strip() or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"invalid VisA {field} path: {value!r}")
    return path


def visa_one_class_records(
    root: str | Path,
    *,
    formal: bool = True,
) -> tuple[Path, dict[str, list[Record]], dict[str, list[Record]]]:
    dataset_root = find_visa_root(root)
    split_path = dataset_root / "split_csv" / "1cls.csv"
    train = {category: [] for category in VISA_CATEGORIES}
    evaluation = {category: [] for category in VISA_CATEGORIES}
    seen_images: set[str] = set()

    with split_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != VISA_FIELDS:
            raise RuntimeError(
                f"VisA 1cls.csv fields must be {VISA_FIELDS}, got {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            category = row["object"].strip()
            split = row["split"].strip()
            label_name = row["label"].strip()
            if category not in train:
                raise RuntimeError(f"unexpected VisA category on line {line_number}: {category}")
            if split not in {"train", "test"} or label_name not in {"normal", "anomaly"}:
                raise RuntimeError(f"invalid VisA split/label on line {line_number}")
            if split == "train" and label_name != "normal":
                raise RuntimeError(f"VisA train must be normal-only: line {line_number}")

            image_relative = _relative_path(row["image"], field="image")
            if not image_relative.parts or image_relative.parts[0] != category:
                raise RuntimeError(f"VisA image/category mismatch on line {line_number}")
            image_key = image_relative.as_posix()
            if image_key in seen_images:
                raise RuntimeError(f"duplicate VisA image in 1cls.csv: {image_key}")
            seen_images.add(image_key)
            image_path = dataset_root.joinpath(*image_relative.parts)

            label = int(label_name == "anomaly")
            mask_value = row["mask"].strip()
            mask_path: Path | None = None
            if label:
                mask_relative = _relative_path(mask_value, field="mask")
                if (
                    not mask_relative.parts
                    or mask_relative.parts[0] != category
                    or mask_relative.stem != image_relative.stem
                ):
                    raise RuntimeError(f"VisA image/mask mismatch on line {line_number}")
                mask_path = dataset_root.joinpath(*mask_relative.parts)
            elif mask_value:
                raise RuntimeError(f"normal VisA row must not declare a mask: line {line_number}")

            if formal:
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                if mask_path is not None and not mask_path.is_file():
                    raise FileNotFoundError(mask_path)

            record = Record(
                path=str(image_path),
                label=label,
                mask_path=None if mask_path is None else str(mask_path),
                category=category,
                dataset="visa",
                domain="source" if split == "train" else "target",
                shift=label_name,
                role="source_normal" if split == "train" else "evaluation",
                mask_capability=(
                    "not_evaluation"
                    if split == "train"
                    else "anomaly_mask" if label else "normal_zero_mask"
                ),
            )
            (train if split == "train" else evaluation)[category].append(record)

    if set(train) != set(VISA_CATEGORIES) or set(evaluation) != set(VISA_CATEGORIES):
        raise RuntimeError("VisA 12-category set is incomplete")
    for category in VISA_CATEGORIES:
        if not train[category] or not evaluation[category]:
            raise RuntimeError(f"VisA category has an empty split: {category}")
        if any(record.label != 0 or record.mask_path is not None for record in train[category]):
            raise RuntimeError(f"VisA train is not source-normal only: {category}")
        if {record.label for record in evaluation[category]} != {0, 1}:
            raise RuntimeError(f"VisA test lacks normal/anomaly support: {category}")
        if any(record.label == 1 and not record.mask_path for record in evaluation[category]):
            raise RuntimeError(f"VisA anomaly mask is missing: {category}")
    return dataset_root, train, evaluation

