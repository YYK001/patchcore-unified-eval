"""Read-only adapters for the official SuperADD external evaluation."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence


ROBUSTAD_LAYOUT = {
    "MetalParts": ("metal_parts_data_dir", {1: "lighting", 2: "position", 3: "rotation", 4: "scale", 5: "background_1", 6: "background_2"}),
    "PCB": ("pcb_data_dir", {1: "lighting", 2: "white_balancing", 3: "rotation", 4: "position", 5: "shadow"}),
    "PiledBags": ("piled_bags_data_dir", {1: "lighting", 2: "background_box_color", 3: "position_rotation", 4: "scale", 5: "shadow"}),
}
AD2_CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")


@dataclass(frozen=True)
class Record:
    path: str
    label: int
    mask_path: str | None
    category: str
    domain: str
    shift: str
    role: str
    mask_capability: str = "not_evaluation"


def _pngs(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.png") if path.is_file())


def _robust_split(category: str, split: Path, domain: str, shift: str, *, evaluation: bool) -> list[Record]:
    metadata = split / "metadata.jsonl"
    if not metadata.is_file(): raise FileNotFoundError(metadata)
    records = []
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line); image = split / str(row["file_name"])
        if not image.is_file(): raise FileNotFoundError(image)
        mask = row.get("mask"); mask_path = None
        if mask:
            candidates = (Path(str(mask)), split / str(mask), split / "masks" / Path(str(mask)).name)
            mask_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[-1])
        records.append(Record(str(image), int(row["label"]), str(mask_path) if mask_path else None, category, domain, shift, "evaluation" if evaluation else "train"))
    return records


def robustad_records(root: Path, categories: Sequence[str] | None = None) -> tuple[dict[str, list[Record]], dict[str, list[Record]]]:
    selected = tuple(categories or ROBUSTAD_LAYOUT)
    if unknown := sorted(set(selected) - set(ROBUSTAD_LAYOUT)): raise ValueError(f"Unknown RobustAD categories: {unknown}")
    train, evaluation = {}, {}
    for category in selected:
        prefix, shifts = ROBUSTAD_LAYOUT[category]; category_root = root / category
        source = _robust_split(category, category_root / f"{prefix}_train", "source", "source_train", evaluation=False)
        train[category] = [record for record in source if record.label == 0]
        records = []
        for index, domain, shift in [(0, "source", "source"), *[(index, "target", shift) for index, shift in sorted(shifts.items())]]:
            records.extend(_robust_split(category, category_root / f"{prefix}_test{index}", domain, shift, evaluation=True))
        grouped: dict[tuple[str, str], list[Record]] = {}
        for record in records: grouped.setdefault((record.domain, record.shift), []).append(record)
        resolved = []
        for key, group in grouped.items():
            bad = [record for record in group if record.label == 1]
            present = [bool(record.mask_path and Path(record.mask_path).is_file()) for record in bad]
            declared = [record.mask_path is not None for record in bad]
            if any(declared) and not all(present): raise ValueError(f"Missing/partial RobustAD masks in {category}/{key}")
            capability = "pixel" if bad and all(present) else "image"
            resolved.extend(replace(record, mask_capability=capability) for record in group)
        evaluation[category] = resolved
    return train, evaluation


def _ad2_root(root: Path) -> Path:
    return root / "mvtec_ad_2" if (root / "mvtec_ad_2").is_dir() else root


def _variant(path: Path) -> str:
    return path.stem.split("_", 1)[1] if "_" in path.stem else "regular"


def _ad2_mask(image: Path, mask_root: Path) -> Path | None:
    candidate = mask_root / f"{image.stem}_mask.png"
    return candidate if candidate.is_file() else None


def mvtec_ad2_records(root: Path, categories: Sequence[str] | None = None) -> tuple[dict[str, list[Record]], dict[str, list[Record]]]:
    root = _ad2_root(root); selected = tuple(categories or AD2_CATEGORIES)
    if unknown := sorted(set(selected) - set(AD2_CATEGORIES)): raise ValueError(f"Unknown MVTec AD 2 categories: {unknown}")
    train, evaluation = {}, {}
    for category in selected:
        category_root = root / category
        required = (category_root / "train" / "good", category_root / "validation" / "good", category_root / "test_public" / "good", category_root / "test_public" / "bad", category_root / "test_public" / "ground_truth" / "bad")
        if missing := [str(path) for path in required if not path.is_dir()]: raise FileNotFoundError(f"Missing MVTec AD 2 directories: {missing}")
        train[category] = [Record(str(path), 0, None, category, "source", "train", "train") for path in _pngs(category_root / "train" / "good")]
        public = [Record(str(path), 0, None, category, "test_public", _variant(path), "evaluation", "pixel") for path in _pngs(category_root / "test_public" / "good")]
        mask_root = category_root / "test_public" / "ground_truth" / "bad"
        for path in _pngs(category_root / "test_public" / "bad"):
            mask = _ad2_mask(path, mask_root)
            public.append(Record(str(path), 1, str(mask) if mask else None, category, "test_public", _variant(path), "evaluation", "pixel"))
        evaluation[category] = public
    return train, evaluation
