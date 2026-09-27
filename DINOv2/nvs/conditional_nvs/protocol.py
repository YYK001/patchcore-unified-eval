from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _without_runtime_fields(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {
            str(key): _without_runtime_fields(value)
            for key, value in payload.items()
            if not str(key).startswith("_")
        }
    if isinstance(payload, (list, tuple)):
        return [_without_runtime_fields(value) for value in payload]
    return payload


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash only persisted protocol settings, never transient run metadata."""

    return stable_hash(_without_runtime_fields(config))


@dataclass(frozen=True)
class ThreeWaySplit(Generic[T]):
    calibration: tuple[T, ...]
    nvs_fit: tuple[T, ...]
    memory: tuple[T, ...]
    split_seed: int
    nvs_split_seed: int
    calibration_fraction: float = 0.20
    nvs_fit_fraction_of_remainder: float = 0.30

    def assert_disjoint(self, key: Callable[[T], str] = str) -> None:
        groups = {
            "calibration": {key(item) for item in self.calibration},
            "nvs_fit": {key(item) for item in self.nvs_fit},
            "memory": {key(item) for item in self.memory},
        }
        for name, values in groups.items():
            if len(values) != len(getattr(self, name)):
                raise AssertionError(f"Duplicate samples inside {name}")
        pairs = (("calibration", "nvs_fit"), ("calibration", "memory"), ("nvs_fit", "memory"))
        for left, right in pairs:
            overlap = groups[left] & groups[right]
            if overlap:
                raise AssertionError(f"{left}/{right} overlap: {sorted(overlap)[:3]}")

    def manifest(self, key: Callable[[T], str] = str) -> dict[str, Any]:
        payload = {
            "split_seed": self.split_seed,
            "nvs_split_seed": self.nvs_split_seed,
            "calibration_fraction": self.calibration_fraction,
            "nvs_fit_fraction_of_remainder": self.nvs_fit_fraction_of_remainder,
            "calibration": [key(item) for item in self.calibration],
            "nvs_fit": [key(item) for item in self.nvs_fit],
            "memory": [key(item) for item in self.memory],
        }
        payload["manifest_hash"] = stable_hash(payload)
        return payload


def split_three_way(
    records: Sequence[T],
    split_seed: int,
    nvs_split_seed: int | None = None,
    calibration_fraction: float = 0.20,
    nvs_fit_fraction_of_remainder: float = 0.30,
    key: Callable[[T], str] = str,
) -> ThreeWaySplit[T]:
    """Create the preregistered 20% / (30%, 70%) image-level split."""

    if len(records) < 3:
        raise ValueError("At least three normal training images are required")
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if not 0.0 < float(nvs_fit_fraction_of_remainder) < 1.0:
        raise ValueError("nvs_fit_fraction_of_remainder must be in (0, 1)")
    nvs_seed = int(split_seed if nvs_split_seed is None else nvs_split_seed)

    first_rng = np.random.default_rng(int(split_seed))
    order = first_rng.permutation(len(records))
    calibration_count = int(round(len(records) * float(calibration_fraction)))
    calibration_count = min(max(1, calibration_count), len(records) - 2)
    calibration_indices = set(order[:calibration_count].tolist())
    remainder_indices = [index for index in range(len(records)) if index not in calibration_indices]

    second_rng = np.random.default_rng(nvs_seed)
    remainder_order = second_rng.permutation(len(remainder_indices))
    nvs_count = int(round(len(remainder_indices) * float(nvs_fit_fraction_of_remainder)))
    nvs_count = min(max(1, nvs_count), len(remainder_indices) - 1)
    nvs_indices = {
        remainder_indices[position] for position in remainder_order[:nvs_count].tolist()
    }
    memory_indices = set(remainder_indices) - nvs_indices
    result = ThreeWaySplit(
        calibration=tuple(records[index] for index in range(len(records)) if index in calibration_indices),
        nvs_fit=tuple(records[index] for index in range(len(records)) if index in nvs_indices),
        memory=tuple(records[index] for index in range(len(records)) if index in memory_indices),
        split_seed=int(split_seed),
        nvs_split_seed=nvs_seed,
        calibration_fraction=float(calibration_fraction),
        nvs_fit_fraction_of_remainder=float(nvs_fit_fraction_of_remainder),
    )
    result.assert_disjoint(key=key)
    if len(result.calibration) + len(result.nvs_fit) + len(result.memory) != len(records):
        raise AssertionError("Three-way split does not cover the input records")
    return result


def write_split_manifest(
    split: ThreeWaySplit[T],
    path: str | Path,
    category: str,
    key: Callable[[T], str] = str,
) -> dict[str, Any]:
    payload = {"category": str(category), **split.manifest(key=key)}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


@dataclass(frozen=True)
class CalibrationState:
    """Immutable calibration fitted only on identity/source calibration data."""

    median: float
    mad: float
    threshold: float
    fit_scope: str = "identity_calibration"

    def normalize(self, scores: np.ndarray) -> np.ndarray:
        return np.maximum(
            (np.asarray(scores, dtype=np.float64) - self.median) / self.mad,
            0.0,
        )

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.normalize(scores) >= self.threshold


def fit_calibration(
    score_maps: np.ndarray,
    image_quantile: float = 0.95,
    mad_epsilon: float = 1.0e-6,
    scope: str = "identity_calibration",
) -> CalibrationState:
    if scope != "identity_calibration":
        raise ValueError("Calibration may only be fitted on identity_calibration")
    maps = np.asarray(score_maps, dtype=np.float64)
    if maps.ndim < 2 or maps.shape[0] == 0:
        raise ValueError("score_maps must contain at least one image")
    median = float(np.median(maps))
    mad = max(float(np.median(np.abs(maps - median))), float(mad_epsilon))
    normalized = np.maximum((maps - median) / mad, 0.0)
    image_max = normalized.reshape(normalized.shape[0], -1).max(axis=1)
    threshold = float(np.quantile(image_max, float(image_quantile)))
    return CalibrationState(median=median, mad=mad, threshold=threshold)


def protocol_metadata(
    category: str,
    seed: int,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    methods: Sequence[str],
) -> dict[str, Any]:
    core = {
        "category": str(category),
        "seed": int(seed),
        "sample_manifest_hash": str(manifest["manifest_hash"]),
        "config_fingerprint": config_fingerprint(config),
        "methods": list(methods),
    }
    return {**core, "protocol_hash": stable_hash(core)}


def completion_is_valid(path: str | Path, expected: Mapping[str, Any]) -> bool:
    marker = Path(path)
    if not marker.is_file():
        return False
    try:
        actual = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    required = (
        "status",
        "category",
        "seed",
        "sample_manifest_hash",
        "config_fingerprint",
        "protocol_hash",
    )
    return actual.get("status") == "complete" and all(
        actual.get(key) == expected.get(key) for key in required if key != "status"
    )


def write_completion(path: str | Path, metadata: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"status": "complete", **dict(metadata)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def select_memory_protocol(
    seed42_rows: Sequence[Mapping[str, Any]],
    confirmation_rows: Sequence[Mapping[str, Any]] | None = None,
    auroc_tolerance: float = 0.002,
) -> dict[str, Any]:
    """Apply the preregistered seed42 top-two then 42/43/44 mean rule."""

    def rank_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        aggregated: list[dict[str, Any]] = []
        protocols = sorted({str(row["memory_protocol"]) for row in rows})
        for protocol in protocols:
            selected = [row for row in rows if str(row["memory_protocol"]) == protocol]
            aurocs = np.asarray([float(row["pixel_AUROC"]) for row in selected])
            fps = np.asarray([float(row["normal_FP"]) for row in selected])
            base = selected[0]
            aggregated.append(
                {
                    "memory_protocol": protocol,
                    "pixel_AUROC_mean": float(np.mean(aurocs)),
                    "pixel_AUROC_std": float(np.std(aurocs)),
                    "normal_FP_mean": float(np.mean(fps)),
                    "capacity": int(base["capacity"]),
                    "inference_ms": float(np.mean([float(row["inference_ms"]) for row in selected])),
                    "seeds": sorted({int(row["seed"]) for row in selected}),
                }
            )
        aggregated.sort(
            key=lambda row: (
                -row["pixel_AUROC_mean"],
                row["normal_FP_mean"],
                row["capacity"],
                row["inference_ms"],
            )
        )
        if not aggregated:
            return aggregated
        best_auroc = aggregated[0]["pixel_AUROC_mean"]
        tolerance_group = [
            row
            for row in aggregated
            if best_auroc - row["pixel_AUROC_mean"] <= auroc_tolerance
        ]
        outside_group = [
            row
            for row in aggregated
            if best_auroc - row["pixel_AUROC_mean"] > auroc_tolerance
        ]
        tolerance_group.sort(
            key=lambda row: (
                row["normal_FP_mean"],
                row["pixel_AUROC_std"],
                row["capacity"],
                row["inference_ms"],
                -row["pixel_AUROC_mean"],
            )
        )
        return tolerance_group + outside_group

    first = rank_rows(seed42_rows)
    if len(first) < 2:
        raise ValueError("Seed42 screening requires at least two memory protocols")
    top_two = [row["memory_protocol"] for row in first[:2]]
    if confirmation_rows is None:
        return {"stage": "seed42", "top_two": top_two, "ranking": first}
    confirmed = [
        row for row in confirmation_rows if str(row["memory_protocol"]) in set(top_two)
    ]
    for protocol in top_two:
        seeds = {int(row["seed"]) for row in confirmed if str(row["memory_protocol"]) == protocol}
        if seeds != {42, 43, 44}:
            raise ValueError(f"{protocol} confirmation must contain paired seeds 42/43/44")
    final = rank_rows(confirmed)
    return {
        "stage": "locked",
        "top_two": top_two,
        "M_locked": final[0]["memory_protocol"],
        "ranking": final,
    }
