from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .protocol import ECDF_EPSILON, LAYERS, LCB_ONE_SIDED_Z, MIN_RELIABILITY_PAIRS


@dataclass(frozen=True)
class ECDFCalibrator:
    sorted_values: dict[int, np.ndarray]
    epsilon: float = ECDF_EPSILON

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[0] != len(LAYERS):
            raise ValueError(f"expected first dimension {len(LAYERS)}, got {values.shape}")
        output = np.empty_like(values, dtype=np.float32)
        for offset, layer in enumerate(LAYERS):
            reference = self.sorted_values[layer]
            if reference.size == 0:
                raise ValueError(f"empty ECDF reference for layer {layer}")
            ranks = np.searchsorted(reference, values[offset], side="right")
            cdf = ranks.astype(np.float64) / float(reference.size)
            output[offset] = -np.log(1.0 - cdf + self.epsilon).astype(np.float32)
        if not np.isfinite(output).all():
            raise FloatingPointError("non-finite ECDF-calibrated distances")
        return output


@dataclass(frozen=True)
class MADCalibrator:
    medians: np.ndarray
    scales: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        shape = (len(LAYERS),) + (1,) * (values.ndim - 1)
        output = (values - self.medians.reshape(shape)) / self.scales.reshape(shape)
        if not np.isfinite(output).all():
            raise FloatingPointError("non-finite MAD-calibrated distances")
        return output.astype(np.float32, copy=False)


def fit_calibrators(samples: Mapping[int, np.ndarray]) -> tuple[ECDFCalibrator, MADCalibrator]:
    sorted_values: dict[int, np.ndarray] = {}
    medians: list[float] = []
    scales: list[float] = []
    for layer in LAYERS:
        values = np.asarray(samples[layer], dtype=np.float32).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size < MIN_RELIABILITY_PAIRS:
            raise ValueError(f"layer {layer} has only {values.size} source-normal calibration distances")
        values.sort()
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        sorted_values[layer] = values
        medians.append(median)
        scales.append(max(1.4826 * mad, ECDF_EPSILON))
    return (
        ECDFCalibrator(sorted_values=sorted_values),
        MADCalibrator(np.asarray(medians, np.float32), np.asarray(scales, np.float32)),
    )


def wilson_lower_bound(successes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    successes = np.asarray(successes, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if successes.shape != counts.shape:
        raise ValueError("success and count arrays must have identical shapes")
    result = np.zeros_like(successes)
    valid = counts > 0
    n = counts[valid]
    p = successes[valid] / n
    z = LCB_ONE_SIDED_Z
    denominator = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    radius = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    result[valid] = np.maximum((centre - radius) / denominator, 0.0)
    return result.astype(np.float32)


@dataclass(frozen=True)
class ReliabilityTable:
    values: np.ndarray
    counts: np.ndarray
    successes: np.ndarray
    fallback_mask: np.ndarray


@dataclass(frozen=True)
class ReliabilityBundle:
    global_values: np.ndarray
    global_counts: np.ndarray
    relation: ReliabilityTable
    feature: ReliabilityTable
    position: ReliabilityTable


class ReliabilityAccumulator:
    def __init__(self, group_sizes: Mapping[str, int]) -> None:
        self._successes = {
            name: np.zeros((size, len(LAYERS)), dtype=np.int64)
            for name, size in group_sizes.items()
        }
        self._counts = {
            name: np.zeros((size, len(LAYERS)), dtype=np.int64)
            for name, size in group_sizes.items()
        }
        self._global_successes = np.zeros(len(LAYERS), dtype=np.int64)
        self._global_counts = np.zeros(len(LAYERS), dtype=np.int64)

    def update(
        self,
        delta_normal: np.ndarray,
        delta_anomaly: np.ndarray,
        labels: Mapping[str, np.ndarray],
        valid_mask: np.ndarray | None = None,
    ) -> None:
        dn = np.asarray(delta_normal, dtype=np.float32)
        da = np.asarray(delta_anomaly, dtype=np.float32)
        if dn.shape != da.shape or dn.ndim != 2 or dn.shape[0] != len(LAYERS):
            raise ValueError(f"expected paired deltas [layer, patch], got {dn.shape} and {da.shape}")
        valid = np.isfinite(dn) & np.isfinite(da)
        if valid_mask is not None:
            vm = np.asarray(valid_mask, dtype=bool).reshape(-1)
            if vm.size != dn.shape[1]:
                raise ValueError("valid mask does not match patch count")
            valid &= vm[None, :]
        success = da > dn
        self._global_successes += (success & valid).sum(axis=1)
        self._global_counts += valid.sum(axis=1)
        for name, group_labels in labels.items():
            group_labels = np.asarray(group_labels, dtype=np.int64).reshape(-1)
            if group_labels.size != dn.shape[1]:
                raise ValueError(f"{name} labels do not match patch count")
            group_count = self._successes[name].shape[0]
            if ((group_labels < 0) | (group_labels >= group_count)).any():
                raise ValueError(f"{name} labels outside [0,{group_count})")
            for layer_offset in range(len(LAYERS)):
                layer_valid = valid[layer_offset]
                self._counts[name][:, layer_offset] += np.bincount(
                    group_labels[layer_valid], minlength=group_count,
                )
                won = layer_valid & success[layer_offset]
                self._successes[name][:, layer_offset] += np.bincount(
                    group_labels[won], minlength=group_count,
                )

    def finalize(self) -> ReliabilityBundle:
        if (self._global_counts < MIN_RELIABILITY_PAIRS).any():
            raise ValueError(f"insufficient global reliability support: {self._global_counts.tolist()}")
        global_values = wilson_lower_bound(self._global_successes, self._global_counts)
        tables: dict[str, ReliabilityTable] = {}
        for name in ("relation", "feature", "position"):
            counts = self._counts[name]
            successes = self._successes[name]
            values = wilson_lower_bound(successes, counts)
            fallback = counts < MIN_RELIABILITY_PAIRS
            values = np.where(fallback, global_values[None, :], values).astype(np.float32)
            tables[name] = ReliabilityTable(values, counts.copy(), successes.copy(), fallback)
        return ReliabilityBundle(
            global_values=global_values,
            global_counts=self._global_counts.copy(),
            relation=tables["relation"],
            feature=tables["feature"],
            position=tables["position"],
        )
