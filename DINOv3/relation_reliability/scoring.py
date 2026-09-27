from __future__ import annotations

from typing import Mapping

import numpy as np

from .calibration import ReliabilityBundle
from .protocol import ECDF_EPSILON, LAYERS, METHODS


def normalized_reliability_weights(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] != len(LAYERS):
        raise ValueError(f"weights must start with {len(LAYERS)} layers, got {values.shape}")
    positive = np.maximum(values, 0.0) + ECDF_EPSILON
    return positive / positive.sum(axis=0, keepdims=True)


def resolve_anchor_labels(
    memory_labels: Mapping[int, np.ndarray],
    top1_indices: np.ndarray,
    ecdf_distances: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(top1_indices, dtype=np.int64)
    distances = np.asarray(ecdf_distances, dtype=np.float32)
    if indices.shape != distances.shape or indices.shape[0] != len(LAYERS):
        raise ValueError("anchor indices and distances must be [4,patch]")
    layer_labels = []
    for offset, layer in enumerate(LAYERS):
        available = np.asarray(memory_labels[layer], dtype=np.int64)
        layer_indices = indices[offset]
        if layer_indices.size and (
            int(layer_indices.min()) < 0
            or int(layer_indices.max()) >= int(available.size)
        ):
            raise IndexError(
                f"layer {layer} top-1 index is outside its normal-memory bank"
            )
        layer_labels.append(available[layer_indices])
    labels = np.stack(layer_labels, axis=0)
    output = np.full(labels.shape[1], -1, dtype=np.int64)
    for patch in range(labels.shape[1]):
        unique, counts = np.unique(labels[:, patch], return_counts=True)
        maximum = int(counts.max())
        if maximum >= 3:
            output[patch] = int(unique[np.argmax(counts)])
            continue
        if maximum == 2:
            candidates = unique[counts == 2]
            if candidates.size == 1:
                output[patch] = int(candidates[0])
                continue
            means = []
            for candidate in candidates:
                supporting = labels[:, patch] == candidate
                means.append(float(distances[supporting, patch].mean()))
            output[patch] = int(candidates[int(np.argmin(means))])
            continue
        # Four different anchors: retain -1 and use global reliability.
    return output


def _conditioned_weights(
    table: np.ndarray,
    global_values: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    values = np.repeat(np.asarray(global_values, np.float32)[:, None], labels.size, axis=1)
    conditioned = labels >= 0
    if conditioned.any():
        if labels[conditioned].max(initial=-1) >= table.shape[0]:
            raise ValueError("condition label exceeds reliability table")
        values[:, conditioned] = np.asarray(table, np.float32)[labels[conditioned]].T
    return normalized_reliability_weights(values)


def score_variants(
    raw_distances: np.ndarray,
    ecdf_distances: np.ndarray,
    mad_distances: np.ndarray,
    reliability: ReliabilityBundle,
    top1_indices: np.ndarray,
    memory_relation_labels: Mapping[int, np.ndarray],
    memory_shuffled_labels: Mapping[int, np.ndarray],
    memory_feature_labels: Mapping[int, np.ndarray],
    memory_position_labels: Mapping[int, np.ndarray],
) -> dict[str, np.ndarray]:
    raw = np.asarray(raw_distances, dtype=np.float32)
    ecdf = np.asarray(ecdf_distances, dtype=np.float32)
    mad = np.asarray(mad_distances, dtype=np.float32)
    if raw.shape != ecdf.shape or raw.shape != mad.shape or raw.ndim != 3:
        raise ValueError("distance tensors must share [4,height,width]")
    flat_ecdf = ecdf.reshape(len(LAYERS), -1)
    flat_indices = np.asarray(top1_indices, dtype=np.int64).reshape(len(LAYERS), -1)

    relation_labels = resolve_anchor_labels(memory_relation_labels, flat_indices, flat_ecdf)
    shuffled_labels = resolve_anchor_labels(memory_shuffled_labels, flat_indices, flat_ecdf)
    feature_labels = resolve_anchor_labels(memory_feature_labels, flat_indices, flat_ecdf)
    position_labels = resolve_anchor_labels(memory_position_labels, flat_indices, flat_ecdf)

    weights = {
        "Global": normalized_reliability_weights(reliability.global_values[:, None]),
        "Relation": _conditioned_weights(reliability.relation.values, reliability.global_values, relation_labels),
        "Shuffled": _conditioned_weights(reliability.relation.values, reliability.global_values, shuffled_labels),
        "Feature": _conditioned_weights(reliability.feature.values, reliability.global_values, feature_labels),
        "Position": _conditioned_weights(reliability.position.values, reliability.global_values, position_labels),
    }
    scores: dict[str, np.ndarray] = {
        "Official": raw.mean(axis=0),
        "L7": raw[0],
        "L15": raw[1],
        "L23": raw[2],
        "L31": raw[3],
        "ECDF-Equal": ecdf.mean(axis=0),
        "MAD-Equal": mad.mean(axis=0),
    }
    for method, method_weights in weights.items():
        fused = (flat_ecdf * method_weights).sum(axis=0)
        scores[method] = fused.reshape(raw.shape[1:])
    if tuple(scores) != METHODS:
        raise AssertionError(f"method contract mismatch: {tuple(scores)}")
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise FloatingPointError("non-finite score variant")
    return scores
