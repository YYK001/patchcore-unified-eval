"""Project-comparable metrics computed from unmodified official SuperADD outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score


CORE_METRICS = ("image_AUROC", "image_AUPR", "normal_image_FP_official_binary", "pixel_AUROC", "pixel_AUPR", "AUPRO_project_reference_256q")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, dict): return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)): return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)): return None if not np.isfinite(value) else float(value)
        if isinstance(value, np.integer): return int(value)
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)


def aupro(masks, maps, max_fpr: float = .30) -> float:
    truth = [np.asarray(mask, dtype=bool) for mask in masks]; maps = [np.asarray(anomaly_map) for anomaly_map in maps]
    components = []
    for mask, anomaly_map in zip(truth, maps, strict=True):
        labels, count = ndimage.label(mask)
        components.extend(anomaly_map[labels == component] for component in range(1, count + 1))
    background_values = np.concatenate([anomaly_map[~mask] for mask, anomaly_map in zip(truth, maps, strict=True)])
    all_scores = np.concatenate([anomaly_map.ravel() for anomaly_map in maps])
    if not components or not background_values.size: return float("nan")
    thresholds = np.unique(np.quantile(all_scores, np.linspace(0, 1, min(256, all_scores.size))))[::-1]
    background_values = np.sort(background_values); fpr = (background_values.size - np.searchsorted(background_values, thresholds, side="left")) / background_values.size
    pro = np.mean([(component.size - np.searchsorted(np.sort(component), thresholds, side="left")) / component.size for component in components], axis=0)
    fpr, pro = np.r_[0., fpr, 1.], np.r_[0., pro, 1.]; order = np.argsort(fpr); fpr, pro = fpr[order], pro[order]
    unique = np.unique(fpr); pro = np.asarray([pro[fpr == value].max() for value in unique]); endpoint = np.interp(max_fpr, unique, pro); keep = unique < max_fpr
    return float(np.trapezoid(np.r_[pro[keep], endpoint], np.r_[unique[keep], max_fpr]) / max_fpr)


def image_metrics(labels: np.ndarray, scores: np.ndarray, binary_flags: np.ndarray) -> dict[str, float]:
    return {
        "image_AUROC": float(roc_auc_score(labels, scores)),
        "image_AUPR": float(average_precision_score(labels, scores)),
        "normal_image_FP_official_binary": float(binary_flags[labels == 0].mean()),
    }


def pixel_metrics(masks, maps, binaries, official_masks) -> dict[str, float]:
    truth = np.concatenate([np.asarray(mask, dtype=bool).ravel() for mask in masks])
    scores = np.concatenate([np.asarray(anomaly_map).ravel() for anomaly_map in maps])
    official_truth = np.concatenate([np.asarray(mask, dtype=bool).ravel() for mask in official_masks])
    prediction = np.concatenate([np.asarray(binary, dtype=bool).ravel() for binary in binaries])
    tp = float(np.logical_and(official_truth, prediction).sum()); fp = float(np.logical_and(~official_truth, prediction).sum()); fn = float(np.logical_and(official_truth, ~prediction).sum())
    return {
        "pixel_AUROC": float(roc_auc_score(truth, scores)),
        "pixel_AUPR": float(average_precision_score(truth, scores)),
        "AUPRO_project_reference_256q": aupro(masks, maps),
        "pixel_F1_binary_external_group": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
    }


def aggregate(rows: list[dict[str, Any]], aggregation: str, metric_names: tuple[str, ...] = CORE_METRICS) -> dict[str, Any]:
    result = {"aggregation": aggregation}
    if metric_names == CORE_METRICS: result["pixel_scope"] = "pixel_evaluable_groups_only"
    for key in metric_names:
        values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        if values: result[key] = float(np.mean(values))
    return result
