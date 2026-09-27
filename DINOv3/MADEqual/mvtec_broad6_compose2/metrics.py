from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from DINOv3.nsrm.m0_scoring import (
    pixel_aupro_native_backend,
    pixel_ranking_metrics_and_thresholds,
    prepare_native_mask_context,
)


FAST_AUPRO_BACKEND = "fast_cuda_aupro_fpr0.3_thresholds200"
COMPONENT_CONNECTIVITY = "4-connected_scipy_ndimage_default"
METRIC_NAMES = (
    "image_AUROC",
    "image_AUPR",
    "pixel_AUROC",
    "pixel_AUPR",
    "AUPRO_at_0p3",
)


def strict_greater_thresholds(thresholds: np.ndarray) -> np.ndarray:
    """Convert float64 thresholds to the first float32 value strictly above each one."""
    values64 = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    values32 = values64.astype(np.float32)
    rounded64 = values32.astype(np.float64)
    advance = rounded64 <= values64
    values32[advance] = np.nextafter(
        values32[advance], np.float32(np.inf), dtype=np.float32,
    )
    return values32.astype(np.float64)


def global_score_range_thresholds(
    scores: np.ndarray,
    *,
    count: int = 200,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("global AUPRO scores must be finite and non-empty")
    if int(count) != 200:
        raise ValueError("frozen MVTec AUPRO requires exactly 200 thresholds")
    return np.linspace(
        float(values.max()),
        float(values.min()),
        num=int(count),
        dtype=np.float64,
    )


def evaluate_fast(
    labels: Sequence[int],
    masks: Sequence[np.ndarray],
    maps: Sequence[np.ndarray],
    *,
    device: torch.device,
    allow_cpu_fallback: bool,
    mask_context: Any | None = None,
) -> dict[str, Any]:
    label_values = np.asarray(labels, dtype=np.int64).reshape(-1)
    mask_values = [np.asarray(mask, dtype=bool) for mask in masks]
    map_values = [np.asarray(score, dtype=np.float32) for score in maps]
    if not map_values or len(map_values) != len(mask_values) or len(map_values) != label_values.size:
        raise ValueError("MVTec labels, masks, and maps must be aligned and non-empty")
    if set(label_values.tolist()) != {0, 1}:
        raise ValueError("MVTec category evaluation requires normal and anomaly images")
    if any(mask.shape != score.shape for mask, score in zip(mask_values, map_values)):
        raise ValueError("MVTec mask/map shapes must match per image")
    if any(not np.isfinite(score).all() for score in map_values):
        raise FloatingPointError("MVTec anomaly maps contain NaN/Inf")

    image_scores = np.asarray([float(score.max()) for score in map_values], dtype=np.float64)
    context = (
        mask_context
        if mask_context is not None
        else prepare_native_mask_context(mask_values, label_values)
    )
    flat_scores = np.concatenate([score.reshape(-1) for score in map_values])
    pixel_auroc, pixel_aupr, _ranking_thresholds, ranking_backend = (
        pixel_ranking_metrics_and_thresholds(
            context.flat_masks,
            flat_scores,
            device=device,
            max_thresholds=200,
            allow_cpu_fallback=allow_cpu_fallback,
        )
    )
    thresholds = global_score_range_thresholds(flat_scores, count=200)
    # Existing streaming counts use >=. Moving every threshold to the first
    # representable float32 value above it implements the frozen score > threshold rule.
    # The reused NSRM backend calls np.trapz, removed by NumPy 2.x. Supply the
    # exact renamed NumPy primitive only for this call; do not duplicate its
    # AUPRO integration or modify the shared backend.
    missing_trapz = not hasattr(np, "trapz")
    if missing_trapz:
        setattr(np, "trapz", np.trapezoid)
    try:
        aupro, count_backend = pixel_aupro_native_backend(
            context.masks,
            map_values,
            thresholds=strict_greater_thresholds(thresholds),
            component_indices_by_image=context.component_indices_by_image,
            flat_scores=flat_scores,
            device=device,
            max_fpr=0.30,
            allow_cpu_fallback=allow_cpu_fallback,
        )
    finally:
        if missing_trapz:
            delattr(np, "trapz")
    if not np.isfinite([pixel_auroc, pixel_aupr, aupro]).all():
        raise FloatingPointError("fast MVTec pixel metrics are not finite")
    return {
        "image_AUROC": float(roc_auc_score(label_values, image_scores)),
        "image_AUPR": float(average_precision_score(label_values, image_scores)),
        "pixel_AUROC": float(pixel_auroc),
        "pixel_AUPR": float(pixel_aupr),
        "AUPRO_at_0p3": float(aupro),
        "aupro_backend": (
            FAST_AUPRO_BACKEND
            if device.type == "cuda" and not allow_cpu_fallback
            else "fast_cpu_aupro_fpr0.3_thresholds200"
        ),
        "pixel_ranking_backend": ranking_backend,
        "aupro_count_backend": count_backend,
        "aupro_threshold_count": int(len(thresholds)),
        "aupro_threshold_sampling": "linear_global_score_range_max_to_min_200",
        "component_connectivity": COMPONENT_CONNECTIVITY,
    }


def category_macro(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    output: list[dict[str, Any]] = []
    for method in methods:
        subset = [row for row in rows if str(row["method"]) == method]
        categories = {str(row["category"]) for row in subset}
        if len(subset) != 15 or len(categories) != 15:
            raise RuntimeError(f"MVTec macro requires 15 unique categories for {method}")
        output.append({
            "method": method,
            "scope": "mvtec_15_category_unweighted_macro",
            "category_count": 15,
            **{
                metric: float(np.mean([float(row[metric]) for row in subset]))
                for metric in METRIC_NAMES
            },
        })
    return output
