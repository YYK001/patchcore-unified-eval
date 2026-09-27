from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_auroc(labels, scores) -> float:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def pixel_aupr(masks, maps) -> float:
    labels = np.asarray(masks, dtype=bool).reshape(-1)
    scores = np.asarray(maps, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    from scipy import ndimage

    labels, count = ndimage.label(np.asarray(mask, dtype=bool))
    return [labels == index for index in range(1, int(count) + 1)]


def pixel_aupro(
    masks: np.ndarray,
    maps: np.ndarray,
    max_fpr: float = 0.30,
    max_thresholds: int = 256,
) -> float:
    masks = np.asarray(masks, dtype=bool)
    maps = np.asarray(maps, dtype=np.float64)
    if masks.shape != maps.shape or masks.ndim != 3:
        raise ValueError("masks/maps must have matching [N,H,W] shapes")
    background = ~masks
    background_count = int(background.sum())
    region_scores = [
        maps[image_index][region]
        for image_index, image_mask in enumerate(masks)
        for region in _components(image_mask)
        if region.any()
    ]
    if background_count == 0 or not region_scores:
        return float("nan")
    flat = maps.reshape(-1)
    thresholds = np.unique(
        np.quantile(flat, np.linspace(0.0, 1.0, min(int(max_thresholds), flat.size)))
    )[::-1]
    background_scores = np.sort(maps[background])
    background_positive = background_count - np.searchsorted(
        background_scores, thresholds, side="left"
    )
    region_overlaps = []
    for scores in region_scores:
        sorted_scores = np.sort(scores)
        region_overlaps.append(
            (scores.size - np.searchsorted(sorted_scores, thresholds, side="left"))
            / scores.size
        )
    fprs = background_positive.astype(np.float64) / background_count
    pros = np.mean(np.stack(region_overlaps, axis=0), axis=0)
    points = [(0.0, 0.0), *zip(fprs.tolist(), pros.tolist()), (1.0, 1.0)]
    points.sort(key=lambda pair: pair[0])
    fpr = np.asarray([point[0] for point in points])
    pro = np.asarray([point[1] for point in points])
    unique_fpr = np.unique(fpr)
    max_pro = np.asarray([pro[fpr == value].max() for value in unique_fpr])
    if unique_fpr[-1] < float(max_fpr):
        unique_fpr = np.append(unique_fpr, float(max_fpr))
        max_pro = np.append(max_pro, max_pro[-1])
    elif float(max_fpr) not in unique_fpr:
        interpolated = np.interp(float(max_fpr), unique_fpr, max_pro)
        keep = unique_fpr < float(max_fpr)
        unique_fpr = np.append(unique_fpr[keep], float(max_fpr))
        max_pro = np.append(max_pro[keep], interpolated)
    else:
        keep = unique_fpr <= float(max_fpr)
        unique_fpr, max_pro = unique_fpr[keep], max_pro[keep]
    return float(np.trapz(max_pro, unique_fpr) / float(max_fpr))


def binary_f1(mask, prediction) -> float:
    mask = np.asarray(mask, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = float(np.logical_and(mask, prediction).sum())
    fp = float(np.logical_and(~mask, prediction).sum())
    fn = float(np.logical_and(mask, ~prediction).sum())
    denominator = 2.0 * tp + fp + fn
    return float(2.0 * tp / denominator) if denominator else float("nan")


def oracle_f1(masks, maps, max_thresholds: int = 512) -> tuple[float, float]:
    labels = np.asarray(masks, dtype=bool).reshape(-1)
    scores = np.asarray(maps, dtype=np.float64).reshape(-1)
    if labels.size != scores.size or scores.size == 0:
        raise ValueError("masks/maps must have the same non-zero number of pixels")
    thresholds = np.unique(
        np.quantile(scores, np.linspace(0.0, 1.0, min(max_thresholds, scores.size)))
    )
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)
    positive_prefix = np.concatenate(([0], np.cumsum(sorted_labels)))
    starts = np.searchsorted(sorted_scores, thresholds, side="left")
    total_positive = int(sorted_labels.sum())
    true_positive = total_positive - positive_prefix[starts]
    predicted_positive = scores.size - starts
    false_positive = predicted_positive - true_positive
    false_negative = total_positive - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.full(thresholds.shape, np.nan, dtype=np.float64),
        where=denominator != 0,
    )
    valid = np.flatnonzero(np.isfinite(f1))
    if valid.size == 0:
        return float("nan"), float("nan")
    best = int(valid[np.argmax(f1[valid])])
    return float(f1[best]), float(thresholds[best])


def localization_metrics(
    masks: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    small_fraction: float = 0.01,
) -> dict[str, float]:
    masks = np.asarray(masks, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    labels = np.asarray(labels)
    recalls, small_recalls, overseg = [], [], []
    for index in np.flatnonzero(labels == 1):
        area = float(masks[index].sum())
        if area == 0:
            continue
        recall = float(np.logical_and(masks[index], predictions[index]).sum() / area)
        recalls.append(recall)
        overseg.append(float(predictions[index].sum() / area))
        if area / masks[index].size <= float(small_fraction):
            small_recalls.append(recall)
    normal = np.flatnonzero(labels == 0)
    normal_fp = (
        float(predictions[normal].reshape(normal.size, -1).any(axis=1).mean())
        if normal.size
        else float("nan")
    )
    return {
        "Recall": float(np.mean(recalls)) if recalls else float("nan"),
        "small_recall": float(np.mean(small_recalls)) if small_recalls else float("nan"),
        "localization_overseg_anomaly_macro": float(np.mean(overseg)) if overseg else float("nan"),
        "localization_test_normal_image_positive_rate": normal_fp,
    }


def evaluate_pixel_metrics(
    masks: np.ndarray,
    maps: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    small_fraction: float = 0.01,
) -> dict[str, float]:
    predictions = np.asarray(maps) >= float(threshold)
    oracle, oracle_threshold = oracle_f1(masks, maps)
    return {
        "pixel_AUROC": safe_auroc(masks, maps),
        "pixel_AUPR": pixel_aupr(masks, maps),
        "pixel_AUPRO": pixel_aupro(masks, maps, max_fpr=0.30),
        "pixel_F1_calibrated": binary_f1(masks, predictions),
        "pixel_F1_oracle": oracle,
        "pixel_threshold_oracle": oracle_threshold,
        **localization_metrics(
            masks, predictions, labels, small_fraction=small_fraction
        ),
    }


def average_relative_drop(source: float, targets) -> float:
    target_values = np.asarray(list(targets), dtype=np.float64)
    if target_values.size == 0:
        return float("nan")
    return float(np.minimum(0.0, target_values - float(source)).mean())
