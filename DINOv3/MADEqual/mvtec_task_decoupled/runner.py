from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from DINOv3.MADEqual.mvtec_broad6_compose2.metrics import (
    FAST_AUPRO_BACKEND,
    METRIC_NAMES,
    category_macro,
    evaluate_fast,
)
from DINOv3.MADEqual.mvtec_broad6_compose2.runner import (
    _atomic_json,
    _load_mask,
    _read_csv,
    _relative,
    _write_csv,
)
from DINOv3.nsrm.m0_scoring import prepare_native_mask_context
from DINOv3.relation_oracle_cache.extract import mad_equal
from DINOv3.relation_reliability.datasets import Record, mvtec_ad_records
from DINOv3.relation_reliability.protocol import LAYERS, MVTEC_AD_CATEGORIES
from DINOv3.relation_reliability.superadd_host import interpolate_score


PROTOCOL_ID = "mad_equal_mvtec_task_decoupled_seed42_v1"
BASE_PROTOCOL_ID = "mad_equal_mvtec_noaug_seed42_v1"
BASE_METHOD = "NOAUG_MAD_EQUAL"
RAW_SHARED = "NOAUG_RAW_SHARED"
MAD_SHARED = "NOAUG_MAD_SHARED"
DUAL_RAW_MAD = "NOAUG_DUAL_RAW_MAD"
DUAL_L31_MAD = "NOAUG_DUAL_L31_MAD"
METHODS = (RAW_SHARED, MAD_SHARED, DUAL_RAW_MAD, DUAL_L31_MAD)
PIXEL_METRICS = ("pixel_AUROC", "pixel_AUPR", "AUPRO_at_0p3")
IMAGE_METRICS = ("image_AUROC", "image_AUPR")
BACKEND_FIELDS = (
    "aupro_backend",
    "pixel_ranking_backend",
    "aupro_count_backend",
    "aupro_threshold_count",
    "aupro_threshold_sampling",
    "component_connectivity",
)


def build_patch_maps(
    raw: np.ndarray,
    medians: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != len(LAYERS):
        raise RuntimeError(f"raw distance shape must be [4,H,W], got {values.shape}")
    if not np.isfinite(values).all():
        raise FloatingPointError("raw distances contain NaN/Inf")
    raw_map = values.mean(axis=0, dtype=np.float32)
    mad_map = np.asarray(mad_equal(values, medians, scales), dtype=np.float32)
    l31_map = values[3]
    expected = values.shape[1:]
    if any(score.shape != expected for score in (raw_map, mad_map, l31_map)):
        raise RuntimeError("task-decoupled patch maps have inconsistent shapes")
    if any(not np.isfinite(score).all() for score in (raw_map, mad_map, l31_map)):
        raise FloatingPointError("task-decoupled anomaly maps contain NaN/Inf")
    return raw_map, mad_map, l31_map


def image_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    target = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if target.shape != values.shape or set(target.tolist()) != {0, 1}:
        raise ValueError("image labels/scores must be aligned with normal and anomaly support")
    if not np.isfinite(values).all():
        raise FloatingPointError("image scores contain NaN/Inf")
    return {
        "image_AUROC": float(roc_auc_score(target, values)),
        "image_AUPR": float(average_precision_score(target, values)),
    }


def assemble_category_metrics(
    category: str,
    *,
    image_count: int,
    normal_count: int,
    anomaly_count: int,
    raw_metrics: dict[str, Any],
    mad_metrics: dict[str, Any],
    l31_image_metrics: dict[str, float],
) -> list[dict[str, Any]]:
    def row(
        method: str,
        image_source: dict[str, Any],
        pixel_source: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "method": method,
            "category": category,
            "image_count": int(image_count),
            "normal_count": int(normal_count),
            "anomaly_count": int(anomaly_count),
            **{metric: float(image_source[metric]) for metric in IMAGE_METRICS},
            **{metric: float(pixel_source[metric]) for metric in PIXEL_METRICS},
            **{
                field: pixel_source[field]
                for field in BACKEND_FIELDS
                if field in pixel_source
            },
        }

    rows = [
        row(RAW_SHARED, raw_metrics, raw_metrics),
        row(MAD_SHARED, mad_metrics, mad_metrics),
        row(DUAL_RAW_MAD, raw_metrics, mad_metrics),
        row(DUAL_L31_MAD, l31_image_metrics, mad_metrics),
    ]
    assert_decoupling_consistency(rows)
    return rows


def assert_decoupling_consistency(rows: Sequence[dict[str, Any]]) -> None:
    by_method = {str(row["method"]): row for row in rows}
    if set(by_method) != set(METHODS):
        raise RuntimeError("task-decoupled method rows are incomplete")
    raw = by_method[RAW_SHARED]
    mad = by_method[MAD_SHARED]
    dual = by_method[DUAL_RAW_MAD]
    dual_l31 = by_method[DUAL_L31_MAD]
    for metric in IMAGE_METRICS:
        if float(dual[metric]) != float(raw[metric]):
            raise RuntimeError(f"Dual Raw/MAD image metric differs from Raw: {metric}")
    for metric in PIXEL_METRICS:
        expected = float(mad[metric])
        if float(dual[metric]) != expected or float(dual_l31[metric]) != expected:
            raise RuntimeError(f"Dual pixel metric differs from MAD: {metric}")


def _load_calibration(base_output: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = _read_csv(base_output / "calibration_stats.csv")
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for category in MVTEC_AD_CATEGORIES:
        selected = {
            int(row["layer"]): row
            for row in rows
            if row.get("category") == category and row.get("method") == BASE_METHOD
        }
        if tuple(sorted(selected)) != tuple(LAYERS):
            raise RuntimeError(f"NoAug MAD calibration is incomplete: {category}")
        output[category] = (
            np.asarray([float(selected[layer]["median"]) for layer in LAYERS], dtype=np.float32),
            np.asarray(
                [float(selected[layer]["mad_scale_1p4826"]) for layer in LAYERS],
                dtype=np.float32,
            ),
        )
    return output


def _load_base_metrics(base_output: Path) -> dict[str, dict[str, str]]:
    rows = [
        row for row in _read_csv(base_output / "category_metrics.csv")
        if row.get("method") == BASE_METHOD
    ]
    if len(rows) != 15 or {row["category"] for row in rows} != set(MVTEC_AD_CATEGORIES):
        raise RuntimeError("base NoAug MAD category metrics are incomplete")
    return {row["category"]: row for row in rows}


def _load_raw(
    path: Path,
    *,
    category: str,
    relative_path: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"required NoAug raw cache is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if "raw_distances" not in payload.files or "output_shape" not in payload.files:
            raise RuntimeError(f"NoAug raw cache arrays are incomplete: {path}")
        raw = np.asarray(payload["raw_distances"], dtype=np.float32)
        output_shape = tuple(int(value) for value in payload["output_shape"].tolist())
        if "category" in payload.files and str(payload["category"].item()) != category:
            raise RuntimeError(f"NoAug raw cache category mismatch: {path}")
        if (
            "relative_path" in payload.files
            and str(payload["relative_path"].item()) != relative_path
        ):
            raise RuntimeError(f"NoAug raw cache order/path mismatch: {path}")
    if raw.ndim != 3 or raw.shape[0] != len(LAYERS):
        raise RuntimeError(f"NoAug raw cache is not [4,H,W]: {path}")
    if len(output_shape) != 2 or min(output_shape) <= 0:
        raise RuntimeError(f"NoAug raw output shape is invalid: {path}")
    if not np.isfinite(raw).all():
        raise FloatingPointError(f"NoAug raw cache contains NaN/Inf: {path}")
    return raw, output_shape


def _compare_base_mad(
    category: str,
    row: dict[str, Any],
    base_row: dict[str, str],
) -> None:
    for metric in METRIC_NAMES:
        if not np.isclose(
            float(row[metric]), float(base_row[metric]), atol=1e-9, rtol=0.0,
        ):
            raise RuntimeError(f"NoAug MAD reproduction failed: {category}/{metric}")


def _report(
    category_rows: Sequence[dict[str, Any]],
    macro_rows: Sequence[dict[str, Any]],
    runtime_seconds: float,
) -> str:
    by_macro = {str(row["method"]): row for row in macro_rows}
    lines = [
        "# MVTec AD 检测—定位任务解耦客观结果",
        "",
        f"Protocol: `{PROTOCOL_ID}`。本实验只读取 NoAug 四层 raw cache 与既有 MAD 参数，不运行 memory、DINO backbone 或 Exact 1-NN。",
        "",
        "## 15类非加权 macro",
        "",
        "| Method | I-AUROC | I-AUPR | P-AUROC | P-AUPR | AUPRO@0.3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = by_macro[method]
        lines.append(
            f"| {method} | "
            + " | ".join(f"{float(row[metric]):.6f}" for metric in METRIC_NAMES)
            + " |"
        )
    lines += ["", "## 逐类别五项指标", ""]
    lines += [
        "| Category | Method | I-AUROC | I-AUPR | P-AUROC | P-AUPR | AUPRO@0.3 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for category in MVTEC_AD_CATEGORIES:
        for method in METHODS:
            row = next(
                value for value in category_rows
                if value["category"] == category and value["method"] == method
            )
            lines.append(
                f"| {category} | {method} | "
                + " | ".join(f"{float(row[metric]):.6f}" for metric in METRIC_NAMES)
                + " |"
            )

    lines += [
        "",
        "## 逐类别分支差值",
        "",
        "| Category | Raw−MAD I-AUROC | Raw−MAD I-AUPR | MAD−Raw P-AUROC | MAD−Raw P-AUPR | MAD−Raw AUPRO | L31−Raw I-AUROC | L31−Raw I-AUPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category in MVTEC_AD_CATEGORIES:
        category_by_method = {
            str(row["method"]): row
            for row in category_rows
            if row["category"] == category
        }
        raw_category = category_by_method[RAW_SHARED]
        mad_category = category_by_method[MAD_SHARED]
        l31_category = category_by_method[DUAL_L31_MAD]
        differences = (
            float(raw_category["image_AUROC"]) - float(mad_category["image_AUROC"]),
            float(raw_category["image_AUPR"]) - float(mad_category["image_AUPR"]),
            float(mad_category["pixel_AUROC"]) - float(raw_category["pixel_AUROC"]),
            float(mad_category["pixel_AUPR"]) - float(raw_category["pixel_AUPR"]),
            float(mad_category["AUPRO_at_0p3"]) - float(raw_category["AUPRO_at_0p3"]),
            float(l31_category["image_AUROC"]) - float(raw_category["image_AUROC"]),
            float(l31_category["image_AUPR"]) - float(raw_category["image_AUPR"]),
        )
        lines.append(
            f"| {category} | "
            + " | ".join(f"{value:+.6f}" for value in differences)
            + " |"
        )

    raw = by_macro[RAW_SHARED]
    mad = by_macro[MAD_SHARED]
    dual = by_macro[DUAL_RAW_MAD]
    l31 = by_macro[DUAL_L31_MAD]
    lines += [
        "",
        "## Macro 分支差值",
        "",
        "| Comparison | I-AUROC | I-AUPR | P-AUROC | P-AUPR | AUPRO@0.3 |",
        "|---|---:|---:|---:|---:|---:|",
        "| Raw shared − MAD shared | "
        + " | ".join(f"{float(raw[m])-float(mad[m]):+.6f}" for m in METRIC_NAMES)
        + " |",
        "| Dual Raw/MAD − Raw shared | "
        + " | ".join(f"{float(dual[m])-float(raw[m]):+.6f}" for m in METRIC_NAMES)
        + " |",
        "| Dual Raw/MAD − MAD shared | "
        + " | ".join(f"{float(dual[m])-float(mad[m]):+.6f}" for m in METRIC_NAMES)
        + " |",
        "| Dual L31/MAD − Dual Raw/MAD | "
        + " | ".join(f"{float(l31[m])-float(dual[m]):+.6f}" for m in METRIC_NAMES)
        + " |",
        "",
        "按定义，Dual Raw/MAD 的图像指标与 Raw shared 完全相同，像素指标与 MAD shared 完全相同；Dual L31/MAD 只替换图像分支。",
        "",
        f"实际后评测运行时间：{runtime_seconds:.3f} 秒。",
        "",
        f"Pixel/AUPRO backend: `{FAST_AUPRO_BACKEND}`。以上仅为客观数值记录，不包含方法有效性、创新性或论文结论。",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    base_output = Path(args.base_output).resolve()
    output = Path(args.output_dir).resolve()
    if output == base_output:
        raise RuntimeError("output-dir must differ from the read-only base-output")
    complete_path = base_output / "complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(f"base output is missing complete.json: {base_output}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        complete.get("status") != "complete"
        or complete.get("protocol_id") != BASE_PROTOCOL_ID
        or int(complete.get("category_count", -1)) != 15
        or int(complete.get("prediction_count", -1)) != 1725
    ):
        raise RuntimeError("base output is not the completed MVTec NoAug formal run")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal task-decoupled evaluation requires CUDA fast metrics")

    calibration = _load_calibration(base_output)
    base_metrics = _load_base_metrics(base_output)
    train, evaluation = mvtec_ad_records(Path(args.mvtec_root))
    del train
    dataset_root = Path(evaluation[MVTEC_AD_CATEGORIES[0]][0].path).resolve().parents[3]
    output.mkdir(parents=True, exist_ok=True)

    all_category_rows: list[dict[str, Any]] = []
    all_sample_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for category in MVTEC_AD_CATEGORIES:
        category_started = time.perf_counter()
        records: Sequence[Record] = evaluation[category]
        medians, scales = calibration[category]
        labels = [int(record.label) for record in records]
        masks: list[np.ndarray] = []
        raw_maps: list[np.ndarray] = []
        mad_maps: list[np.ndarray] = []
        l31_scores: list[float] = []
        cache_root = base_output / "categories" / category / "raw_cache" / "test"
        for index, record in enumerate(records):
            relative_path = _relative(record, dataset_root)
            raw, output_shape = _load_raw(
                cache_root / f"test_{index:04d}.npz",
                category=category,
                relative_path=relative_path,
            )
            raw_patch, mad_patch, l31_patch = build_patch_maps(raw, medians, scales)
            raw_final = np.asarray(interpolate_score(raw_patch, output_shape), dtype=np.float32)
            mad_final = np.asarray(interpolate_score(mad_patch, output_shape), dtype=np.float32)
            l31_final = np.asarray(interpolate_score(l31_patch, output_shape), dtype=np.float32)
            if any(not np.isfinite(value).all() for value in (raw_final, mad_final, l31_final)):
                raise FloatingPointError(f"interpolated anomaly map contains NaN/Inf: {relative_path}")
            masks.append(_load_mask(record, output_shape))
            raw_maps.append(raw_final)
            mad_maps.append(mad_final)
            raw_score = float(raw_final.max())
            mad_score = float(mad_final.max())
            l31_score = float(l31_final.max())
            l31_scores.append(l31_score)
            all_sample_rows.append({
                "category": category,
                "relative_path": relative_path,
                "label": int(record.label),
                "raw_image_score": raw_score,
                "mad_image_score": mad_score,
                "l31_image_score": l31_score,
            })
        context = prepare_native_mask_context(masks, np.asarray(labels, dtype=np.int64))
        raw_metrics = evaluate_fast(
            labels,
            masks,
            raw_maps,
            device=device,
            allow_cpu_fallback=False,
            mask_context=context,
        )
        mad_metrics = evaluate_fast(
            labels,
            masks,
            mad_maps,
            device=device,
            allow_cpu_fallback=False,
            mask_context=context,
        )
        l31_metrics = image_metrics(labels, l31_scores)
        rows = assemble_category_metrics(
            category,
            image_count=len(records),
            normal_count=sum(label == 0 for label in labels),
            anomaly_count=sum(label == 1 for label in labels),
            raw_metrics=raw_metrics,
            mad_metrics=mad_metrics,
            l31_image_metrics=l31_metrics,
        )
        _compare_base_mad(
            category,
            next(row for row in rows if row["method"] == MAD_SHARED),
            base_metrics[category],
        )
        all_category_rows.extend(rows)
        runtime_rows.append({
            "category": category,
            "stage": "raw_cache_post_evaluation",
            "fast_pixel_evaluations": 2,
            "seconds": time.perf_counter() - category_started,
        })
        print(f"[MVTEC-TASK-DECOUPLED] category {category} complete", flush=True)

    macro_rows = category_macro(all_category_rows)
    assert_decoupling_consistency(macro_rows)
    total_seconds = time.perf_counter() - started
    runtime_rows.append({
        "category": "ALL",
        "stage": "total",
        "fast_pixel_evaluations": 30,
        "seconds": total_seconds,
    })
    _write_csv(output / "category_metrics.csv", all_category_rows)
    _write_csv(output / "macro_metrics.csv", macro_rows)
    _write_csv(output / "sample_scores.csv", all_sample_rows)
    _write_csv(output / "runtime.csv", runtime_rows)
    (output / "objective_report.md").write_text(
        _report(all_category_rows, macro_rows, total_seconds), encoding="utf-8",
    )
    _atomic_json(output / "complete.json", {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "run_type": "post_evaluation",
        "dataset": "MVTec AD",
        "seed": 42,
        "category_count": 15,
        "prediction_count": 1725,
        "methods": list(METHODS),
        "layers": list(LAYERS),
        "base_output": str(base_output),
        "base_mad_reproduced": True,
        "memory_rebuilt": False,
        "backbone_run": False,
        "exact_1nn_run": False,
        "fast_pixel_evaluations_per_category": 2,
        "aupro_backend": FAST_AUPRO_BACKEND,
        "runtime_seconds": total_seconds,
    })
    print(f"MVTEC_TASK_DECOUPLED_COMPLETE={output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MVTec AD Raw/MAD detection-localization task decoupling",
    )
    parser.add_argument("--base-output", required=True)
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
