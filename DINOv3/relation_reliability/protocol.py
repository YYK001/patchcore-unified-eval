from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROTOCOL_ID = "relation_reliability_exact_superadd_seed42_v6"
SEED = 42
LAYERS = (7, 15, 23, 31)
MEMORY_PER_LAYER = 100_000
PATCH_SIZE = 640
PATCH_OVERLAP = 128
MODEL_PATCH_SIZE = 16
BACKBONE = "dinov3_vith16plus"
RELATION_AGGREGATION = "H1"
RELATION_DIM_PER_LAYER = 11
JOINT_RELATION_DIM = 44
KMEANS_CLUSTERS = 64
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300
CODEBOOK_MAX_PATCHES = 50_000
MIN_RELIABILITY_PAIRS = 256
ECDF_EPSILON = 1e-8
LCB_CONFIDENCE = 0.95
LCB_ONE_SIDED_Z = 1.6448536269514722
AUPRO_METRIC = "pixel_AUPRO_adeval_nstrips200"
POSITION_BINS = 8
FEATURE_PCA_COMPONENTS = 32
THRESHOLD_FRACTION = 8
TRANSFORM_IDS = tuple(range(1, 9))
PSEUDO_FAMILIES = ("scratch", "stain", "texture_replacement", "local_missing")

MVTEC_AD_CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
)
ROBUSTAD_CATEGORIES = ("MetalParts", "PCB", "PiledBags")

METHODS = (
    "Official",
    "L7", "L15", "L23", "L31",
    "ECDF-Equal",
    "MAD-Equal",
    "Global",
    "Relation",
    "Shuffled",
    "Feature",
    "Position",
)


@dataclass(frozen=True)
class FrozenProtocol:
    protocol: str = PROTOCOL_ID
    seed: int = SEED
    backbone: str = BACKBONE
    layers: tuple[int, ...] = LAYERS
    memory_budget_cap_per_category_per_layer: int = MEMORY_PER_LAYER
    memory_budget_semantics: str = "min(candidate_count,100000)_no_duplicate_padding"
    patch_size: int = PATCH_SIZE
    patch_overlap: int = PATCH_OVERLAP
    small_image_policy: str = (
        "official_resize_unmodified_when_both_sides_ge_640_else_"
        "isotropic_min_side_640_ceil_patch16"
    )
    relation_aggregation: str = RELATION_AGGREGATION
    relation_dim_per_layer: int = RELATION_DIM_PER_LAYER
    joint_relation_dim: int = JOINT_RELATION_DIM
    kmeans_clusters: int = KMEANS_CLUSTERS
    kmeans_n_init: int = KMEANS_N_INIT
    kmeans_max_iter: int = KMEANS_MAX_ITER
    codebook_max_physical_patches: int = CODEBOOK_MAX_PATCHES
    reliability_lcb: str = "one_sided_Wilson_95_percent"
    reliability_lcb_sampling_unit: str = "valid_physical_patch_x_transform_x_proxy_family_pair"
    reliability_dependence_interpretation: str = "descriptive_patch_pair_LCB_not_image_level_confidence_interval"
    aupro_backend: str = "ADEval_CUDA_nstrips200"
    aupro_package: str = "adeval==1.1.0"
    min_reliability_pairs: int = MIN_RELIABILITY_PAIRS
    transforms: tuple[int, ...] = TRANSFORM_IDS
    pseudo_families: tuple[str, ...] = PSEUDO_FAMILIES
    methods: tuple[str, ...] = METHODS
    mvtec_ad_categories: tuple[str, ...] = MVTEC_AD_CATEGORIES
    robustad_categories: tuple[str, ...] = ROBUSTAD_CATEGORIES
    test_relation_assignment: str = "four_top1_normal_memory_anchors_only"
    two_two_tie_break: str = "smaller_supporting_layer_mean_ECDF_distance"
    all_distinct_fallback: str = "global_layer_reliability"
    layer_weight_temperature: str = "none_direct_normalization"
    feature_execution_backend: str = "gpu_resident_torch_no_numpy_roundtrip"
    calibration_nn_scope: str = "exact_defect_patch_queries_normal_uses_four_family_union"
    intervention_microbatch_policy: str = "official_per_image_forward_batch1"
    preprocessing_rng_consumption: str = "one_fixed_brightness_draw_per_image_serial_execution"
    nn_distance_matrix_max_bytes: int = 1 << 30
    intervention_query_aggregation: str = "per_transform_group_one_exact_1nn_call_per_layer"
    official_candidate_storage: str = "ram_first_16GiB_headroom_disk_fallback"
    cpu_prefetch_queue: int = 2
    execution_policy: str = "complete_all_categories_no_result_dependent_stopping"
    final_judgment_policy: str = "descriptive_only_no_preregistered_pass_fail_gate"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_u64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def validate_superadd_config(config: dict[str, Any]) -> None:
    expected = {
        "backbone": BACKBONE,
        "layers": list(LAYERS),
        "max_database_size": MEMORY_PER_LAYER,
        "patch_size": PATCH_SIZE,
        "patch_overlap": PATCH_OVERLAP,
        "threshold_fraction": THRESHOLD_FRACTION,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"SuperADD configuration violates the frozen protocol: {mismatches}")


def protocol_fingerprint(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = FrozenProtocol().as_dict()
    if extra:
        payload["runtime_inputs"] = extra
    return {"protocol": payload, "sha256": stable_json_hash(payload)}
