from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


RADIAL_EDGES = (0.0, 2.0, 4.0, 8.0, 16.0, float("inf"))
RANK_FRACTIONS = (0.01, 0.05, 0.20, 0.50, 1.0)


def _patch_geometry(side: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    y, x = torch.meshgrid(
        torch.arange(side, device=device),
        torch.arange(side, device=device),
        indexing="ij",
    )
    coords = torch.stack((y.flatten(), x.flatten()), dim=1).float()
    distance = torch.cdist(coords, coords, p=2)
    return coords, distance


def _mass_by_bins(weights: torch.Tensor, bins: torch.Tensor, n_bins: int) -> torch.Tensor:
    parts = []
    for idx in range(int(n_bins)):
        parts.append((weights * (bins == idx)).sum(dim=-1))
    return torch.stack(parts, dim=-1)


def relation_signature(
    attention: torch.Tensor,
    grid: tuple[int, int],
    aggregation: str = "H1",
) -> torch.Tensor:
    """Convert patch self-attention into a deterministic per-patch ARS vector.

    Input attention is [B, heads, cls+register+patches, cls+register+patches].
    The output is [B, patches, 11] for H1 or [B, patches, 44] for H4.
    """
    if attention.ndim != 4:
        raise ValueError("NSRM relation_signature expects [B, H, N, N] attention")
    side_h, side_w = grid
    patches = side_h * side_w
    prefix = attention.shape[-1] - patches
    if prefix < 1:
        raise ValueError("Attention tensor has no CLS prefix")
    weights = attention[:, :, prefix:, prefix:].float()
    weights = weights.clone()
    weights.diagonal(dim1=-2, dim2=-1).zero_()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    _, distances = _patch_geometry(side_h, weights.device)
    radial_bins = torch.bucketize(
        distances,
        torch.tensor(RADIAL_EDGES[1:-1], device=weights.device),
        right=True,
    )
    radial = _mass_by_bins(weights, radial_bins, len(RADIAL_EDGES) - 1).permute(0, 2, 1, 3)

    order = torch.argsort(weights, dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(patches, device=weights.device).view(1, 1, 1, -1)
    ranks.scatter_(-1, order, rank_values.expand_as(order))
    thresholds = torch.tensor(
        [max(1, math.ceil(patches * fraction)) for fraction in RANK_FRACTIONS[:-1]],
        device=ranks.device,
        dtype=torch.long,
    )
    rank_bins = torch.bucketize(ranks, thresholds, right=True)
    rank_mass = _mass_by_bins(weights, rank_bins, len(RANK_FRACTIONS)).permute(0, 2, 1, 3)

    entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
    entropy = (entropy / math.log(max(2, patches))).permute(0, 2, 1).unsqueeze(-1)
    per_head = torch.cat((radial, rank_mass, entropy), dim=-1)  # [B,P,H,11]
    if aggregation == "H1":
        return per_head.mean(dim=2)
    if aggregation != "H4":
        raise ValueError(f"Unknown ARS aggregation: {aggregation}")
    heads = per_head.shape[2]
    if heads % 4 != 0:
        raise ValueError("H4 requires the number of heads to be divisible by 4")
    grouped = per_head.reshape(per_head.shape[0], patches, 4, heads // 4, per_head.shape[-1]).mean(dim=3)
    return grouped.reshape(per_head.shape[0], patches, -1)


@dataclass(frozen=True)
class RelationAudit:
    category: str
    layer: int
    aggregation: str
    transform_id: int
    rho: float
    cluster_consistency: float
    random_pair_consistency: float


def normalize_relation_trajectories(raw: np.ndarray) -> np.ndarray:
    """Median/IQR-scale using physical-patch medians, then L2-normalize."""
    if raw.ndim != 4:
        raise ValueError("raw relation trajectories must be [images, transforms, patches, dim]")
    physical_median = np.median(raw, axis=1)
    center = np.median(physical_median.reshape(-1, raw.shape[-1]), axis=0)
    q1 = np.percentile(physical_median.reshape(-1, raw.shape[-1]), 25, axis=0)
    q3 = np.percentile(physical_median.reshape(-1, raw.shape[-1]), 75, axis=0)
    scale = np.maximum(q3 - q1, 1e-8)
    normalized = np.clip((raw - center) / scale, -5.0, 5.0)
    norm = np.linalg.norm(normalized, axis=-1, keepdims=True)
    return normalized / np.maximum(norm, 1e-8)


def cluster_consistency(
    identity: np.ndarray,
    transformed: np.ndarray,
    seed: int,
    clusters: int = 64,
) -> tuple[float, float]:
    if identity.shape != transformed.shape:
        raise ValueError("identity and transformed signatures must have the same shape")
    flat_identity = identity.reshape(-1, identity.shape[-1])
    flat_transformed = transformed.reshape(-1, transformed.shape[-1])
    n_clusters = min(int(clusters), max(2, flat_identity.shape[0]))
    # Import lazily so the protocol/geometry helpers remain usable in a minimal
    # environment; the server runtime pins scikit-learn for the actual P0 run.
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(4096, flat_identity.shape[0]),
        n_init=10,
        max_iter=300,
        random_state=int(seed),
    )
    labels = model.fit_predict(flat_identity)
    transformed_labels = model.predict(flat_transformed)
    consistency = float(np.mean(labels == transformed_labels))
    rng = np.random.default_rng(int(seed))
    shuffled = flat_transformed[rng.permutation(len(flat_transformed))]
    random_labels = model.predict(shuffled)
    random_consistency = float(np.mean(labels == random_labels))
    return consistency, random_consistency


def fit_relation_clusters(identity: np.ndarray, seed: int, clusters: int = 64):
    from sklearn.cluster import MiniBatchKMeans

    flat_identity = identity.reshape(-1, identity.shape[-1])
    n_clusters = min(int(clusters), max(2, flat_identity.shape[0]))
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(4096, flat_identity.shape[0]),
        n_init=10,
        max_iter=300,
        random_state=int(seed),
    ).fit(flat_identity)


def cluster_consistency_from_model(
    model,
    identity: np.ndarray,
    transformed: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    flat_identity = identity.reshape(-1, identity.shape[-1])
    flat_transformed = transformed.reshape(-1, transformed.shape[-1])
    labels = model.predict(flat_identity)
    transformed_labels = model.predict(flat_transformed)
    consistency = float(np.mean(labels == transformed_labels))
    rng = np.random.default_rng(int(seed))
    shuffled = flat_transformed[rng.permutation(len(flat_transformed))]
    random_labels = model.predict(shuffled)
    random_consistency = float(np.mean(labels == random_labels))
    return consistency, random_consistency

