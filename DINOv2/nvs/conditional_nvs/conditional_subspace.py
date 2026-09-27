from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .subspace import fit_centered_basis, fit_uncentered_basis


@dataclass(frozen=True)
class PrototypeModel:
    centers: torch.Tensor
    labels: torch.Tensor
    feature_bases: torch.Tensor
    delta_bases: torch.Tensor
    fallback: torch.Tensor
    unique_patch_counts: torch.Tensor
    unique_image_counts: torch.Tensor
    global_feature_basis: torch.Tensor
    global_delta_basis: torch.Tensor
    rank: int
    sr_gain: torch.Tensor | None = None
    sr_instability: torch.Tensor | None = None
    sr_weights: torch.Tensor | None = None

    def statistics(self) -> dict:
        counts = torch.bincount(self.labels, minlength=self.centers.shape[0])
        fallback_patch_count = int(counts[self.fallback].sum())
        stats = {
            "prototype_count": int(self.centers.shape[0]),
            "fallback_count": int(self.fallback.sum()),
            "fallback_fraction": float(self.fallback.float().mean()),
            "fallback_patch_fraction": float(
                fallback_patch_count / max(1, int(self.labels.numel()))
            ),
            "member_counts": counts.detach().cpu().tolist(),
            "unique_original_patch_counts": self.unique_patch_counts.detach().cpu().tolist(),
            "unique_image_counts": self.unique_image_counts.detach().cpu().tolist(),
            "fallback_ids": torch.nonzero(self.fallback, as_tuple=True)[0]
            .detach()
            .cpu()
            .tolist(),
        }
        if self.sr_weights is not None:
            weights = self.sr_weights.detach().float().cpu()
            stats["sr_cnvs"] = {
                "enabled": True,
                "weight_mean": float(weights.mean()),
                "weight_median": float(weights.median()),
                "weight_min": float(weights.min()),
                "weight_max": float(weights.max()),
                "near_zero_fraction": float((weights <= 1.0e-3).float().mean()),
                "weights": weights.tolist(),
                "gain": None
                if self.sr_gain is None
                else self.sr_gain.detach().float().cpu().tolist(),
                "instability": None
                if self.sr_instability is None
                else self.sr_instability.detach().float().cpu().tolist(),
            }
        return stats


def spherical_kmeans(
    values: torch.Tensor,
    clusters: int,
    seed: int,
    max_iter: int = 50,
    tolerance: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = F.normalize(values.reshape(-1, values.shape[-1]).float(), dim=-1)
    n = int(values.shape[0])
    k = min(max(1, int(clusters)), n)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    initial = torch.randperm(n, generator=generator)[:k].to(values.device)
    centers = values[initial].clone()
    labels = torch.full((n,), -1, dtype=torch.long, device=values.device)
    for _ in range(int(max_iter)):
        new_labels = (values @ centers.T).argmax(dim=1)
        new_centers = torch.zeros_like(centers)
        new_centers.index_add_(0, new_labels, values)
        counts = torch.bincount(new_labels, minlength=k)
        empty = torch.nonzero(counts == 0, as_tuple=True)[0]
        if empty.numel():
            nearest_similarity = (values @ centers.T).max(dim=1).values
            replacements = torch.argsort(nearest_similarity)[: empty.numel()]
            new_centers[empty] = values[replacements]
        new_centers = F.normalize(new_centers, dim=-1)
        shift = torch.linalg.norm(new_centers - centers, dim=1).max().item()
        converged = torch.equal(labels, new_labels) or shift <= float(tolerance)
        labels, centers = new_labels, new_centers
        if converged:
            break
    return centers.contiguous(), labels.contiguous()


def assign_prototypes(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    device = values.device
    normalized = F.normalize(values.reshape(-1, values.shape[-1]).float(), dim=-1)
    normalized_centers = F.normalize(
        centers.float().to(device, non_blocking=True), dim=-1
    )
    return (normalized @ normalized_centers.T).argmax(dim=1)


def _explained_ratio(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1, values.shape[-1]).float()
    basis = basis.float().to(flat.device, non_blocking=True)
    projected_energy = (flat @ basis.T).square().sum(dim=1)
    total_energy = flat.square().sum(dim=1).clamp_min(1.0e-12)
    return projected_energy / total_energy


def _projector_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.float()
    right = right.float().to(left.device, non_blocking=True)
    rank = max(1, min(int(left.shape[0]), int(right.shape[0])))
    overlap = left @ right.T
    distance_sq = (2.0 * rank - 2.0 * overlap.square().sum()).clamp_min(0.0)
    return torch.sqrt(distance_sq / max(1.0, 2.0 * float(rank)))


def estimate_sr_weights(
    deltas: torch.Tensor,
    labels: torch.Tensor,
    image_ids: torch.Tensor,
    global_basis: torch.Tensor,
    local_bases: torch.Tensor,
    fallback: torch.Tensor,
    rank: int,
    seed: int,
    bootstrap_repeats: int = 5,
    bootstrap_fraction: float = 0.8,
    weight_epsilon: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate SR-CNVS prototype trust weights from nvs_fit normal deltas only.

    g_c is an image-level two-fold cross-fit explanation gain of the local
    delta basis over the final global basis. v_c is bootstrap projector
    instability against the final local projector. No calibration, test image,
    anomaly mask, or threshold statistic enters this estimate.
    """

    device = deltas.device
    labels = labels.long().to(device, non_blocking=True)
    image_ids = image_ids.long().to(device, non_blocking=True)
    fallback = fallback.bool().to(device, non_blocking=True)
    k = int(local_bases.shape[0])
    rank = int(rank)
    gains = torch.zeros(k, dtype=torch.float32, device=device)
    instabilities = torch.zeros(k, dtype=torch.float32, device=device)
    weights = torch.zeros(k, dtype=torch.float32, device=device)
    minimum_patches = max(8 * rank, 64)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 7919)
    bootstrap_fraction = min(max(float(bootstrap_fraction), 0.05), 1.0)
    bootstrap_repeats = max(1, int(bootstrap_repeats))

    for prototype in range(k):
        members = torch.nonzero(labels == prototype, as_tuple=True)[0]
        unique_images = torch.unique(image_ids[members])
        if (
            fallback[prototype]
            or members.numel() < minimum_patches
            or unique_images.numel() < 4
        ):
            continue

        order = torch.randperm(unique_images.numel(), generator=generator).to(device)
        shuffled_images = unique_images[order]
        folds = torch.chunk(shuffled_images, 2)
        fold_gains: list[torch.Tensor] = []
        for held_images in folds:
            if held_images.numel() == 0:
                continue
            held_mask = torch.isin(image_ids[members], held_images)
            held_members = members[held_mask]
            train_members = members[~held_mask]
            if train_members.numel() < rank or held_members.numel() == 0:
                continue
            train_delta = deltas[train_members].reshape(-1, deltas.shape[-1])
            if train_delta.shape[0] < rank:
                continue
            local_basis = fit_uncentered_basis(train_delta, rank)
            held_delta = deltas[held_members].reshape(-1, deltas.shape[-1])
            local_e = _explained_ratio(held_delta, local_basis)
            global_e = _explained_ratio(held_delta, global_basis)
            fold_gains.append((local_e - global_e).mean())
        if fold_gains:
            gains[prototype] = torch.stack(fold_gains).mean()

        distances: list[torch.Tensor] = []
        final_basis = local_bases[prototype]
        unique_count = int(unique_images.numel())
        sample_count = max(2, int(round(unique_count * bootstrap_fraction)))
        sample_count = min(sample_count, unique_count)
        for _ in range(bootstrap_repeats):
            sample_order = torch.randperm(unique_count, generator=generator)[
                :sample_count
            ].to(device)
            sampled_images = unique_images[sample_order]
            sample_members = members[torch.isin(image_ids[members], sampled_images)]
            if sample_members.numel() < rank:
                continue
            sample_delta = deltas[sample_members].reshape(-1, deltas.shape[-1])
            if sample_delta.shape[0] < rank:
                continue
            sample_basis = fit_uncentered_basis(sample_delta, rank)
            distances.append(_projector_distance(sample_basis, final_basis))
        if not distances:
            continue
        instabilities[prototype] = torch.stack(distances).mean()

        positive_gain = torch.clamp(gains[prototype], min=0.0)
        if positive_gain > 0:
            weights[prototype] = positive_gain / (
                positive_gain + instabilities[prototype] + float(weight_epsilon)
            )

    return gains, instabilities, weights


def fit_prototype_model(
    original_features: torch.Tensor,
    deltas: torch.Tensor,
    image_ids: torch.Tensor,
    prototypes: int,
    rank: int,
    seed: int,
    max_iter: int = 50,
    stability_regularization: bool = False,
    bootstrap_repeats: int = 5,
    bootstrap_fraction: float = 0.8,
    weight_epsilon: float = 1.0e-8,
) -> PrototypeModel:
    """Fit shared appearance prototypes and D1/D3 bases.

    ``original_features`` is [N,D], ``deltas`` is [N,T,D], and ``image_ids``
    identifies the source image of each original patch.
    """

    device = original_features.device
    originals = F.normalize(original_features.float(), dim=-1)
    deltas = deltas.float().to(device, non_blocking=True)
    image_ids = image_ids.long().to(device, non_blocking=True)
    if originals.ndim != 2 or deltas.ndim != 3:
        raise ValueError("Expected originals [N,D] and deltas [N,T,D]")
    if originals.shape[0] != deltas.shape[0] or image_ids.numel() != originals.shape[0]:
        raise ValueError("Original patch, delta, and image-id counts must match")
    centers, labels = spherical_kmeans(
        originals, clusters=prototypes, seed=seed, max_iter=max_iter
    )
    _, global_feature_basis = fit_centered_basis(originals, rank)
    global_delta_basis = fit_uncentered_basis(
        deltas.reshape(-1, deltas.shape[-1]), rank
    )
    k = int(centers.shape[0])
    feature_bases: list[torch.Tensor] = []
    delta_bases: list[torch.Tensor] = []
    fallback = torch.zeros(k, dtype=torch.bool, device=device)
    patch_counts = torch.zeros(k, dtype=torch.long, device=device)
    image_counts = torch.zeros(k, dtype=torch.long, device=device)
    minimum_patches = max(8 * int(rank), 64)
    for prototype in range(k):
        members = torch.nonzero(labels == prototype, as_tuple=True)[0]
        patch_counts[prototype] = members.numel()
        image_counts[prototype] = torch.unique(image_ids[members]).numel()
        use_fallback = (
            members.numel() < minimum_patches or image_counts[prototype].item() < 4
        )
        fallback[prototype] = use_fallback
        if use_fallback:
            feature_bases.append(global_feature_basis)
            delta_bases.append(global_delta_basis)
        else:
            _, feature_basis = fit_centered_basis(originals[members], rank)
            delta_basis = fit_uncentered_basis(
                deltas[members].reshape(-1, deltas.shape[-1]), rank
            )
            feature_bases.append(feature_basis)
            delta_bases.append(delta_basis)

    final_delta_bases = torch.stack(delta_bases)
    sr_gain = sr_instability = sr_weights = None
    if stability_regularization:
        sr_gain, sr_instability, sr_weights = estimate_sr_weights(
            deltas=deltas,
            labels=labels,
            image_ids=image_ids,
            global_basis=global_delta_basis,
            local_bases=final_delta_bases,
            fallback=fallback,
            rank=rank,
            seed=seed,
            bootstrap_repeats=bootstrap_repeats,
            bootstrap_fraction=bootstrap_fraction,
            weight_epsilon=weight_epsilon,
        )

    return PrototypeModel(
        centers=centers,
        labels=labels,
        feature_bases=torch.stack(feature_bases),
        delta_bases=final_delta_bases,
        fallback=fallback,
        unique_patch_counts=patch_counts,
        unique_image_counts=image_counts,
        global_feature_basis=global_feature_basis,
        global_delta_basis=global_delta_basis,
        rank=int(rank),
        sr_gain=sr_gain,
        sr_instability=sr_instability,
        sr_weights=sr_weights,
    )


def prototype_by_mstar(memory_prototype_ids: torch.Tensor, nn_indices: torch.Tensor) -> torch.Tensor:
    device = nn_indices.device
    return memory_prototype_ids.long().to(device, non_blocking=True)[nn_indices.long()]


def prototype_by_topk_vote_k5(
    memory_prototype_ids: torch.Tensor, topk_indices: torch.Tensor
) -> torch.Tensor:
    if topk_indices.shape[-1] != 5:
        raise ValueError("topk vote protocol is fixed to k=5")
    device = topk_indices.device
    memory_ids = memory_prototype_ids.long().to(device, non_blocking=True)
    votes = memory_ids[topk_indices.long()]
    flat = votes.reshape(-1, 5)
    prototype_count = max(1, int(memory_ids.max().item()) + 1)
    counts = F.one_hot(flat, num_classes=prototype_count).sum(dim=1)
    return counts.argmax(dim=1).reshape(votes.shape[:-1])


def conditional_residual(
    deviation: torch.Tensor, bases: torch.Tensor, prototype_ids: torch.Tensor
) -> torch.Tensor:
    flat = deviation.reshape(-1, deviation.shape[-1]).float()
    ids = prototype_ids.reshape(-1).long().to(flat.device, non_blocking=True)
    selected = bases.float().to(flat.device, non_blocking=True)[ids]
    coefficients = torch.einsum("nd,nrd->nr", flat, selected)
    projected = torch.einsum("nr,nrd->nd", coefficients, selected)
    return torch.linalg.norm(flat - projected, dim=-1).reshape(deviation.shape[:-1])


def score_conditional_deviation(
    deviation: torch.Tensor,
    model: PrototypeModel,
    prototype_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "D1_Proto": conditional_residual(
            deviation, model.feature_bases, prototype_ids
        ),
        "D3_NVSProto": conditional_residual(
            deviation, model.delta_bases, prototype_ids
        ),
    }


def sr_conditional_residual(
    deviation: torch.Tensor,
    model: PrototypeModel,
    prototype_ids: torch.Tensor,
) -> torch.Tensor:
    if model.sr_weights is None:
        raise RuntimeError("SR-CNVS weights have not been fitted")
    flat = deviation.reshape(-1, deviation.shape[-1]).float()
    ids = prototype_ids.reshape(-1).long().to(flat.device, non_blocking=True)
    global_basis = model.global_delta_basis.float().to(flat.device, non_blocking=True)
    local_bases = model.delta_bases.float().to(flat.device, non_blocking=True)[ids]
    weights = model.sr_weights.float().to(flat.device, non_blocking=True)[ids].unsqueeze(1)

    global_projection = (flat @ global_basis.T) @ global_basis
    local_coefficients = torch.einsum("nd,nrd->nr", flat, local_bases)
    local_projection = torch.einsum("nr,nrd->nd", local_coefficients, local_bases)
    sr_projection = (1.0 - weights) * global_projection + weights * local_projection
    return torch.linalg.norm(flat - sr_projection, dim=-1).reshape(deviation.shape[:-1])


def score_sr_deviation(
    deviation: torch.Tensor,
    model: PrototypeModel,
    prototype_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {"SR_CNVS": sr_conditional_residual(deviation, model, prototype_ids)}