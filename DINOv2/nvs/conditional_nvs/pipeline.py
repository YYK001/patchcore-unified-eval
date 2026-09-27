from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .conditional_subspace import (
    PrototypeModel,
    assign_prototypes,
    fit_prototype_model,
    prototype_by_mstar,
    prototype_by_topk_vote_k5,
    score_conditional_deviation,
    score_sr_deviation,
    spherical_kmeans,
)
from .memory import MemoryBuildResult, build_memory, tensor_sha256
from .protocol import CalibrationState, fit_calibration
from .subspace import (
    fit_centered_basis,
    fit_uncentered_basis,
    score_unified_deviation,
)
from .whitening import Whitener, fit_whitener

CORE_METHODS = (
    "D0_NN",
    "D1_Global",
    "D1_Proto",
    "D2_NVSGlobal",
    "D3_NVSProto",
)


@dataclass(frozen=True)
class FeatureSplit:
    memory: torch.Tensor
    nvs_fit_original: torch.Tensor
    nvs_fit_transformed: tuple[torch.Tensor, ...]


def _cosine_topk(
    queries: torch.Tensor,
    bank: torch.Tensor,
    k: int,
    query_chunk_size: int,
    bank_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    queries = F.normalize(queries.float(), dim=-1)
    bank = F.normalize(bank.float().to(queries.device, non_blocking=True), dim=-1)
    k = min(int(k), bank.shape[0])
    all_values, all_indices = [], []
    for start in range(0, queries.shape[0], int(query_chunk_size)):
        query = queries[start : start + int(query_chunk_size)]
        values = torch.full(
            (query.shape[0], k), -float("inf"), device=queries.device
        )
        indices = torch.full(
            (query.shape[0], k), -1, dtype=torch.long, device=queries.device
        )
        for bank_start in range(0, bank.shape[0], int(bank_chunk_size)):
            similarity = query @ bank[bank_start : bank_start + int(bank_chunk_size)].T
            local_k = min(k, similarity.shape[1])
            local_values, local_indices = torch.topk(similarity, local_k, dim=1)
            merged_values = torch.cat([values, local_values], dim=1)
            merged_indices = torch.cat([indices, local_indices + bank_start], dim=1)
            values, order = torch.topk(merged_values, k, dim=1)
            indices = torch.gather(merged_indices, 1, order)
        all_values.append(values)
        all_indices.append(indices)
    return torch.cat(all_values), torch.cat(all_indices)


def _euclidean_topk(
    queries: torch.Tensor,
    bank: torch.Tensor,
    k: int,
    query_chunk_size: int,
    bank_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    queries = queries.float()
    bank = bank.float().to(queries.device, non_blocking=True)
    k = min(int(k), bank.shape[0])
    all_values, all_indices = [], []
    for start in range(0, queries.shape[0], int(query_chunk_size)):
        query = queries[start : start + int(query_chunk_size)]
        values = torch.full(
            (query.shape[0], k), float("inf"), device=queries.device
        )
        indices = torch.full(
            (query.shape[0], k), -1, dtype=torch.long, device=queries.device
        )
        for bank_start in range(0, bank.shape[0], int(bank_chunk_size)):
            block = bank[bank_start : bank_start + int(bank_chunk_size)]
            distances = torch.cdist(query, block).square()
            local_k = min(k, distances.shape[1])
            local_values, local_indices = torch.topk(
                distances, local_k, dim=1, largest=False
            )
            merged_values = torch.cat([values, local_values], dim=1)
            merged_indices = torch.cat([indices, local_indices + bank_start], dim=1)
            values, order = torch.topk(merged_values, k, dim=1, largest=False)
            indices = torch.gather(merged_indices, 1, order)
        all_values.append(values)
        all_indices.append(indices)
    return torch.cat(all_values), torch.cat(all_indices)


def _fit_model_in_appearance_space(
    originals: torch.Tensor,
    deltas: torch.Tensor,
    image_ids: torch.Tensor,
    appearances: torch.Tensor,
    prototypes: int,
    rank: int,
    seed: int,
    stability_regularization: bool = False,
    bootstrap_repeats: int = 5,
    bootstrap_fraction: float = 0.8,
    weight_epsilon: float = 1.0e-8,
) -> PrototypeModel:
    """Mahalanobis-full variant: rebuild clusters while keeping raw delta bases."""

    centers, labels = spherical_kmeans(appearances, prototypes, seed)
    _, global_feature = fit_centered_basis(originals, rank)
    global_delta = fit_uncentered_basis(deltas.reshape(-1, deltas.shape[-1]), rank)
    minimum = max(8 * int(rank), 64)
    feature_bases, delta_bases = [], []
    device = originals.device
    fallback = torch.zeros(centers.shape[0], dtype=torch.bool, device=device)
    patch_counts = torch.zeros(centers.shape[0], dtype=torch.long, device=device)
    image_counts = torch.zeros(centers.shape[0], dtype=torch.long, device=device)
    for prototype in range(centers.shape[0]):
        members = torch.nonzero(labels == prototype, as_tuple=True)[0]
        patch_counts[prototype] = members.numel()
        image_counts[prototype] = torch.unique(image_ids[members]).numel()
        fallback[prototype] = (
            members.numel() < minimum or image_counts[prototype].item() < 4
        )
        if fallback[prototype]:
            feature_bases.append(global_feature)
            delta_bases.append(global_delta)
        else:
            _, feature_basis = fit_centered_basis(originals[members], rank)
            delta_basis = fit_uncentered_basis(
                deltas[members].reshape(-1, deltas.shape[-1]), rank
            )
            feature_bases.append(feature_basis)
            delta_bases.append(delta_basis)
    from .conditional_subspace import estimate_sr_weights

    final_delta_bases = torch.stack(delta_bases)
    sr_gain = sr_instability = sr_weights = None
    if stability_regularization:
        sr_gain, sr_instability, sr_weights = estimate_sr_weights(
            deltas=deltas,
            labels=labels,
            image_ids=image_ids,
            global_basis=global_delta,
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
        global_feature_basis=global_feature,
        global_delta_basis=global_delta,
        rank=int(rank),
        sr_gain=sr_gain,
        sr_instability=sr_instability,
        sr_weights=sr_weights,
    )


class ConditionalNVSPipeline:
    """Feature-level D0/D1/D2/D3 pipeline shared by CPU tests and GPU runners."""

    def __init__(
        self,
        rank: int = 8,
        prototypes: int = 128,
        memory_strategy: str = "kcenter",
        memory_capacity: int = 30_000,
        seed: int = 42,
        prototype_selection: str = "proto_by_mstar",
        search_mode: str = "cosine",
        query_chunk_size: int = 4096,
        bank_chunk_size: int = 8192,
        memory_candidate_size: int = 50_000,
        memory_kcenter_chunk_size: int = 8192,
        memory_kcenter_block_size: int = 10_000,
        memory_large_k_batch_select: int = 64,
        compute_device: str | torch.device | None = None,
        whitener_rho: float = 0.99,
        whitener_shrinkage: float = 0.07,
        whitener_relative_floor: float = 1.0e-8,
        whitener_max_components: int | None = None,
        stability_regularization: bool = False,
        sr_bootstrap_repeats: int = 5,
        sr_bootstrap_fraction: float = 0.8,
        sr_weight_epsilon: float = 1.0e-8,
        kcenter_state_cache_path: str | None = None,
        kcenter_state_cache_interval: int = 250,
    ) -> None:
        if prototype_selection not in {
            "proto_by_mstar",
            "proto_by_topk_vote_k5",
        }:
            raise ValueError("Unsupported prototype selection")
        if search_mode not in {"cosine", "maha_select", "maha_full"}:
            raise ValueError("search_mode must be cosine/maha_select/maha_full")
        self.rank = int(rank)
        self.prototypes = int(prototypes)
        self.memory_strategy = str(memory_strategy)
        self.memory_capacity = int(memory_capacity)
        self.seed = int(seed)
        self.prototype_selection = prototype_selection
        self.search_mode = search_mode
        self.query_chunk_size = int(query_chunk_size)
        self.bank_chunk_size = int(bank_chunk_size)
        self.memory_candidate_size = int(memory_candidate_size)
        self.memory_kcenter_chunk_size = int(memory_kcenter_chunk_size)
        self.kcenter_chunk_size = self.memory_kcenter_chunk_size
        self.memory_kcenter_block_size = int(memory_kcenter_block_size)
        self.memory_large_k_batch_select = int(memory_large_k_batch_select)
        self.kcenter_state_cache_path = kcenter_state_cache_path
        self.kcenter_state_cache_interval = int(kcenter_state_cache_interval)
        self.compute_device = torch.device(compute_device or "cpu")
        if self.compute_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA compute_device requested but CUDA is unavailable")
        self.whitener_rho = float(whitener_rho)
        self.whitener_shrinkage = float(whitener_shrinkage)
        self.whitener_relative_floor = float(whitener_relative_floor)
        self.whitener_max_components = whitener_max_components
        self.stability_regularization = bool(stability_regularization)
        self.sr_bootstrap_repeats = int(sr_bootstrap_repeats)
        self.sr_bootstrap_fraction = float(sr_bootstrap_fraction)
        self.sr_weight_epsilon = float(sr_weight_epsilon)
        self.memory_result: MemoryBuildResult | None = None
        self.prototype_model: PrototypeModel | None = None
        self.memory_prototype_ids: torch.Tensor | None = None
        self.whitener: Whitener | None = None
        self.search_bank: torch.Tensor | None = None
        self.calibrations: dict[str, CalibrationState] = {}

    def _normalized(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            values.float().to(self.compute_device, non_blocking=True), dim=-1
        )

    def fit(self, features: FeatureSplit) -> "ConditionalNVSPipeline":
        memory_features = self._normalized(features.memory).reshape(
            -1, features.memory.shape[-1]
        )
        originals_3d = self._normalized(features.nvs_fit_original)
        if len(features.nvs_fit_transformed) != 13:
            raise ValueError("NVS fitting requires exactly 13 deterministic transforms")
        transformed = tuple(
            self._normalized(values) for values in features.nvs_fit_transformed
        )
        if any(values.shape != originals_3d.shape for values in transformed):
            raise ValueError("Transformed and original nvs_fit features must align")
        original_flat = originals_3d.reshape(-1, originals_3d.shape[-1])
        deltas = torch.stack(
            [
                values.reshape_as(original_flat) - original_flat
                for values in transformed
            ],
            dim=1,
        )
        image_ids = torch.arange(
            originals_3d.shape[0], device=self.compute_device
        ).repeat_interleave(int(originals_3d.shape[1]))

        candidate_indices = None
        memory_for_selection = memory_features
        if self.search_mode in {"maha_select", "maha_full"}:
            provisional = build_memory(
                memory_features,
                strategy="random",
                capacity=min(50_000, memory_features.shape[0]),
                seed=self.seed,
            )
            candidate_indices = provisional.candidate_indices
            self.whitener = fit_whitener(
                memory_features[
                    candidate_indices.to(self.compute_device, non_blocking=True)
                ],
                rho=self.whitener_rho,
                shrinkage=self.whitener_shrinkage,
                relative_floor=self.whitener_relative_floor,
                max_components=self.whitener_max_components,
            )
            if self.search_mode == "maha_full":
                memory_for_selection = self.whitener.transform(memory_features)

        selected_result = build_memory(
            memory_for_selection,
            strategy=self.memory_strategy,
            capacity=self.memory_capacity,
            seed=self.seed,
            candidate_indices=candidate_indices,
            candidate_size=self.memory_candidate_size,
            block_size=self.memory_kcenter_block_size,
            chunk_size=self.memory_kcenter_chunk_size,
            large_k_batch_select=self.memory_large_k_batch_select,
            kcenter_state_cache_path=self.kcenter_state_cache_path,
            kcenter_state_cache_interval=self.kcenter_state_cache_interval,
        )
        self.memory_result = MemoryBuildResult(
            memory_bank=memory_features[
                selected_result.selected_memory_indices.to(
                    self.compute_device, non_blocking=True
                )
            ].contiguous(),
            candidate_indices=selected_result.candidate_indices,
            selected_memory_indices=selected_result.selected_memory_indices,
            strategy=selected_result.strategy,
            capacity=selected_result.capacity,
            build_seconds=selected_result.build_seconds,
            algorithm=selected_result.algorithm,
        )
        if self.whitener is None:
            appearance_features = original_flat
            self.search_bank = self.memory_result.memory_bank
        else:
            appearance_features = self.whitener.transform(original_flat)
            self.search_bank = self.whitener.transform(self.memory_result.memory_bank)
        if self.search_mode == "maha_full":
            self.prototype_model = _fit_model_in_appearance_space(
                original_flat,
                deltas,
                image_ids,
                appearance_features,
                self.prototypes,
                self.rank,
                self.seed,
                stability_regularization=self.stability_regularization,
                bootstrap_repeats=self.sr_bootstrap_repeats,
                bootstrap_fraction=self.sr_bootstrap_fraction,
                weight_epsilon=self.sr_weight_epsilon,
            )
        else:
            self.prototype_model = fit_prototype_model(
                original_flat,
                deltas,
                image_ids=image_ids,
                prototypes=self.prototypes,
                rank=self.rank,
                seed=self.seed,
                stability_regularization=self.stability_regularization,
                bootstrap_repeats=self.sr_bootstrap_repeats,
                bootstrap_fraction=self.sr_bootstrap_fraction,
                weight_epsilon=self.sr_weight_epsilon,
            )
            appearance_features = original_flat
        memory_appearance = (
            self.memory_result.memory_bank
            if self.search_mode != "maha_full"
            else self.whitener.transform(self.memory_result.memory_bank)
        )
        self.memory_prototype_ids = assign_prototypes(
            memory_appearance, self.prototype_model.centers
        )
        return self

    def _require_fitted(self) -> None:
        if (
            self.memory_result is None
            or self.prototype_model is None
            or self.memory_prototype_ids is None
            or self.search_bank is None
        ):
            raise RuntimeError("Pipeline has not been fitted")

    def score_patch_features(
        self, query_features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self._require_fitted()
        assert self.memory_result is not None
        assert self.prototype_model is not None
        assert self.memory_prototype_ids is not None
        assert self.search_bank is not None
        shape = query_features.shape
        query = self._normalized(query_features).reshape(-1, shape[-1])
        query_search = (
            query if self.whitener is None else self.whitener.transform(query)
        )
        needed_k = 5 if self.prototype_selection == "proto_by_topk_vote_k5" else 1
        if self.whitener is None:
            nearest_values, nearest_indices = _cosine_topk(
                query_search,
                self.search_bank,
                needed_k,
                self.query_chunk_size,
                self.bank_chunk_size,
            )
            d0 = 1.0 - nearest_values[:, 0]
        else:
            nearest_values, nearest_indices = _euclidean_topk(
                query_search,
                self.search_bank,
                needed_k,
                self.query_chunk_size,
                self.bank_chunk_size,
            )
            d0 = nearest_values[:, 0].sqrt()
        nearest = self.memory_result.memory_bank[
            nearest_indices[:, 0].to(self.memory_result.memory_bank.device)
        ]
        deviation = query - nearest
        if self.prototype_selection == "proto_by_mstar":
            prototype_ids = prototype_by_mstar(
                self.memory_prototype_ids, nearest_indices[:, 0]
            )
        else:
            prototype_ids = prototype_by_topk_vote_k5(
                self.memory_prototype_ids, nearest_indices
            )
        scores = {
            "D0_NN": d0,
            **score_unified_deviation(
                deviation,
                self.prototype_model.global_feature_basis,
                self.prototype_model.global_delta_basis,
            ),
            **score_conditional_deviation(
                deviation, self.prototype_model, prototype_ids
            ),
        }
        if self.stability_regularization:
            scores.update(score_sr_deviation(deviation, self.prototype_model, prototype_ids))
        return {
            method: scores[method].reshape(shape[:-1]).float()
            for method in self.method_names()
            if method in scores
        }

    def calibrate(
        self,
        calibration_features: torch.Tensor,
        image_quantile: float = 0.95,
        mad_epsilon: float = 1.0e-6,
    ) -> Mapping[str, CalibrationState]:
        raw = self.score_patch_features(calibration_features)
        self.calibrations = {
            method: fit_calibration(
                values.detach().cpu().numpy(),
                image_quantile=image_quantile,
                mad_epsilon=mad_epsilon,
                scope="identity_calibration",
            )
            for method, values in raw.items()
        }
        return dict(self.calibrations)

    def normalize_scores(
        self, scores: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        missing = set(scores) - set(self.calibrations)
        if missing:
            raise RuntimeError(
                f"Independent calibration is required for every method; missing {sorted(missing)}"
            )
        normalized: dict[str, torch.Tensor] = {}
        for method in self.method_names():
            if method not in scores:
                continue
            values = scores[method]
            calibration = self.calibrations[method]
            z = (
                values.detach().to(dtype=torch.float64)
                - float(calibration.median)
            ) / float(calibration.mad)
            normalized[method] = torch.clamp(z, min=0.0).to(dtype=torch.float32)
        return normalized

    def method_names(self) -> tuple[str, ...]:
        return CORE_METHODS + (("SR_CNVS",) if self.stability_regularization else ())

    def sr_weight_rows(self) -> list[dict]:
        self._require_fitted()
        assert self.prototype_model is not None
        model = self.prototype_model
        if model.sr_weights is None:
            return []
        counts = torch.bincount(model.labels, minlength=model.centers.shape[0]).detach().cpu()
        gains = model.sr_gain.detach().float().cpu() if model.sr_gain is not None else None
        instabilities = (
            model.sr_instability.detach().float().cpu()
            if model.sr_instability is not None
            else None
        )
        weights = model.sr_weights.detach().float().cpu()
        fallback = model.fallback.detach().cpu()
        unique_images = model.unique_image_counts.detach().cpu()
        rows = []
        for prototype in range(int(model.centers.shape[0])):
            rows.append(
                {
                    "prototype": prototype,
                    "member_count": int(counts[prototype]),
                    "unique_image_count": int(unique_images[prototype]),
                    "fallback": bool(fallback[prototype]),
                    "g_c": None if gains is None else float(gains[prototype]),
                    "v_c": None
                    if instabilities is None
                    else float(instabilities[prototype]),
                    "w_c": float(weights[prototype]),
                }
            )
        return rows

    def fuse(
        self,
        normalized_scores: Mapping[str, torch.Tensor],
        alpha: float = 0.25,
    ) -> torch.Tensor:
        if float(alpha) not in {0.25, 0.50}:
            raise ValueError("Only preregistered alpha 0.25/0.50 are supported")
        return (1.0 - float(alpha)) * normalized_scores[
            "D3_NVSProto"
        ] + float(alpha) * normalized_scores["D0_NN"]

    def state_summary(self) -> dict:
        self._require_fitted()
        assert self.memory_result is not None
        assert self.prototype_model is not None
        return {
            "rank": self.rank,
            "prototypes": self.prototypes,
            "prototype_selection": self.prototype_selection,
            "search_mode": self.search_mode,
            "compute_device": str(self.compute_device),
            "methods": list(self.method_names()),
            "stability_regularization": {
                "enabled": self.stability_regularization,
                "bootstrap_repeats": self.sr_bootstrap_repeats,
                "bootstrap_fraction": self.sr_bootstrap_fraction,
                "weight_epsilon": self.sr_weight_epsilon,
            },
            "memory_entries": self.memory_result.capacity,
            "memory_strategy": self.memory_result.strategy,
            "memory_algorithm": self.memory_result.algorithm,
            "memory_build_seconds": self.memory_result.build_seconds,
            "candidate_indices_sha256": tensor_sha256(
                self.memory_result.candidate_indices.to(torch.int64)
            ),
            "selected_memory_indices_sha256": tensor_sha256(
                self.memory_result.selected_memory_indices.to(torch.int64)
            ),
            "kcenter_state_cache_path": self.kcenter_state_cache_path,
            "candidate_indices": self.memory_result.candidate_indices.detach().cpu().tolist(),
            "selected_memory_indices": self.memory_result.selected_memory_indices.detach().cpu().tolist(),
            "prototype_statistics": self.prototype_model.statistics(),
            "whitener": None
            if self.whitener is None
            else {
                "rho": self.whitener.rho,
                "shrinkage": self.whitener.shrinkage,
                "relative_floor": self.whitener.relative_floor,
                "eigenvalue_floor": self.whitener.eigenvalue_floor,
                "jitter": self.whitener.jitter,
                "components": int(self.whitener.reducer_components.shape[0]),
            },
        }
