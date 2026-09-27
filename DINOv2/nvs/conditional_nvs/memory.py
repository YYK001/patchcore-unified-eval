from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import torch
import torch.nn.functional as F


MEMORY_PROTOCOLS = {
    "M_R5": ("random", 5_000),
    "M_K5": ("kcenter", 5_000),
    "M_R10": ("random", 10_000),
    "M_K10": ("kcenter", 10_000),
    "M_MRK10": ("kcenter_merge_reduce", 10_000),
    "M_IBK10": ("kcenter_image_balanced", 10_000),
    "M_R30": ("random", 30_000),
    "M_K30": ("kcenter", 30_000),
    "M_F0": ("full", 0),
}


@dataclass(frozen=True)
class MemoryBuildResult:
    memory_bank: torch.Tensor
    candidate_indices: torch.Tensor
    selected_memory_indices: torch.Tensor
    strategy: str
    capacity: int
    build_seconds: float
    algorithm: str

    def state_dict(self) -> dict:
        return {
            "memory_bank": self.memory_bank,
            "candidate_indices": self.candidate_indices,
            "selected_memory_indices": self.selected_memory_indices,
            "strategy": self.strategy,
            "capacity": self.capacity,
            "build_seconds": self.build_seconds,
            "algorithm": self.algorithm,
        }


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Stable SHA256 over a tensor's contiguous CPU bytes."""

    array = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _kcenter_metadata(
    values: torch.Tensor,
    k: int,
    seed: int,
    chunk_size: int,
    batch_select: int,
    values_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "strict_greedy_kcenter_state_v1",
        "algorithm": "exact_farthest_first_topk1"
        if int(batch_select) == 1
        else "batched_farthest_first_topk",
        "n": int(values.shape[0]),
        "dim": int(values.shape[1]),
        "k": int(k),
        "seed": int(seed),
        "chunk_size": int(chunk_size),
        "batch_select": int(batch_select),
        "dtype": "torch.float32",
        "values_sha256": str(values_sha256),
    }


def _state_digest(
    metadata: dict[str, Any],
    selected: torch.Tensor,
    min_distances: torch.Tensor,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    hasher.update(
        selected.detach().contiguous().cpu().to(torch.int64).numpy().tobytes(order="C")
    )
    hasher.update(
        min_distances.detach()
        .contiguous()
        .cpu()
        .to(torch.float32)
        .numpy()
        .tobytes(order="C")
    )
    return hasher.hexdigest()


def _torch_load_state(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _load_strict_kcenter_state(
    path: Path,
    metadata: dict[str, Any],
    k: int,
    n: int,
) -> tuple[int, torch.Tensor, torch.Tensor] | None:
    if not path.exists():
        return None
    payload = _torch_load_state(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid k-center state cache: {path}")
    if payload.get("schema") != "strict_greedy_kcenter_state_v1":
        raise RuntimeError(f"Unsupported k-center state cache schema: {path}")
    if payload.get("metadata") != metadata:
        raise RuntimeError(f"K-center state cache fingerprint mismatch: {path}")
    selected_count = int(payload.get("selected_count", 0))
    selected = payload.get("selected")
    min_distances = payload.get("min_distances")
    if not isinstance(selected, torch.Tensor) or not isinstance(
        min_distances, torch.Tensor
    ):
        raise RuntimeError(f"Malformed k-center state cache tensors: {path}")
    selected = selected.detach().cpu().to(torch.long).contiguous()
    min_distances = min_distances.detach().cpu().to(torch.float32).contiguous()
    if selected_count < 1 or selected_count > int(k):
        raise RuntimeError(f"Invalid k-center cached selected_count: {path}")
    if selected.shape != (selected_count,) or min_distances.shape != (int(n),):
        raise RuntimeError(f"Invalid k-center cached tensor shapes: {path}")
    if torch.unique(selected).numel() != selected_count:
        raise RuntimeError(f"K-center cached selected indices are not unique: {path}")
    if int(selected.min()) < 0 or int(selected.max()) >= int(n):
        raise RuntimeError(f"K-center cached selected indices out of range: {path}")
    if not torch.isfinite(min_distances).all():
        raise RuntimeError(f"K-center cached distances contain non-finite values: {path}")
    if not torch.all(min_distances[selected] == -1.0):
        raise RuntimeError(f"K-center cached selected distances are not masked: {path}")
    expected_digest = _state_digest(metadata, selected, min_distances)
    if payload.get("state_sha256") != expected_digest:
        raise RuntimeError(f"K-center state cache digest mismatch: {path}")
    return selected_count, selected, min_distances


def _save_strict_kcenter_state(
    path: Path,
    metadata: dict[str, Any],
    selected: torch.Tensor,
    selected_count: int,
    min_distances: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_cpu = (
        selected[: int(selected_count)].detach().cpu().to(torch.long).contiguous()
    )
    distances_cpu = min_distances.detach().cpu().to(torch.float32).contiguous()
    payload = {
        "schema": "strict_greedy_kcenter_state_v1",
        "metadata": metadata,
        "selected_count": int(selected_count),
        "selected": selected_cpu,
        "min_distances": distances_cpu,
    }
    payload["state_sha256"] = _state_digest(metadata, selected_cpu, distances_cpu)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def shared_candidate_indices(total: int, seed: int, size: int = 50_000) -> torch.Tensor:
    if total <= 0:
        raise ValueError("total must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(int(total), generator=generator)[: min(int(total), int(size))]


def _squared_euclidean_to_centers(
    values: torch.Tensor,
    centers: torch.Tensor,
    chunk_size: int,
    value_norms: torch.Tensor | None = None,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    center_norm = (centers * centers).sum(dim=1)
    for start in range(0, values.shape[0], int(chunk_size)):
        block = values[start : start + int(chunk_size)]
        block_norm = (
            (block * block).sum(dim=1, keepdim=True)
            if value_norms is None
            else value_norms[start : start + int(chunk_size)].unsqueeze(1)
        )
        distances = (
            block_norm
            + center_norm.unsqueeze(0)
            - 2.0 * block @ centers.T
        ).clamp_min_(0.0)
        output.append(distances.min(dim=1).values)
    return torch.cat(output)


def _minimum_in_place(target: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    if hasattr(target, "minimum_"):
        return target.minimum_(update)
    torch.minimum(target, update, out=target)
    return target


def greedy_kcenter_indices(
    values: torch.Tensor,
    k: int,
    seed: int,
    chunk_size: int = 8192,
    batch_select: int = 1,
    state_cache_path: str | Path | None = None,
    state_cache_interval: int = 250,
) -> torch.Tensor:
    """Deterministic farthest-first selection.

    ``batch_select=1`` is exact greedy. Larger batches are the bounded-cost
    approximation used inside the 30k merge-reduce path. The exact path avoids
    a full selected-mask write each iteration: selected centers are represented
    only by setting their current distance to ``-1``.
    """

    values = values.float().contiguous()
    n = int(values.shape[0])
    k = min(max(1, int(k)), n)
    batch = max(1, int(batch_select))
    cache_path = Path(state_cache_path) if state_cache_path is not None else None
    cache_interval = max(1, int(state_cache_interval))
    values_sha = tensor_sha256(values) if cache_path is not None else ""
    metadata = (
        _kcenter_metadata(values, k, seed, chunk_size, batch, values_sha)
        if cache_path is not None
        else None
    )

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    first = int(torch.randint(n, (1,), generator=generator).item())
    selected = torch.empty(k, dtype=torch.long, device=values.device)
    value_norms = (values * values).sum(dim=1)

    loaded = (
        _load_strict_kcenter_state(cache_path, metadata, k, n)
        if cache_path is not None and metadata is not None
        else None
    )
    if loaded is None:
        selected[0] = first
        selected_count = 1
        min_distances = _squared_euclidean_to_centers(
            values, values[first : first + 1], chunk_size, value_norms
        )
        min_distances[first] = -1.0
    else:
        selected_count, cached_selected, cached_min_distances = loaded
        if int(cached_selected[0]) != first:
            raise RuntimeError("K-center state cache first index mismatch")
        selected[:selected_count] = cached_selected.to(values.device, non_blocking=True)
        min_distances = cached_min_distances.to(values.device, non_blocking=True)
        if selected_count == k:
            return cached_selected.cpu()

    while selected_count < k:
        count = min(batch, k - selected_count)
        candidates = torch.topk(min_distances, k=count, largest=True).indices
        selected[selected_count : selected_count + count] = candidates
        update = _squared_euclidean_to_centers(
            values, values[candidates], chunk_size, value_norms
        )
        _minimum_in_place(min_distances, update)
        min_distances[candidates] = -1.0
        selected_count += count
        if (
            cache_path is not None
            and metadata is not None
            and (selected_count == k or selected_count % cache_interval == 0)
        ):
            _save_strict_kcenter_state(
                cache_path, metadata, selected, selected_count, min_distances
            )
    return selected.cpu()


def merge_reduce_kcenter_indices(
    values: torch.Tensor,
    k: int,
    seed: int,
    block_size: int = 50_000,
    oversample: float = 2.0,
    chunk_size: int = 8192,
    batch_select: int = 64,
) -> torch.Tensor:
    """Blockwise merge-reduce k-center for 30k banks.

    This function never calls full-pool exact greedy k-center. Each block is
    reduced to its proportional share of a 2K coreset and only that coreset is
    compressed to K with batched farthest-first updates.
    """

    values = values.float().contiguous()
    n = int(values.shape[0])
    k = min(max(1, int(k)), n)
    if k == n:
        return torch.arange(n)
    candidates: list[torch.Tensor] = []
    for block_number, start in enumerate(range(0, n, int(block_size))):
        stop = min(n, start + int(block_size))
        block_n = stop - start
        block_k = min(
            block_n,
            max(1, int(math.ceil(float(oversample) * k * block_n / n))),
        )
        local = greedy_kcenter_indices(
            values[start:stop],
            block_k,
            seed=int(seed) + 104729 * block_number,
            chunk_size=chunk_size,
            batch_select=max(1, int(batch_select)),
        )
        candidates.append(local + start)
    merged = torch.unique(torch.cat(candidates), sorted=True)
    if merged.numel() <= k:
        missing = torch.tensor(
            [index for index in range(n) if index not in set(merged.tolist())],
            dtype=torch.long,
        )
        return torch.cat([merged, missing[: k - merged.numel()]])
    final_local = greedy_kcenter_indices(
        values[merged.to(values.device, non_blocking=True)],
        k,
        seed=int(seed) + 1_000_003,
        chunk_size=chunk_size,
        batch_select=max(1, int(batch_select)),
    )
    return merged[final_local]


def group_balanced_kcenter_indices(
    values: torch.Tensor,
    group_indices: torch.Tensor,
    k: int,
    seed: int,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Select near-equal per-group quotas with within-group exact k-center."""

    values = values.float().contiguous()
    groups = group_indices.long().cpu().reshape(-1)
    n = int(values.shape[0])
    if groups.numel() != n:
        raise ValueError("group_indices must align with values")
    k = min(max(1, int(k)), n)
    unique, counts = torch.unique(groups, sorted=True, return_counts=True)
    quotas = torch.zeros_like(counts)
    allocated = 0
    while allocated < k:
        progressed = False
        for index in range(int(unique.numel())):
            if quotas[index] >= counts[index]:
                continue
            quotas[index] += 1
            allocated += 1
            progressed = True
            if allocated == k:
                break
        if not progressed:
            raise AssertionError("Unable to allocate balanced group quotas")

    selected: list[torch.Tensor] = []
    for ordinal, (group, quota) in enumerate(
        zip(unique.tolist(), quotas.tolist())
    ):
        if quota <= 0:
            continue
        members = torch.nonzero(groups == int(group), as_tuple=True)[0]
        local = greedy_kcenter_indices(
            values[members.to(values.device, non_blocking=True)],
            int(quota),
            seed=int(seed) + 104729 * ordinal,
            chunk_size=chunk_size,
            batch_select=1,
        )
        selected.append(members[local])
    output = torch.cat(selected).long().cpu()
    if output.numel() != k or torch.unique(output).numel() != k:
        raise AssertionError("Balanced k-center did not return K unique indices")
    return output


def build_memory(
    features: torch.Tensor,
    strategy: str,
    capacity: int,
    seed: int,
    candidate_indices: torch.Tensor | None = None,
    group_indices: torch.Tensor | None = None,
    candidate_size: int = 50_000,
    block_size: int = 50_000,
    chunk_size: int = 8192,
    large_k_batch_select: int = 64,
    normalize_features: bool = True,
    kcenter_state_cache_path: str | Path | None = None,
    kcenter_state_cache_interval: int = 250,
) -> MemoryBuildResult:
    flat = features.reshape(-1, features.shape[-1]).float()
    if normalize_features:
        flat = F.normalize(flat, dim=-1)
    flat = flat.contiguous()
    n = int(flat.shape[0])
    if n == 0:
        raise ValueError("Cannot build an empty memory bank")
    strategy = str(strategy).lower()
    target = n if strategy == "full" or int(capacity) <= 0 else min(int(capacity), n)
    start = perf_counter()

    if strategy == "full":
        candidates = torch.arange(n)
        selected = candidates.clone()
        algorithm = "full"
    elif target >= 30_000 and n > target:
        candidates = torch.arange(n)
        if strategy == "random":
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            selected = torch.randperm(n, generator=generator)[:target]
            algorithm = "full_pool_random"
        elif strategy == "kcenter":
            selected = merge_reduce_kcenter_indices(
                flat,
                target,
                seed=seed,
                block_size=block_size,
                chunk_size=chunk_size,
                batch_select=large_k_batch_select,
            )
            algorithm = "merge_reduce_kcenter_gamma2"
        else:
            raise ValueError(f"Unsupported memory strategy: {strategy}")
    else:
        candidates = (
            shared_candidate_indices(n, seed, candidate_size)
            if candidate_indices is None
            else candidate_indices.long().cpu().clone()
        )
        if candidates.ndim != 1 or candidates.numel() == 0:
            raise ValueError("candidate_indices must be a non-empty vector")
        if int(candidates.min()) < 0 or int(candidates.max()) >= n:
            raise IndexError("candidate_indices out of range")
        target = min(target, int(candidates.numel()))
        if strategy == "random":
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            local = torch.randperm(candidates.numel(), generator=generator)[:target]
            algorithm = "shared_candidate_random"
        elif strategy == "kcenter":
            batch_select = 1 if target <= 10_000 else large_k_batch_select
            local = greedy_kcenter_indices(
                flat[candidates.to(flat.device, non_blocking=True)],
                target,
                seed=seed,
                chunk_size=chunk_size,
                batch_select=batch_select,
                state_cache_path=kcenter_state_cache_path
                if batch_select == 1
                else None,
                state_cache_interval=kcenter_state_cache_interval,
            )
            algorithm = "shared_candidate_greedy_kcenter"
        elif strategy == "kcenter_merge_reduce":
            local = merge_reduce_kcenter_indices(
                flat[candidates.to(flat.device, non_blocking=True)],
                target,
                seed=seed,
                block_size=block_size,
                chunk_size=chunk_size,
                batch_select=large_k_batch_select,
            )
            algorithm = "shared_candidate_merge_reduce_kcenter_gamma2"
        elif strategy == "kcenter_image_balanced":
            if group_indices is None:
                raise ValueError("Image-balanced k-center requires group_indices")
            groups = group_indices.long().cpu().reshape(-1)
            if groups.numel() != n:
                raise ValueError("group_indices must align with flattened features")
            local = group_balanced_kcenter_indices(
                flat[candidates.to(flat.device, non_blocking=True)],
                groups[candidates],
                target,
                seed=seed,
                chunk_size=chunk_size,
            )
            algorithm = "shared_candidate_image_balanced_kcenter"
        else:
            raise ValueError(f"Unsupported memory strategy: {strategy}")
        selected = candidates[local]
    return MemoryBuildResult(
        memory_bank=flat[selected.to(flat.device, non_blocking=True)].contiguous(),
        candidate_indices=candidates.contiguous(),
        selected_memory_indices=selected.contiguous(),
        strategy=strategy,
        capacity=int(selected.numel()),
        build_seconds=float(perf_counter() - start),
        algorithm=algorithm,
    )


def build_protocol_memory(
    features: torch.Tensor,
    protocol: str,
    seed: int,
    candidate_indices: torch.Tensor | None = None,
    **kwargs,
) -> MemoryBuildResult:
    if protocol not in MEMORY_PROTOCOLS:
        raise KeyError(f"Unknown memory protocol {protocol!r}")
    strategy, capacity = MEMORY_PROTOCOLS[protocol]
    return build_memory(
        features,
        strategy=strategy,
        capacity=capacity,
        seed=seed,
        candidate_indices=candidate_indices,
        **kwargs,
    )


def augmented_memory_candidates(
    original_features: torch.Tensor,
    transformed_features: Iterable[torch.Tensor],
) -> torch.Tensor:
    chunks = [original_features.reshape(-1, original_features.shape[-1])]
    chunks.extend(
        values.reshape(-1, values.shape[-1]) for values in transformed_features
    )
    return F.normalize(torch.cat(chunks, dim=0).float(), dim=-1)

def matched_augmented_memory_candidates(
    memory_original: torch.Tensor,
    nvs_fit_original: torch.Tensor,
    nvs_fit_transformed: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Build AugMem from exactly the feature information available to D2.

    D2 retrieves from ``memory_original`` and fits its delta basis from the
    aligned ``nvs_fit_original`` plus 13 transformed nvs_fit tensors. AugMem
    receives those same tensors, but uses them as a direct retrieval pool.
    It must not receive transformed memory-split images.
    """

    transformed = tuple(nvs_fit_transformed)
    if len(transformed) != 13:
        raise ValueError("Matched AugMem requires exactly 13 nvs_fit transforms")
    if memory_original.ndim != 3 or nvs_fit_original.ndim != 3:
        raise ValueError("Original feature tensors must have shape [N,P,C]")
    if memory_original.shape[1:] != nvs_fit_original.shape[1:]:
        raise ValueError("memory and nvs_fit patch feature shapes must match")
    if any(values.shape != nvs_fit_original.shape for values in transformed):
        raise ValueError("Transformed nvs_fit features must align with originals")
    originals = torch.cat([memory_original, nvs_fit_original], dim=0)
    return augmented_memory_candidates(originals, transformed)
