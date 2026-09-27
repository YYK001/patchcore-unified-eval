from __future__ import annotations

import torch


def _validate(values: torch.Tensor, rank: int) -> tuple[torch.Tensor, int]:
    if values.ndim != 2:
        raise ValueError("values must have shape [N, D]")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("values must be non-empty")
    return values.float(), min(max(1, int(rank)), min(values.shape))


def fit_centered_basis(
    values: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    values, rank = _validate(values, rank)
    mean = values.mean(dim=0)
    _, _, vh = torch.linalg.svd(values - mean, full_matrices=False)
    return mean.contiguous(), vh[:rank].contiguous()


def fit_uncentered_basis(values: torch.Tensor, rank: int) -> torch.Tensor:
    """Fit a linear subspace through the origin (no mean subtraction)."""

    values, rank = _validate(values, rank)
    _, _, vh = torch.linalg.svd(values, full_matrices=False)
    return vh[:rank].contiguous()


def projection(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    values = values.float()
    basis = basis.float().to(values.device, non_blocking=True)
    return (values @ basis.T) @ basis


def residual_norm(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    values = values.float()
    return torch.linalg.norm(values - projection(values, basis), dim=-1)


def score_unified_deviation(
    deviation: torch.Tensor,
    feature_basis: torch.Tensor,
    delta_basis: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """D1 and D2 both act on exactly the same d=q-m*."""

    return {
        "D1_Global": residual_norm(deviation, feature_basis),
        "D2_NVSGlobal": residual_norm(deviation, delta_basis),
    }
