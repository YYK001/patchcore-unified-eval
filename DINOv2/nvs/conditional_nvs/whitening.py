from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Whitener:
    reducer_mean: torch.Tensor
    reducer_components: torch.Tensor
    covariance_mean: float
    cholesky: torch.Tensor
    rho: float
    shrinkage: float
    relative_floor: float
    eigenvalue_floor: float
    jitter: float

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        shape = values.shape
        device = values.device
        flat = values.reshape(-1, shape[-1]).float()
        mean = self.reducer_mean.to(device, non_blocking=True)
        components = self.reducer_components.to(device, non_blocking=True)
        cholesky = self.cholesky.to(device, non_blocking=True)
        reduced = (flat - mean) @ components.T
        whitened = torch.linalg.solve_triangular(
            cholesky, reduced.T, upper=False
        ).T
        return whitened.reshape(*shape[:-1], -1)

    def state_dict(self) -> dict:
        return {
            "reducer_mean": self.reducer_mean,
            "reducer_components": self.reducer_components,
            "covariance_mean": self.covariance_mean,
            "cholesky": self.cholesky,
            "rho": self.rho,
            "shrinkage": self.shrinkage,
            "relative_floor": self.relative_floor,
            "eigenvalue_floor": self.eigenvalue_floor,
            "jitter": self.jitter,
        }


def fit_whitener(
    candidate_features: torch.Tensor,
    rho: float = 0.99,
    shrinkage: float = 0.07,
    relative_floor: float = 1.0e-8,
    max_components: int | None = None,
) -> Whitener:
    values = candidate_features.reshape(-1, candidate_features.shape[-1]).double()
    device = values.device
    if values.shape[0] < 2:
        raise ValueError("At least two candidate features are required")
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / max(1, values.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    total = eigenvalues.sum()
    if total <= 0:
        component_count = 1
    else:
        cumulative = torch.cumsum(eigenvalues, dim=0) / total
        component_count = int(
            torch.searchsorted(
                cumulative, torch.tensor(float(rho), device=device, dtype=cumulative.dtype)
            ).item()
        ) + 1
    if max_components is not None:
        component_count = min(component_count, int(max_components))
    components = eigenvectors[:, :component_count].T.contiguous()
    reduced = centered @ components.T
    reduced_covariance = reduced.T @ reduced / max(1, reduced.shape[0] - 1)
    covariance_mean = float(torch.trace(reduced_covariance) / component_count)
    shrunk = (1.0 - float(shrinkage)) * reduced_covariance + float(
        shrinkage
    ) * covariance_mean * torch.eye(
        component_count, dtype=torch.float64, device=device
    )
    floor = float(relative_floor) * max(abs(covariance_mean), 1.0e-12)
    evals, evecs = torch.linalg.eigh(shrunk)
    shrunk = (evecs * evals.clamp_min(floor).unsqueeze(0)) @ evecs.T
    jitter = 0.0
    identity = torch.eye(component_count, dtype=torch.float64, device=device)
    attempt = 0.0
    while True:
        try:
            cholesky = torch.linalg.cholesky(shrunk + attempt * identity)
            jitter = attempt
            break
        except RuntimeError:
            attempt = 1.0e-12 if attempt == 0.0 else attempt * 10.0
            if attempt > 1.0:
                raise RuntimeError("Adaptive Cholesky jitter exceeded 1")
    return Whitener(
        reducer_mean=mean.float(),
        reducer_components=components.float(),
        covariance_mean=covariance_mean,
        cholesky=cholesky.float(),
        rho=float(rho),
        shrinkage=float(shrinkage),
        relative_floor=float(relative_floor),
        eigenvalue_floor=floor,
        jitter=float(jitter),
    )


@dataclass(frozen=True)
class LedoitWolfWhitener:
    mean: torch.Tensor
    cholesky: torch.Tensor
    shrinkage: float
    relative_floor: float
    eigenvalue_floor: float
    jitter: float
    empirical_eigenvalues: torch.Tensor
    shrunk_eigenvalues: torch.Tensor
    fit_samples: int

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        shape = values.shape
        flat = values.reshape(-1, shape[-1]).float()
        mean = self.mean.to(flat.device, non_blocking=True)
        cholesky = self.cholesky.to(flat.device, non_blocking=True)
        whitened = torch.linalg.solve_triangular(
            cholesky, (flat - mean).T, upper=False
        ).T
        return whitened.reshape(*shape[:-1], -1)

    @staticmethod
    def _spectrum(values: torch.Tensor) -> dict[str, float]:
        eigenvalues = values.detach().double().cpu().clamp_min(0.0)
        total = eigenvalues.sum().clamp_min(1.0e-30)
        probabilities = eigenvalues / total
        positive = probabilities > 0
        effective_rank = torch.exp(
            -(probabilities[positive] * probabilities[positive].log()).sum()
        )
        participation = total.square() / eigenvalues.square().sum().clamp_min(1.0e-30)
        maximum = float(eigenvalues.max())
        minimum = float(eigenvalues.min())
        return {
            "min": minimum,
            "p01": float(torch.quantile(eigenvalues, 0.01)),
            "p05": float(torch.quantile(eigenvalues, 0.05)),
            "median": float(torch.quantile(eigenvalues, 0.50)),
            "p95": float(torch.quantile(eigenvalues, 0.95)),
            "p99": float(torch.quantile(eigenvalues, 0.99)),
            "max": maximum,
            "condition_number": maximum / max(minimum, 1.0e-30),
            "effective_rank": float(effective_rank),
            "participation_ratio": float(participation),
        }

    def summary(self) -> dict:
        return {
            "estimator": "ledoit_wolf",
            "fit_samples": int(self.fit_samples),
            "dimensions": int(self.mean.numel()),
            "shrinkage": float(self.shrinkage),
            "relative_floor": float(self.relative_floor),
            "eigenvalue_floor": float(self.eigenvalue_floor),
            "jitter": float(self.jitter),
            "empirical_spectrum": self._spectrum(self.empirical_eigenvalues),
            "shrunk_spectrum": self._spectrum(self.shrunk_eigenvalues),
            "empirical_eigenvalues": self.empirical_eigenvalues.detach().cpu().tolist(),
            "shrunk_eigenvalues": self.shrunk_eigenvalues.detach().cpu().tolist(),
        }


def fit_ledoit_wolf_whitener(
    candidate_features: torch.Tensor,
    relative_floor: float = 1.0e-8,
) -> LedoitWolfWhitener:
    """Fit full-dimensional Ledoit-Wolf whitening on normal source features.

    The shrinkage coefficient follows the standard non-blocked Ledoit-Wolf
    formula. Large matrix products stay on the input device; only the small
    covariance matrix is promoted to float64 for stabilization and Cholesky.
    """

    values = candidate_features.reshape(-1, candidate_features.shape[-1]).float()
    if values.shape[0] < 2:
        raise ValueError("At least two candidate features are required")
    if not 0.0 < float(relative_floor) < 1.0:
        raise ValueError("relative_floor must be in (0, 1)")
    device = values.device
    sample_count, dimensions = int(values.shape[0]), int(values.shape[1])
    mean = values.mean(dim=0)
    centered = values - mean
    squared = centered.square()
    cross = centered.T @ centered
    empirical = cross / float(sample_count)
    empirical_trace = squared.sum(dim=0) / float(sample_count)
    mu = empirical_trace.sum() / float(dimensions)
    delta_raw = cross.square().sum().double() / float(sample_count**2)
    beta_raw = (squared.T @ squared).sum().double()
    beta = (
        beta_raw / float(sample_count) - delta_raw
    ) / float(dimensions * sample_count)
    delta = (
        delta_raw
        - 2.0 * mu.double() * empirical_trace.sum().double()
        + float(dimensions) * mu.double().square()
    ) / float(dimensions)
    beta = torch.minimum(beta.clamp_min(0.0), delta.clamp_min(0.0))
    shrinkage = 0.0 if float(delta) <= 0.0 else float(beta / delta)
    empirical64 = empirical.double()
    covariance = (1.0 - shrinkage) * empirical64
    covariance.diagonal().add_(shrinkage * float(mu))
    empirical_eigenvalues = torch.linalg.eigvalsh(empirical64).clamp_min(0.0)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    scale = max(float(eigenvalues.max()), 1.0e-30)
    floor = float(relative_floor) * scale
    eigenvalues = eigenvalues.clamp_min(floor)
    stabilized = (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T
    identity = torch.eye(dimensions, dtype=torch.float64, device=device)
    attempt = 0.0
    while True:
        try:
            cholesky = torch.linalg.cholesky(stabilized + attempt * identity)
            break
        except RuntimeError:
            attempt = floor if attempt == 0.0 else attempt * 10.0
            if attempt > scale:
                raise RuntimeError("Adaptive Cholesky jitter exceeded covariance scale")
    return LedoitWolfWhitener(
        mean=mean.float(),
        cholesky=cholesky.float(),
        shrinkage=float(shrinkage),
        relative_floor=float(relative_floor),
        eigenvalue_floor=float(floor),
        jitter=float(attempt),
        empirical_eigenvalues=empirical_eigenvalues.float(),
        shrunk_eigenvalues=eigenvalues.float(),
        fit_samples=sample_count,
    )
