"""
Gradient projectors for GaLore and Proximal GaLore.

GaLoreProjector:
    Периодически вычисляет SVD градиента и проецирует его на подпространство
    фиксированного ранга r (Truncated SVD). Соответствует оригинальной статье
    "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection"
    (Zhao et al., 2024).

ProximalGaLoreProjector:
    Вместо жёсткого усечения сингулярных чисел применяет мягкое пороговое
    отсечение (Singular Value Thresholding) — проксимальный оператор ядерной
    нормы. Ранг подпространства адаптируется динамически в процессе обучения.
"""

import torch
from typing import Optional, Tuple


class GaLoreProjector:
    """
    Standard GaLore projector — fixed-rank truncated SVD.

    At every `update_freq` steps recomputes the top-r singular vectors of the
    gradient and uses them as the projection basis.

    Parameters
    ----------
    rank : int
        Target rank of the low-rank projection.
    update_freq : int
        How often (in optimizer steps) to recompute the projection basis.
    scale : float
        Scaling factor applied when projecting back to full space.
    """

    def __init__(self, rank: int, update_freq: int = 200, scale: float = 1.0):
        self.rank = rank
        self.update_freq = update_freq
        self.scale = scale

        self.ortho_matrix: Optional[torch.Tensor] = None
        self.step = 0
        self._full_shape: Optional[Tuple[int, ...]] = None
        self._sv_history: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(self, grad: torch.Tensor) -> torch.Tensor:
        """Project full gradient → low-rank subspace."""
        self._maybe_update_projection(grad)
        self.step += 1

        if grad.shape[0] >= grad.shape[1]:
            return self.ortho_matrix.T @ grad
        else:
            return grad @ self.ortho_matrix

    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        """Project low-rank update → full parameter space."""
        assert self._full_shape is not None, "Call project() first"
        if self._full_shape[0] >= self._full_shape[1]:
            return (self.ortho_matrix @ low_rank_grad) * self.scale
        else:
            return (low_rank_grad @ self.ortho_matrix.T) * self.scale

    def get_effective_rank(self) -> int:
        """Return the current effective rank (always equals self.rank)."""
        return self.rank

    def get_singular_values(self, grad: torch.Tensor) -> torch.Tensor:
        """Compute and return singular values of the gradient (for analysis)."""
        _, S, _ = torch.linalg.svd(grad.detach(), full_matrices=False)
        return S

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_update_projection(self, grad: torch.Tensor) -> None:
        if self.ortho_matrix is None or self.step % self.update_freq == 0:
            self.ortho_matrix = self._compute_projection(grad)
            self._full_shape = grad.shape

    def _compute_projection(self, grad: torch.Tensor) -> torch.Tensor:
        """Truncated SVD → orthogonal projection matrix."""
        U, S, Vh = torch.linalg.svd(grad.detach(), full_matrices=False)
        self._sv_history.append(S.cpu())

        if grad.shape[0] >= grad.shape[1]:
            return U[:, : self.rank].detach()
        else:
            return Vh[: self.rank, :].T.detach()


class ProximalGaLoreProjector:
    """
    Proximal GaLore projector — dynamic rank via Singular Value Thresholding.

    Applies the proximal operator of the nuclear norm (SVT):
        σ_i  →  max(σ_i − λ, 0)
    The number of surviving singular values determines the effective rank.

    Parameters
    ----------
    threshold : float
        Soft-thresholding parameter λ for SVT.
    update_freq : int
        How often (in optimizer steps) to recompute the projection basis.
    scale : float
        Scaling factor applied when projecting back to full space.
    min_rank : int
        Minimum allowed rank (prevents complete collapse to rank-0).
    """

    def __init__(
        self,
        threshold: float,
        update_freq: int = 200,
        scale: float = 1.0,
        min_rank: int = 1,
    ):
        self.threshold = threshold
        self.update_freq = update_freq
        self.scale = scale
        self.min_rank = min_rank

        self.ortho_matrix: Optional[torch.Tensor] = None
        self.effective_rank: int = 0
        self.step = 0
        self._full_shape: Optional[Tuple[int, ...]] = None
        self._sv_history: list = []
        self._rank_history: list = []
        self._thresholded_sv: Optional[torch.Tensor] = None
        self._basis_updated: bool = False
        self._prev_ortho_matrix: Optional[torch.Tensor] = None
        self._prev_full_shape: Optional[Tuple[int, ...]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(self, grad: torch.Tensor) -> torch.Tensor:
        """Project full gradient → low-rank subspace (rank determined by SVT)."""
        self._maybe_update_projection(grad)
        self.step += 1

        if self.effective_rank == 0:
            # Fallback — keep at least a 1-d projection
            if grad.shape[0] >= grad.shape[1]:
                return torch.zeros(
                    self.min_rank, grad.shape[1], device=grad.device, dtype=grad.dtype
                )
            else:
                return torch.zeros(
                    grad.shape[0], self.min_rank, device=grad.device, dtype=grad.dtype
                )

        if grad.shape[0] >= grad.shape[1]:
            return self.ortho_matrix.T @ grad
        else:
            return grad @ self.ortho_matrix

    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        """Project low-rank update → full parameter space."""
        assert self._full_shape is not None, "Call project() first"
        if self.effective_rank == 0:
            return torch.zeros(
                self._full_shape, device=low_rank_grad.device, dtype=low_rank_grad.dtype
            )

        if self._full_shape[0] >= self._full_shape[1]:
            return (self.ortho_matrix @ low_rank_grad) * self.scale
        else:
            return (low_rank_grad @ self.ortho_matrix.T) * self.scale

    def get_effective_rank(self) -> int:
        """Return the current effective rank (changes dynamically)."""
        return self.effective_rank

    def get_singular_values(self, grad: torch.Tensor) -> torch.Tensor:
        """Compute and return singular values of the gradient (for analysis)."""
        _, S, _ = torch.linalg.svd(grad.detach(), full_matrices=False)
        return S

    def consume_prev_basis(self) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, ...]]]:
        """Return previous basis info and reset the update flag."""
        if not self._basis_updated:
            return None, None
        self._basis_updated = False
        return self._prev_ortho_matrix, self._prev_full_shape

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_update_projection(self, grad: torch.Tensor) -> None:
        if self.ortho_matrix is None or self.step % self.update_freq == 0:
            self._prev_ortho_matrix = self.ortho_matrix
            self._prev_full_shape = self._full_shape
            self.ortho_matrix, self.effective_rank, self._thresholded_sv = (
                self._compute_projection(grad)
            )
            self._full_shape = grad.shape
            self._rank_history.append(self.effective_rank)
            self._basis_updated = True

    def _compute_projection(
        self, grad: torch.Tensor
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """SVT + truncation → dynamic-rank orthogonal projection matrix."""
        U, S, Vh = torch.linalg.svd(grad.detach(), full_matrices=False)
        self._sv_history.append(S.cpu())

        # Relative soft thresholding: scale threshold by the max singular value
        # This prevents rank collapse as gradient norms decay during training.
        threshold_val = self.threshold * S[0]
        S_thresh = torch.relu(S - threshold_val)
        mask = S_thresh > 0
        eff_rank = max(int(mask.sum().item()), self.min_rank)

        if grad.shape[0] >= grad.shape[1]:
            ortho = U[:, :eff_rank].detach()
        else:
            ortho = Vh[:eff_rank, :].T.detach()

        return ortho, eff_rank, S_thresh[:eff_rank].detach()
