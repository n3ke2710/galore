"""
Optimizers that integrate GaLore gradient projection.

StandardAdamW:
    Обычный AdamW (baseline для сравнения).

GaLoreAdamW:
    AdamW с проекцией градиента на подпространство фиксированного ранга
    (Truncated SVD). Состояния оптимизатора (m, v) хранятся в проецированном
    пространстве, что экономит память.

ProximalGaLoreAdamW:
    AdamW с Proximal GaLore — вместо жёсткого усечения ранга используется
    Singular Value Thresholding (проксимальный оператор ядерной нормы).
    Ранг адаптируется динамически в процессе обучения.
"""

import torch
from torch.optim import Optimizer
from .projector import GaLoreProjector, ProximalGaLoreProjector


# ======================================================================
#  Standard AdamW  (baseline)
# ======================================================================


class StandardAdamW(Optimizer):
    """
    Standard AdamW optimizer (decoupled weight decay).

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float
        Learning rate.
    betas : tuple[float, float]
        Coefficients for running mean / variance of gradients.
    eps : float
        Term for numerical stability.
    weight_decay : float
        Decoupled weight decay coefficient.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Decoupled weight decay
                p.mul_(1.0 - lr * wd)

                # Momentum & variance
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bc1 = 1.0 - beta1 ** state["step"]
                bc2 = 1.0 - beta2 ** state["step"]

                step_size = lr / bc1
                denom = (exp_avg_sq.sqrt() / (bc2**0.5)).add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ======================================================================
#  GaLore AdamW  (fixed rank)
# ======================================================================


class GaLoreAdamW(Optimizer):
    """
    AdamW with GaLore gradient projection (Zhao et al., 2024).

    For 2-D parameters (weight matrices), gradients are projected to a
    fixed-rank subspace before the Adam update.  Optimizer states (m, v)
    are maintained in the low-rank space → memory savings.

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float
        Learning rate.
    betas : tuple[float, float]
        Coefficients for running mean / variance of gradients.
    eps : float
        Term for numerical stability.
    weight_decay : float
        Decoupled weight decay coefficient.
    rank : int
        Target rank for gradient projection.
    update_proj_gap : int
        How often (steps) to recompute the projection via SVD.
    galore_scale : float
        Multiplicative scale applied to the projected-back update.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        rank: int = 128,
        update_proj_gap: int = 200,
        galore_scale: float = 1.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            update_proj_gap=update_proj_gap,
            galore_scale=galore_scale,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # ---- Lazy init ----
                if len(state) == 0:
                    state["step"] = 0
                    if grad.dim() == 2:
                        state["projector"] = GaLoreProjector(
                            rank=min(group["rank"], min(grad.shape)),
                            update_freq=group["update_proj_gap"],
                            scale=group["galore_scale"],
                        )
                        state["use_galore"] = True
                    else:
                        state["use_galore"] = False

                state["step"] += 1

                if state["use_galore"]:
                    self._galore_step(p, grad, state, lr, beta1, beta2, eps, wd)
                else:
                    self._standard_step(p, grad, state, lr, beta1, beta2, eps, wd)

        return loss

    # ------------------------------------------------------------------

    @staticmethod
    def _galore_step(p, grad, state, lr, beta1, beta2, eps, wd):
        proj = state["projector"]

        # Project gradient → low-rank
        low_rank_grad = proj.project(grad)

        # Init / resize optimizer states in low-rank space
        if "exp_avg" not in state or state["exp_avg"].shape != low_rank_grad.shape:
            state["exp_avg"] = torch.zeros_like(low_rank_grad)
            state["exp_avg_sq"] = torch.zeros_like(low_rank_grad)

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        # Adam update  (in low-rank space)
        exp_avg.mul_(beta1).add_(low_rank_grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad, value=1.0 - beta2)

        bc1 = 1.0 - beta1 ** state["step"]
        bc2 = 1.0 - beta2 ** state["step"]

        step_size = lr / bc1
        denom = (exp_avg_sq.sqrt() / (bc2**0.5)).add_(eps)
        norm_grad = exp_avg / denom

        # Project back to full space
        full_update = proj.project_back(norm_grad)

        # Decoupled weight decay  (full space)
        p.mul_(1.0 - lr * wd)
        p.add_(full_update, alpha=-step_size)

    @staticmethod
    def _standard_step(p, grad, state, lr, beta1, beta2, eps, wd):
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        p.mul_(1.0 - lr * wd)
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bc1 = 1.0 - beta1 ** state["step"]
        bc2 = 1.0 - beta2 ** state["step"]
        step_size = lr / bc1
        denom = (exp_avg_sq.sqrt() / (bc2**0.5)).add_(eps)

        p.addcdiv_(exp_avg, denom, value=-step_size)


# ======================================================================
#  Proximal GaLore AdamW  (dynamic rank via nuclear norm / SVT)
# ======================================================================


class ProximalGaLoreAdamW(Optimizer):
    """
    AdamW with Proximal GaLore — dynamic rank adaptation.

    Uses Singular Value Thresholding (the proximal operator of the nuclear
    norm) to automatically determine the rank of gradient projection at
    each SVD recomputation step.

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float
        Learning rate.
    betas : tuple[float, float]
        Coefficients for running mean / variance of gradients.
    eps : float
        Term for numerical stability.
    weight_decay : float
        Decoupled weight decay coefficient.
    threshold : float
        Soft-thresholding parameter λ for SVT.
    update_proj_gap : int
        How often (steps) to recompute the projection via SVD + SVT.
    galore_scale : float
        Multiplicative scale applied to the projected-back update.
    min_rank : int
        Minimum allowed rank (prevents collapse to rank-0).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        threshold: float = 0.03,
        update_proj_gap: int = 200,
        galore_scale: float = 1.0,
        min_rank: int = 1,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            threshold=threshold,
            update_proj_gap=update_proj_gap,
            galore_scale=galore_scale,
            min_rank=min_rank,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # ---- Lazy init ----
                if len(state) == 0:
                    state["step"] = 0
                    if grad.dim() == 2:
                        state["projector"] = ProximalGaLoreProjector(
                            threshold=group["threshold"],
                            update_freq=group["update_proj_gap"],
                            scale=group["galore_scale"],
                            min_rank=group["min_rank"],
                        )
                        state["use_galore"] = True
                    else:
                        state["use_galore"] = False

                state["step"] += 1

                if state["use_galore"]:
                    self._proximal_galore_step(
                        p, grad, state, lr, beta1, beta2, eps, wd
                    )
                else:
                    self._standard_step(p, grad, state, lr, beta1, beta2, eps, wd)

        return loss

    # ------------------------------------------------------------------

    @staticmethod
    def _proximal_galore_step(p, grad, state, lr, beta1, beta2, eps, wd):
        proj = state["projector"]

        # Project gradient → low-rank  (rank determined by SVT)
        low_rank_grad = proj.project(grad)

        # Init / resize optimizer states in low-rank space
        if "exp_avg" not in state or state["exp_avg"].shape != low_rank_grad.shape:
            state["exp_avg"] = torch.zeros_like(low_rank_grad)
            state["exp_avg_sq"] = torch.zeros_like(low_rank_grad)

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        # Adam update  (in low-rank space)
        exp_avg.mul_(beta1).add_(low_rank_grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(
            low_rank_grad, low_rank_grad, value=1.0 - beta2
        )

        bc1 = 1.0 - beta1 ** state["step"]
        bc2 = 1.0 - beta2 ** state["step"]
        step_size = lr / bc1
        denom = (exp_avg_sq.sqrt() / (bc2**0.5)).add_(eps)
        norm_grad = exp_avg / denom

        # Project back to full space
        full_update = proj.project_back(norm_grad)

        # Decoupled weight decay  (full space)
        p.mul_(1.0 - lr * wd)
        p.add_(full_update, alpha=-step_size)

    @staticmethod
    def _standard_step(p, grad, state, lr, beta1, beta2, eps, wd):
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        p.mul_(1.0 - lr * wd)
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bc1 = 1.0 - beta1 ** state["step"]
        bc2 = 1.0 - beta2 ** state["step"]
        step_size = lr / bc1
        denom = (exp_avg_sq.sqrt() / (bc2**0.5)).add_(eps)

        p.addcdiv_(exp_avg, denom, value=-step_size)
