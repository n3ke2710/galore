"""
Utilities for training, tracking metrics, and computing memory footprint.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class TrainingTracker:
    """
    Collects per-step metrics during training for later visualization.

    Fields
    ------
    losses : list[float]
        Training loss at each logged step.
    times : list[float]
        Elapsed time in seconds for each logged metric.
    train_accuracies : list[float]
        Train accuracy list.
    test_accuracies : list[float]
        Test accuracy list.
    arg_updates : list[float]
        Norm of weight change ||w_k - w_{k-1}||.
    arg_dist_start : list[float]
        Distance from initial weights ||w_k - w_0||.
    ranks : dict[str, list[int]]
        Per-layer effective rank over time  (layer_name → list of ranks).
    singular_values : dict[str, list[Tensor]]
        Per-layer gradient singular values snapshots.
    memory : dict[str, float]
        Optimizer state memory footprint (key → bytes).
    """

    losses: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    test_accuracies: List[float] = field(default_factory=list)
    arg_updates: List[float] = field(default_factory=list)
    arg_dist_start: List[float] = field(default_factory=list)
    ranks: Dict[str, List[int]] = field(default_factory=dict)
    singular_values: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    memory_history: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)

    def log_loss(self, loss: float, time_val: Optional[float] = None) -> None:
        self.losses.append(loss)
        if time_val is not None:
            self.times.append(time_val)

    def log_accuracies(self, train: float, test: float) -> None:
        self.train_accuracies.append(train)
        self.test_accuracies.append(test)

    def log_arg_metrics(self, update_norm: float, dist_start: float) -> None:
        self.arg_updates.append(update_norm)
        self.arg_dist_start.append(dist_start)

    def log_grad_norm(self, model: nn.Module) -> None:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self.grad_norms.append(total_norm**0.5)

    def log_rank(self, layer_name: str, rank: int) -> None:
        if layer_name not in self.ranks:
            self.ranks[layer_name] = []
        self.ranks[layer_name].append(rank)

    def log_singular_values(
        self, layer_name: str, sv: torch.Tensor
    ) -> None:
        if layer_name not in self.singular_values:
            self.singular_values[layer_name] = []
        self.singular_values[layer_name].append(sv.cpu().detach())

    def log_memory(self, memory_mb: float) -> None:
        self.memory_history.append(memory_mb)


def compute_memory_footprint(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """
    Estimate the memory footprint of optimizer states, including tensors
    hidden inside objects (like projectors).

    Returns
    -------
    dict with keys:
        - "total_bytes" : int — total state size in bytes
        - "total_mb"    : float — total in MiB
        - "n_state_tensors" : int
        - "per_param"   : list of dicts with per-parameter info
    """
    total_bytes = 0
    n_tensors = 0
    per_param = []

    def count_tensors(obj):
        nonlocal total_bytes, n_tensors, param_bytes
        if isinstance(obj, torch.Tensor):
            size = obj.nelement() * obj.element_size()
            total_bytes += size
            param_bytes += size
            n_tensors += 1
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                count_tensors(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                count_tensors(v)
        elif hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                count_tensors(v)

    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            param_bytes = 0
            count_tensors(state)
            per_param.append(
                {"shape": tuple(p.shape), "state_bytes": param_bytes}
            )

    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "n_state_tensors": n_tensors,
        "per_param": per_param,
    }


def collect_projector_ranks(optimizer: torch.optim.Optimizer) -> Dict[str, int]:
    """
    Walk through optimizer state and extract current effective rank
    from every projector attached to a parameter.
    """
    ranks = {}
    idx = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            proj = state.get("projector", None)
            if proj is not None:
                ranks[f"layer_{idx}_{tuple(p.shape)}"] = proj.get_effective_rank()
            idx += 1
    return ranks


def collect_sv_histories(optimizer: torch.optim.Optimizer) -> Dict[str, list]:
    """
    Walk through optimizer state and extract singular value histories
    from every projector.
    """
    histories = {}
    idx = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            proj = state.get("projector", None)
            if proj is not None and hasattr(proj, "_sv_history"):
                key = f"layer_{idx}_{tuple(p.shape)}"
                histories[key] = proj._sv_history
            idx += 1
    return histories


def collect_rank_histories(optimizer: torch.optim.Optimizer) -> Dict[str, list]:
    """
    Walk through optimizer state and extract rank histories
    from ProximalGaLore projectors.
    """
    histories = {}
    idx = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            proj = state.get("projector", None)
            if proj is not None and hasattr(proj, "_rank_history"):
                key = f"layer_{idx}_{tuple(p.shape)}"
                histories[key] = proj._rank_history
            idx += 1
    return histories
