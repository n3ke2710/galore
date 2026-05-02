#!/usr/bin/env python3
"""
Large-scale experiment runner for Proximal GaLore.

Focus:
  - Multiple datasets (MNIST, FashionMNIST, CIFAR10)
  - Multiple models (MLP, CNN variants)
  - Multiple seeds and hyperparameter sweeps
  - Consolidated CSV/JSON outputs for analysis
"""

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from galore_framework import (
    StandardAdamW,
    GaLoreAdamW,
    ProximalGaLoreAdamW,
    compute_memory_footprint,
)

# ---------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class DatasetSpec:
    name: str
    in_channels: int
    input_size: int
    num_classes: int


DATASETS: Dict[str, DatasetSpec] = {
    "MNIST": DatasetSpec("MNIST", in_channels=1, input_size=28, num_classes=10),
    "FashionMNIST": DatasetSpec("FashionMNIST", in_channels=1, input_size=28, num_classes=10),
    "CIFAR10": DatasetSpec("CIFAR10", in_channels=3, input_size=32, num_classes=10),
}


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CIFARCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def build_model(model_name: str, spec: DatasetSpec) -> nn.Module:
    if model_name == "mlp":
        input_dim = spec.in_channels * spec.input_size * spec.input_size
        return MLP(input_dim=input_dim, hidden_dim=512, num_classes=spec.num_classes)
    if model_name == "cnn":
        if spec.name == "CIFAR10":
            return CIFARCNN(num_classes=spec.num_classes)
        return SimpleCNN(in_channels=spec.in_channels, num_classes=spec.num_classes)
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def get_transforms(dataset_name: str):
    if dataset_name in {"MNIST", "FashionMNIST"}:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    if dataset_name == "CIFAR10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_loaders(dataset_name: str, batch_size: int, data_dir: str, subset: int) -> Tuple[DataLoader, DataLoader]:
    transform = get_transforms(dataset_name)
    dataset_cls = getattr(torchvision.datasets, dataset_name)
    train_data = dataset_cls(root=data_dir, train=True, download=True, transform=transform)
    test_data = dataset_cls(root=data_dir, train=False, download=True, transform=transform)

    if subset > 0 and subset < len(train_data):
        indices = torch.randperm(len(train_data))[:subset]
        train_data = Subset(train_data, indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def train_one_run(
    model: nn.Module,
    optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
) -> Tuple[Dict[str, float], List[float]]:
    criterion = nn.CrossEntropyLoss()
    model.to(DEVICE)

    start_time = time.perf_counter()
    total_steps = 0
    last_loss = 0.0
    epoch_losses: List[float] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, dynamic_ncols=True)
        for images, labels in progress:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
            running_loss += loss.item() * images.size(0)
            total_steps += 1
        epoch_losses.append(running_loss / len(train_loader.dataset))

    total_time = time.perf_counter() - start_time
    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)

    mem = compute_memory_footprint(optimizer)

    metrics = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "final_loss": last_loss,
        "time_sec": total_time,
        "steps": total_steps,
        "mem_mb": mem["total_mb"],
    }

    return metrics, epoch_losses


def plot_loss_curve(losses: List[float], out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------

def build_optimizer(opt_name: str, model_params, lr: float, weight_decay: float, cfg: Dict[str, float]):
    if opt_name == "adamw":
        return StandardAdamW(model_params, lr=lr, weight_decay=weight_decay)
    if opt_name == "galore":
        return GaLoreAdamW(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            rank=int(cfg["rank"]),
            update_proj_gap=int(cfg["update_proj_gap"]),
            galore_scale=float(cfg["galore_scale"]),
        )
    if opt_name == "prox":
        return ProximalGaLoreAdamW(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            threshold=float(cfg["threshold"]),
            update_proj_gap=int(cfg["update_proj_gap"]),
            galore_scale=float(cfg["galore_scale"]),
            min_rank=int(cfg["min_rank"]),
        )
    raise ValueError(f"Unknown optimizer: {opt_name}")


def make_sweep_grid() -> List[Dict[str, float]]:
    grid = []
    for rank in [8, 16, 32, 64]:
        grid.append({
            "opt": "galore",
            "rank": rank,
            "update_proj_gap": 200,
            "galore_scale": 1.0,
        })
    for threshold in [0.01, 0.03, 0.05]:
        for min_rank in [2, 4]:
            grid.append({
                "opt": "prox",
                "threshold": threshold,
                "min_rank": min_rank,
                "update_proj_gap": 200,
                "galore_scale": 1.0,
            })
    grid.append({
        "opt": "adamw",
    })
    return grid


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_summary_row(csv_path: str, row: Dict[str, float]) -> None:
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_sweep(args) -> None:
    ensure_dir(args.results_dir)
    ensure_dir(os.path.join(args.results_dir, "runs"))

    datasets = [d.strip() for d in args.datasets.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    sweep_grid = make_sweep_grid()

    for dataset_name in datasets:
        spec = DATASETS[dataset_name]
        train_loader, test_loader = get_loaders(
            dataset_name, args.batch_size, args.data_dir, args.subset
        )

        for model_name in models:
            for cfg in sweep_grid:
                opt_name = cfg["opt"]

                for seed in seeds:
                    seed_everything(seed)
                    model = build_model(model_name, spec)
                    optimizer = build_optimizer(
                        opt_name, model.parameters(), args.lr, args.weight_decay, cfg
                    )

                    metrics, losses = train_one_run(
                        model, optimizer, train_loader, test_loader, args.epochs
                    )

                    row = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "optimizer": opt_name,
                        "seed": seed,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        **{k: v for k, v in cfg.items() if k != "opt"},
                        **metrics,
                    }

                    summary_path = os.path.join(args.results_dir, "summary.csv")
                    write_summary_row(summary_path, row)

                    run_id = f"{dataset_name}_{model_name}_{opt_name}_seed{seed}_{int(time.time())}"
                    run_path = os.path.join(args.results_dir, "runs", f"{run_id}.json")
                    with open(run_path, "w") as f:
                        json.dump(row, f, indent=2)

                    plot_path = os.path.join(args.results_dir, "runs", f"{run_id}_loss.png")
                    plot_loss_curve(
                        losses,
                        plot_path,
                        title=f"{dataset_name} {model_name} {opt_name} seed={seed}",
                    )

                    print(
                        f"Done: {dataset_name}/{model_name}/{opt_name} seed={seed} "
                        f"acc={metrics['test_acc']:.2f}% mem={metrics['mem_mb']:.1f}MB"
                    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run large-scale GaLore sweeps")
    parser.add_argument("--datasets", type=str, default="MNIST,FashionMNIST,CIFAR10")
    parser.add_argument("--models", type=str, default="mlp,cnn")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--subset", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--results-dir", type=str, default="./results/sweeps")
    return parser.parse_args()


def main():
    args = parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
