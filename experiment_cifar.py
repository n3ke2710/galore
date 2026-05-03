#!/usr/bin/env python3
"""
Experiment: GaLore vs Proximal GaLore on CIFAR-10 with ResNet-18.

Tracks:
  1. Loss vs Iterations / Time
  2. Argument convergence (Update norm and Distance from start) vs Iterations / Time
  3. Train / Test Accuracy
  4. GaLore-specific metrics (Ranks, Memory, Singular Values)
"""

import os
import copy
import time
from typing import Dict

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from galore_framework import (
    StandardAdamW,
    GaLoreAdamW,
    ProximalGaLoreAdamW,
    TrainingTracker,
    compute_memory_footprint,
)
from galore_framework.utils import collect_projector_ranks

# ======================================================================
#  Config
# ======================================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = "results"

# Training Hyperparams
EPOCHS = 50  # 50 epochs should be enough to see the trend (~1-1.5 hours total for all 3 methods on T4)
BATCH_SIZE = 128
LR = 5e-3
WEIGHT_DECAY = 1e-4

# GaLore Params
GALORE_RANK = 128  # ResNet has larger matrices (e.g. 512x512, 256x256), rank 128 gives good compression
UPDATE_PROJ_GAP = 200
GALORE_SCALE = 0.25

# Proximal GaLore Params
SVT_THRESHOLD = 0.03  # Relative threshold (3% of max singular value)
MIN_RANK = 4

# ======================================================================
#  Model
# ======================================================================

def get_resnet18_cifar10():
    """Adapts standard ResNet-18 for 32x32 CIFAR-10 images."""
    model = torchvision.models.resnet18(num_classes=10)
    # Replace the initial 7x7 conv and maxpool with a simpler 3x3 conv
    # to avoid drastically reducing the spatial dimensions of 32x32 images.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

# ======================================================================
#  Helper Functions
# ======================================================================

def get_param_diff(model1, model2) -> float:
    """Compute ||w1 - w2||."""
    dist = 0.0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        dist += (p1 - p2).norm()**2
    return torch.sqrt(dist).item()

def evaluate(model, loader):
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

# ======================================================================
#  Training Loop
# ======================================================================

def train_cifar(model, optimizer, train_loader, test_loader, epochs, tracker, model_name="model"):
    criterion = nn.CrossEntropyLoss()
    
    initial_model = copy.deepcopy(model)
    prev_model = copy.deepcopy(model)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    
    start_time = time.perf_counter()
    global_step = 0
    
    print(f"  Training {model_name}...")
    
    for epoch in range(epochs):
        model.train()
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            elapsed = time.perf_counter() - start_time
            
            # To save somewhat on computation, we only compute norms periodically or rely on small freq.
            if global_step % 10 == 0:
                update_norm = get_param_diff(model, prev_model)
                dist_start = get_param_diff(model, initial_model)
                tracker.log_arg_metrics(update_norm, dist_start)
                with torch.no_grad():
                    for p_prev, p_curr in zip(prev_model.parameters(), model.parameters()):
                        p_prev.copy_(p_curr)

            tracker.log_loss(loss.item(), elapsed)
            tracker.log_grad_norm(model)
            global_step += 1
            
        # End of epoch metrics
        train_acc = evaluate(model, train_loader)
        test_acc = evaluate(model, test_loader)
        tracker.log_accuracies(train_acc, test_acc)
        
        mem_mb = compute_memory_footprint(optimizer)["total_mb"]
        tracker.log_memory(mem_mb)
        
        ranks = collect_projector_ranks(optimizer)
        for name, r in ranks.items():
            tracker.log_rank(name, r)
            
        print(f"    Epoch {epoch+1:02d}/{epochs} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Mem: {mem_mb:.2f} MB")

    return tracker

# ======================================================================
#  Plotting
# ======================================================================

def plot_benchmark(results: Dict[str, TrainingTracker], model_tag: str, save_dir: str):
    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    fig.suptitle(f"CIFAR-10 Benchmark: {model_tag}", fontsize=20, fontweight='bold')
    
    colors = {"AdamW": "#2196F3", "GaLore AdamW": "#FF9800", "Proximal GaLore": "#4CAF50"}
    
    # 1. Loss vs Iterations
    ax = axes[0, 0]
    for name, tracker in results.items():
        # Smoothed loss
        win = max(1, len(tracker.losses) // 200)
        y = np.convolve(tracker.losses, np.ones(win)/win, mode='valid')
        ax.plot(y, label=name, color=colors.get(name))
    ax.set_title("Loss vs Iterations (Smoothed)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("CrossEntropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Loss vs Time
    ax = axes[0, 1]
    for name, tracker in results.items():
        win = max(1, len(tracker.losses) // 200)
        y = np.convolve(tracker.losses, np.ones(win)/win, mode='valid')
        x = tracker.times[win-1:]
        ax.plot(x, y, label=name, color=colors.get(name))
    ax.set_title("Loss vs Time (Smoothed)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CrossEntropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Update Norm vs Iterations
    ax = axes[1, 0]
    for name, tracker in results.items():
        win = max(1, len(tracker.arg_updates) // 50)
        y = np.convolve(tracker.arg_updates, np.ones(win)/win, mode='valid')
        ax.plot(y, label=name, color=colors.get(name))
    ax.set_title("Update Norm ||w_k - w_{k-1}|| vs Iterations")
    ax.set_xlabel("Tracked Steps (every 10th)")
    ax.set_ylabel("Step Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Distance from Start vs Iterations
    ax = axes[1, 1]
    for name, tracker in results.items():
        ax.plot(tracker.arg_dist_start, label=name, color=colors.get(name))
    ax.set_title("Distance from Start ||w_k - w_initial|| vs Iterations")
    ax.set_xlabel("Tracked Steps (every 10th)")
    ax.set_ylabel("Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Test Accuracy vs Epochs
    ax = axes[2, 0]
    for name, tracker in results.items():
        x = np.arange(1, len(tracker.test_accuracies) + 1)
        ax.plot(x, tracker.test_accuracies, label=f"{name} (Test)", marker='o', markersize=4, color=colors.get(name))
        ax.plot(x, tracker.train_accuracies, label=f"{name} (Train)", linestyle='--', alpha=0.5, color=colors.get(name))
    ax.set_title("Accuracy vs Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Optimizer State Memory vs Epochs
    ax = axes[2, 1]
    for name, tracker in results.items():
        if len(tracker.memory_history) > 0:
            ax.plot(range(1, len(tracker.memory_history) + 1), tracker.memory_history, label=name, marker='o', markersize=4, color=colors.get(name))
    ax.set_title("Optimizer State Memory vs Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Memory (MB)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Hide unused subplots
    axes[3, 0].axis('off')
    axes[3, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(save_dir, f"{model_tag.lower()}_benchmark.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Saved consolidated benchmark at {path}")


# ======================================================================
#  Main Execution
# ======================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    # Patch CIFAR-10 URL to avoid "HTTP Error 503: Service Unavailable" from cs.toronto.edu
    torchvision.datasets.CIFAR10.url = "https://ossci-datasets.s3.amazonaws.com/cifar/cifar-10-python.tar.gz"

    # Data Augmentation for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_cifar_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_cifar_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_cifar_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_cifar_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\n" + "="*60)
    print(f"  Architecture: ResNet-18 (CIFAR-10 Adaptation)")
    print("="*60)
    
    results = {}
    memory_results = {}
    
    # 1. AdamW
    print("\n── AdamW ──")
    model = get_resnet18_cifar10().to(DEVICE)
    opt = StandardAdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    tracker = train_cifar(model, opt, train_loader, test_loader, EPOCHS, TrainingTracker(), "ResNet18_AdamW")
    results["AdamW"] = tracker
    memory_results["AdamW"] = compute_memory_footprint(opt)["total_mb"]

    # 2. GaLore
    print("\n── GaLore AdamW ──")
    model = get_resnet18_cifar10().to(DEVICE)
    opt = GaLoreAdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        rank=GALORE_RANK,
        update_proj_gap=UPDATE_PROJ_GAP
    )
    tracker = train_cifar(model, opt, train_loader, test_loader, EPOCHS, TrainingTracker(), "ResNet18_GaLore")
    results["GaLore AdamW"] = tracker
    memory_results["GaLore AdamW"] = compute_memory_footprint(opt)["total_mb"]
    
    # 3. Proximal GaLore
    print("\n── Proximal GaLore ──")
    model = get_resnet18_cifar10().to(DEVICE)
    opt = ProximalGaLoreAdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        threshold=SVT_THRESHOLD,
        update_proj_gap=UPDATE_PROJ_GAP,
        min_rank=MIN_RANK
    )
    tracker = train_cifar(model, opt, train_loader, test_loader, EPOCHS, TrainingTracker(), "ResNet18_Proximal")
    results["Proximal GaLore"] = tracker
    memory_results["Proximal GaLore"] = compute_memory_footprint(opt)["total_mb"]

    # Result Aggregation
    arch_name = "resnet18_cifar10"
    arch_dir = os.path.join(RESULTS_DIR, arch_name)
    os.makedirs(arch_dir, exist_ok=True)
    plot_benchmark(results, arch_name, arch_dir)
    
    print(f"\nSummary for {arch_name}:")
    print(f"{'Method':<20} {'Test Acc (%)':>15} {'Mem Avg (MB)':>15} {'Mem Peak (MB)':>15} {'Mem Final (MB)':>15} {'Time (s)':>15}")
    for name, tracker in results.items():
        if len(tracker.memory_history) > 0:
            avg_mem = sum(tracker.memory_history) / len(tracker.memory_history)
            peak_mem = max(tracker.memory_history)
            final_mem = tracker.memory_history[-1]
        else:
            avg_mem = peak_mem = final_mem = memory_results.get(name, 0.0)
            
        total_time = tracker.times[-1] if len(tracker.times) > 0 else 0.0
        print(f"{name:<20} {tracker.test_accuracies[-1]:>15.2f} {avg_mem:>15.3f} {peak_mem:>15.3f} {final_mem:>15.3f} {total_time:>15.1f}")

if __name__ == "__main__":
    main()