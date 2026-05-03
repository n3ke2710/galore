#!/usr/bin/env python3
"""
Experiment: GaLore vs Proximal GaLore on MNIST (MLP and CNN).

Tracks:
  1. Loss vs Iterations / Time
  2. Argument convergence (Update norm and Distance from start) vs Iterations / Time
  3. Train / Test Accuracy
  4. GaLore-specific metrics (Ranks, Memory, Singular Values)
"""

import os
import copy
import time
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from galore_framework.utils import (
    collect_projector_ranks,
    collect_sv_histories,
    collect_rank_histories,
)

# ======================================================================
#  Config
# ======================================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = "results"

# Training Hyperparams
EPOCHS = 10
BATCH_SIZE = 128
LR = 5e-3
WEIGHT_DECAY = 1e-3

# GaLore Params
GALORE_RANK = 64
UPDATE_PROJ_GAP = 250
GALORE_SCALE = 0.25

# Proximal GaLore Params
SVT_THRESHOLD = 0.03
MIN_RANK = 4

# ======================================================================
#  Models
# ======================================================================

class MNIST_MLP(nn.Module):
    """Simple MLP for MNIST."""
    def __init__(self, input_dim=784, hidden_dim=512, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)

class MNIST_CNN(nn.Module):
    """LeNet-style CNN for MNIST."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ======================================================================
#  Helper Functions
# ======================================================================

def get_param_norm(model: nn.Module) -> float:
    """Compute the Frobenius norm of all parameters concatenated."""
    return torch.sqrt(sum(p.norm()**2 for p in model.parameters())).item()

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

def train_mnist(model, optimizer, train_loader, test_loader, epochs, tracker, model_name="model"):
    criterion = nn.CrossEntropyLoss()
    
    # Store initial cache to compute distance from start
    initial_model = copy.deepcopy(model)
    prev_model = copy.deepcopy(model)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    
    start_time = time.perf_counter()
    global_step = 0
    
    print(f"  Training {model_name}...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 1. Update Norm ||w_k - w_{k-1}|| and ||w_k - w_0||
            # Note: We compute this BEFORE step for w_{k-1} or AFTER for w_k.
            # Usually we take the step, then compare.
            optimizer.step()
            scheduler.step()
            
            elapsed = time.perf_counter() - start_time
            
            # Metrics every step can be expensive for CNN, but user asked for it. 
            # We'll compute weight metrics every few steps or every step? Let's do every step for iterations as requested.
            update_norm = get_param_diff(model, prev_model)
            dist_start = get_param_diff(model, initial_model)
            
            tracker.log_loss(loss.item(), elapsed)
            tracker.log_arg_metrics(update_norm, dist_start)
            tracker.log_grad_norm(model)
            
            # Update prev_model (expensive copy, maybe just sub?)
            with torch.no_grad():
                for p_prev, p_curr in zip(prev_model.parameters(), model.parameters()):
                    p_prev.copy_(p_curr)
            
            global_step += 1
        
        # End of epoch metrics
        train_acc = evaluate(model, train_loader)
        test_acc = evaluate(model, test_loader)
        tracker.log_accuracies(train_acc, test_acc)
        
        # Track memory at the end of each epoch
        mem_mb = compute_memory_footprint(optimizer)["total_mb"]
        tracker.log_memory(mem_mb)
        
        # GaLore Ranks
        ranks = collect_projector_ranks(optimizer)
        for name, r in ranks.items():
            tracker.log_rank(name, r)
            
        print(f"    Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # Final Singular Values extraction
    # (Simplified: just call a helper later)
    return tracker

# ======================================================================
#  Plotting
# ======================================================================

def plot_mnist_benchmark(results: Dict[str, TrainingTracker], model_tag: str, save_dir: str):
    """
    Consolidated 4x2 grid of plots.
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    fig.suptitle(f"MNIST Benchmark: {model_tag}", fontsize=20, fontweight='bold')
    
    colors = {"AdamW": "#2196F3", "GaLore AdamW": "#FF9800", "Proximal GaLore": "#4CAF50"}
    
    # 1. Loss vs Iterations
    ax = axes[0, 0]
    for name, tracker in results.items():
        ax.plot(tracker.losses, label=name, color=colors.get(name))
    ax.set_title("Loss vs Iterations")
    ax.set_xlabel("Steps")
    ax.set_ylabel("CrossEntropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Loss vs Time
    ax = axes[0, 1]
    for name, tracker in results.items():
        ax.plot(tracker.times, tracker.losses, label=name, color=colors.get(name))
    ax.set_title("Loss vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CrossEntropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Update Norm ||w_k - w_{k-1}|| vs Time
    ax = axes[1, 0]
    for name, tracker in results.items():
        # Smoothen update norms for better visualization
        win = 20
        y = np.convolve(tracker.arg_updates, np.ones(win)/win, mode='valid')
        x = tracker.times[win-1:]
        ax.plot(x, y, label=name, color=colors.get(name))
    ax.set_title("Update Norm ||w_k - w_{k-1}|| vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Step Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Update Norm vs Iterations
    ax = axes[1, 1]
    for name, tracker in results.items():
        win = 20
        y = np.convolve(tracker.arg_updates, np.ones(win)/win, mode='valid')
        ax.plot(y, label=name, color=colors.get(name))
    ax.set_title("Update Norm ||w_k - w_{k-1}|| vs Iterations")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Step Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Distance from Start vs Iterations
    ax = axes[2, 0]
    for name, tracker in results.items():
        ax.plot(tracker.arg_dist_start, label=name, color=colors.get(name))
    ax.set_title("Distance from Start ||w_k - w_initial|| vs Iterations")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Test Accuracy vs Epochs (Iterations scaled)
    ax = axes[2, 1]
    for name, tracker in results.items():
        steps_per_epoch = len(tracker.losses) // len(tracker.test_accuracies)
        x = np.arange(1, len(tracker.test_accuracies) + 1) * steps_per_epoch
        ax.plot(x, tracker.test_accuracies, label=f"{name} (Test)", marker='o', color=colors.get(name))
        ax.plot(x, tracker.train_accuracies, label=f"{name} (Train)", linestyle='--', alpha=0.5, color=colors.get(name))
    ax.set_title("Accuracy vs Iterations")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7. Optimizer State Memory vs Epochs
    ax = axes[3, 0]
    for name, tracker in results.items():
        if len(tracker.memory_history) > 0:
            ax.plot(range(1, len(tracker.memory_history) + 1), tracker.memory_history, label=name, marker='o', color=colors.get(name))
    ax.set_title("Optimizer State Memory vs Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Memory (MB)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Hide the 8th subplot
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

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_mnist_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_mnist_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Subsampling for faster runs if needed, but let's use full for MNIST (it's small anyway)
    # Actually, for 10 epochs on multiple optimizers and models, a small subset is better for a quick demo.
    subset_indices = torch.randperm(len(train_mnist_data))[:10000]
    train_subset = torch.utils.data.Subset(train_mnist_data, subset_indices)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_mnist_data, batch_size=BATCH_SIZE, shuffle=False)

    architectures = {
        "MLP": MNIST_MLP(),
        "CNN": MNIST_CNN()
    }

    for arch_name, base_model in architectures.items():
        print(f"\n" + "="*60)
        print(f"  Architecture: {arch_name}")
        print("="*60)
        
        results = {}
        memory_results = {}
        
        # 1. AdamW
        print("\n── AdamW ──")
        model = copy.deepcopy(base_model).to(DEVICE)
        opt = StandardAdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        tracker = train_mnist(model, opt, train_loader, test_loader, EPOCHS, TrainingTracker(), f"{arch_name}_AdamW")
        results["AdamW"] = tracker
        memory_results["AdamW"] = compute_memory_footprint(opt)["total_mb"]

        # 2. GaLore
        print("\n── GaLore AdamW ──")
        model = copy.deepcopy(base_model).to(DEVICE)
        opt = GaLoreAdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            rank=GALORE_RANK,
            update_proj_gap=UPDATE_PROJ_GAP
        )
        tracker = train_mnist(model, opt, train_loader, test_loader, EPOCHS, TrainingTracker(), f"{arch_name}_GaLore")
        results["GaLore AdamW"] = tracker
        memory_results["GaLore AdamW"] = compute_memory_footprint(opt)["total_mb"]
        
        # 3. Proximal GaLore
        print("\n── Proximal GaLore ──")
        model = copy.deepcopy(base_model).to(DEVICE)
        opt = ProximalGaLoreAdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            threshold=SVT_THRESHOLD,
            update_proj_gap=UPDATE_PROJ_GAP,
            min_rank=MIN_RANK
        )
        tracker = train_mnist(model, opt, train_loader, test_loader, EPOCHS, TrainingTracker(), f"{arch_name}_Proximal")
        results["Proximal GaLore"] = tracker
        memory_results["Proximal GaLore"] = compute_memory_footprint(opt)["total_mb"]

        # Final Plotting for this architecture
        arch_dir = os.path.join(RESULTS_DIR, arch_name.lower())
        os.makedirs(arch_dir, exist_ok=True)
        plot_mnist_benchmark(results, arch_name, arch_dir)
        
        # Also save memory and summary
        print(f"\nSummary for {arch_name}:")
        print(f"{'Method':<20} {'Test Acc (%)':>15} {'Mem Avg (MB)':>15} {'Mem Peak (MB)':>15} {'Mem Final (MB)':>15} {'Time (s)':>15}")
        for name, tracker in results.items():
            if len(tracker.memory_history) > 0:
                avg_mem = sum(tracker.memory_history) / len(tracker.memory_history)
                peak_mem = max(tracker.memory_history)
                final_mem = tracker.memory_history[-1]
            else:
                # Fallback if history is empty for some reason
                avg_mem = peak_mem = final_mem = memory_results[name]
                
            total_time = tracker.times[-1] if len(tracker.times) > 0 else 0.0
            print(f"{name:<20} {tracker.test_accuracies[-1]:>15.2f} {avg_mem:>15.3f} {peak_mem:>15.3f} {final_mem:>15.3f} {total_time:>15.1f}")

if __name__ == "__main__":
    main()
