#!/usr/bin/env python3
"""
LLM Experiment: GPT-2 on WikiText-2 with Proximal GaLore.

Trains a GPT-2 model using ProximalGaLoreAdamW and tracks per-layer rank
evolution, loss, and optimizer memory. All raw data is saved to JSON/CSV
so plots can be regenerated without re-training.

Usage:
    # Full pipeline: train + plot
    python llm_experiment.py

    # Only regenerate plots from saved data
    python llm_experiment.py --plot-only

    # Custom training params
    python llm_experiment.py --max-steps 2000 --log-every 50 --lr 1e-4
"""

import argparse
import json
import csv
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm

# ── Framework imports ────────────────────────────────────────────────
from galore_framework import (
    ProximalGaLoreAdamW,
    compute_memory_footprint,
    collect_projector_ranks,
    collect_rank_histories,
)

# =====================================================================
#  Constants
# =====================================================================

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
RESULTS_DIR = os.path.join("results", "gpt2")

# =====================================================================
#  Logging setup
# =====================================================================

logger = logging.getLogger("llm_experiment")


def setup_logging(results_dir: str, level: int = logging.INFO) -> None:
    """Configure dual logging: console (INFO) + file (DEBUG) with timestamps."""
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — DEBUG and above (full trace)
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "experiment.log")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Logging initialised  →  %s", log_path)

# =====================================================================
#  1. Data: WikiText-2 tokenized for Causal LM
# =====================================================================

class CausalLMDataset(Dataset):
    """Pre-tokenized chunks for causal language modelling."""

    def __init__(self, token_ids: torch.Tensor, block_size: int):
        n = len(token_ids) - block_size
        self.input_ids = token_ids[:n].unfold(0, block_size, block_size)
        self.labels = token_ids[1:n + 1].unfold(0, block_size, block_size)
        # Trim to equal length
        min_len = min(len(self.input_ids), len(self.labels))
        self.input_ids = self.input_ids[:min_len]
        self.labels = self.labels[:min_len]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


def load_wikitext2(block_size: int = 128, batch_size: int = 8):
    """Load WikiText-2-raw-v1 via HuggingFace and return a DataLoader."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    logger.info("[data] Loading wikitext-2-raw-v1 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize everything into one long sequence
    all_ids: List[int] = []
    for row in ds:
        text = row["text"]
        if text.strip():
            all_ids.extend(tokenizer.encode(text))

    token_ids = torch.tensor(all_ids, dtype=torch.long)
    logger.info("[data] Total tokens: %s  |  block_size=%d", f"{len(token_ids):,}", block_size)

    dataset = CausalLMDataset(token_ids, block_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    return loader, tokenizer


# =====================================================================
#  2. Model helpers
# =====================================================================

def build_gpt2():
    """Initialise a fresh GPT-2 (124 M) model."""
    from transformers import GPT2LMHeadModel, GPT2Config

    config = GPT2Config()          # default 124M params
    model = GPT2LMHeadModel(config)
    logger.info("[model] GPT-2 params: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    return model


def make_param_groups(model: nn.Module):
    """
    Split parameters into two groups:
      - galore_params: 2-D weight matrices (Linear layers) → use GaLore
      - regular_params: everything else (Embedding, LayerNorm, biases)
    """
    galore_params = []
    regular_params = []

    galore_layer_names = []
    regular_layer_names = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 2 and "wte" not in name and "wpe" not in name:
            galore_params.append(p)
            galore_layer_names.append(name)
        else:
            regular_params.append(p)
            regular_layer_names.append(name)

    logger.info("[params] GaLore layers: %d  |  Regular: %d", len(galore_params), len(regular_params))
    logger.debug("[params] GaLore layer names: %s", galore_layer_names)
    logger.debug("[params] Regular layer names: %s", regular_layer_names)
    return galore_params, regular_params, galore_layer_names


# =====================================================================
#  3. Training loop
# =====================================================================

def train(
    max_steps: int = 1500,
    log_every: int = 50,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    threshold: float = 0.03,
    update_proj_gap: int = 100,
    galore_scale: float = 1.0,
    min_rank: int = 2,
    block_size: int = 128,
    batch_size: int = 8,
    results_dir: str = RESULTS_DIR,
):
    """Run the full training pipeline and save raw metrics to disk."""
    os.makedirs(results_dir, exist_ok=True)
    setup_logging(results_dir)

    logger.info("═" * 60)
    logger.info("  Hyperparameters:")
    logger.info("    max_steps=%d  log_every=%d  lr=%g", max_steps, log_every, lr)
    logger.info("    threshold=%g  update_proj_gap=%d  min_rank=%d", threshold, update_proj_gap, min_rank)
    logger.info("    galore_scale=%g  weight_decay=%g", galore_scale, weight_decay)
    logger.info("    block_size=%d  batch_size=%d", block_size, batch_size)
    logger.info("═" * 60)

    # ── Data ──
    loader, tokenizer = load_wikitext2(block_size=block_size, batch_size=batch_size)

    # ── Model ──
    model = build_gpt2().to(DEVICE)

    # ── Optimizer ──
    galore_params, regular_params, galore_layer_names = make_param_groups(model)

    optimizer = ProximalGaLoreAdamW(
        [
            {"params": galore_params},
            {"params": regular_params},
        ],
        lr=lr,
        weight_decay=weight_decay,
        threshold=threshold,
        update_proj_gap=update_proj_gap,
        galore_scale=galore_scale,
        min_rank=min_rank,
    )

    # ── Tracking containers ──
    step_log: List[Dict[str, Any]] = []          # per-step: step, loss, time
    rank_snapshots: Dict[str, List[int]] = defaultdict(list)
    rank_snapshot_steps: List[int] = []
    mem_log: List[Dict[str, Any]] = []

    # Map param id → human-readable layer name
    param_id_to_name = {}
    for name, p in model.named_parameters():
        param_id_to_name[id(p)] = name

    # ── Training ──
    model.train()
    global_step = 0
    epoch = 0
    t0 = time.perf_counter()

    logger.info("")
    logger.info("═" * 60)
    logger.info("  Training GPT-2 on WikiText-2  |  device=%s", DEVICE)
    logger.info("  max_steps=%d  lr=%g  threshold=%g", max_steps, lr, threshold)
    logger.info("═" * 60)
    logger.info("")

    while global_step < max_steps:
        epoch += 1
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True)
        for input_ids, labels in pbar:
            if global_step >= max_steps:
                break

            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            elapsed = time.perf_counter() - t0

            # ── Log every N steps ──
            if global_step % log_every == 0 or global_step == 1:
                loss_val = loss.item()
                step_log.append({
                    "step": global_step,
                    "loss": loss_val,
                    "time_sec": round(elapsed, 2),
                })

                # Collect per-layer ranks
                ranks_now = _collect_named_ranks(optimizer, param_id_to_name)
                rank_snapshot_steps.append(global_step)
                for lname, rval in ranks_now.items():
                    rank_snapshots[lname].append(rval)

                # Memory
                mem = compute_memory_footprint(optimizer)
                mem_log.append({
                    "step": global_step,
                    "mem_mb": round(mem["total_mb"], 2),
                })

                # Log to file + console
                logger.info(
                    "step=%d  loss=%.4f  mem=%.1fMB  elapsed=%.1fs  ranks_snapshot=%d_layers",
                    global_step, loss_val, mem["total_mb"], elapsed, len(ranks_now),
                )
                logger.debug("  ranks: %s", ranks_now)

                pbar.set_postfix(loss=f"{loss_val:.4f}", mem=f"{mem['total_mb']:.1f}MB")

    total_time = time.perf_counter() - t0
    logger.info("")
    logger.info("[train] Completed %d steps in %.1fs", global_step, total_time)
    logger.info("")

    # ── Save raw data ──
    _save_raw_data(results_dir, step_log, rank_snapshots, rank_snapshot_steps, mem_log,
                   galore_layer_names, {
                       "max_steps": max_steps, "lr": lr, "threshold": threshold,
                       "update_proj_gap": update_proj_gap, "galore_scale": galore_scale,
                       "min_rank": min_rank, "block_size": block_size,
                       "batch_size": batch_size, "weight_decay": weight_decay,
                       "total_time_sec": round(total_time, 2), "device": DEVICE,
                   })

    logger.info("[save] Raw data saved to %s/", results_dir)
    return results_dir


def _collect_named_ranks(optimizer, param_id_to_name: Dict[int, str]) -> Dict[str, int]:
    """Collect ranks with human-readable layer names."""
    ranks = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            proj = state.get("projector", None)
            if proj is not None:
                name = param_id_to_name.get(id(p), f"param_{id(p)}")
                # Shorten transformer. prefix for readability
                short = name.replace("transformer.", "").replace(".weight", "")
                ranks[short] = proj.get_effective_rank()
    return ranks


# =====================================================================
#  4. Raw data persistence
# =====================================================================

def _save_raw_data(
    results_dir: str,
    step_log: list,
    rank_snapshots: dict,
    rank_snapshot_steps: list,
    mem_log: list,
    galore_layer_names: list,
    config: dict,
):
    """Dump all metrics as JSON / CSV for reproducible re-plotting."""
    # 4a. metrics.json — loss + time per step
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(step_log, f, indent=2)

    # 4b. metrics.csv — same data, CSV format
    with open(os.path.join(results_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "time_sec"])
        writer.writeheader()
        writer.writerows(step_log)

    # 4c. rank_history.json — {layer_name: [rank_at_step_1, rank_at_step_2, ...]}
    with open(os.path.join(results_dir, "rank_history.json"), "w") as f:
        json.dump({
            "steps": rank_snapshot_steps,
            "ranks": dict(rank_snapshots),
        }, f, indent=2)

    # 4d. memory.json
    with open(os.path.join(results_dir, "memory.json"), "w") as f:
        json.dump(mem_log, f, indent=2)

    # 4e. config.json — hyperparameters
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 4f. layer_names.json
    with open(os.path.join(results_dir, "layer_names.json"), "w") as f:
        json.dump(galore_layer_names, f, indent=2)


# =====================================================================
#  5. Plotting (reads from saved JSON — fully independent of training)
# =====================================================================

# ── Plotting style constants ─────────────────────────────────────────
FONT_FAMILY = "serif"
CMAP_HEATMAP = "inferno"
FIG_DPI = 200


def _apply_style():
    """Set publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def plot_all(results_dir: str = RESULTS_DIR):
    """Read saved JSONs and generate all plots."""
    _apply_style()

    metrics = _load_json(results_dir, "metrics.json")
    rank_data = _load_json(results_dir, "rank_history.json")
    mem_data = _load_json(results_dir, "memory.json")

    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_rank_heatmap(rank_data, plots_dir)
    plot_loss_curve(metrics, plots_dir)
    plot_memory_curve(mem_data, plots_dir)

    logger.info("[plot] All plots saved to %s/", plots_dir)


def _load_json(results_dir: str, filename: str):
    path = os.path.join(results_dir, filename)
    with open(path) as f:
        return json.load(f)


# ── 5a. Rank heatmap ─────────────────────────────────────────────────

def _sort_layer_names(names: List[str]) -> List[str]:
    """Sort layers: group by block number, then Attention before MLP."""
    def sort_key(name: str):
        import re
        # Extract block number
        m = re.search(r"h\.(\d+)", name)
        block = int(m.group(1)) if m else 999
        # Attention layers first (attn), then MLP
        if "attn" in name:
            sublayer = 0
            if "c_attn" in name:
                detail = 0
            elif "c_proj" in name:
                detail = 1
            else:
                detail = 2
        elif "mlp" in name:
            sublayer = 1
            if "c_fc" in name:
                detail = 0
            elif "c_proj" in name:
                detail = 1
            else:
                detail = 2
        else:
            sublayer = 2
            detail = 0
        return (block, sublayer, detail, name)

    return sorted(names, key=sort_key)


def plot_rank_heatmap(rank_data: dict, plots_dir: str):
    """
    Main figure: heatmap of effective rank evolution across layers.
    X = training step, Y = layer, color = rank.
    """
    steps = rank_data["steps"]
    ranks = rank_data["ranks"]

    layer_names = _sort_layer_names(list(ranks.keys()))
    n_layers = len(layer_names)
    n_steps = len(steps)

    if n_layers == 0 or n_steps == 0:
        logger.warning("[plot] No rank data to plot.")
        return

    # Build matrix (layers × steps)
    matrix = np.zeros((n_layers, n_steps))
    for i, lname in enumerate(layer_names):
        vals = ranks[lname]
        matrix[i, :len(vals)] = vals

    # ── Pretty labels: shorten layer names ──
    pretty_labels = []
    for name in layer_names:
        label = name
        # e.g. "h.0.attn.c_attn" → "B0 Attn c_attn"
        import re
        m = re.match(r"h\.(\d+)\.(attn|mlp)\.(.+)", label)
        if m:
            block, module, sub = m.groups()
            mod_label = "Attn" if module == "attn" else "MLP"
            label = f"B{block} {mod_label} {sub}"
        pretty_labels.append(label)

    # ── Figure ──
    height = max(8, n_layers * 0.35)
    fig, ax = plt.subplots(figsize=(14, height))

    im = ax.imshow(
        matrix, aspect="auto", cmap=CMAP_HEATMAP,
        interpolation="nearest",
    )

    # Axes
    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_ylabel("Layer", fontweight="bold")
    ax.set_title(
        "Proximal GaLore: Effective Rank Evolution by Layer",
        fontsize=15, fontweight="bold", pad=12,
    )

    # X ticks — show subset of steps
    tick_step = max(1, n_steps // 15)
    xtick_idx = list(range(0, n_steps, tick_step))
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels([str(steps[i]) for i in xtick_idx], rotation=45, ha="right")

    # Y ticks
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(pretty_labels)

    # Add horizontal separators between blocks
    prev_block = None
    for i, name in enumerate(layer_names):
        import re
        m = re.search(r"h\.(\d+)", name)
        block = int(m.group(1)) if m else -1
        if prev_block is not None and block != prev_block:
            ax.axhline(y=i - 0.5, color="white", linewidth=1.5, alpha=0.7)
        prev_block = block

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Effective Rank", fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(plots_dir, "rank_heatmap.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("  → %s", out_path)


# ── 5b. Loss curve ───────────────────────────────────────────────────

def plot_loss_curve(metrics: list, plots_dir: str):
    """Smoothed training loss vs. step."""
    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]

    # Exponential moving average for smoothing
    smoothed = _ema(losses, alpha=0.15)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, alpha=0.25, color="#5e81ac", linewidth=1, label="Raw")
    ax.plot(steps, smoothed, color="#bf616a", linewidth=2.2, label="Smoothed (EMA)")
    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    ax.set_title("GPT-2 Training Loss (Proximal GaLore)", fontsize=14, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.set_xlim(steps[0], steps[-1])

    fig.tight_layout()
    out_path = os.path.join(plots_dir, "train_loss.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("  → %s", out_path)


def _ema(values: list, alpha: float = 0.1) -> list:
    """Exponential moving average."""
    result = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        result.append(s)
    return result


# ── 5c. Memory curve ─────────────────────────────────────────────────

def plot_memory_curve(mem_data: list, plots_dir: str):
    """Optimizer state memory vs. step."""
    steps = [m["step"] for m in mem_data]
    mem_mb = [m["mem_mb"] for m in mem_data]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(steps, mem_mb, alpha=0.3, color="#a3be8c")
    ax.plot(steps, mem_mb, color="#a3be8c", linewidth=2.2, marker="o", markersize=3)
    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_ylabel("Optimizer State Memory (MB)", fontweight="bold")
    ax.set_title("Optimizer Memory Footprint", fontsize=14, fontweight="bold")
    ax.set_xlim(steps[0], steps[-1])

    fig.tight_layout()
    out_path = os.path.join(plots_dir, "memory.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("  → %s", out_path)


# =====================================================================
#  6. CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="GPT-2 + Proximal GaLore experiment")
    p.add_argument("--plot-only", action="store_true",
                    help="Skip training, only regenerate plots from saved data")
    p.add_argument("--results-dir", type=str, default=RESULTS_DIR)

    # Training hyperparameters
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--threshold", type=float, default=0.03)
    p.add_argument("--update-proj-gap", type=int, default=100)
    p.add_argument("--galore-scale", type=float, default=1.0)
    p.add_argument("--min-rank", type=int, default=2)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    # Ensure logging is set up even in --plot-only mode
    setup_logging(args.results_dir)
    logger.info("Script started  |  args=%s", vars(args))

    if not args.plot_only:
        train(
            max_steps=args.max_steps,
            log_every=args.log_every,
            lr=args.lr,
            weight_decay=args.weight_decay,
            threshold=args.threshold,
            update_proj_gap=args.update_proj_gap,
            galore_scale=args.galore_scale,
            min_rank=args.min_rank,
            block_size=args.block_size,
            batch_size=args.batch_size,
            results_dir=args.results_dir,
        )

    plot_all(results_dir=args.results_dir)
    logger.info("Script finished successfully")


if __name__ == "__main__":
    main()
