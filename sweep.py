#!/usr/bin/env python3
"""
Hyperparameter Sweep for GaLore and Proximal GaLore.

Runs multiple configurations (different SVT thresholds for Proximal,
different fixed ranks for GaLore) and generates a Pareto front comparison
(Memory vs Perplexity) to find the optimal trade-off.
"""

import os
import glob
import json
import argparse
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the training logic from the main script
from llm_experiment import train_single, load_wikitext2, setup_logging, logger, _apply_style, OPT_META

class DummyArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def run_sweep(base_args: DummyArgs, sweep_dir: str):
    os.makedirs(sweep_dir, exist_ok=True)
    setup_logging(sweep_dir)
    logger.info("Starting Hyperparameter Sweep...")

    loader, _ = load_wikitext2(block_size=base_args.block_size, batch_size=base_args.batch_size)

    # 1. Baseline AdamW
    logger.info("--- Running AdamW Baseline ---")
    args = DummyArgs(**base_args.__dict__)
    train_single("adamw", loader, args, os.path.join(sweep_dir, "adamw_baseline"))

    # 2. Sweep standard GaLore ranks
    galore_ranks = [16, 64, 128]
    for r in galore_ranks:
        logger.info(f"--- Running GaLore (rank={r}) ---")
        args = DummyArgs(**base_args.__dict__)
        args.rank = r
        train_single("galore", loader, args, os.path.join(sweep_dir, f"galore_rank{r}"))

    # 3. Sweep Proximal GaLore thresholds
    # Now that we use relative thresholding, reasonable values are 0.01 to 0.15
    prox_thresholds = [0.01, 0.05, 0.10, 0.15]
    for th in prox_thresholds:
        logger.info(f"--- Running Proximal GaLore (threshold={th}) ---")
        args = DummyArgs(**base_args.__dict__)
        args.threshold = th
        train_single("prox", loader, args, os.path.join(sweep_dir, f"prox_thresh{th}"))

    logger.info("Sweep execution complete.")


def plot_sweep_results(sweep_dir: str):
    _apply_style()
    plots_dir = os.path.join(sweep_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    results = []
    
    # Read all subdirectories robustly
    for d_name in os.listdir(sweep_dir):
        d = os.path.join(sweep_dir, d_name)
        if not os.path.isdir(d) or d_name == "plots": 
            continue
            
        # train_single artificially appends opt_name to the results_dir,
        # so the actual files are inside a nested subfolder (e.g., galore_rank16/galore/)
        subfolders = [f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))]
        if not subfolders:
            # Fallback if somehow they are in the root
            actual_d = d
        else:
            actual_d = os.path.join(d, subfolders[0])
            
        cfg_path = os.path.join(actual_d, "config.json")
        met_path = os.path.join(actual_d, "metrics.json")
        if not os.path.exists(cfg_path) or not os.path.exists(met_path):
            continue
            
        with open(cfg_path) as f: cfg = json.load(f)
        with open(met_path) as f: metrics = json.load(f)
        
        last = metrics[-1]
        name = os.path.basename(d.rstrip("/"))
        
        opt = cfg["optimizer"]
        label = f"AdamW" if opt == "adamw" else (
            f"GaLore (r={cfg['rank']})" if opt == "galore" else f"Proximal (th={cfg['threshold']})"
        )
        
        results.append({
            "name": name,
            "opt": opt,
            "label": label,
            "ppl": cfg.get("final_ppl", last["ppl"]),
            "mem_mb": last["mem_mb"],
            "time": cfg.get("total_time_sec", 0),
            "loss_curve": [m["loss"] for m in metrics],
            "steps": [m["step"] for m in metrics],
        })

    if not results:
        logger.warning("No results found to plot.")
        return

    # ── 1. Pareto Front: Memory vs Perplexity ──
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {"adamw": OPT_META["adamw"]["color"], 
              "galore": OPT_META["galore"]["color"], 
              "prox": OPT_META["prox"]["color"]}
              
    markers = {"adamw": "X", "galore": "o", "prox": "s"}
    
    for r in results:
        ax.scatter(r["mem_mb"], r["ppl"], color=colors[r["opt"]], 
                   marker=markers[r["opt"]], s=150, edgecolors='black', zorder=5)
        ax.annotate(r["label"], (r["mem_mb"], r["ppl"]), 
                    xytext=(8, 8), textcoords="offset points", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax.set_xlabel("Optimizer Memory Footprint (MB)", fontweight="bold")
    ax.set_ylabel("Final Perplexity (Lower is Better)", fontweight="bold")
    ax.set_title("Memory vs Perplexity Trade-off (Pareto Front)", fontsize=15, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Invert x axis if you want "better" (lower memory) on the right? 
    # Usually bottom-left is the best (low memory, low PPL).
    ax.set_xlim(left=0)
    
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "pareto_front.png"))
    plt.close(fig)

    # ── 2. Loss Curves Sweep ──
    fig, ax = plt.subplots(figsize=(12, 7))
    for r in results:
        ax.plot(r["steps"], r["loss_curve"], color=colors[r["opt"]], 
                linewidth=2.0 if r["opt"] == "adamw" else 1.5, 
                alpha=0.8, label=r["label"])
        
    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    ax.set_title("Sweep Loss Curves", fontsize=15, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "sweep_loss_curves.png"))
    plt.close(fig)

    logger.info(f"Sweep plots saved to {plots_dir}")
    
    # Generate Markdown Table
    md_table = "| Model Configuration | Final Perplexity | Optimizer Memory (MB) | Time (s) |\n"
    md_table += "|---------------------|-----------------|-----------------------|----------|\n"
    
    # Sort by PPL
    results.sort(key=lambda x: x["ppl"])
    for r in results:
        md_table += f"| {r['label']} | {r['ppl']:.2f} | {r['mem_mb']:.1f} | {r['time']:.1f} |\n"
        
    with open(os.path.join(sweep_dir, "sweep_summary.md"), "w") as f:
        f.write("# Hyperparameter Sweep Results\n\n")
        f.write(md_table)
    
    logger.info("Markdown summary generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=str, default="results/sweep")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    base_args = DummyArgs(
        epochs=args.epochs,
        lr=5e-5,
        weight_decay=0.01,
        seed=42,
        rank=128,          # Default value, overridden in sweep
        threshold=0.05,    # Default value, overridden in sweep
        update_proj_gap=200,
        galore_scale=1.0,
        min_rank=4,
        block_size=256,
        batch_size=16,
        log_every=25,
        model_size="124M"  # Change to 'tiny' for quick local tests
    )

    if not args.plot_only:
        run_sweep(base_args, args.sweep_dir)
        
    plot_sweep_results(args.sweep_dir)
