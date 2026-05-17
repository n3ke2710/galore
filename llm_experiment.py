#!/usr/bin/env python3
"""
LLM Experiment: GPT-2 on WikiText-2 — comparing all 3 optimizers.

Trains GPT-2 (124M) with StandardAdamW, GaLoreAdamW, and ProximalGaLoreAdamW,
logging loss, perplexity, memory, and per-layer rank evolution.
All raw data → JSON/CSV so plots can be regenerated without re-training.

Usage:
    python llm_experiment.py                    # full pipeline
    python llm_experiment.py --plot-only        # regenerate plots only
    python llm_experiment.py --optimizers prox  # train only one optimizer
"""

import argparse, json, csv, logging, os, sys, time, re, math, copy
from collections import defaultdict
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from galore_framework import (
    StandardAdamW, GaLoreAdamW, ProximalGaLoreAdamW,
    compute_memory_footprint, collect_projector_ranks,
)

# =====================================================================
#  Constants
# =====================================================================
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
RESULTS_DIR = os.path.join("results", "gpt2")

# Optimizer display names & colors (for plots)
OPT_META = {
    "adamw": {"label": "AdamW (baseline)", "color": "#4c72b0"},
    "galore": {"label": "GaLore (fixed rank)", "color": "#55a868"},
    "prox":  {"label": "Proximal GaLore (SVT)", "color": "#c44e52"},
}

# =====================================================================
#  Logging
# =====================================================================
logger = logging.getLogger("llm_experiment")

def setup_logging(results_dir: str) -> None:
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    logger.addHandler(ch)
    os.makedirs(results_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(results_dir, "experiment.log"),
                             mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info("Logging → %s/experiment.log", results_dir)

# =====================================================================
#  1. Data
# =====================================================================
class CausalLMDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, block_size: int):
        n_full = (len(token_ids) - 1) // block_size
        self.input_ids = token_ids[: n_full * block_size].view(n_full, block_size)
        self.labels = token_ids[1: n_full * block_size + 1].view(n_full, block_size)
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, i): return self.input_ids[i], self.labels[i]

def load_wikitext2(block_size: int = 256, batch_size: int = 16):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    logger.info("[data] Loading wikitext-2-raw-v1 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    all_ids: List[int] = []
    for row in ds:
        if row["text"].strip():
            all_ids.extend(tok.encode(row["text"]))
    token_ids = torch.tensor(all_ids, dtype=torch.long)
    logger.info("[data] Tokens: %s  block_size=%d  batch_size=%d",
                f"{len(token_ids):,}", block_size, batch_size)
    dataset = CausalLMDataset(token_ids, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    logger.info("[data] Batches/epoch: %d  samples: %d", len(loader), len(dataset))
    return loader, tok

# =====================================================================
#  2. Model
# =====================================================================
def build_gpt2():
    from transformers import GPT2LMHeadModel, GPT2Config
    config = GPT2Config()  # 124M
    model = GPT2LMHeadModel(config)
    n = sum(p.numel() for p in model.parameters())
    logger.info("[model] GPT-2 params: %s", f"{n:,}")
    return model

def make_param_groups(model: nn.Module):
    """Split: 2D Linear weights → GaLore group, rest → standard group."""
    galore_params, regular_params = [], []
    galore_names, regular_names = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.dim() == 2 and "wte" not in name and "wpe" not in name:
            galore_params.append(p); galore_names.append(name)
        else:
            regular_params.append(p); regular_names.append(name)
    logger.info("[params] GaLore: %d  Regular: %d", len(galore_params), len(regular_params))
    return galore_params, regular_params, galore_names

# =====================================================================
#  3. Optimizer factory
# =====================================================================
def build_optimizer(opt_name: str, model: nn.Module, lr: float, wd: float,
                    rank: int, threshold: float, update_proj_gap: int,
                    galore_scale: float, min_rank: int):
    galore_p, regular_p, galore_names = make_param_groups(model)
    param_groups = [{"params": galore_p}, {"params": regular_p}]

    if opt_name == "adamw":
        opt = StandardAdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "galore":
        opt = GaLoreAdamW(param_groups, lr=lr, weight_decay=wd,
                          rank=rank, update_proj_gap=update_proj_gap,
                          galore_scale=galore_scale)
    elif opt_name == "prox":
        opt = ProximalGaLoreAdamW(param_groups, lr=lr, weight_decay=wd,
                                  threshold=threshold, update_proj_gap=update_proj_gap,
                                  galore_scale=galore_scale, min_rank=min_rank)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    return opt, galore_names

# =====================================================================
#  4. Training loop (single optimizer)
# =====================================================================
def seed_everything(seed: int):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def _collect_named_ranks(optimizer, pid_to_name: dict) -> dict:
    ranks = {}
    for g in optimizer.param_groups:
        for p in g["params"]:
            st = optimizer.state.get(p, {})
            proj = st.get("projector", None)
            if proj is not None:
                name = pid_to_name.get(id(p), f"p_{id(p)}")
                short = name.replace("transformer.", "").replace(".weight", "")
                ranks[short] = proj.get_effective_rank()
    return ranks

def train_single(opt_name: str, loader: DataLoader, args, results_dir: str):
    """Train GPT-2 with a single optimizer, return path to results subdir."""
    sub_dir = os.path.join(results_dir, opt_name)
    os.makedirs(sub_dir, exist_ok=True)

    seed_everything(args.seed)
    model = build_gpt2().to(DEVICE)

    optimizer, galore_names = build_optimizer(
        opt_name, model, lr=args.lr, wd=args.weight_decay,
        rank=args.rank, threshold=args.threshold,
        update_proj_gap=args.update_proj_gap,
        galore_scale=args.galore_scale, min_rank=args.min_rank,
    )

    pid_to_name = {id(p): n for n, p in model.named_parameters()}

    step_log = []
    rank_snapshots = defaultdict(list)
    rank_steps = []
    mem_log = []

    model.train()
    global_step = 0
    epoch = 0
    t0 = time.perf_counter()

    logger.info("")
    logger.info("═" * 60)
    logger.info("  [%s] Starting training  |  device=%s", opt_name.upper(), DEVICE)
    logger.info("  epochs=%d  lr=%g  steps/epoch≈%d", args.epochs, args.lr, len(loader))
    logger.info("═" * 60)

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"[{opt_name}] Epoch {epoch}/{args.epochs}",
                    leave=False, dynamic_ncols=True)
        for input_ids, labels in pbar:
            input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            elapsed = time.perf_counter() - t0

            if global_step % args.log_every == 0 or global_step == 1:
                lv = loss.item()
                ppl = math.exp(min(lv, 20))  # cap to avoid overflow
                mem = compute_memory_footprint(optimizer)
                step_log.append({"step": global_step, "epoch": epoch,
                                 "loss": round(lv, 5), "ppl": round(ppl, 2),
                                 "mem_mb": round(mem["total_mb"], 2),
                                 "time_sec": round(elapsed, 2)})

                # Ranks (only meaningful for galore/prox)
                if opt_name in ("galore", "prox"):
                    ranks_now = _collect_named_ranks(optimizer, pid_to_name)
                    rank_steps.append(global_step)
                    for ln, rv in ranks_now.items():
                        rank_snapshots[ln].append(rv)

                logger.info("[%s] step=%d  loss=%.4f  ppl=%.1f  mem=%.1fMB  t=%.0fs",
                            opt_name, global_step, lv, ppl, mem["total_mb"], elapsed)
                logger.debug("  ranks: %s", dict(rank_snapshots) if rank_snapshots else "N/A")
                pbar.set_postfix(loss=f"{lv:.4f}", ppl=f"{ppl:.1f}")

    total_time = time.perf_counter() - t0
    logger.info("[%s] Done: %d steps in %.1fs (%.1f steps/s)",
                opt_name, global_step, total_time, global_step / total_time)

    # ── Eval perplexity on last batch (quick proxy) ──
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
        final_ppl = math.exp(min(out.loss.item(), 20))
    logger.info("[%s] Final eval ppl ≈ %.2f", opt_name, final_ppl)

    # ── Save ──
    _save_run(sub_dir, step_log, rank_snapshots, rank_steps, galore_names,
              {"optimizer": opt_name, "epochs": args.epochs, "lr": args.lr,
               "weight_decay": args.weight_decay, "rank": args.rank,
               "threshold": args.threshold, "update_proj_gap": args.update_proj_gap,
               "galore_scale": args.galore_scale, "min_rank": args.min_rank,
               "block_size": args.block_size, "batch_size": args.batch_size,
               "seed": args.seed, "total_steps": global_step,
               "total_time_sec": round(total_time, 2), "device": DEVICE,
               "final_ppl": round(final_ppl, 2)})
    logger.info("[%s] Saved to %s/", opt_name, sub_dir)

def _save_run(d, step_log, rank_snapshots, rank_steps, galore_names, config):
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(step_log, f, indent=2)
    with open(os.path.join(d, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step","epoch","loss","ppl","mem_mb","time_sec"])
        w.writeheader(); w.writerows(step_log)
    with open(os.path.join(d, "rank_history.json"), "w") as f:
        json.dump({"steps": rank_steps, "ranks": dict(rank_snapshots)}, f, indent=2)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(d, "layer_names.json"), "w") as f:
        json.dump(galore_names, f, indent=2)

# =====================================================================
#  5. Orchestrator
# =====================================================================
def train_all(args):
    setup_logging(args.results_dir)
    logger.info("═" * 60)
    logger.info("  Hyperparameters:")
    logger.info("    epochs=%d  lr=%g  batch=%d  block=%d  seed=%d",
                args.epochs, args.lr, args.batch_size, args.block_size, args.seed)
    logger.info("    rank=%d  threshold=%g  update_proj_gap=%d  min_rank=%d",
                args.rank, args.threshold, args.update_proj_gap, args.min_rank)
    logger.info("    optimizers=%s", args.optimizers)
    logger.info("═" * 60)

    loader, _ = load_wikitext2(block_size=args.block_size, batch_size=args.batch_size)
    opt_list = [o.strip() for o in args.optimizers.split(",")]

    for opt_name in opt_list:
        train_single(opt_name, loader, args, args.results_dir)

    logger.info("All training complete!")

# =====================================================================
#  6. Plotting — reads JSON, fully independent of training
# =====================================================================
FONT_FAMILY = "serif"
CMAP_HEATMAP = "inferno"
FIG_DPI = 200

def _apply_style():
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": 12,
        "axes.titlesize": 15, "axes.labelsize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 9,
        "legend.fontsize": 11, "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.25,
    })

def _ema(vals, alpha=0.12):
    out, s = [], vals[0]
    for v in vals:
        s = alpha * v + (1 - alpha) * s; out.append(s)
    return out

def _load(path):
    with open(path) as f: return json.load(f)

def _sort_layer_names(names):
    def key(n):
        m = re.search(r"h\.(\d+)", n)
        blk = int(m.group(1)) if m else 999
        sub = 0 if "attn" in n else (1 if "mlp" in n else 2)
        det = 0 if "c_attn" in n else (1 if "c_fc" in n else 2)
        return (blk, sub, det, n)
    return sorted(names, key=key)

def _pretty_label(name):
    m = re.match(r"h\.(\d+)\.(attn|mlp)\.(.+)", name)
    if m:
        blk, mod, sub = m.groups()
        return f"B{blk} {'Attn' if mod == 'attn' else 'MLP'} {sub}"
    return name

# ── 6a. Comparison: loss curves ──────────────────────────────────────
def plot_loss_comparison(results_dir, plots_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    for opt in ("adamw", "galore", "prox"):
        p = os.path.join(results_dir, opt, "metrics.json")
        if not os.path.exists(p): continue
        m = _load(p)
        steps = [e["step"] for e in m]; losses = [e["loss"] for e in m]
        sm = _ema(losses)
        meta = OPT_META[opt]
        ax.plot(steps, losses, alpha=0.15, color=meta["color"], linewidth=0.8)
        ax.plot(steps, sm, color=meta["color"], linewidth=2.5, label=meta["label"])
    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    ax.set_title("GPT-2 Training Loss Comparison", fontsize=16, fontweight="bold")
    ax.legend(framealpha=0.9, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss_comparison.png")); plt.close(fig)
    logger.info("  → loss_comparison.png")

# ── 6b. Comparison: perplexity curves ────────────────────────────────
def plot_ppl_comparison(results_dir, plots_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    for opt in ("adamw", "galore", "prox"):
        p = os.path.join(results_dir, opt, "metrics.json")
        if not os.path.exists(p): continue
        m = _load(p)
        steps = [e["step"] for e in m]; ppls = [e["ppl"] for e in m]
        sm = _ema(ppls)
        meta = OPT_META[opt]
        ax.plot(steps, ppls, alpha=0.15, color=meta["color"], linewidth=0.8)
        ax.plot(steps, sm, color=meta["color"], linewidth=2.5, label=meta["label"])
    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_ylabel("Perplexity", fontweight="bold")
    ax.set_title("GPT-2 Perplexity Comparison", fontsize=16, fontweight="bold")
    ax.set_yscale("log"); ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "ppl_comparison.png")); plt.close(fig)
    logger.info("  → ppl_comparison.png")

# ── 6c. Comparison: memory ───────────────────────────────────────────
def plot_memory_comparison(results_dir, plots_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    for opt in ("adamw", "galore", "prox"):
        p = os.path.join(results_dir, opt, "metrics.json")
        if not os.path.exists(p): continue
        m = _load(p)
        steps = [e["step"] for e in m]; mem = [e["mem_mb"] for e in m]
        meta = OPT_META[opt]
        ax.fill_between(steps, mem, alpha=0.15, color=meta["color"])
        ax.plot(steps, mem, color=meta["color"], linewidth=2.2, label=meta["label"])
    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_ylabel("Optimizer State Memory (MB)", fontweight="bold")
    ax.set_title("Optimizer Memory Footprint Comparison", fontsize=16, fontweight="bold")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "memory_comparison.png")); plt.close(fig)
    logger.info("  → memory_comparison.png")

# ── 6d. Rank heatmap (per optimizer) ─────────────────────────────────
def plot_rank_heatmap(results_dir, plots_dir, opt_name: str):
    p = os.path.join(results_dir, opt_name, "rank_history.json")
    if not os.path.exists(p): return
    rd = _load(p)
    steps, ranks = rd["steps"], rd["ranks"]
    if not ranks or not steps: return

    layer_names = _sort_layer_names(list(ranks.keys()))
    n_layers, n_steps = len(layer_names), len(steps)
    matrix = np.zeros((n_layers, n_steps))
    for i, ln in enumerate(layer_names):
        v = ranks[ln]; matrix[i, :len(v)] = v

    labels = [_pretty_label(n) for n in layer_names]
    height = max(8, n_layers * 0.35)
    fig, ax = plt.subplots(figsize=(16, height))
    im = ax.imshow(matrix, aspect="auto", cmap=CMAP_HEATMAP, interpolation="nearest")
    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_ylabel("Layer", fontweight="bold")
    title = OPT_META.get(opt_name, {}).get("label", opt_name)
    ax.set_title(f"Effective Rank Evolution — {title}", fontsize=15, fontweight="bold", pad=12)

    tick_s = max(1, n_steps // 15)
    idx = list(range(0, n_steps, tick_s))
    ax.set_xticks(idx); ax.set_xticklabels([str(steps[i]) for i in idx], rotation=45, ha="right")
    ax.set_yticks(range(n_layers)); ax.set_yticklabels(labels)

    # Block separators
    prev_blk = None
    for i, nm in enumerate(layer_names):
        m = re.search(r"h\.(\d+)", nm)
        blk = int(m.group(1)) if m else -1
        if prev_blk is not None and blk != prev_blk:
            ax.axhline(y=i - 0.5, color="white", linewidth=1.5, alpha=0.7)
        prev_blk = blk

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Effective Rank", fontweight="bold")
    fig.tight_layout()
    fname = f"rank_heatmap_{opt_name}.png"
    fig.savefig(os.path.join(plots_dir, fname)); plt.close(fig)
    logger.info("  → %s", fname)

# ── 6e. Summary bar chart ────────────────────────────────────────────
def plot_summary_bar(results_dir, plots_dir):
    """Final loss, final PPL, memory — as grouped bar chart."""
    data = {}
    for opt in ("adamw", "galore", "prox"):
        p = os.path.join(results_dir, opt, "config.json")
        mp = os.path.join(results_dir, opt, "metrics.json")
        if not os.path.exists(p) or not os.path.exists(mp): continue
        cfg = _load(p); metrics = _load(mp)
        last = metrics[-1]
        data[opt] = {"ppl": cfg.get("final_ppl", last["ppl"]),
                     "mem_mb": last["mem_mb"],
                     "time": cfg.get("total_time_sec", 0)}
    if len(data) < 2: return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    opts = list(data.keys())
    colors = [OPT_META[o]["color"] for o in opts]
    labels = [OPT_META[o]["label"] for o in opts]

    for ax, metric, title, ylabel in zip(
        axes,
        ["ppl", "mem_mb", "time"],
        ["Final Perplexity", "Optimizer Memory", "Training Time"],
        ["Perplexity", "MB", "Seconds"],
    ):
        vals = [data[o][metric] for o in opts]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Optimizer Comparison Summary", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "summary_comparison.png")); plt.close(fig)
    logger.info("  → summary_comparison.png")

# ── Plot orchestrator ────────────────────────────────────────────────
def plot_all(results_dir: str = RESULTS_DIR):
    setup_logging(results_dir)
    _apply_style()
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info("[plot] Generating plots from %s ...", results_dir)

    plot_loss_comparison(results_dir, plots_dir)
    plot_ppl_comparison(results_dir, plots_dir)
    plot_memory_comparison(results_dir, plots_dir)
    plot_summary_bar(results_dir, plots_dir)
    for opt in ("galore", "prox"):
        plot_rank_heatmap(results_dir, plots_dir, opt)

    logger.info("[plot] All plots saved to %s/", plots_dir)

# =====================================================================
#  7. CLI
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="GPT-2 × 3 optimizers experiment")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    p.add_argument("--optimizers", type=str, default="adamw,galore,prox",
                   help="Comma-separated: adamw,galore,prox")
    # Scale: 3 epochs × ~2000 steps/epoch × 3 opts ≈ 18k steps → ~1-2h on H200
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    # GaLore-specific
    p.add_argument("--rank", type=int, default=128, help="Fixed rank for GaLore")
    p.add_argument("--threshold", type=float, default=0.03, help="SVT λ for ProxGaLore")
    p.add_argument("--update-proj-gap", type=int, default=200)
    p.add_argument("--galore-scale", type=float, default=1.0)
    p.add_argument("--min-rank", type=int, default=4)
    # Data
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()

def main():
    args = parse_args()
    if not args.plot_only:
        train_all(args)
    plot_all(results_dir=args.results_dir)

if __name__ == "__main__":
    main()
