#!/usr/bin/env python3
"""
Smoke-tests for llm_experiment.py components.

Validates that:
  1. Framework imports work (optimizers, projectors, utils)
  2. Parameter grouping splits 2D vs non-2D correctly
  3. ProximalGaLoreAdamW trains a tiny GPT-2 without errors
  4. Rank collection returns sensible data
  5. Raw data saving/loading roundtrips correctly
  6. Plotting functions run without errors on synthetic data

Run:
    python test_llm_experiment.py
"""

import json
import os
import shutil
import sys
import tempfile
import traceback

import torch
import numpy as np

# ── Counters ──
_passed = 0
_failed = 0


def _ok(name: str):
    global _passed
    _passed += 1
    print(f"  ✅  {name}")


def _fail(name: str, err: Exception):
    global _failed
    _failed += 1
    print(f"  ❌  {name}: {err}")
    traceback.print_exc()


# =====================================================================
#  Test 1: Framework imports
# =====================================================================

def test_imports():
    name = "Framework imports"
    try:
        from galore_framework import (
            StandardAdamW,
            GaLoreAdamW,
            ProximalGaLoreAdamW,
            GaLoreProjector,
            ProximalGaLoreProjector,
            compute_memory_footprint,
            collect_projector_ranks,
            collect_rank_histories,
        )
        _ok(name)
    except Exception as e:
        _fail(name, e)


# =====================================================================
#  Test 2: Parameter grouping
# =====================================================================

def test_param_groups():
    name = "Parameter group splitting"
    try:
        from transformers import GPT2LMHeadModel, GPT2Config

        config = GPT2Config(n_layer=2, n_head=2, n_embd=64)
        model = GPT2LMHeadModel(config)

        from llm_experiment import make_param_groups
        galore_p, regular_p, galore_names = make_param_groups(model)

        # Galore params must all be 2D and NOT embeddings
        for p in galore_p:
            assert p.dim() == 2, f"Expected 2D, got {p.dim()}D"

        # Regular params include embeddings, layernorm, biases
        has_1d = any(p.dim() == 1 for p in regular_p)
        assert has_1d, "Regular group should contain 1D params (biases, LN)"

        # No overlap
        galore_ids = set(id(p) for p in galore_p)
        regular_ids = set(id(p) for p in regular_p)
        assert len(galore_ids & regular_ids) == 0, "Overlap between groups"

        assert len(galore_p) > 0, "No GaLore params found"
        assert len(regular_p) > 0, "No regular params found"

        _ok(name)
    except Exception as e:
        _fail(name, e)


# =====================================================================
#  Test 3: Tiny GPT-2 training (3 steps)
# =====================================================================

def test_tiny_training():
    name = "Tiny GPT-2 training (3 steps)"
    try:
        from transformers import GPT2LMHeadModel, GPT2Config
        from galore_framework import ProximalGaLoreAdamW, compute_memory_footprint
        from llm_experiment import make_param_groups

        device = "cpu"
        config = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=100)
        model = GPT2LMHeadModel(config).to(device)

        galore_p, regular_p, _ = make_param_groups(model)

        optimizer = ProximalGaLoreAdamW(
            [{"params": galore_p}, {"params": regular_p}],
            lr=1e-3,
            threshold=0.03,
            update_proj_gap=2,
            min_rank=1,
        )

        # Fake data
        input_ids = torch.randint(0, 100, (2, 16))
        labels = torch.randint(0, 100, (2, 16))

        losses = []
        for step in range(3):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # Loss should be finite
        assert all(np.isfinite(l) for l in losses), f"Non-finite loss: {losses}"

        # Memory footprint should be > 0
        mem = compute_memory_footprint(optimizer)
        assert mem["total_mb"] > 0, "Memory footprint is 0"

        _ok(name)
    except Exception as e:
        _fail(name, e)


# =====================================================================
#  Test 4: Rank collection
# =====================================================================

def test_rank_collection():
    name = "Rank collection"
    try:
        from transformers import GPT2LMHeadModel, GPT2Config
        from galore_framework import ProximalGaLoreAdamW, collect_projector_ranks
        from llm_experiment import make_param_groups, _collect_named_ranks

        config = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=100)
        model = GPT2LMHeadModel(config)

        galore_p, regular_p, _ = make_param_groups(model)
        optimizer = ProximalGaLoreAdamW(
            [{"params": galore_p}, {"params": regular_p}],
            lr=1e-3, threshold=0.03, update_proj_gap=2, min_rank=1,
        )

        # Do 1 step to initialize projectors
        input_ids = torch.randint(0, 100, (2, 16))
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # collect_projector_ranks (generic)
        ranks_generic = collect_projector_ranks(optimizer)
        assert len(ranks_generic) > 0, "No ranks collected (generic)"

        # _collect_named_ranks (named)
        param_id_to_name = {id(p): n for n, p in model.named_parameters()}
        ranks_named = _collect_named_ranks(optimizer, param_id_to_name)
        assert len(ranks_named) > 0, "No ranks collected (named)"

        # All ranks should be >= 1 (min_rank)
        for layer, rank in ranks_named.items():
            assert rank >= 1, f"{layer} has rank {rank} < 1"

        _ok(name)
    except Exception as e:
        _fail(name, e)


# =====================================================================
#  Test 5: Data save / load roundtrip
# =====================================================================

def test_data_roundtrip():
    name = "Data save/load roundtrip"
    try:
        from llm_experiment import _save_raw_data

        tmpdir = tempfile.mkdtemp(prefix="test_llm_")

        step_log = [
            {"step": 50, "loss": 10.5, "time_sec": 1.2},
            {"step": 100, "loss": 9.3, "time_sec": 2.4},
        ]
        rank_snapshots = {
            "h.0.attn.c_attn": [32, 28],
            "h.0.mlp.c_fc": [16, 14],
        }
        rank_snapshot_steps = [50, 100]
        mem_log = [
            {"step": 50, "mem_mb": 12.5},
            {"step": 100, "mem_mb": 12.3},
        ]
        config = {"lr": 1e-4, "threshold": 0.03}

        _save_raw_data(tmpdir, step_log, rank_snapshots,
                       rank_snapshot_steps, mem_log, ["layer1"], config)

        # Verify files exist
        for fname in ["metrics.json", "metrics.csv", "rank_history.json",
                       "memory.json", "config.json", "layer_names.json"]:
            fpath = os.path.join(tmpdir, fname)
            assert os.path.exists(fpath), f"Missing {fname}"

        # Verify JSON roundtrip
        with open(os.path.join(tmpdir, "rank_history.json")) as f:
            loaded = json.load(f)
        assert loaded["steps"] == rank_snapshot_steps
        assert loaded["ranks"]["h.0.attn.c_attn"] == [32, 28]

        with open(os.path.join(tmpdir, "metrics.json")) as f:
            loaded_metrics = json.load(f)
        assert len(loaded_metrics) == 2
        assert loaded_metrics[0]["loss"] == 10.5

        shutil.rmtree(tmpdir)
        _ok(name)
    except Exception as e:
        _fail(name, e)


# =====================================================================
#  Test 6: Plotting on synthetic data
# =====================================================================

def test_plotting():
    name = "Plotting from synthetic data"
    try:
        from llm_experiment import plot_rank_heatmap, plot_loss_curve, plot_memory_curve

        tmpdir = tempfile.mkdtemp(prefix="test_llm_plots_")
        plots_dir = os.path.join(tmpdir, "plots")
        os.makedirs(plots_dir)

        # Synthetic rank data
        steps = list(range(50, 550, 50))
        rank_data = {
            "steps": steps,
            "ranks": {
                "h.0.attn.c_attn": [np.random.randint(10, 50) for _ in steps],
                "h.0.attn.c_proj": [np.random.randint(5, 30) for _ in steps],
                "h.0.mlp.c_fc":    [np.random.randint(15, 60) for _ in steps],
                "h.0.mlp.c_proj":  [np.random.randint(8, 40) for _ in steps],
                "h.1.attn.c_attn": [np.random.randint(10, 50) for _ in steps],
                "h.1.mlp.c_fc":    [np.random.randint(15, 60) for _ in steps],
            },
        }

        metrics = [{"step": s, "loss": 10.0 - s * 0.01 + np.random.randn() * 0.3,
                     "time_sec": s * 0.1} for s in steps]
        mem_data = [{"step": s, "mem_mb": 12.0 + np.random.randn() * 0.2} for s in steps]

        plot_rank_heatmap(rank_data, plots_dir)
        plot_loss_curve(metrics, plots_dir)
        plot_memory_curve(mem_data, plots_dir)

        # Verify PNGs exist
        for fname in ["rank_heatmap.png", "train_loss.png", "memory.png"]:
            fpath = os.path.join(plots_dir, fname)
            assert os.path.exists(fpath), f"Missing plot {fname}"
            assert os.path.getsize(fpath) > 1000, f"Plot {fname} too small"

        shutil.rmtree(tmpdir)
        _ok(name)
    except Exception as e:
        _fail(name, e)


# =====================================================================
#  Test 7: Layer name sorting
# =====================================================================

def test_layer_sorting():
    name = "Layer name sorting (Attn before MLP)"
    try:
        from llm_experiment import _sort_layer_names

        names = [
            "h.1.mlp.c_fc", "h.0.attn.c_proj", "h.0.mlp.c_fc",
            "h.0.attn.c_attn", "h.1.attn.c_attn", "h.1.mlp.c_proj",
        ]
        sorted_names = _sort_layer_names(names)

        # Block 0 should come before Block 1
        b0 = [n for n in sorted_names if "h.0" in n]
        b1 = [n for n in sorted_names if "h.1" in n]
        assert sorted_names.index(b0[0]) < sorted_names.index(b1[0])

        # Within a block, attn should come before mlp
        for block_names in [b0, b1]:
            attn_idx = [sorted_names.index(n) for n in block_names if "attn" in n]
            mlp_idx = [sorted_names.index(n) for n in block_names if "mlp" in n]
            if attn_idx and mlp_idx:
                assert max(attn_idx) < min(mlp_idx), \
                    f"Attn not before MLP: attn={attn_idx}, mlp={mlp_idx}"

        _ok(name)
    except Exception as e:
        _fail(name, e)


# =====================================================================
#  Runner
# =====================================================================

def main():
    print()
    print("=" * 56)
    print("  LLM Experiment — Smoke Tests")
    print("=" * 56)
    print()

    test_imports()
    test_param_groups()
    test_tiny_training()
    test_rank_collection()
    test_data_roundtrip()
    test_plotting()
    test_layer_sorting()

    print()
    print("-" * 56)
    total = _passed + _failed
    print(f"  Results:  {_passed}/{total} passed", end="")
    if _failed:
        print(f",  {_failed} FAILED")
    else:
        print("  ✨  All clear!")
    print("-" * 56)
    print()

    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
