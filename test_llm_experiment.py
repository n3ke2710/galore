#!/usr/bin/env python3
"""Smoke-tests for llm_experiment.py — validates all components work."""

import json, os, shutil, sys, tempfile, traceback
import torch, numpy as np

_passed = _failed = 0
def _ok(n):
    global _passed; _passed += 1; print(f"  ✅  {n}")
def _fail(n, e):
    global _failed; _failed += 1; print(f"  ❌  {n}: {e}"); traceback.print_exc()

def test_imports():
    try:
        from galore_framework import (StandardAdamW, GaLoreAdamW, ProximalGaLoreAdamW,
            compute_memory_footprint, collect_projector_ranks)
        _ok("Framework imports")
    except Exception as e: _fail("Framework imports", e)

def test_param_groups():
    try:
        from transformers import GPT2LMHeadModel, GPT2Config
        from llm_experiment import make_param_groups
        model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=64))
        gp, rp, gn = make_param_groups(model)
        assert all(p.dim() == 2 for p in gp), "GaLore params must be 2D"
        assert any(p.dim() == 1 for p in rp), "Regular must have 1D"
        assert len(gp) > 0 and len(rp) > 0
        _ok("Parameter group splitting")
    except Exception as e: _fail("Parameter group splitting", e)

def test_all_optimizers():
    """Train 3 steps with each optimizer — no crashes."""
    try:
        from transformers import GPT2LMHeadModel, GPT2Config
        from llm_experiment import build_optimizer
        cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=100)
        ids = torch.randint(0, 100, (2, 16))
        for opt_name in ("adamw", "galore", "prox"):
            model = GPT2LMHeadModel(cfg)
            opt, _ = build_optimizer(opt_name, model, lr=1e-3, wd=0.01,
                rank=8, threshold=0.03, update_proj_gap=2, galore_scale=1.0, min_rank=1)
            for _ in range(3):
                out = model(input_ids=ids, labels=ids)
                out.loss.backward(); opt.step(); opt.zero_grad()
            assert np.isfinite(out.loss.item()), f"{opt_name} loss not finite"
        _ok("All 3 optimizers train OK")
    except Exception as e: _fail("All 3 optimizers train OK", e)

def test_rank_collection():
    try:
        from transformers import GPT2LMHeadModel, GPT2Config
        from llm_experiment import build_optimizer, _collect_named_ranks
        cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=100)
        model = GPT2LMHeadModel(cfg)
        opt, _ = build_optimizer("prox", model, lr=1e-3, wd=0.01,
            rank=8, threshold=0.03, update_proj_gap=2, galore_scale=1.0, min_rank=1)
        ids = torch.randint(0, 100, (2, 16))
        out = model(input_ids=ids, labels=ids); out.loss.backward(); opt.step(); opt.zero_grad()
        pid = {id(p): n for n, p in model.named_parameters()}
        ranks = _collect_named_ranks(opt, pid)
        assert len(ranks) > 0, "No ranks"
        assert all(r >= 1 for r in ranks.values())
        _ok("Rank collection")
    except Exception as e: _fail("Rank collection", e)

def test_data_roundtrip():
    try:
        from llm_experiment import _save_run
        d = tempfile.mkdtemp(prefix="test_llm_")
        _save_run(d,
            [{"step":50,"epoch":1,"loss":10.5,"ppl":100.0,"mem_mb":12.5,"time_sec":1.2}],
            {"h.0.attn.c_attn": [32]}, [50], ["layer1"],
            {"optimizer":"prox","lr":1e-4})
        for f in ("metrics.json","metrics.csv","rank_history.json","config.json"):
            assert os.path.exists(os.path.join(d,f)), f"Missing {f}"
        loaded = json.load(open(os.path.join(d,"rank_history.json")))
        assert loaded["ranks"]["h.0.attn.c_attn"] == [32]
        shutil.rmtree(d)
        _ok("Data save/load roundtrip")
    except Exception as e: _fail("Data save/load roundtrip", e)

def test_plotting():
    try:
        from llm_experiment import (plot_loss_comparison, plot_ppl_comparison,
            plot_memory_comparison, plot_rank_heatmap, plot_summary_bar)
        d = tempfile.mkdtemp(prefix="test_plots_")
        steps = list(range(25, 275, 25))
        # Create fake data for 3 optimizers
        for opt in ("adamw", "galore", "prox"):
            od = os.path.join(d, opt); os.makedirs(od)
            metrics = [{"step":s,"epoch":1,"loss":10.0-s*0.01+np.random.randn()*0.3,
                        "ppl":100-s*0.1,"mem_mb":12+np.random.randn()*0.2,
                        "time_sec":s*0.1} for s in steps]
            json.dump(metrics, open(os.path.join(od,"metrics.json"),"w"))
            json.dump({"optimizer":opt,"final_ppl":50.0,"total_time_sec":30},
                      open(os.path.join(od,"config.json"),"w"))
            if opt in ("galore","prox"):
                ranks = {"h.0.attn.c_attn":[np.random.randint(10,50) for _ in steps],
                         "h.0.mlp.c_fc":[np.random.randint(5,30) for _ in steps]}
                json.dump({"steps":steps,"ranks":ranks},
                          open(os.path.join(od,"rank_history.json"),"w"))
        pd = os.path.join(d, "plots"); os.makedirs(pd)
        plot_loss_comparison(d, pd)
        plot_ppl_comparison(d, pd)
        plot_memory_comparison(d, pd)
        plot_summary_bar(d, pd)
        plot_rank_heatmap(d, pd, "prox")
        for f in ("loss_comparison.png","ppl_comparison.png","memory_comparison.png",
                   "summary_comparison.png","rank_heatmap_prox.png"):
            assert os.path.exists(os.path.join(pd,f)) and os.path.getsize(os.path.join(pd,f))>1000, f
        shutil.rmtree(d)
        _ok("Plotting (all 5 charts)")
    except Exception as e: _fail("Plotting", e)

def test_layer_sorting():
    try:
        from llm_experiment import _sort_layer_names
        names = ["h.1.mlp.c_fc","h.0.attn.c_proj","h.0.mlp.c_fc",
                 "h.0.attn.c_attn","h.1.attn.c_attn"]
        s = _sort_layer_names(names)
        assert s.index("h.0.attn.c_attn") < s.index("h.0.mlp.c_fc")
        assert s.index("h.0.mlp.c_fc") < s.index("h.1.attn.c_attn")
        _ok("Layer sorting")
    except Exception as e: _fail("Layer sorting", e)

def main():
    print("\n" + "=" * 56)
    print("  LLM Experiment — Smoke Tests")
    print("=" * 56 + "\n")
    test_imports()
    test_param_groups()
    test_all_optimizers()
    test_rank_collection()
    test_data_roundtrip()
    test_plotting()
    test_layer_sorting()
    print(f"\n{'-'*56}")
    t = _passed + _failed
    print(f"  Results:  {_passed}/{t} passed", end="")
    print(f",  {_failed} FAILED" if _failed else "  ✨  All clear!")
    print(f"{'-'*56}\n")
    sys.exit(1 if _failed else 0)

if __name__ == "__main__":
    main()
