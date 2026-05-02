#!/usr/bin/env python3
"""
Transformer language model on WikiText-2 with AdamW vs GaLore vs Proximal GaLore.
"""

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator

from galore_framework import StandardAdamW, GaLoreAdamW, ProximalGaLoreAdamW, compute_memory_footprint


DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class Config:
    epochs: int = 10
    batch_size: int = 20
    bptt: int = 35
    lr: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 42

    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # GaLore / Proximal GaLore
    galore_rank: int = 64
    update_proj_gap: int = 200
    svt_threshold: float = 0.03
    min_rank: int = 4


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerLM(nn.Module):
    def __init__(
        self,
        ntokens: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.embedding = nn.Embedding(ntokens, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntokens)

        self.init_weights()

    def init_weights(self) -> None:
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return self.decoder(output)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_vocab() -> Tuple[torchtext.vocab.Vocab, torchtext.data.utils._SimpleTokenizer]:
    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)

    train_iter = WikiText2(split="train")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab, tokenizer


def data_process(raw_text_iter, vocab, tokenizer):
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: torch.Tensor, batch_size: int) -> torch.Tensor:
    seq_len = data.size(0) // batch_size
    data = data[: seq_len * batch_size]
    data = data.view(batch_size, seq_len).t().contiguous()
    return data


def get_batch(source: torch.Tensor, i: int, bptt: int) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data, target


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


def evaluate(model: nn.Module, data_source: torch.Tensor, criterion: nn.Module, bptt: int) -> float:
    model.eval()
    total_loss = 0.0
    ntokens = model.decoder.out_features
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            src_mask = generate_square_subsequent_mask(data.size(0)).to(DEVICE)
            output = model(data, src_mask)
            output = output.view(-1, ntokens)
            total_loss += criterion(output, targets).item() * data.size(0)
    return total_loss / (len(data_source) - 1)


def train_epoch(
    model: nn.Module,
    data_source: torch.Tensor,
    optimizer,
    criterion: nn.Module,
    bptt: int,
    clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    ntokens = model.decoder.out_features
    start = time.perf_counter()

    progress = tqdm(range(0, data_source.size(0) - 1, bptt), leave=False, dynamic_ncols=True)
    for batch, i in enumerate(progress):
        data, targets = get_batch(data_source, i, bptt)
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        src_mask = generate_square_subsequent_mask(data.size(0)).to(DEVICE)

        optimizer.zero_grad()
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item() * data.size(0)

        if batch % 200 == 0 and batch > 0:
            elapsed = time.perf_counter() - start
            avg_loss = total_loss / (bptt * batch)
            print(f"  step {batch:4d} | loss {avg_loss:.3f} | ppl {math.exp(avg_loss):.2f} | {elapsed:.1f}s")

    return total_loss / (len(data_source) - 1)


def build_optimizer(cfg: Config, opt_name: str, params):
    if opt_name == "adamw":
        return StandardAdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt_name == "galore":
        return GaLoreAdamW(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            rank=cfg.galore_rank,
            update_proj_gap=cfg.update_proj_gap,
        )
    if opt_name == "prox":
        return ProximalGaLoreAdamW(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            threshold=cfg.svt_threshold,
            update_proj_gap=cfg.update_proj_gap,
            min_rank=cfg.min_rank,
        )
    raise ValueError(f"Unknown optimizer: {opt_name}")


def write_summary_row(path: str, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_epoch_log(path: str, train_losses: list, val_losses: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "train_ppl", "val_ppl"],
        )
        writer.writeheader()
        for idx, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow(
                {
                    "epoch": idx,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_ppl": math.exp(train_loss),
                    "val_ppl": math.exp(val_loss),
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Transformer LM on WikiText-2")
    parser.add_argument("--opt", type=str, default="adamw", choices=["adamw", "galore", "prox"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--results-dir", type=str, default="./results/transformer")
    args = parser.parse_args()

    cfg = Config(epochs=args.epochs)
    set_seed(cfg.seed)

    print("Loading WikiText-2...")
    vocab, tokenizer = build_vocab()
    ntokens = len(vocab)

    train_iter, valid_iter, test_iter = WikiText2()
    train_data = data_process(train_iter, vocab, tokenizer)
    val_data = data_process(valid_iter, vocab, tokenizer)
    test_data = data_process(test_iter, vocab, tokenizer)

    train_data = batchify(train_data, cfg.batch_size)
    val_data = batchify(val_data, cfg.batch_size)
    test_data = batchify(test_data, cfg.batch_size)

    model = TransformerLM(
        ntokens=ntokens,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    ).to(DEVICE)

    optimizer = build_optimizer(cfg, args.opt, model.parameters())
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    start_time = time.perf_counter()

    train_losses = []
    val_losses = []
    for epoch in range(1, cfg.epochs + 1):
        print(f"Epoch {epoch}/{cfg.epochs}")
        train_loss = train_epoch(model, train_data, optimizer, criterion, cfg.bptt, clip=0.5)
        val_loss = evaluate(model, val_data, criterion, cfg.bptt)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        print(f"  train loss {train_loss:.3f} | val loss {val_loss:.3f} | val ppl {math.exp(val_loss):.2f}")

    test_loss = evaluate(model, test_data, criterion, cfg.bptt)
    total_time = time.perf_counter() - start_time

    mem = compute_memory_footprint(optimizer)
    os.makedirs(args.results_dir, exist_ok=True)

    result = {
        "optimizer": args.opt,
        "epochs": cfg.epochs,
        "seed": cfg.seed,
        "train_ppl": math.exp(train_loss),
        "val_ppl": math.exp(best_val_loss),
        "test_ppl": math.exp(test_loss),
        "time_sec": total_time,
        "mem_mb": mem["total_mb"],
        "device": DEVICE,
        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "num_layers": cfg.num_layers,
        "dim_feedforward": cfg.dim_feedforward,
        "batch_size": cfg.batch_size,
        "bptt": cfg.bptt,
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("WikiText-2 Transformer LM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(args.results_dir, f"wikitext2_{args.opt}_loss.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    epoch_log_path = os.path.join(args.results_dir, f"wikitext2_{args.opt}_epochs.csv")
    write_epoch_log(epoch_log_path, train_losses, val_losses)

    summary_path = os.path.join(args.results_dir, "summary.csv")
    write_summary_row(summary_path, result)

    out_path = os.path.join(args.results_dir, f"wikitext2_{args.opt}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Done. Test ppl {result['test_ppl']:.2f}. Saved {out_path}")


if __name__ == "__main__":
    main()
