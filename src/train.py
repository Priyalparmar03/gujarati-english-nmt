"""
Trains the from-scratch transformer.

Usage:
    python -m src.train --data_dir data/ --spm_model data/spm_shared.model \
        --out_dir checkpoints/ --epochs 20 --max_tokens 25000 --direction en2gu

    # resume an interrupted run
    python -m src.train --data_dir data/ --spm_model data/spm_shared.model \
        --out_dir checkpoints/ --epochs 20 --resume checkpoints/last.pt
"""
import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.transformer import Seq2SeqTransformer
from src.model.tokenizer import SharedTokenizer, PAD_ID
from src.data.dataset import ParallelTextDataset, TokenBudgetBatchSampler, collate_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NoamScheduler:
    """LR schedule from 'Attention Is All You Need', section 5.3."""

    def __init__(self, optimizer, d_model, warmup_steps=4000, step_num=0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = step_num

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.optimizer.step()

    def state_dict(self):
        return {"step_num": self.step_num}

    def load_state_dict(self, sd):
        self.step_num = sd["step_num"]


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_tokens = 0.0, 0
    use_amp = scaler is not None and device.type == "cuda"

    for src, tgt in loader:
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        if is_train:
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # NoamScheduler.step() calls optimizer.step() internally, so
                # step the scaler around it explicitly instead
                scaler.step(optimizer)
                scaler.update()
                scheduler.step_num += 1
                lr = (scheduler.d_model ** -0.5) * min(
                    scheduler.step_num ** -0.5, scheduler.step_num * scheduler.warmup_steps ** -1.5
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scheduler.step()

        n_tok = (tgt_out != PAD_ID).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok

    return total_loss / max(1, total_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--spm_model", default="data/spm_shared.model")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=25000,
                     help="token budget per batch (src+tgt), not sentence count")
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=4000)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--direction", choices=["en2gu", "gu2en"], default="en2gu",
                     help="en2gu: English source -> Gujarati target (default, matches the app goal). "
                          "gu2en: reverse. Pick the one direction you're reporting in the paper; "
                          "match Panchal et al.'s direction if comparing against them directly.")
    ap.add_argument("--resume", default=None, help="path to a checkpoint (e.g. checkpoints/last.pt) to resume from")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = SharedTokenizer(args.spm_model)

    en_train, gu_train = f"{args.data_dir}/train.en", f"{args.data_dir}/train.gu"
    en_dev, gu_dev = f"{args.data_dir}/dev.en", f"{args.data_dir}/dev.gu"
    if args.direction == "en2gu":
        train_src, train_tgt, dev_src, dev_tgt = en_train, gu_train, en_dev, gu_dev
    else:
        train_src, train_tgt, dev_src, dev_tgt = gu_train, en_train, gu_dev, en_dev

    train_ds = ParallelTextDataset(train_src, train_tgt, tok, args.max_len)
    dev_ds = ParallelTextDataset(dev_src, dev_tgt, tok, args.max_len)

    train_sampler = TokenBudgetBatchSampler(train_ds, max_tokens=args.max_tokens, shuffle=True)
    dev_sampler = TokenBudgetBatchSampler(dev_ds, max_tokens=args.max_tokens, shuffle=False)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_sampler=dev_sampler, collate_fn=collate_fn)
    print(f"direction: {args.direction} | train batches/epoch: {len(train_sampler)} | dev batches: {len(dev_sampler)}")

    model = Seq2SeqTransformer(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len + 2,
        pad_id=PAD_ID,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.1f}M | vocab: {tok.vocab_size} | device: {device}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, args.d_model, args.warmup_steps)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(args.amp and device.type == "cuda"))

    start_epoch = 1
    best_dev_loss = float("inf")
    log_path = f"{args.out_dir}/training_log.csv"

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_dev_loss = ckpt.get("best_dev_loss", float("inf"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_ppl", "dev_loss", "dev_ppl", "lr", "seconds"])

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, scaler)
        dev_loss = run_epoch(model, dev_loader, criterion, device)
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:02d} | train_loss {train_loss:.4f} (ppl {math.exp(train_loss):.2f}) "
            f"| dev_loss {dev_loss:.4f} (ppl {math.exp(dev_loss):.2f}) | lr {cur_lr:.2e} | {elapsed:.0f}s"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.4f}", f"{math.exp(train_loss):.2f}",
                f"{dev_loss:.4f}", f"{math.exp(dev_loss):.2f}", f"{cur_lr:.6e}", f"{elapsed:.0f}",
            ])

        is_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)

        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_dev_loss": best_dev_loss,
            "args": vars(args),
        }
        # keep only best.pt and last.pt on disk — not one file per epoch
        torch.save(ckpt, f"{args.out_dir}/last.pt")
        if is_best:
            torch.save(ckpt, f"{args.out_dir}/best.pt")


if __name__ == "__main__":
    main()
