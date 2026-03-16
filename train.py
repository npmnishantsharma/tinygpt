"""
train.py — Next-token pretraining loop.

Features:
  - Cosine LR decay with linear warm-up
  - Gradient accumulation  (simulates larger batch on limited RAM)
  - bfloat16 autocast      (faster on CPUs with AVX512 BF16 support)
  - Gradient norm clipping (stabilises training)
  - Auto-resume from latest checkpoint
  - Val loss tracking + saves best model

Run:
    python train.py
"""

import os
import time
import math
import json
import glob

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler   # unused on CPU but imported for completeness

from config import model_cfg, train_cfg, tok_cfg, ModelConfig
from dataset import get_dataloaders
from model import GPT


# ─── Device setup ────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# bfloat16 autocast on CPU (supported on modern CPUs; gracefully skips if not)
USE_AMP = train_cfg.dtype == "bfloat16"
AMP_DTYPE = torch.bfloat16 if USE_AMP else torch.float32

os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
os.makedirs(train_cfg.log_dir, exist_ok=True)


# ─── Learning rate schedule ──────────────────────────────────────────────────

def get_lr(step: int) -> float:
    """
    Linear warm-up then cosine decay to min_lr.
    This is the standard schedule used by GPT-style models.
    """
    if step < train_cfg.warmup_iters:
        return train_cfg.learning_rate * step / max(1, train_cfg.warmup_iters)
    if step >= train_cfg.max_iters:
        return train_cfg.min_lr
    # Cosine decay
    progress = (step - train_cfg.warmup_iters) / (train_cfg.max_iters - train_cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return train_cfg.min_lr + coeff * (train_cfg.learning_rate - train_cfg.min_lr)


# ─── Checkpoint helpers ──────────────────────────────────────────────────────

def save_checkpoint(model: GPT, optimizer: torch.optim.Optimizer,
                    step: int, val_loss: float, tag: str = "") -> str:
    suffix = tag or f"step{step:05d}"
    path = os.path.join(train_cfg.checkpoint_dir, f"pretrain_{suffix}.pt")
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
        "model_cfg": model_cfg.__dict__,
    }, path)
    return path


def load_latest_checkpoint(model: GPT, optimizer: torch.optim.Optimizer):
    """Returns (start_step, best_val_loss) or (0, inf) if no checkpoint found."""
    checkpoints = glob.glob(os.path.join(train_cfg.checkpoint_dir, "pretrain_step*.pt"))
    if not checkpoints:
        return 0, float("inf")
    latest = max(checkpoints, key=lambda p: int(p.split("step")[-1].split(".")[0]))
    print(f"Resuming from {latest}")
    ckpt = torch.load(latest, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["step"], ckpt["val_loss"]


# ─── Validation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_val_loss(model: GPT, val_loader) -> float:
    model.eval()
    losses = []
    loader_iter = iter(val_loader)
    for _ in range(train_cfg.eval_iters):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            break
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


# ─── Main training loop ──────────────────────────────────────────────────────

def train():
    # Model
    model = GPT(model_cfg).to(DEVICE)
    print(f"Parameters: {model.num_parameters():,}")

    # Optimiser: AdamW with decoupled weight decay
    # Only apply weight decay to weight matrices, not biases or LayerNorm params
    decay_params     = [p for n, p in model.named_parameters()
                        if p.dim() >= 2 and p.requires_grad]
    no_decay_params  = [p for n, p in model.named_parameters()
                        if p.dim() < 2  and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params,    "weight_decay": train_cfg.weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        fused=False,   # fused kernel not available on CPU
    )

    # Auto-resume
    start_step = 0
    best_val_loss = float("inf")
    if train_cfg.resume:
        start_step, best_val_loss = load_latest_checkpoint(model, optimizer)

    train_loader, val_loader = get_dataloaders()
    train_iter = iter(train_loader)

    log_path = os.path.join(train_cfg.log_dir, "train_log.jsonl")
    log_file = open(log_path, "a")

    print(f"\nStarting training from step {start_step} / {train_cfg.max_iters}")
    print(f"Effective batch size: {train_cfg.batch_size * train_cfg.grad_accum_steps} sequences\n")

    model.train()
    optimizer.zero_grad()
    t0 = time.time()

    for step in range(start_step, train_cfg.max_iters):
        # ── LR update ────────────────────────────────────────────────────────
        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ── Gradient accumulation ────────────────────────────────────────────
        # Accumulate gradients over `grad_accum_steps` micro-batches before
        # a single optimizer.step().  This simulates a larger effective batch
        # without needing more RAM — critical on a memory-constrained CPU.
        accum_loss = 0.0
        for micro_step in range(train_cfg.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
                _, loss = model(x, y)

            # Scale loss by accumulation steps so gradients average correctly
            loss = loss / train_cfg.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        # ── Gradient clipping + optimizer step ───────────────────────────────
        # Clipping the gradient norm prevents occasional exploding gradients.
        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        # ── Logging ──────────────────────────────────────────────────────────
        if step % train_cfg.log_interval == 0:
            elapsed = time.time() - t0
            print(f"step {step:5d} | loss {accum_loss:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
            log_file.write(json.dumps({"step": step, "loss": accum_loss, "lr": lr}) + "\n")
            log_file.flush()
            t0 = time.time()

        # ── Validation + best model save ─────────────────────────────────────
        if step % train_cfg.eval_interval == 0 and step > 0:
            val_loss = estimate_val_loss(model, val_loader)
            print(f"  ↳ val loss: {val_loss:.4f} {'✓ new best' if val_loss < best_val_loss else ''}")
            log_file.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, val_loss, tag="best")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if step % train_cfg.save_interval == 0 and step > 0:
            path = save_checkpoint(model, optimizer, step, best_val_loss)
            print(f"  ↳ checkpoint saved: {path}")

    # Final save
    path = save_checkpoint(model, optimizer, train_cfg.max_iters, best_val_loss, tag="final")
    print(f"\nTraining complete. Final checkpoint: {path}")
    log_file.close()


if __name__ == "__main__":
    train()
