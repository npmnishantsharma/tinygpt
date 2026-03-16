"""
sft.py — Supervised Fine-Tuning for basic chat formatting.

What SFT does:
  Takes the pretrained model and continues training it on a small set of
  manually written prompt-response pairs formatted with special tokens.
  This teaches the model to "follow a chat template" — when it sees
  <|user|> it should respond after <|assistant|>.

Data format (data/sft_examples.jsonl):
  One JSON object per line:
  {"prompt": "What is a transformer?", "response": "A transformer is ..."}

Chat template used during both SFT and inference:
  <|user|> {prompt} <|assistant|> {response} <|end|>

Run:
    python sft.py
"""

import os
import json
import math
import torch
import torch.nn as nn
import sentencepiece as spm

from config import model_cfg, sft_cfg, tok_cfg, SFTConfig
from model import GPT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = sft_cfg.dtype == "bfloat16"
AMP_DTYPE = torch.bfloat16 if USE_AMP else torch.float32


# ─── Chat template ───────────────────────────────────────────────────────────

def format_example(prompt: str, response: str) -> str:
    """Wraps a prompt/response pair in the chat template."""
    return (
        f"{sft_cfg.user_token} {prompt.strip()} "
        f"{sft_cfg.assistant_token} {response.strip()} "
        f"{sft_cfg.end_token}"
    )


# ─── SFT Dataset ─────────────────────────────────────────────────────────────

class SFTDataset(torch.utils.data.Dataset):
    """
    Each example is a full chat turn (prompt + response).
    Loss is only computed on the *response* tokens — the model is not
    penalised for "predicting" the prompt, since that's given as context.

    Implementation: we return the full token sequence plus a mask that
    is 0 for prompt tokens and 1 for response tokens.
    """

    def __init__(self, path: str, sp: spm.SentencePieceProcessor, block_size: int):
        self.examples = []
        self.block_size = block_size

        if not os.path.exists(path):
            print(f"WARNING: SFT data file not found at {path}")
            print("Creating a tiny synthetic example so the script can run.")
            print("Replace data/sft_examples.jsonl with real examples for good results.\n")
            self._create_synthetic(path)

        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                prompt   = obj["prompt"]
                response = obj["response"]

                # Encode with special token strings directly in text
                # (SentencePiece was trained with these as user_defined_symbols)
                text = format_example(prompt, response)
                ids  = sp.encode(text)

                # Identify where the response starts
                assistant_tok_id = sp.piece_to_id(sft_cfg.assistant_token)
                try:
                    split_idx = ids.index(assistant_tok_id) + 1
                except ValueError:
                    split_idx = len(ids) // 2  # fallback

                # Truncate / pad to block_size
                ids = ids[:block_size + 1]
                if len(ids) < 2:
                    continue

                tokens = torch.tensor(ids, dtype=torch.long)
                x = tokens[:-1]
                y = tokens[1:]

                # Loss mask: 1 only for response tokens
                mask = torch.zeros_like(y, dtype=torch.float)
                mask[max(0, split_idx - 1):] = 1.0

                self.examples.append((x, y, mask))

    def _create_synthetic(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        synthetic = [
            {"prompt": "Hello, who are you?",
             "response": "I am TinyGPT, a small language model trained from scratch."},
            {"prompt": "What can you do?",
             "response": "I can answer questions and generate text based on my training data."},
            {"prompt": "What is a neural network?",
             "response": "A neural network is a computing system loosely inspired by the brain."},
            {"prompt": "Explain gradient descent.",
             "response": "Gradient descent is an optimisation method that minimises loss by following the steepest downhill direction of the loss surface."},
            {"prompt": "What is tokenisation?",
             "response": "Tokenisation splits raw text into smaller units called tokens before feeding them to a language model."},
        ]
        with open(path, "w") as f:
            for ex in synthetic:
                f.write(json.dumps(ex) + "\n")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def sft_collate(batch):
    """Pad sequences in a batch to the same length."""
    xs, ys, masks = zip(*batch)
    max_len = max(x.size(0) for x in xs)

    def pad(seq, val=0):
        p = torch.full((max_len,), val, dtype=seq.dtype)
        p[:len(seq)] = seq
        return p

    return (
        torch.stack([pad(x) for x in xs]),
        torch.stack([pad(y) for y in ys]),
        torch.stack([pad(m, val=0.0) for m in masks]),
    )


# ─── LR schedule (same cosine as pretrain) ───────────────────────────────────

def get_lr(step: int) -> float:
    cfg = sft_cfg
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / max(1, cfg.warmup_iters)
    progress = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ─── SFT training ────────────────────────────────────────────────────────────

def run_sft():
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(f"{tok_cfg.model_prefix}.model")

    # Load base pretrained model
    print(f"Loading pretrained checkpoint: {sft_cfg.base_checkpoint}")
    ckpt = torch.load(sft_cfg.base_checkpoint, map_location=DEVICE)
    model = GPT(model_cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"Pretrain val_loss was: {ckpt.get('val_loss', '?'):.4f}")

    # Dataset
    dataset = SFTDataset(sft_cfg.sft_data_file, sp, model_cfg.block_size)
    print(f"SFT examples: {len(dataset)}")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=sft_cfg.batch_size,
        shuffle=True, collate_fn=sft_collate, drop_last=False,
    )

    # Optimiser (lower LR than pretraining — we don't want to forget too much)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=sft_cfg.learning_rate,
        weight_decay=sft_cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    model.train()
    optimizer.zero_grad()
    step = 0

    print(f"\nStarting SFT for {sft_cfg.max_iters} steps...\n")

    while step < sft_cfg.max_iters:
        for x, y, mask in loader:
            if step >= sft_cfg.max_iters:
                break
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            mask = mask.to(DEVICE)

            # Update LR
            lr = get_lr(step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            for micro in range(sft_cfg.grad_accum_steps):
                with torch.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
                    logits, _ = model(x)   # (B, T, vocab)
                    # Masked cross-entropy: ignore prompt tokens
                    B, T, V = logits.shape
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, V),
                        y.view(-1),
                        reduction="none",
                    )
                    # Apply response mask
                    loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)
                    loss = loss / sft_cfg.grad_accum_steps

                loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), sft_cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            if step % 50 == 0:
                print(f"SFT step {step:4d} | loss {loss.item() * sft_cfg.grad_accum_steps:.4f} | lr {lr:.2e}")
            step += 1

    # Save
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "model_cfg": model_cfg.__dict__,
    }, sft_cfg.output_checkpoint)
    print(f"\nSFT complete. Saved to {sft_cfg.output_checkpoint}")


if __name__ == "__main__":
    run_sft()
