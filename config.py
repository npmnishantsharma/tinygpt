"""
config.py — Single source of truth for all hyperparameters.

Every number that matters lives here. Change values here only;
the rest of the codebase reads from this file.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@dataclass
class TokenizerConfig:
    vocab_size: int = 4096          # BPE vocabulary size; 4k is enough for small corpora
    model_prefix: str = "tokenizer/spm"  # SentencePiece output prefix
    model_type: str = "bpe"         # "bpe" or "unigram"
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # ~10M parameters with these defaults
    vocab_size: int = 4096          # must match TokenizerConfig.vocab_size
    block_size: int = 256           # context window length (tokens)
    n_layer: int = 6                # number of Transformer blocks
    n_head: int = 6                 # attention heads (n_embd must be divisible)
    n_embd: int = 384               # embedding / hidden dimension
    dropout: float = 0.1            # applied in attention + MLP (overfitting guard)
    bias: bool = False              # no bias in Linear layers (cleaner, faster)

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head


# ---------------------------------------------------------------------------
# Training (Pretraining)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # --- Data ---
    data_dir: str = "data/processed"
    val_fraction: float = 0.05      # 5% of tokens held out for validation

    # --- Batching ---
    batch_size: int = 4             # sequences per batch (keep low for CPU)
    block_size: int = 256           # must match ModelConfig.block_size
    grad_accum_steps: int = 8       # effective batch = batch_size * grad_accum_steps = 32

    # --- Optimiser ---
    learning_rate: float = 3e-4
    min_lr: float = 3e-5            # cosine decay floor
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0          # gradient norm clipping

    # --- Schedule ---
    max_iters: int = 5000           # total optimiser steps
    warmup_iters: int = 200         # linear LR warm-up

    # --- Eval & Checkpointing ---
    eval_interval: int = 200        # evaluate val loss every N steps
    eval_iters: int = 50            # batches averaged for val loss estimate
    save_interval: int = 500        # save checkpoint every N steps
    checkpoint_dir: str = "checkpoints"
    resume: bool = True             # auto-resume from latest checkpoint if found

    # --- Precision ---
    # bfloat16 works on modern CPUs (AVX512 BF16). Falls back to float32 if not.
    dtype: str = "bfloat16"         # "float32" | "bfloat16"

    # --- Logging ---
    log_dir: str = "logs"
    log_interval: int = 50


# ---------------------------------------------------------------------------
# Supervised Fine-Tuning (SFT)
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    sft_data_file: str = "data/sft_examples.jsonl"   # one {"prompt":..,"response":..} per line
    base_checkpoint: str = "checkpoints/pretrain_best.pt"
    output_checkpoint: str = "checkpoints/sft_final.pt"

    batch_size: int = 2
    grad_accum_steps: int = 4
    max_iters: int = 500
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    warmup_iters: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dtype: str = "bfloat16"

    # Chat template tokens (baked into the tokenizer as special pieces or via raw strings)
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    end_token: str = "<|end|>"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    checkpoint: str = "checkpoints/sft_final.pt"
    max_new_tokens: int = 200
    temperature: float = 0.8        # > 1 = more random, < 1 = sharper
    top_k: int = 40                 # keep only top-k logits before sampling
    top_p: float = 0.9              # nucleus sampling threshold


# ---------------------------------------------------------------------------
# Instantiated defaults (imported by other modules)
# ---------------------------------------------------------------------------

tok_cfg   = TokenizerConfig()
model_cfg = ModelConfig()
train_cfg = TrainConfig()
sft_cfg   = SFTConfig()
inf_cfg   = InferenceConfig()
