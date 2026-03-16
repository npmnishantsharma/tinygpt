"""
model.py — Causal Transformer (GPT-style), ~10M parameters.

Architecture (identical to GPT-2 small, just smaller):
  Token embedding  →  positional embedding
  → N × TransformerBlock(CausalSelfAttention, LayerNorm, MLP)
  → final LayerNorm
  → linear head (vocab logits, weight-tied to embedding)

Key design notes:
  - Pre-norm (LayerNorm before sub-layers, not after) — more stable training.
  - Causal mask built once and cached as a buffer.
  - Weight tying: the output projection shares weights with token embeddings,
    saving ~1.5M params and improving generalisation on small data.
  - Dropout in attention and MLP for overfitting control.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, model_cfg


# ─── Causal Self-Attention ────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    All three projections (Q, K, V) are batched into one matrix multiply for
    efficiency, then split.  The causal mask ensures position i can only
    attend to positions ≤ i (autoregressive property).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head  = cfg.n_head
        self.n_embd  = cfg.n_embd
        self.dropout = cfg.dropout

        # Batched Q, K, V projection
        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # Output projection
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Causal mask: upper-triangular matrix (1 = masked / -inf)
        # Registered as a buffer so it moves with .to(device) automatically
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape   # batch, sequence length, channels (n_embd)

        # Compute Q, K, V in one shot, then split
        qkv = self.qkv_proj(x)                        # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)       # each (B, T, C)

        # Reshape to (B, n_head, T, head_size)
        def reshape(t):
            return t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(k.size(-1))
        att = (q @ k.transpose(-2, -1)) * scale        # (B, n_head, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Weighted sum of values
        y = att @ v                                     # (B, n_head, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble heads

        return self.resid_drop(self.out_proj(y))


# ─── MLP (Feed-Forward Network) ──────────────────────────────────────────────

class MLP(nn.Module):
    """
    Two-layer feed-forward network with GELU activation.
    Hidden size is 4× the embedding size (standard GPT ratio).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.fc1   = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.act   = nn.GELU()
        self.fc2   = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop  = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


# ─── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block: LayerNorm → Attention → residual +
                                 LayerNorm → MLP       → residual
    Pre-norm (normalise *before* sub-layer) is more training-stable than
    the original post-norm formulation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp  = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # attention residual
        x = x + self.mlp(self.ln2(x))    # MLP residual
        return x


# ─── GPT (full model) ────────────────────────────────────────────────────────

class GPT(nn.Module):
    """
    Causal language model.

    Parameter count (defaults from config.py):
      vocab_size=4096, block_size=256, n_layer=6, n_head=6, n_embd=384
      → ~10.6M parameters
    """

    def __init__(self, cfg: ModelConfig = model_cfg):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb   = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop      = nn.Dropout(cfg.dropout)

        self.blocks    = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_final  = nn.LayerNorm(cfg.n_embd)

        # Output head: projects hidden states → vocabulary logits
        self.lm_head   = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: share parameters between embedding and output projection.
        # Rationale: both matrices map between token-space and embedding-space,
        # so sharing reduces parameters and improves generalisation.
        self.lm_head.weight = self.token_emb.weight

        # Initialise weights (GPT-2 style)
        self.apply(self._init_weights)
        # Scale residual projections down by sqrt(2 * n_layer) — stabilises
        # deep network training by preventing residual stream blow-up
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "fc2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,            # (B, T) token IDs
        targets: torch.Tensor = None, # (B, T) shifted targets for loss
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, \
            f"Sequence length {T} exceeds block_size {self.cfg.block_size}"

        # Positional indices 0..T-1
        pos = torch.arange(T, device=idx.device)

        # Combine token + position embeddings
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy over all positions
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,          # (1, T) context tokens
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature, top-k, and nucleus (top-p)
        sampling.  Returns the full sequence (prompt + generated tokens).
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if context is too long
            idx_cond = idx[:, -self.cfg.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # last token logits

            # Top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                kth = torch.topk(logits, top_k_val).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            # Nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens once cumulative prob exceeds top_p
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)

        return idx

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Quick check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    m = GPT()
    print(f"Model parameters: {m.num_parameters():,}")
    x = torch.randint(0, model_cfg.vocab_size, (2, model_cfg.block_size))
    logits, loss = m(x, x)
    print(f"Forward pass OK — logits: {logits.shape}, loss: {loss.item():.4f}")
