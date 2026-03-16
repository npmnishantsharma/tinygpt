"""
dataset.py — Tokenize raw text → packed binary token arrays.

Design choices:
  - All tokens packed into one flat array (no padding waste).
  - 5% held out as validation, sliced from the end (not shuffled —
    language models should see contiguous text during training).
  - A LanguageModelDataset samples random fixed-length windows at runtime,
    which acts as data augmentation for small corpora.

Run (prepare data):
    python dataset.py

Import (used by train.py):
    from dataset import get_dataloaders
"""

import os
import glob
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

from config import tok_cfg, train_cfg


# ─── Tokenize & save ─────────────────────────────────────────────────────────

def build_token_array(output_dir: str = "data/processed") -> None:
    """
    Read all .md files, tokenize with SentencePiece, and save two .bin files:
      train.bin  — ~95% of tokens
      val.bin    — ~5%  of tokens
    Each file is a flat array of uint16 token IDs (max vocab 65535).
    """
    os.makedirs(output_dir, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(f"{tok_cfg.model_prefix}.model")

    raw_dir = "data/raw"
    md_files = sorted(
        glob.glob(os.path.join(raw_dir, "**/*.md"), recursive=True)
        + glob.glob(os.path.join(raw_dir, "*.md"))
    )
    if not md_files:
        raise FileNotFoundError(f"No .md files in {raw_dir}/")

    all_ids: list[int] = []
    for path in md_files:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        # Encode; add BOS at start and EOS at end of each document
        ids = [tok_cfg.bos_id] + sp.encode(text) + [tok_cfg.eos_id]
        all_ids.extend(ids)

    arr = np.array(all_ids, dtype=np.uint16)
    total = len(arr)
    val_n = max(train_cfg.block_size * 2, int(total * train_cfg.val_fraction))
    train_n = total - val_n

    train_arr = arr[:train_n]
    val_arr = arr[train_n:]

    train_arr.tofile(os.path.join(output_dir, "train.bin"))
    val_arr.tofile(os.path.join(output_dir, "val.bin"))

    print(f"Total tokens : {total:,}")
    print(f"Train tokens : {train_n:,}")
    print(f"Val   tokens : {val_n:,}")
    print(f"Saved to {output_dir}/")


# ─── Dataset class ───────────────────────────────────────────────────────────

class LanguageModelDataset(Dataset):
    """
    Samples random contiguous windows of length `block_size` from a flat
    token array.  `__len__` is the number of non-overlapping windows
    (conservative estimate), but `__getitem__` uses random offsets so each
    epoch sees different windows — effective augmentation on small data.
    """

    def __init__(self, bin_path: str, block_size: int):
        data = np.fromfile(bin_path, dtype=np.uint16)
        # Store as int32 so torch can handle it cleanly
        self.data = torch.from_numpy(data.astype(np.int32))
        self.block_size = block_size
        self.n = len(self.data)

    def __len__(self) -> int:
        # Each index maps to a non-overlapping window; random offset in __getitem__
        return (self.n - self.block_size - 1) // self.block_size

    def __getitem__(self, idx: int):
        # Random start within the valid range — better than a fixed stride
        max_start = self.n - self.block_size - 1
        start = torch.randint(0, max_start, (1,)).item()
        x = self.data[start : start + self.block_size].long()
        y = self.data[start + 1 : start + self.block_size + 1].long()
        return x, y


# ─── DataLoader factory ──────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str = train_cfg.data_dir,
    block_size: int = train_cfg.block_size,
    batch_size: int = train_cfg.batch_size,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader)."""

    train_ds = LanguageModelDataset(os.path.join(data_dir, "train.bin"), block_size)
    val_ds   = LanguageModelDataset(os.path.join(data_dir, "val.bin"),   block_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    return train_loader, val_loader


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_token_array()
    # Quick smoke-test
    train_dl, val_dl = get_dataloaders()
    x, y = next(iter(train_dl))
    print(f"Batch shapes — x: {x.shape}, y: {y.shape}  dtype: {x.dtype}")
    print("dataset.py OK")
