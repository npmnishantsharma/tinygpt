"""
tokenizer_train.py — Train a BPE tokenizer with SentencePiece.

SentencePiece learns a byte-pair encoding directly from raw text.
It handles unknown characters gracefully and doesn't need pre-tokenisation.

Run:
    python tokenizer_train.py
"""

import os
import glob
import sentencepiece as spm
from config import tok_cfg

# ─── 1. Collect all .md files into a single flat text corpus ────────────────

RAW_DIR = "data/raw"
CORPUS_FILE = "data/corpus_for_tokenizer.txt"

os.makedirs("data", exist_ok=True)
os.makedirs("tokenizer", exist_ok=True)

md_files = glob.glob(os.path.join(RAW_DIR, "**/*.md"), recursive=True)
md_files += glob.glob(os.path.join(RAW_DIR, "*.md"))

if not md_files:
    raise FileNotFoundError(
        f"No .md files found in {RAW_DIR}/. "
        "Add at least one markdown file before running this script."
    )

print(f"Found {len(md_files)} .md file(s). Merging into corpus...")

total_chars = 0
with open(CORPUS_FILE, "w", encoding="utf-8") as out:
    for path in md_files:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
            out.write(text + "\n")
            total_chars += len(text)

print(f"Corpus size: {total_chars:,} characters ({total_chars / 1e6:.2f} MB)")

# ─── 2. Train SentencePiece BPE model ───────────────────────────────────────

# Special tokens we'll use for the chat template (SFT phase)
user_defined_symbols = ["<|user|>", "<|assistant|>", "<|end|>", "<|json|>"]

print(f"Training SentencePiece BPE with vocab_size={tok_cfg.vocab_size}...")

spm.SentencePieceTrainer.train(
    input=CORPUS_FILE,
    model_prefix=tok_cfg.model_prefix,
    vocab_size=tok_cfg.vocab_size,
    model_type=tok_cfg.model_type,         # BPE
    pad_id=tok_cfg.pad_id,
    unk_id=tok_cfg.unk_id,
    bos_id=tok_cfg.bos_id,
    eos_id=tok_cfg.eos_id,
    user_defined_symbols=",".join(user_defined_symbols),
    character_coverage=0.9995,             # cover almost all unicode chars
    num_threads=4,
    input_sentence_size=2_000_000,         # max sentences sampled for training
    shuffle_input_sentence=True,
    normalization_rule_name="nmt_nfkc_cf", # lower-case + unicode normalisation
    # Small corpora can't always support large vocab sizes. Let SentencePiece
    # automatically pick a smaller effective vocab instead of erroring.
    hard_vocab_limit=False,
)

print(f"Tokenizer saved to {tok_cfg.model_prefix}.model and .vocab")

# ─── 3. Quick sanity check ──────────────────────────────────────────────────

sp = spm.SentencePieceProcessor()
sp.load(f"{tok_cfg.model_prefix}.model")

sample = "Hello! This is a test of the TinyGPT tokenizer."
tokens = sp.encode(sample)
decoded = sp.decode(tokens)

print(f"\nSample encode/decode test:")
print(f"  Input  : {sample!r}")
print(f"  Tokens : {tokens}")
print(f"  Decoded: {decoded!r}")
print(f"  Match  : {sample.lower() == decoded.lower()}")   # nmt_nfkc_cf lowercases
print("\nTokenizer training complete.")
