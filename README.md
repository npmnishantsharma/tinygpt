# TinyGPT — A ~10M Parameter Language Model From Scratch

A complete, educational GPT-style transformer trained on plain `.md` files.
Runs entirely on CPU (no CUDA required).

---

## Hardware Note
Intel HD Graphics 3000 has **no CUDA support**. All training runs on CPU using
PyTorch's native `bfloat16` autocast where available. A 10M-param model on a
small dataset (< 5MB text) trains comfortably in a few hours overnight.

---

## Folder Structure

```
tinygpt/
├── data/
│   ├── raw/          ← put your .md files here
│   └── processed/    ← auto-generated tokenized .bin files
├── tokenizer/        ← trained SentencePiece model lives here
├── checkpoints/      ← model checkpoints saved here
├── logs/             ← training loss curves
├── config.py         ← all hyperparameters in one place
├── tokenizer_train.py← train BPE tokenizer with SentencePiece
├── dataset.py        ← data prep, train/val split, DataLoader
├── model.py          ← causal Transformer (GPT architecture)
├── train.py          ← pretraining loop
├── sft.py            ← supervised fine-tuning for chat format
├── chat.py           ← minimal CLI inference script
└── requirements.txt
```

---

## Weekend Execution Steps

### Day 1 — Setup & Pretrain

```bash
# 1. Install dependencies (5 min)
pip install -r requirements.txt

# 2. Put your .md files in data/raw/
#    Any plain English markdown works: docs, notes, wiki exports, books.
#    Aim for at least 1–5 MB of text for meaningful training.

# 3. Train the tokenizer (1 min)
python tokenizer_train.py

# 4. Preprocess data into token bins (1 min)
python dataset.py

# 5. Start pretraining (overnight on CPU, ~2-6 hrs for small data)
python train.py
```

### Day 2 — Fine-tune & Chat

```bash
# 6. Run supervised fine-tuning on chat examples (30 min)
python sft.py

# 7. Chat with your model
python chat.py --checkpoint checkpoints/sft_final.pt
```

---

## Quick Tips

- **More data = better model.** Even 2–3 MB of coherent English text is enough
  to see meaningful learning. Wikipedia article exports work great.
- **Overfitting?** Reduce `n_layer` or increase `dropout` in `config.py`.
- **Too slow?** Reduce `block_size` to 128 or `batch_size` to 2.
- **JSON output:** Pass `--json` flag to `chat.py` for structured responses.
- Checkpoints auto-save every `save_interval` steps. Training is resumable.
