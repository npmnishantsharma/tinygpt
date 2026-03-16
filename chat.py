"""
chat.py — Minimal CLI for local inference with TinyGPT.

Usage:
    # Normal chat
    python chat.py

    # Specify checkpoint
    python chat.py --checkpoint checkpoints/sft_final.pt

    # JSON output mode (model is prompted to respond in JSON)
    python chat.py --json

    # One-shot generation (non-interactive)
    python chat.py --prompt "What is attention?" --max_tokens 150

Options:
    --checkpoint   Path to .pt checkpoint file
    --max_tokens   Max new tokens to generate (default: 200)
    --temperature  Sampling temperature 0.1–2.0 (default: 0.8)
    --top_k        Top-k sampling cutoff (default: 40)
    --top_p        Nucleus sampling threshold (default: 0.9)
    --json         Ask the model to respond in JSON format
    --prompt       Single prompt (non-interactive mode)
"""

import argparse
import json
import sys
import torch
import sentencepiece as spm

from config import tok_cfg, sft_cfg, inf_cfg, model_cfg
from model import GPT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Load model ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str | None = None) -> GPT:
    print(f"Loading checkpoint: {checkpoint_path}")
    target_device = device or DEVICE
    ckpt = torch.load(checkpoint_path, map_location=target_device, weights_only=False)

    # Reconstruct config from checkpoint if available (handles modified configs)
    from config import ModelConfig
    cfg_dict = ckpt.get("model_cfg", model_cfg.__dict__)
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items()
                         if k in ModelConfig.__dataclass_fields__})

    model = GPT(cfg).to(target_device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    params = model.num_parameters()
    print(f"Model loaded — {params:,} parameters\n")
    return model


# ─── Tokenizer helpers ───────────────────────────────────────────────────────

def load_tokenizer() -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(f"{tok_cfg.model_prefix}.model")
    return sp


def encode(sp: spm.SentencePieceProcessor, text: str, device: str | None = None) -> torch.Tensor:
    ids = sp.encode(text)
    dev = device or DEVICE
    return torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)   # (1, T)


def decode(sp: spm.SentencePieceProcessor, ids: torch.Tensor) -> str:
    return sp.decode(ids.squeeze(0).tolist())


# ─── Notebook-friendly helpers ────────────────────────────────────────────────

def init_chat(
    checkpoint: str | None = None,
    device: str | None = None,
) -> tuple[GPT, spm.SentencePieceProcessor]:
    """
    Convenience helper for Kaggle/Colab notebooks.

    Usage (in a notebook cell):

        from chat import init_chat, chat_once
        model, sp = init_chat()  # or init_chat(\"checkpoints/my_ckpt.pt\")
        print(chat_once(model, sp, \"Hello!\"))}

    """
    ckpt_path = checkpoint or inf_cfg.checkpoint
    model = load_model(ckpt_path, device=device)
    sp = load_tokenizer()
    return model, sp


@torch.no_grad()
def chat_once(
    model: GPT,
    sp: spm.SentencePieceProcessor,
    prompt_text: str,
    json_mode: bool = False,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str | None = None,
) -> str:
    """
    Single-turn chat helper for notebooks.

    Example:

        model, sp = init_chat()
        reply = chat_once(model, sp, \"Explain attention.\")
        print(reply)
    """
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens or inf_cfg.max_new_tokens,
        temperature=temperature or inf_cfg.temperature,
        top_k=top_k or inf_cfg.top_k,
        top_p=top_p or inf_cfg.top_p,
    )
    prompt = build_prompt(prompt_text, json_mode=json_mode)
    response = generate_response(model, sp, prompt, device=device, **gen_kwargs)
    if json_mode:
        response = try_parse_json(response)
    return response


# ─── Build prompt ────────────────────────────────────────────────────────────

def build_prompt(user_text: str, json_mode: bool = False) -> str:
    """
    Format the user's input using the same chat template used during SFT.
    In JSON mode we prepend an instruction to make the model attempt JSON output.
    """
    if json_mode:
        user_text = (
            f"Respond only with valid JSON. {user_text}"
        )
    # Template: <|user|> ... <|assistant|>   (no response — model fills this in)
    return f"{sft_cfg.user_token} {user_text.strip()} {sft_cfg.assistant_token}"


# ─── Stream generation ───────────────────────────────────────────────────────

@torch.no_grad()
def generate_response(
    model: GPT,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: str | None = None,
) -> str:
    idx = encode(sp, prompt, device=device)
    prompt_len = idx.size(1)

    output_ids = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Decode only the newly generated tokens
    new_ids = output_ids[0, prompt_len:]
    text = decode(sp, new_ids)

    # Trim at end-of-turn token if present
    for stop in [sft_cfg.end_token, sft_cfg.user_token]:
        if stop in text:
            text = text.split(stop)[0]

    return text.strip()


# ─── JSON post-processing ────────────────────────────────────────────────────

def try_parse_json(text: str) -> str:
    """Attempt to parse and pretty-print JSON from the model output."""
    # Find first { or [
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj = json.loads(text[i:])
                return json.dumps(obj, indent=2)
            except json.JSONDecodeError:
                pass
    return text  # return raw text if no valid JSON found


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TinyGPT Chat")
    parser.add_argument("--checkpoint",   default=inf_cfg.checkpoint)
    parser.add_argument("--max_tokens",   type=int,   default=inf_cfg.max_new_tokens)
    parser.add_argument("--temperature",  type=float, default=inf_cfg.temperature)
    parser.add_argument("--top_k",        type=int,   default=inf_cfg.top_k)
    parser.add_argument("--top_p",        type=float, default=inf_cfg.top_p)
    parser.add_argument("--json",         action="store_true", help="JSON output mode")
    parser.add_argument("--prompt",       default=None, help="Single prompt (non-interactive)")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    sp    = load_tokenizer()

    gen_kwargs = dict(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # ── Single prompt mode ────────────────────────────────────────────────────
    if args.prompt:
        prompt = build_prompt(args.prompt, json_mode=args.json)
        response = generate_response(model, sp, prompt, **gen_kwargs)
        if args.json:
            response = try_parse_json(response)
        print(response)
        return

    # ── Interactive chat loop ─────────────────────────────────────────────────
    print("TinyGPT — type your message, 'quit' to exit, 'reset' to clear history.")
    print("─" * 60)

    conversation_context = ""   # simple stateless context (no memory beyond block_size)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            conversation_context = ""
            print("[Context cleared]")
            continue

        prompt = build_prompt(user_input, json_mode=args.json)

        # Prepend short recent context (last ~100 tokens worth of characters)
        # This gives minimal multi-turn coherence without complex state management
        context_limit = model_cfg.block_size * 3  # rough char limit
        if conversation_context:
            full_prompt = conversation_context[-context_limit:] + "\n" + prompt
        else:
            full_prompt = prompt

        response = generate_response(model, sp, full_prompt, **gen_kwargs)

        if args.json:
            response = try_parse_json(response)

        print(f"\nTinyGPT: {response}")

        # Append to context for next turn
        conversation_context += f"\n{prompt} {response} {sft_cfg.end_token}"


if __name__ == "__main__":
    main()
