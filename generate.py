import argparse
from pathlib import Path

import torch

from config import config
from dataset import CharTokenizer
from model import GPT


def build_device(device_flag: str) -> torch.device:
    device_flag = device_flag.lower()
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine.")
        return torch.device("cuda")
    raise ValueError("Unsupported device. Use 'cpu' or 'cuda'.")


def validate_prompt(prompt: str, tokenizer: CharTokenizer) -> None:
    invalid_chars = sorted({ch for ch in prompt if ch not in tokenizer.stoi})
    if invalid_chars:
        invalid_display = ", ".join(repr(ch) for ch in invalid_chars)
        raise ValueError(
            f"Prompt contains characters not in the saved vocabulary: {invalid_display}"
        )


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint["tokenizer"]
    tokenizer = CharTokenizer("".join(vocab))
    ckpt_config = checkpoint.get("config", {})

    model = GPT(
        vocab_size=len(tokenizer.vocab),
        block_size=ckpt_config.get("block_size", config.block_size),
        n_layer=ckpt_config.get("n_layer", config.n_layer),
        n_head=ckpt_config.get("n_head", config.n_head),
        n_embd=ckpt_config.get("n_embd", config.n_embd),
        dropout=ckpt_config.get("dropout", config.dropout),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return tokenizer, model, ckpt_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Shakespeare-like text from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=config.sample_temperature)
    parser.add_argument("--top_k", type=int, default=config.sample_top_k)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = build_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer, model, ckpt_config = load_checkpoint(args.checkpoint, device)
    validate_prompt(args.prompt, tokenizer)

    prompt_tokens = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    if prompt_tokens.size(1) > model.block_size:
        prompt_tokens = prompt_tokens[:, -model.block_size :]

    generated = model.generate(
        prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(generated[0].tolist())

    if args.output_file:
        Path(args.output_file).write_text(text, encoding="utf-8")
        print(f"Generated text saved to {args.output_file}")
    else:
        print(text)


if __name__ == "__main__":
    main()
