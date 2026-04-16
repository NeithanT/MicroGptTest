import argparse
import torch

from config import config
from dataset import CharTokenizer
from model import GPT


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Shakespeare-like text from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=config.sample_temperature)
    parser.add_argument("--top_k", type=int, default=config.sample_top_k)
    args = parser.parse_args()

    device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    vocab = checkpoint["tokenizer"]
    tokenizer = CharTokenizer("".join(vocab))

    model = GPT(
        vocab_size=len(tokenizer.vocab),
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prompt_tokens = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    generated = model.generate(prompt_tokens, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    text = tokenizer.decode(generated[0].tolist())

    print(text)


if __name__ == "__main__":
    main()
