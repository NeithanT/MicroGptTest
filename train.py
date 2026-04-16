import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from config import config
from dataset import CharTokenizer, ShakespeareDataset, load_text
from model import GPT


def estimate_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model: GPT, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    return float(sum(losses) / len(losses)) if losses else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small GPT model on Shakespeare text.")
    parser.add_argument("--data_file", type=str, default=config.data_file)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--block_size", type=int, default=config.block_size)
    parser.add_argument("--max_epochs", type=int, default=config.max_epochs)
    parser.add_argument("--learning_rate", type=float, default=config.learning_rate)
    parser.add_argument("--save_path", type=str, default=config.save_path)
    parser.add_argument("--eval_interval", type=int, default=config.eval_interval)
    parser.add_argument("--eval_iters", type=int, default=config.eval_iters)
    parser.add_argument("--sample_length", type=int, default=config.sample_length)
    parser.add_argument("--sample_temperature", type=float, default=config.sample_temperature)
    parser.add_argument("--sample_top_k", type=int, default=config.sample_top_k)
    args = parser.parse_args()

    device = config.device
    print(f"Using device: {device}")

    text = load_text(args.data_file)
    tokenizer = CharTokenizer(text)
    dataset = ShakespeareDataset(text, tokenizer, args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = GPT(
        vocab_size=len(tokenizer.vocab),
        block_size=args.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
    ).to(device)

    print(f"Vocab size: {len(tokenizer.vocab)}")
    print(f"Model parameters: {estimate_parameters(model):,}")
    print(f"Dataset size: {len(dataset)} sequences")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=config.weight_decay)

    val_split = max(1, int(len(dataset) * 0.05))
    if len(dataset) > 2 * val_split:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_split, val_split])
    else:
        train_dataset, val_dataset = dataset, dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                elapsed = time.time() - start_time
                print(
                    f"step {global_step:>5} | epoch {epoch}/{args.max_epochs} | "
                    f"train loss {loss.item():.4f} | val loss {val_loss:.4f} | elapsed {elapsed:.1f}s"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "tokenizer": tokenizer.vocab,
                            "config": vars(config),
                        },
                        args.save_path,
                    )
                    print(f"Saved best checkpoint to {args.save_path}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} complete | avg train loss {avg_epoch_loss:.4f}")

    final_path = args.save_path
    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.vocab,
            "config": vars(config),
        },
        final_path,
    )
    print(f"Training complete. Model saved to {final_path}")

    prompt = "ROMEO:"
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=args.sample_length, temperature=args.sample_temperature, top_k=args.sample_top_k)
    sample_text = tokenizer.decode(out[0].tolist())
    print("\nSample generation:\n")
    print(sample_text)


if __name__ == "__main__":
    main()
