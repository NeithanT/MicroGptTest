import argparse
import os
import re
import time
from pathlib import Path
from typing import cast

import torch
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from config import config
from dataset import CharTokenizer, ShakespeareDataset, load_text
from model import GPT


def estimate_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model: GPT, dataloader: DataLoader, device: torch.device, use_cuda: bool) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_cuda):
                _, loss = model(x, y)
            losses.append(loss.item())
    return float(sum(losses) / len(losses)) if losses else 0.0


def build_device(device_flag: str) -> torch.device:
    device_flag = device_flag.lower()
    if device_flag == "auto":
        return config.device
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine.")
        return torch.device("cuda")
    raise ValueError("Unsupported device. Use 'auto', 'cpu', or 'cuda'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small GPT model on Shakespeare text.")
    parser.add_argument("--data_file", type=str, default=config.data_file)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--block_size", type=int, default=config.block_size)
    parser.add_argument("--max_epochs", type=int, default=config.max_epochs)
    parser.add_argument("--learning_rate", type=float, default=config.learning_rate)
    parser.add_argument("--save_path", type=str, default=config.save_path)
    parser.add_argument("--save_epoch_checkpoints", action="store_true", default=True,
                        help="Save a checkpoint file after each epoch.")
    parser.add_argument("--no_save_epoch_checkpoints", action="store_false", dest="save_epoch_checkpoints",
                        help="Do not save checkpoint files after each epoch.")
    parser.add_argument("--keep_last", type=int, default=0,
                        help="Keep only the last N epoch checkpoints. 0 keeps all.")
    parser.add_argument("--save_best_path", type=str, default=None,
                        help="Optional path to save the best validation checkpoint separately.")
    parser.add_argument("--eval_interval", type=int, default=config.eval_interval)
    parser.add_argument("--eval_iters", type=int, default=config.eval_iters)
    parser.add_argument("--sample_length", type=int, default=config.sample_length)
    parser.add_argument("--sample_temperature", type=float, default=config.sample_temperature)
    parser.add_argument("--sample_top_k", type=int, default=config.sample_top_k)
    parser.add_argument("--stride", type=int, default=1,
                        help="Step size between blocks in the training dataset. Use block_size for non-overlapping sequences.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers for background data loading.")
    parser.add_argument("--pin_memory", action="store_true", default=True,
                        help="Pin memory for faster host to GPU transfers.")
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory",
                        help="Disable pin_memory for DataLoader.")
    parser.add_argument("--compile_model", action="store_true", default=True,
                        help="Compile the model with torch.compile() when available.")
    parser.add_argument("--no_compile_model", action="store_false", dest="compile_model",
                        help="Disable torch.compile() even if available.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to train on; auto selects CUDA if available.")
    args = parser.parse_args()

    checkpoint_config = vars(config).copy()
    checkpoint_config.update(
        data_file=args.data_file,
        batch_size=args.batch_size,
        block_size=args.block_size,
        stride=args.stride,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        save_path=args.save_path,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        sample_length=args.sample_length,
        sample_temperature=args.sample_temperature,
        sample_top_k=args.sample_top_k,
    )

    device = build_device(args.device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {device}")

    text = load_text(args.data_file)
    tokenizer = CharTokenizer(text)
    dataset = ShakespeareDataset(text, tokenizer, args.block_size, stride=args.stride)

    save_best_path = args.save_best_path or f"{os.path.splitext(args.save_path)[0]}_best{os.path.splitext(args.save_path)[1] or '.pt'}"

    def save_checkpoint(path: str) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "tokenizer": tokenizer.vocab,
                "config": checkpoint_config,
            },
            path,
        )

    def cleanup_epoch_checkpoints(path: str, keep_last: int) -> None:
        if keep_last <= 0:
            return
        base_name = os.path.splitext(os.path.basename(path))[0]
        ext = os.path.splitext(path)[1]
        parent = os.path.dirname(path) or "."
        epoch_files = []
        for epoch_file in Path(parent).glob(f"{base_name}_epoch_*{ext}"):
            match = re.search(r"_epoch_(\d+)", epoch_file.name)
            if match:
                epoch_files.append((int(match.group(1)), epoch_file))
        epoch_files.sort(key=lambda item: item[0])
        for _, old_file in epoch_files[:-keep_last]:
            old_file.unlink(missing_ok=True)

    model = GPT(
        vocab_size=len(tokenizer.vocab),
        block_size=args.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
    )
    model = cast(GPT, model.to(device))

    if args.compile_model and hasattr(torch, "compile"):
        model = cast(GPT, torch.compile(model))

    print(f"Vocab size: {len(tokenizer.vocab)}")
    print(f"Model parameters: {estimate_parameters(model):,}")
    print(f"Dataset size: {len(dataset)} sequences")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler(device="cuda", enabled=use_cuda)

    val_split = max(1, int(len(dataset) * 0.05))
    if len(dataset) > 2 * val_split:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_split, val_split])
    else:
        train_dataset, val_dataset = dataset, dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_cuda):
                logits, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, use_cuda)
                elapsed = time.time() - start_time
                print(
                    f"step {global_step:>5} | epoch {epoch}/{args.max_epochs} | "
                    f"train loss {loss.item():.4f} | val loss {val_loss:.4f} | elapsed {elapsed:.1f}s"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(save_best_path)
                    print(f"Saved best checkpoint to {save_best_path}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} complete | avg train loss {avg_epoch_loss:.4f}")

        if args.save_epoch_checkpoints:
            epoch_path = f"{os.path.splitext(args.save_path)[0]}_epoch_{epoch}{os.path.splitext(args.save_path)[1] or '.pt'}"
            save_checkpoint(epoch_path)
            print(f"Saved epoch checkpoint to {epoch_path}")
            cleanup_epoch_checkpoints(args.save_path, args.keep_last)

    final_path = args.save_path
    save_checkpoint(final_path)
    print(f"Training complete. Model saved to {final_path}")
    print(f"Best validation checkpoint: {save_best_path}")

    prompt = "ROMEO:"
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=args.sample_length, temperature=args.sample_temperature, top_k=args.sample_top_k)
    sample_text = tokenizer.decode(out[0].tolist())
    print("\nSample generation:\n")
    print(sample_text)


if __name__ == "__main__":
    main()
