import os
from typing import List

import torch
from torch.utils.data import Dataset


class CharTokenizer:
    def __init__(self, text: str):
        self.vocab = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(self.itos[tok] for tok in tokens)


def load_text(data_file: str) -> str:
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    with open(data_file, "r", encoding="utf-8") as fp:
        text = fp.read()
    if not text:
        raise ValueError(f"Data file is empty: {data_file}")
    return text


class ShakespeareDataset(Dataset):
    def __init__(self, text: str, tokenizer: CharTokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

    def __len__(self) -> int:
        return max(0, self.data.size(0) - self.block_size)

    def __getitem__(self, idx: int):
        start = idx
        end = idx + self.block_size
        x = self.data[start:end]
        y = self.data[start + 1 : end + 1]
        return x, y
