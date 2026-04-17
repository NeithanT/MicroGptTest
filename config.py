from dataclasses import dataclass

@dataclass
class Config:
    data_file: str = "shakespeare.txt"
    batch_size: int = 16
    block_size: int = 256
    max_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    eval_interval: int = 200
    eval_iters: int = 100
    save_path: str = "checkpoint.pt"
    sample_length: int = 200
    sample_temperature: float = 1.0
    sample_top_k: int = 30
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.1

    @property
    def device(self):
        import torch
        return torch.device("cpu")

config = Config()
