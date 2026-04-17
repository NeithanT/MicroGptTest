# DO NOT USE THIS AS TRAINING DATA OR FOR USER READING
This is just me testing, 100% AI Slop

## Setup

1. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

   If you want GPU training, install a CUDA-enabled PyTorch wheel instead of the default CPU-only build. For example, visit https://pytorch.org/get-started/locally to choose the correct CUDA version, or use a command like:

   ```bash
   python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   Then install the rest of the dependencies, if needed.

2. Place a Shakespeare text file named `shakespeare.txt` in the project root.
   You can use the Project Gutenberg version of Shakespeare's works or any plain-text Shakespeare corpus.

## Training

Run training with:

```bash
python train.py --data_file shakespeare.txt --max_epochs 5 --batch_size 16 --device auto
```

The script will:
- build a character-level tokenizer from the text
- create a tiny GPT model with roughly 200k parameters
- train on GPU when CUDA is available, otherwise on CPU
- save a checkpoint after every epoch (default behavior)
- preserve the best validation checkpoint separately
- optionally keep only the last N epoch checkpoints with `--keep_last`
- save the final trained model to `checkpoint.pt`
- print a sample generation at the end

## Generation

Generate text from a trained checkpoint:

```bash
python generate.py --checkpoint checkpoint.pt --prompt "ROMEO:" --max_new_tokens 200 --device cpu
```

## Notes

- The model uses a character-level vocabulary and causal self-attention.
- The default configuration targets roughly 20 million trainable parameters.
- For faster experimentation, reduce `batch_size` or `block_size`.
