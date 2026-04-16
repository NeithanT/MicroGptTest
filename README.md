# mini-gpt

A small PyTorch GPT-style model for training on Shakespeare text.

## Setup

1. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Place a Shakespeare text file named `shakespeare.txt` in the project root.
   You can use the Project Gutenberg version of Shakespeare's works or any plain-text Shakespeare corpus.

## Training

Run training with:

```bash
python train.py --data_file shakespeare.txt --max_epochs 5 --batch_size 16
```

The script will:
- build a character-level tokenizer from the text
- create a small GPT model with ~20M parameters
- train on CPU
- save the best checkpoint to `checkpoint.pt`
- print a sample generation at the end

## Generation

Generate text from a trained checkpoint:

```bash
python generate.py --checkpoint checkpoint.pt --prompt "ROMEO:" --max_new_tokens 200
```

## Notes

- The model uses a character-level vocabulary and causal self-attention.
- The default configuration targets roughly 20 million trainable parameters.
- For faster experimentation, reduce `batch_size` or `block_size`.
