# Shakespeare GPT

A character-level GPT model trained on Shakespeare's complete works, based on Andrej Karpathy's tutorial: "Let's build GPT: from scratch, in code, spelled out."

## Project Structure

```
shakespeareGPT/
├── config.py           # Model configuration and hyperparameters
├── tokenizer.py        # Character-level tokenizer
├── dataset.py          # Shakespeare dataset loader
├── model.py            # GPT model architecture (Transformer blocks)
├── train.py            # Training script
├── generate.py         # Text generation script
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python train.py
```

This will:
- Load the Shakespeare text data
- Train the model for 10,000 iterations
- Validate on a held-out test set
- Save the trained model to `shakespeareGPT/shakespeareGPT.pth`
- Generate a loss plot saved as `losses.png`

### Generating Text

After training, generate new Shakespeare-style text:

```bash
python generate.py
```

This will:
- Load the trained model
- Generate text samples of various lengths (100, 300, 500, 700, 1000 tokens)
- Save the output to `generated.txt`

## Model Architecture

- **Model Type**: Character-level GPT (Transformer decoder)
- **Embedding Dimension**: 256
- **Number of Heads**: 8
- **Number of Layers**: 3
- **Context Length**: 256 characters
- **Feed-Forward Dimension**: 1024 (4x embed dim)
- **Dropout**: 0.1 (attention and blocks)

## Hyperparameters

- **Batch Size**: 64
- **Learning Rate**: 3e-4
- **Optimizer**: AdamW
- **Training Iterations**: 10,000
- **Validation Iterations**: 1,000

## Data

The model expects Shakespeare text data at:
`/kaggle/input/shakespeare-full-text/shakespeare.txt`

For local use, update the data path in `train.py` and `generate.py`.

## Notes

- The model uses a character-level tokenizer, so it learns to generate text character by character
- Training/validation split is 90/10
- Random seed is set to 1357 for reproducibility