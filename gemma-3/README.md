# Gemma 3 Model

This directory contains a custom implementation of the Gemma 3 model architecture.

## Model Configuration (`config.py`)

The default configuration (`Gemma3Config`) defines a "Mini" variant suitable for testing and small-scale experiments:

- **Parameters**: ~29M
- **Hidden Size**: 768
- **Layers**: 24
- **Attention Heads**: 12
- **Vocab Size**: 25600 (or matched to tokenizer)
- **Sequence Length**: 2048
- **Attention**: Global/Local sliding window attention (5:1 ratio)

## Training (`train.py`)

The training script initializes the model and trains it on the TinyShakespeare dataset.

### Usage
Run the training script from the project root:
```bash
python gemma-3/train.py
```

### Features
- **Automatic Device Selection**: Uses CUDA if available, else CPU.
- **Gradient Accumulation**: Simulates larger batch sizes.
- **Mixed Precision**: Uses `torch.cuda.amp` for efficiency.
- **TensorBoard Logging**: Metrics are logged to `experiments/gemma-3/tinyshakespeare/baseline`.
- **Checkpointing**: Saves the model to `experiments` at the end of training.

## Architecture Details (`model.py`)
The implementation includes:
- RMSNorm
- Rotary Positional Embeddings (RoPE)
- Sliding Window Attention with Global Attention layers
- Gated GeLU MLP
