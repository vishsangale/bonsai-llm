# Data Pipeline

This directory contains scripts for data downloading, loading, preprocessing, and cleaning for the Bonsai LLM project.

## Components

### TinyShakespeare Dataset (`tinyshakespeare.py`)

Handles the downloading and loading of the [TinyShakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) dataset.

#### Usage

**Download the dataset:**
Run the script directly to download the dataset to `datasets/tinyshakespeare/input.txt`.
```bash
python data_pipeline/tinyshakespeare.py
```

**Load in code:**
```python
from data_pipeline.tinyshakespeare import read_tinyshakespeare

# Get training split (90%)
train_data = read_tinyshakespeare(split='train')

# Get validation split (10%)
val_data = read_tinyshakespeare(split='val')
```
