# Small LLMs Lab

A laboratory for training small Language Models (LLMs) from scratch. This project currently features a custom implementation of the Gemma 3 architecture.

## Project Structure

- **`gemma-3/`**: Implementation of the Gemma 3 model, configuration, and training loop.
- **`data_pipeline/`**: Scripts for fetching and processing datasets (currently TinyShakespeare).
- **`datasets/`**: Directory where downloaded datasets are stored.
- **`experiments/`**: Stores training logs (TensorBoard) and model checkpoints.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ and a virtual environment set up.

Dependencies:
- `torch`
- `transformers`
- `numpy`
- `requests`
- `tqdm`
- `tensorboard`
- `tiktoken`
- `datasets`

### Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install torch transformers numpy requests tqdm tensorboard tiktoken datasets
   ```

### Running the Project

1. **Prepare the Data**:
   ```bash
   python data_pipeline/prepare_data.py --dataset tinyshakespeare
   ```

2. **Start Training**:
   ```bash
   python gemma-3/train.py
   ```

3. **Monitor Progress**:
   Open a new terminal, activate the environment, and run TensorBoard:
   ```bash
   tensorboard --logdir experiments
   ```
