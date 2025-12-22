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
- `datasets`

### Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install torch transformers numpy requests tqdm tensorboard datasets lm_eval
   ```

### Evaluation

To run industry standard benchmarks (HellaSwag, PIQA, ARC, etc.), you can use the evaluation script:

```bash
python gemma-3/eval_harness.py --checkpoint experiments/gemma-3/fineweb/baseline/gemma3_step_99000.pt --tasks hellaswag,piqa,arc_easy
```

These benchmarks are also run periodically during training (configurable via `--training.eval_harness_steps`).

### Running the Project
 
 1. **Data Preparation**:
    The pipeline is split into downloading raw data and preparing model-specific tokenized datasets.
 
    **a. TinyShakespeare**
    ```bash
    # 1. Download raw data
    python data_pipeline/prepare_data.py --dataset tinyshakespeare
    
    # 2. Tokenize for Gemma-3
    python gemma-3/prepare_data.py --dataset tinyshakespeare
    ```
 
    **b. FineWeb (Edu)**
    ```bash
    # Stream and tokenize directly (no separate download step needed)
    python gemma-3/prepare_data.py --dataset fineweb
    ```
 
 2. **Start Training**:
    Run the training script. You can override any configuration parameter via CLI.
 
    **Basic Run (`tinyshakespeare`)**:
    ```bash
    python gemma-3/train.py
    ```
 
    **FineWeb Run**:
    ```bash
    python gemma-3/train.py --dataset.dataset_name=fineweb
    ```
 
    **Advanced CLI Usage**:
    Override any config field using `--section.key=value` syntax.
    ```bash
    python gemma-3/train.py --training.batch_size=4 --training.max_steps=5000 --model.gradient_checkpointing=False
    ```
 
 3. **Monitor Progress**:
    Open a new terminal, activate the environment, and run TensorBoard:
    ```bash
    tensorboard --logdir experiments
    ```

## Training Results

### FineWeb Training Run (100k steps)

We trained the Gemma-3 small model on the FineWeb dataset for 100,000 steps.

**Configuration:**
- **Model:** Gemma-3 (Approx 1B parameters scale down)
- **Dataset:** FineWeb Edu
- **Batch Size:** 2 (local), Gradient Accumulation: 16
- **Max Steps:** 100,000
- **Optimizer:** AdamW

**Final Metrics (Step 99,000):**
- **Validation Loss:** 3.8147
- **Perplexity:** 45.3633

These results indicate the model has successfully learned from the FineWeb dataset, achieving a perplexity of ~45.36.

### Benchmark Results (Zero-shot)

We evaluated the model on industry-standard benchmarks using `lm-evaluation-harness`.

| Task | Metric | Value | Random Baseline |
| :--- | :--- | :--- | :--- |
| **ARC-Easy** | Accuracy | **41.20%** | ~25% |
| **PIQA** | Accuracy | **57.78%** | ~50% |
| **HellaSwag** | Accuracy | **26.51%** | ~25% |

*Note: Results are zero-shot. HellaSwag remains challenging for smaller models at this training stage.*

