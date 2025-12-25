import os
import argparse
import sys
import numpy as np
import torch
from transformers import AutoTokenizer

# Add available paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared download util
from data_pipeline.prepare_data import prepare_raw_data

# Configuration
# This points to bonsai-llm/datasets
BASE_DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')
# Output specifically to datasets/gemma-3
MODEL_DATASETS_DIR = os.path.join(BASE_DATASETS_DIR, 'gemma-3')

def process_tinyshakespeare(dataset_name, tokenizer_path):
    # 1. Ensure raw data exists using shared pipeline
    prepare_raw_data(dataset_name)
    
    raw_dir = os.path.join(BASE_DATASETS_DIR, dataset_name)
    output_dir = os.path.join(MODEL_DATASETS_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load Tokenizer
    print(f"Loading tokenizer: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        # Fallback to gpt2 if user just wants to test flow, but warn heavily
        # Actually for Gemma-3 we really want the right tokenizer.
        # But if it requires auth and we don't have it, we might be stuck.
        # Assuming user has permissions or uses a public equivalent.
        raise e

    # 3. Read and Tokenize
    input_file = os.path.join(raw_dir, 'input.txt')
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
        
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    print(f"Tokenizing {len(train_data)} chars train, {len(val_data)} chars val...")
    
    # encode_ordinary is for tiktoken, transformers use encode / call
    # We use tokenizer.encode which returns list of ints
    train_ids = tokenizer.encode(train_data, add_special_tokens=False) # raw text usually
    val_ids = tokenizer.encode(val_data, add_special_tokens=False)
    
    print(f"Train tokens: {len(train_ids)}")
    print(f"Val tokens: {len(val_ids)}")
    
    # 4. Save as uint32
    # Gemma vocab is 256k, fits in uint32 (up to 4B)
    train_ids = np.array(train_ids, dtype=np.uint32)
    val_ids = np.array(val_ids, dtype=np.uint32)
    
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))
    
    print(f"Saved to {output_dir}")

def process_fineweb(dataset_name, tokenizer_path, token_limit, num_proc=1):
    print(f"Processing FineWeb ({dataset_name})...")
    from datasets import load_dataset
    from tqdm import tqdm
    import numpy as np
    import array
    
    # We ignore num_proc as we are back to single process, but keep arg for compatibility
    
    output_dir = os.path.join(MODEL_DATASETS_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading tokenizer: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        raise e

    hf_dataset_name = "HuggingFaceFW/fineweb-edu"
    hf_subset = "sample-10BT"
    
    print(f"Streaming {hf_dataset_name} ({hf_subset})...")
    ds = load_dataset(hf_dataset_name, name=hf_subset, split="train", streaming=True)
    
    train_file = os.path.join(output_dir, 'train.bin')
    val_file = os.path.join(output_dir, 'val.bin')
    
    # Overwrite output files
    open(train_file, 'wb').close()
    open(val_file, 'wb').close()
    
    # Use array.array for efficient memory usage (4 bytes per int)
    train_buffer = array.array('I')
    val_buffer = array.array('I')
    
    # 10M tokens buffer (approx 40MB per buffer in RAM + small overhead)
    buffer_size = 100_000_000 
    
    total_tokens = 0
    train_tokens = 0
    val_tokens = 0
    
    print(f"Processing {'all' if token_limit <= 0 else f'up to {token_limit}'} tokens...")

    for i, entry in enumerate(tqdm(ds)):
        if token_limit > 0 and total_tokens >= token_limit:
            break
            
        text = entry['text']
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Split logic: 10% to val (every 10th sample)
        if i % 10 == 0:
            val_buffer.extend(tokens)
            val_tokens += len(tokens)
        else:
            train_buffer.extend(tokens)
            train_tokens += len(tokens)
            
        total_tokens += len(tokens)

        # Flush buffers
        if len(train_buffer) >= buffer_size:
            with open(train_file, 'ab') as f:
                f.write(train_buffer.tobytes())
            train_buffer = array.array('I')

        if len(val_buffer) >= buffer_size:
            with open(val_file, 'ab') as f:
                f.write(val_buffer.tobytes())
            val_buffer = array.array('I')
            
    # Final flush
    if len(train_buffer) > 0:
        with open(train_file, 'ab') as f:
            f.write(train_buffer.tobytes())
    if len(val_buffer) > 0:
        with open(val_file, 'ab') as f:
            f.write(val_buffer.tobytes())
            
    print(f"Finished. {train_tokens} train, {val_tokens} val tokens saved to {output_dir}")

def prepare_dataset(dataset_name, tokenizer_path, token_limit, num_proc=None):
    if dataset_name == 'tinyshakespeare':
        process_tinyshakespeare(dataset_name, tokenizer_path)
    elif dataset_name == 'fineweb':
        process_fineweb(dataset_name, tokenizer_path, token_limit, num_proc)
    else:
        print(f"Dataset {dataset_name} not implemented for Gemma-3 yet.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for Gemma-3')
    parser.add_argument('--dataset', type=str, required=True, choices=['tinyshakespeare', 'fineweb'], help='Dataset to process')
    parser.add_argument('--tokenizer_path', type=str, default="google/gemma-3-1b-pt", help='Path to tokenizer')
    parser.add_argument('--token_limit', type=int, default=100000000, help='Number of tokens to process from FineWeb. Set to 0 for full dataset (sample-10BT is ~10B tokens)')
    
    parser.add_argument('--num_proc', type=int, default=int(os.cpu_count()/2), help='Number of processes to use')
    
    args = parser.parse_args()
    prepare_dataset(args.dataset, args.tokenizer_path, args.token_limit, args.num_proc)

    # Flush buffers and force exit to avoid PyGILState_Release errors with some libraries
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
