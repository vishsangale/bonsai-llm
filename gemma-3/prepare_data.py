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

def process_fineweb(dataset_name, tokenizer_path):
    print(f"Processing FineWeb ({dataset_name})...")
    # For FineWeb, we stream directly from HF instead of downloading a raw file first
    from datasets import load_dataset
    from tqdm import tqdm

    output_dir = os.path.join(MODEL_DATASETS_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if files already exist
    if os.path.exists(os.path.join(output_dir, 'train.bin')) and os.path.exists(os.path.join(output_dir, 'val.bin')):
        print(f"FineWeb binaries already exist in {output_dir}. Skipping.")
        return

    print(f"Loading tokenizer: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        raise e

    # Configs
    # Using sample-10BT for tractable training/verification
    hf_dataset_name = "HuggingFaceFW/fineweb-edu"
    hf_subset = "sample-10BT" 
    
    print(f"Streaming {hf_dataset_name} ({hf_subset})...")
    ds = load_dataset(hf_dataset_name, name=hf_subset, split="train", streaming=True)
    
    # We will assume a simple split: first X tokens for val, rest for train
    # Or to ensure diversity, every Nth sample.
    # Given it's streaming, let's just grab enough data for a "bonsai" model.
    # Let's target ~100M tokens for this example? Or just let it run until user stops?
    # Better: process for a fixed amount or all of it. sample-10BT is 10B tokens, which is huge for local.
    # Let's implement a limit or just process a chunk. 
    # Current implementation: Process 100k samples.
    
    train_file = os.path.join(output_dir, 'train.bin')
    val_file = os.path.join(output_dir, 'val.bin')
    
    # Clean files
    open(train_file, 'wb').close()
    open(val_file, 'wb').close()
    
    train_tokens_count = 0
    val_tokens_count = 0
    
    train_buffer = []
    val_buffer = []
    buffer_size = 100 * 1024 # 100k tokens flush
    
    # Validation ratio: 1/100
    val_ratio = 100
    
    limit_samples = 500 # Enough for verification and small training
    print(f"Processing up to {limit_samples} samples...")

    for i, entry in tqdm(enumerate(ds)):
        if i >= limit_samples:
            break
            
        text = entry['text']
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Split logic: simple every 100th doc to val
        if i % val_ratio == 0:
            val_buffer.extend(tokens)
        else:
            train_buffer.extend(tokens)
            
        # Flush if buffer full
        if len(train_buffer) >= buffer_size:
            arr = np.array(train_buffer, dtype=np.uint32)
            with open(train_file, 'ab') as f:
                f.write(arr.tobytes())
            train_tokens_count += len(arr)
            train_buffer = []
            
        if len(val_buffer) >= buffer_size:
            arr = np.array(val_buffer, dtype=np.uint32)
            with open(val_file, 'ab') as f:
                f.write(arr.tobytes())
            val_tokens_count += len(arr)
            val_buffer = []
            
    # Flush remainder
    if train_buffer:
        with open(train_file, 'ab') as f:
            f.write(np.array(train_buffer, dtype=np.uint32).tobytes())
        train_tokens_count += len(train_buffer)
        
    if val_buffer:
        with open(val_file, 'ab') as f:
            f.write(np.array(val_buffer, dtype=np.uint32).tobytes())
        val_tokens_count += len(val_buffer)
        
    print(f"Saved {train_tokens_count} train tokens, {val_tokens_count} val tokens to {output_dir}")

def prepare_dataset(dataset_name, tokenizer_path):
    if dataset_name == 'tinyshakespeare':
        process_tinyshakespeare(dataset_name, tokenizer_path)
    elif dataset_name == 'fineweb':
        process_fineweb(dataset_name, tokenizer_path)
    else:
        print(f"Dataset {dataset_name} not implemented for Gemma-3 yet.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for Gemma-3')
    parser.add_argument('--dataset', type=str, required=True, choices=['tinyshakespeare', 'fineweb'], help='Dataset to process')
    parser.add_argument('--tokenizer_path', type=str, default="google/gemma-3-1b-pt", help='Path to tokenizer')
    
    args = parser.parse_args()
    prepare_dataset(args.dataset, args.tokenizer_path)

    # Flush buffers and force exit to avoid PyGILState_Release errors with some libraries
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
