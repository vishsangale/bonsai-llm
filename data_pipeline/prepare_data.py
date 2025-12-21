import os
import argparse
import requests
from tqdm import tqdm
from datasets import load_dataset
import shutil

# Configuration
# This points to bonsai-llm/datasets
DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')

def download_tinyshakespeare(dataset_dir):
    """Downloads the TinyShakespeare dataset."""
    os.makedirs(dataset_dir, exist_ok=True)
    file_path = os.path.join(dataset_dir, 'input.txt')
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    
    if not os.path.exists(file_path):
        print(f"Downloading TinyShakespeare to {file_path}...")
        response = requests.get(data_url, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print("TinyShakespeare input.txt already exists.")
    
    return file_path

def download_fineweb(dataset_dir):
    """
    For FineWeb, we don't necessarily download the whole thing to a text file
    because it's huge. But if the user wants to 'prepare' it, we might just
    ensure the directory exists or download a small sample if requested.
    
    For now, this is a placeholder or a small sample download if we want to support it similarly.
    """
    print("FineWeb download handling is delegated to streaming in model-specific scripts usually.")
    print("However, ensuring directory exists.")
    os.makedirs(dataset_dir, exist_ok=True)
    # Could download a sample jsonl here if needed.

def prepare_raw_data(dataset_name):
    """Downloads the raw dataset."""
    dataset_dir = os.path.join(DATASETS_DIR, dataset_name)
    
    print(f"Ensuring raw data for: {dataset_name} in {dataset_dir}")
    
    if dataset_name == 'tinyshakespeare':
        download_tinyshakespeare(dataset_dir)
    elif dataset_name == 'fineweb':
        download_fineweb(dataset_dir)
    else:
        print(f"Unknown dataset: {dataset_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download raw datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['tinyshakespeare', 'fineweb'], help='Dataset to download')
    
    args = parser.parse_args()
    prepare_raw_data(args.dataset)
