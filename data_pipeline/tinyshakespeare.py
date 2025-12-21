import os
import requests
import numpy as np
from tqdm import tqdm

DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'tinyshakespeare')

def download_tinyshakespeare():
    """Downloads the TinyShakespeare dataset to the datasets directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, 'input.txt')
    
    if not os.path.exists(file_path):
        print(f"Downloading TinyShakespeare to {file_path}...")
        response = requests.get(DATA_URL, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print("TinyShakespeare dataset already exists.")
    
    return file_path

def read_tinyshakespeare(split='train', split_ratio=0.9):
    """Reads the raw text from the dataset.
    
    Args:
        split (str): 'train' or 'val'
        split_ratio (float): Ratio of data to use for training.
    """
    file_path = os.path.join(DATA_DIR, 'input.txt')
    if not os.path.exists(file_path):
        download_tinyshakespeare()
        
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    n = len(text)
    split_idx = int(n * split_ratio)
    
    if split == 'train':
        return text[:split_idx]
    elif split == 'val':
        return text[split_idx:]
    else:
        raise ValueError(f"Unknown split: {split}")

if __name__ == "__main__":
    download_tinyshakespeare()
