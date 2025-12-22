
import torch
import os
import sys
import numpy as np

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from gemma_3.config import Gemma3Config
    from gemma_3.model import Gemma3ForCausalLM
    # We need PretokenizedDataset. It's in train.py but importing train might run code or be messy.
    # Let's just redefine it or hack import. redefining is safer/cleaner for a script.
except ImportError:
    from config import Gemma3Config
    from model import Gemma3ForCausalLM

from torch.utils.data import Dataset, DataLoader

class PretokenizedDataset(Dataset):
    def __init__(self, bin_path, seq_length):
        self.seq_length = seq_length
        if not os.path.exists(bin_path):
             raise FileNotFoundError(f"Dataset file not found: {bin_path}")
        self.tokens = np.memmap(bin_path, dtype=np.uint32, mode='r')
        self.num_chunks = len(self.tokens) // self.seq_length

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        chunk = self.tokens[start:end]
        input_ids = torch.from_numpy(chunk.astype(np.int64))
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}

def evaluate():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Defaults
    dataset_name = "fineweb"
    checkpoint_path = os.path.join(project_root, "experiments", "gemma-3", dataset_name, "baseline", "gemma3_step_99000.pt")
    
    # Dataset paths
    dataset_dir = os.path.join(project_root, 'datasets', 'gemma-3', dataset_name)
    val_path = os.path.join(dataset_dir, 'val.bin')
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    config = Gemma3Config()
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = Gemma3ForCausalLM(config.model).to(device)
    
    # Load Weights
    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Fix keys if model was compiled (remove _orig_mod prefix)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    
    # Optimize
    torch.set_float32_matmul_precision('high')
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    
    model.eval()

    # Data
    val_dataset = PretokenizedDataset(val_path, config.dataset.seq_length)
    val_dataloader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    total_loss = 0
    num_batches = 0
    max_val_batches = 100 # Evaluate on 100 batches to be reasonably accurate
    
    print("Starting evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= max_val_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]
            
            total_loss += loss.item()
            num_batches += 1
            
            if i % 10 == 0:
                print(f"Batch {i}: Loss {loss.item():.4f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Final Validation Loss: {avg_loss:.4f}")
    print(f"Perplexity: {np.exp(avg_loss):.4f}")

if __name__ == "__main__":
    evaluate()
