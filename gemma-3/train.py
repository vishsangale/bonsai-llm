import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
import sys
import numpy as np

# Fix for CUDA memory fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Add parent dir to path to import config/model
# Add parent dir to path to import data_pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from gemma_3.config import Gemma3Config
    from gemma_3.model import Gemma3ForCausalLM
except ImportError:
    # Fallback for when running directly inside gemma-3 or if package name issue
    from config import Gemma3Config
    from model import Gemma3ForCausalLM

# DATASETS_DIR
DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'gemma-3')

class PretokenizedDataset(Dataset):
    def __init__(self, bin_path, seq_length):
        self.seq_length = seq_length
        # Load memory mapped
        # Check if file exists
        if not os.path.exists(bin_path):
             raise FileNotFoundError(f"Dataset file not found: {bin_path}")
        
        self.tokens = np.memmap(bin_path, dtype=np.uint32, mode='r')
        self.num_chunks = len(self.tokens) // self.seq_length

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        
        # Get chunk (numpy array)
        chunk = self.tokens[start:end]
        
        # Convert to tensor (int64/long for torch)
        # Copy is necessary because memmap is not writeable and torch might want to own memory or just casting
        input_ids = torch.from_numpy(chunk.astype(np.int64))
        labels = input_ids.clone()

        return {"input_ids": input_ids, "labels": labels}

from torch.utils.tensorboard import SummaryWriter

def parse_args(config):
    # Very simple generic arg parser for --key=value
    # Supports nested keys like --training.batch_size=4
    import sys
    
    for arg in sys.argv[1:]:
        if arg.startswith('--') and '=' in arg:
            key_full, value_str = arg[2:].split('=', 1)
            
            # Traverse config to find the leaf attribute
            keys = key_full.split('.')
            obj = config
            try:
                # Go down to parent object
                for k in keys[:-1]:
                    obj = getattr(obj, k)
                
                leaf_key = keys[-1]
                # Get current value to infer type
                current_val = getattr(obj, leaf_key)
                target_type = type(current_val)
                
                # Cast value
                if target_type == bool:
                    new_val = value_str.lower() in ('true', '1', 'yes', 'on')
                else:
                    new_val = target_type(value_str)
                    
                print(f"Config override: {key_full} = {new_val} (was {current_val})")
                setattr(obj, leaf_key, new_val)
                
            except AttributeError:
                print(f"Warning: Config key '{key_full}' not found. Ignoring.")
            except ValueError:
                print(f"Warning: Could not cast '{value_str}' to type {target_type} for key '{key_full}'. Ignoring.")

def train():
    config = Gemma3Config()
    parse_args(config)
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer
    # Trying to use gemma tokenizer, but if not authenticated/exist, fallback to something simple or gpt2
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer_path)
    except Exception as e:
        print(f"Warning: Could not load {config.dataset.tokenizer_path} due to error: {e}")
        print("Falling back to 'gpt2'")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Update config vocab size to match tokenizer
        config.model.vocab_size = tokenizer.vocab_size

    # Load Data
    # Resolve dataset directory. 
    if config.dataset.dataset_path:
        # If absolute, use as is. If relative, assume relative to project root.
        if os.path.isabs(config.dataset.dataset_path):
            dataset_dir = config.dataset.dataset_path
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dataset_dir = os.path.join(project_root, config.dataset.dataset_path)
    else:
        # Generate automatically based on dataset_name
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_dir = os.path.join(project_root, 'datasets', 'gemma-3', config.dataset.dataset_name)
    
    train_path = os.path.join(dataset_dir, 'train.bin')
    val_path = os.path.join(dataset_dir, 'val.bin')
    
    print(f"Loading datasets from {dataset_dir}...")
    
    try:
        dataset = PretokenizedDataset(train_path, config.dataset.seq_length)
        val_dataset = PretokenizedDataset(val_path, config.dataset.seq_length)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please run 'python gemma-3/prepare_data.py --dataset {config.dataset.dataset_name}' first.")
        return
    
    # Optimize standard matmul precision
    torch.set_float32_matmul_precision('high')

    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize Model
    model = Gemma3ForCausalLM(config.model).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Compile model
    print("Compiling model...")
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, fused=True)
    scaler = torch.amp.GradScaler('cuda')
    
    # Resolve output directory
    if config.training.output_dir:
        if not os.path.isabs(config.training.output_dir):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config.training.output_dir = os.path.join(project_root, config.training.output_dir)
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config.training.output_dir = os.path.join(project_root, "experiments", "gemma-3", config.dataset.dataset_name, "baseline")

    # TensorBoard
    writer = SummaryWriter(log_dir=config.training.output_dir)

    def evaluate():
        model.eval()
        total_loss = 0
        num_batches = 0
        max_val_batches = 50 
        
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                if i >= max_val_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss
                
                total_loss += loss.item()
                num_batches += 1
        
        model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0

    model.train()
    step = 0
    
    print("Starting training...")
    # Ensure output directory exists for checkpoints
    os.makedirs(config.training.output_dir, exist_ok=True)

    import time
    start_time = time.time()
    last_log_time = time.time()

    for epoch in range(10): # Arbitrary epochs, loop controlled by max_steps
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            
            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Batch size * sequence length
            tokens_per_step = input_ids.numel()

            if step % config.training.logging_steps == 0:
                current_time = time.time()
                elapsed_time = current_time - last_log_time
                # Avoid division by zero for the very first step if it's super fast, though unlikely
                if step == 0:
                     steps_logged = 1
                else:
                     steps_logged = config.training.logging_steps
                
                avg_step_time = elapsed_time / steps_logged
                tokens_per_sec = (tokens_per_step * steps_logged) / elapsed_time

                print(f"Step {step}: Loss = {loss.item():.4f}, Avg Time = {avg_step_time:.3f}s, Tokens/sec = {tokens_per_sec:.2f}")
                writer.add_scalar("Training/Loss", loss.item(), step)
                writer.add_scalar("Performance/TimePerStep", avg_step_time, step)
                writer.add_scalar("Performance/TokensPerSec", tokens_per_sec, step)
                
                last_log_time = current_time
            
            if step % config.training.eval_steps == 0:
                 val_loss = evaluate()
                 val_ppl = np.exp(val_loss)
                 print(f"Step {step}: Validation Loss = {val_loss:.4f}, Perplexity = {val_ppl:.4f}")
                 writer.add_scalar("Validation/Loss", val_loss, step)
                 writer.add_scalar("Validation/Perplexity", val_ppl, step)

            if step > 0 and step % config.training.eval_harness_steps == 0:
                 print(f"Running Eval Harness at step {step}...")
                 try:
                     import lm_eval
                     from lm_eval.models.huggingface import HFLM
                     
                     # Re-wrap model. HFLM is lightweight wrapper.
                     # We use the raw model (not compiled) if possible, but compiled might work.
                     # HFLM expects a model that returns CausalLMOutput-like object or dict with logits.
                     # We updated model to return CausalLMOutputWithPast, so it should work.
                     
                     # Note: HFLM might assume it owns the model. We pass it in.
                     # Using batch_size from training config might be too aggressive for heavier eval tasks, 
                     # but let's try.
                     
                     # IMPORTANT: HFLM expects 'device' attribute or we pass it. 
                     # We added 'device' property to model.
                     
                     hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=config.training.batch_size)
                     
                     results = lm_eval.simple_evaluate(
                        model=hflm,
                        tasks=config.training.eval_harness_tasks,
                        limit=None # Evaluate all or set a limit in config if needed
                     )
                     
                     # Log results
                     for task, metrics in results['results'].items():
                         for metric_name, value in metrics.items():
                             if isinstance(value, (int, float)):
                                 # Metric name often has none, e.g. "acc,none"
                                 clean_metric = metric_name.split(',')[0]
                                 writer.add_scalar(f"Eval/{task}/{clean_metric}", value, step)
                                 print(f"Eval {task} {clean_metric}: {value:.4f}")
                                 
                 except Exception as e:
                     print(f"Eval Harness Failed: {e}")
                     import traceback
                     traceback.print_exc()

            if step > 0 and step % config.training.save_steps == 0:
                 print(f"Saving checkpoint at step {step}...")
                 torch.save(model.state_dict(), os.path.join(config.training.output_dir, f"gemma3_step_{step}.pt"))

            step += 1
            if step >= config.training.max_steps:
                print("Max steps reached. Saving model...")
                torch.save(model.state_dict(), os.path.join(config.training.output_dir, f"gemma3_step_{step}.pt"))
                writer.close()
                return

if __name__ == "__main__":
    train()
