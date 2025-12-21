import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
import sys

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

from data_pipeline.tinyshakespeare import read_tinyshakespeare

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.tokens = tokenizer.encode(text, add_special_tokens=False)
        # Convert to tensor immediately
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)

    def __len__(self):
        return len(self.tokens) // self.seq_length

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        chunk = self.tokens[start:end]
        
        if len(chunk) < self.seq_length:
             # Padding or repeat logic
             return self.__getitem__(0)

        input_ids = chunk
        labels = chunk

        return {"input_ids": input_ids, "labels": labels}

from torch.utils.tensorboard import SummaryWriter

def train():
    config = Gemma3Config()
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
    train_text = read_tinyshakespeare(split='train')
    val_text = read_tinyshakespeare(split='val')
    
    dataset = TextDataset(train_text, tokenizer, config.dataset.seq_length)
    val_dataset = TextDataset(val_text, tokenizer, config.dataset.seq_length)
    
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    # Initialize Model
    model = Gemma3ForCausalLM(config.model).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    
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
                
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, labels=labels)
                    loss = outputs["loss"]
                
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

            with torch.cuda.amp.autocast():
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]

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
                 print(f"Step {step}: Validation Loss = {val_loss:.4f}")
                 writer.add_scalar("Validation/Loss", val_loss, step)

            step += 1
            if step >= config.training.max_steps:
                print("Max steps reached. Saving model...")
                torch.save(model.state_dict(), os.path.join(config.training.output_dir, "gemma3_mini.pt"))
                writer.close()
                return

if __name__ == "__main__":
    train()
