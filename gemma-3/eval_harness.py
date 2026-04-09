
import argparse
import sys
import os
import torch
import torch.nn as nn
from typing import List, Optional

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from gemma_3.config import Gemma3Config
    from gemma_3.model import Gemma3ForCausalLM
except ImportError:
    from config import Gemma3Config
    from model import Gemma3ForCausalLM

from transformers import AutoTokenizer

try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
except ImportError:
    print("Error: lm_eval not installed. Please run 'pip install lm_eval'")
    sys.exit(1)

def run_eval(checkpoint_path, tasks, limit=None, batch_size=None):
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    config = Gemma3Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Model
    model = Gemma3ForCausalLM(config.model).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Fix keys if model was compiled (remove _orig_mod prefix)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    
    model.eval()
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer_path)

    # Determine batch size
    if batch_size is None:
        batch_size = config.training.batch_size
    print(f"Using batch size: {batch_size}")

    print("Wrapping model with HFLM...")
    # Wrap in HFLM
    # Note: We pass the instantiated model and tokenizer
    # HFLM handles batching and device placement logic
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    
    print(f"Running evaluation on tasks: {tasks}")
    # Set max_gen_toks to avoid context limits error on generative tasks
    # Model max length is 2048. If task requests 2048 gen, context becomes 0.
    gen_kwargs = {"max_gen_toks": 1024}
    
    results = lm_eval.simple_evaluate(
        model=hflm,
        tasks=tasks,
        limit=limit,
        gen_kwargs=gen_kwargs
    )
    
    import json
    print(json.dumps(results["results"], indent=2))
    
    # Pretty print summary
    for task, metrics in results["results"].items():
        print(f"Task: {task}")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                 print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tasks", type=str, default="hellaswag,mmlu_pro", help="Comma separated list of tasks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per task for testing")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    if args.tasks == "all":
        task_manager = TaskManager()
        all_task_names = task_manager.all_tasks
        tasks_list = []
        print(f"Found {len(all_task_names)} potential tasks. Filtering available ones...")
        for task_name in all_task_names:
            try:
                # Try to load the task configuration/class to check for dependencies
                # This is a heuristic; simple_evaluate will re-load them, but this catches init errors
                task_manager.load_task_or_group(task_name)
                tasks_list.append(task_name)
            except Exception as e:
                # print(f"Skipping task '{task_name}': {e}") # Optional: verify verbose output
                pass
        print(f"Selected {len(tasks_list)} tasks out of {len(all_task_names)}.")
    else:
        tasks_list = args.tasks.split(",")
    
    run_eval(args.checkpoint, tasks_list, args.limit, args.batch_size)
