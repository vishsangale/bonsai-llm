from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    vocab_size: int = 256000
    hidden_size: int = 768        # Original 1B: 2048
    intermediate_size: int = 3072 # 4x hidden_size. Original 1B: 16384 (8x)
    num_hidden_layers: int = 24   # Original 1B: 18
    num_attention_heads: int = 12 # Original 1B: 8
    num_key_value_heads: int = 1  # Original 1B: 1
    head_dim: int = 64            # Original 1B: 256
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    
    # RoPE
    rope_theta_global: float = 1000000.0
    rope_theta_local: float = 10000.0
    gradient_checkpointing: bool = True # Enable to save memory
    
    # Attention
    attention_window_size: int = 512 # Sliding window size
    # 5:1 ratio: Local, Local, Local, Local, Local, Global
    # Indices of global layers. 24 layers -> 5, 11, 17, 23 (0-indexed)
    global_attn_layer_indices: List[int] = field(default_factory=lambda: [5, 11, 17, 23])
    
    # Init
    initializer_range: float = 0.02

@dataclass
class TrainingConfig:
    batch_size: int = 2
    gradient_accumulation_steps: int = 16 # Increase accum steps to keep effective batch size similar
    learning_rate: float = 3e-4
    max_steps: int = 100000
    warmup_steps: int = 1000
    logging_steps: int = 10
    eval_steps: int = 1000
    save_steps: int = 5000
    output_dir: Optional[str] = None
    device: str = "cuda"
    
    # Eval Harness
    eval_harness_steps: int = 10000
    eval_harness_tasks: List[str] = field(default_factory=lambda: ["hellaswag", "piqa"])

@dataclass
class DatasetConfig:
    dataset_name: str = "tinyshakespeare"
    dataset_path: Optional[str] = None
    tokenizer_path: str = "google/gemma-3-1b-pt" # User requested tokenizer path
    seq_length: int = 1024

@dataclass
class Gemma3Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
