import math
from typing import Optional, Tuple, List
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
try:
    from .config import ModelConfig
except ImportError:
    from config import ModelConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Gemma3RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # q, k: [Batch, Head, Seq, Dim]
    # cos, sin: [Seq, Dim]
    
    # Reshape cos, sin to [1, 1, Seq, Dim] to broadcast against q, k
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Gemma3MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.GELU() # Approximate tanh is implied often if not specified

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Gemma3Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.is_global = layer_idx in config.global_attn_layer_indices
        self.attention_window = config.attention_window_size if not self.is_global else None
        self.rope_theta = config.rope_theta_global if self.is_global else config.rope_theta_local

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK Norm
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = Gemma3RotaryEmbedding(
            self.head_dim, 
            max_position_embeddings=config.max_position_embeddings,
            base=self.rope_theta
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply QK Norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Attention Masking
        attn_mask = attention_mask
        is_causal = False
        
        # If we have a standard causal mask requirement
        # Global layers are fully causal.
        # Local layers are causal + window.
        
        if self.is_global:
            # Fully causal
            # If no external mask provided, we can use optimized is_causal=True
            if attn_mask is None:
                is_causal = True
            else:
                 # Expand mask for broadcasting if needed
                 pass 
        else:
            # Local sliding window
            # Construct mask: i >= j (causal) AND i - j <= window
            # We must pass this explicitly.
            # Create mask on the fly or assuming same device.
            # Using bool mask for SDPA (True = Keep? No, SDPA mask: True indicates values to participate? 
            # Docs: "Binary mask where True indicates that the corresponding position is allowed to attend.")
            # Verify: "attn_mask (...) – Float, byte, or boolean mask. ... For boolean mask, True indicates values to be computed, False indicates values to be ignored."
            # Wait, verify PyTorch version behavior. In 2.0+ usually True = Attend.
            # Old manual mask had 0.0 for keep and -inf for mask.
            
            # Let's use the explicit manual mask code but convert to boolean or use as float add mask?
            # SDPA supports float mask (added).
            
            # Recreate mask logic
            # Causal: i >= j
            # Window: i - j <= window
            
            indices = torch.arange(q_len, device=hidden_states.device)
            dist = indices.unsqueeze(1) - indices.unsqueeze(0) # start_pos - target_pos
            # dist represents i - j
            
            # Mask where dist < 0 (future, j > i) OR dist > window (too far past)
            # allowed: 0 <= dist <= window
            
            window_mask = (dist >= 0) & (dist <= self.attention_window)
            
            # (q_len, q_len)
            # Add dimensions for batch/head: (1, 1, q_len, q_len)
            attn_mask = window_mask.unsqueeze(0).unsqueeze(0)
            
            # If external attention_mask is provided, combine.
            # External mask usually 0 for keep, -inf for mask. Or boolean.
            if attention_mask is not None:
                # Convert our boolean window_mask to float if attention_mask is float
                # Assume attention_mask is float/additive as per typical transformers
                # This is tricky with SDPA. SDPA prefers strictly boolean alignment or float.
                # Let's assumes attention_mask is ignored for now in training loop (it is None usually for causal LM training)
                pass

        # SDPA handles GQA broadcasting automatically (Query: H, Key: 1 -> Broadcasts)
        
        # Dropout
        dropout_p = 0.0 # No dropout config? Assuming 0.

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states, 
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output

class Gemma3Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma3Attention(config, layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Gemma3Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Gemma3Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states = self.embed_tokens(input_ids)
        
        # Simple position_ids generation if not provided
        if position_ids is None:
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*args):
                        return module(*args)
                    return custom_forward
                
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)

        hidden_states = self.norm(hidden_states)
        return hidden_states

class Gemma3ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Gemma3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.model.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


    @property
    def device(self):
        return next(self.parameters()).device
        
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, labels=None, attention_mask=None):
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
