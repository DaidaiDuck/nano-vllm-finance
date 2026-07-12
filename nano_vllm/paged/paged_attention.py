import torch 
from flash_attn import flash_attn_with_kvcache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


class AttnContext:          # ENGINE_CTX (Similar to vLLM's attn_metadata)
    block_table = None      # [1, max_blocks] 
    cache_seqlens = None    # [1] 

ENGINE_CTX = AttnContext()  # Set ENGINE_CTX before each GPU forward step. 


def paged_attn_forward(
    self,                                       # Qwen2Attention's instance 
    hidden_states: torch.Tensor,                # shape: [batch, seq, hidden_size]  
    position_embeddings: tuple[torch.Tensor, torch.Tensor], 
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]: 
    """
    [derek.sun]
    Need to maintain same inputs and outputs as transformers/models/qwen2/modeling_qwen2.py Qwen2Attention#forward.
    Because we still use HuggingFace's model. So we need to maintain the same input and output parameters as Qwen2Attention#forward 
    for custom paged_attn_forward method. 
    """
    input_shape = hidden_states.shape[: -1]             # [batch, seq] 
    hidden_shape = (*input_shape, -1, self.head_dim)    # [batch, seq, -1, head_dim] 

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # [batch, num_heads, seq, head_dim] 
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)      # [batch, num_kv, heads, seq, head_dim] 
    values_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) 

    # Apply RoPE 
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)   # [batch, num_heads, seq, head_dim]

    q = query_states.transpose(1, 2)     # [batch, seq, num_heads, head_dim]
    k = key_states.transpose(1, 2)       # [batch, seq, num_kv_heads, head_dim]
    v = values_states.transpose(1, 2)    # [batch, seq, num_kv_heads, head_dim]

    # Use flash_attn_with_kvcache to write paged cache + calculate attention in just one step 
    attn_output = flash_attn_with_kvcache(
        q = q, 
        k_cache = self.k_cache,
        v_cache = self.v_cache, 
        k = k, 
        v = v, 
        cache_seqlens = ENGINE_CTX.cache_seqlens,
        block_table = ENGINE_CTX.block_table,
        causal = True, 
    ) # -> [batch, seq, num_heads, head_dim]

    attn_output = attn_output.reshape(*input_shape, -1).contiguous() # [batch, seq, hidden_size]
    attn_output = self.o_proj(attn_output)
    return attn_output, None 
