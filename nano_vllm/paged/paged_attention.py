import torch 
from flash_attn import flash_attn_varlen_func
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from typing_extensions import deprecated

class AttnContext:          # ENGINE_CTX (Similar to vLLM's attn_metadata)
    block_table = None      # [batch_size, max_blocks] 
    cu_seqlens_q = None     # [batch+1]
    seq_lens_k = None       # [batch]
    slot_mapping = None     # [sum of num_scheduled_tokens] 
    max_seqlen_q = None     # int 
    max_seqlen_k = None     # int


ENGINE_CTX = AttnContext()  # Set ENGINE_CTX before each GPU forward step. 


def batch_attn_forward(
    self,
    hidden_states: torch.Tensor,         # shape: [batch, seq, hidden_size]  
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    **kwargs 
):
    """
    Use batch_attn_forward since M4. 
    """
    input_shape = hidden_states.shape[:-1]              # [batch, seq]
    hidden_shape = (*input_shape, -1, self.head_dim)    # [batch, seq, -1, head_dim]

    # 1. Calculate q,k,v 
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1,2) # [batch, num_heads, seq, head_dim]
    key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1,2) # [batch, num_kv_heads, seq, head_dim]
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1,2) # [batch, num_kv_heads, seq, head_dim]

    # Apply RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin) # [batch, num_heads, seq, head_dim]

    q = query_states.transpose(1,2)[0]     # [seq, num_heads, head_dim]
    k = key_states.transpose(1,2)[0]       # [seq, num_kv_heads, head_dim]
    v = value_states.transpose(1,2)[0]     # [seq, num_kv_heads, head_dim]
    
    # 2. Write new token's KV into KVCache.
    ctx = ENGINE_CTX    
    n_kv = k.shape[1]
    k_flat = self.k_cache.view(-1, n_kv, self.head_dim) # Convert cache shape from [num_blocks, block_size, n_kv_heads, head_dim] to [num_blocks * block_size, n_kv_heads, head_dim]
    v_flat = self.v_cache.view(-1, n_kv, self.head_dim) 
    k_flat[ctx.slot_mapping] = k 
    v_flat[ctx.slot_mapping] = v 

    # 3. Execute GPU Forward 
    out = flash_attn_varlen_func(
        q=q,                   
        k=self.k_cache, # flash_attn_varlen_func will read history k and v from KVCache. 
        v=self.v_cache,
        cu_seqlens_q=ctx.cu_seqlens_q,
        seqused_k=ctx.seq_lens_k,
        max_seqlen_q=ctx.max_seqlen_q, 
        max_seqlen_k=ctx.max_seqlen_k,
        block_table=ctx.block_table,
        causal = True,
    ) # -> [batch, seq, num_heads, head_dim]

    # Convert output's shape from [batch, seq, num_heads, head_dim] to [batch, seq, hidden_size]
    attn_output = out.reshape(*input_shape, -1).contiguous() #TODO: when to use contiguous? 
    attn_output = self.o_proj(attn_output)
    return attn_output, None 
    

    