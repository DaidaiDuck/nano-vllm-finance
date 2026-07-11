from numpy import block

import pytest
import torch

flash_attn = pytest.importorskip("flash_attn") 
from flash_attn import flash_attn_with_kvcache 

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA" 
)

N_HEADS = 4     # Query heads
N_KV = 2        # KV heads
HEAD_DIM = 64   # Must be a multiple of 8. 
BLOCK_SIZE = 16 # How many tokens in one block 
NUM_BLOCKS = 8  
DTYPE = torch.bfloat16
DEV = "cuda" 

def _rand(*shape):
    return torch.randn(*shape, dtype=DTYPE, device=DEV) 

def _empty_cache():
    k = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, N_KV, HEAD_DIM, dtype=DTYPE, device=DEV) 
    v = torch.zeros_like(k) 
    return k, v

def _block_table(n_blocks):     # [1, n_blocks] int32 cuda 
    return torch.tensor([list(range(n_blocks))], dtype=torch.int32, device=DEV) 

def _seqlens(n):                # [1] int32 cuda
    return torch.tensor([n], dtype=torch.int32, device=DEV) 

def test_prefill_shape_and_append():
    seqlen = 10 
    q = _rand(1, seqlen, N_HEADS, HEAD_DIM)
    k = _rand(1, seqlen, N_HEADS, HEAD_DIM)
    v = _rand(1, seqlen, N_HEADS, HEAD_DIM)
    k_cache, v_cache = _empty_cache() 
    block_table = _block_table(2)
    out = flash_attn_with_kvcache(
        q=q, k=k, v=v, k_cache=k_cache, v_cache=v_cache, 
        cache_seqlens=_seqlens(0),   # cache has zero token at first
        block_table=block_table,
        casual=True,
    )
    
    assert out.shape == (1, seqlen, N_HEADS, HEAD_DIM) # Output shape 
    assert out.dtype == DTYPE 
    assert torch.allclose(k_cache[0, :seqlen], k[0], atol=1e-2)
    assert torch.allclose(v_cache[0, :seqlen], v[0], atol=1e-2) 

def test_decode_shape_after_prefill(): 
    k_cache, v_cache = _empty_cache() 
    block_table = _block_table(2) 
    # Prefill 10 tokens
    flash_attn_with_kvcache(
        q=_rand(1, 10, N_HEADS, HEAD_DIM),
        k=_rand(1, 10, N_KV, HEAD_DIM), 
        v=_rand(1, 10, N_KV, HEAD_DIM), 
        k_cache=k_cache, v_cache=v_cache,
        cache_seqlens=_seqlens(0), # Prefill starts with 0 token
        block_table=block_table, 
        casual=True, 
    )
    # Decode the 11st token 
    out = flash_attn_with_kvcache(
        q=_rand(1, 1, N_HEADS, HEAD_DIM), # In decode stage, decode one token each step. 
        k=_rand(1, 1, N_KV, HEAD_DIM), 
        v=_rand(1, 1, N_KV, HEAD_DIM), 
        k_cache=k_cache, v_cache=v_cache,
        cache_seqlens=_seqlens(10),       # 10 tokens have been prefilled and cached. 
        block_table=block_table,
        causal=True,
    )
    assert out.shape == [1, 11, N_HEADS, HEAD_DIM]







