import pytest
from nano_vllm.paged.block_pool import BlockPool
from nano_vllm.paged.kv_cache_manager import KVCacheManager
from nano_vllm.core.types import Request


"""
This file covers BlockPool allocation, free_blocks, 
"""
BLOCK_SIZE = 16

def _req(rid="r1", num_computed=0):
    return Request(id=rid, prompt="", num_computed_tokens=num_computed) 

def _free(pool: BlockPool):   # Get number of free blocks. 
    return pool.free_block_queue.get_num_free_blocks() 

def test_pool_allocate_and_free_roundtrip():
    pool = BlockPool(num_gpu_blocks=100) 
    assert _free(pool) == 100 
    blocks = pool.get_new_blocks(7)
    assert len(blocks) == 7
    assert len({b.block_id for b in blocks}) == 7 # Ensure no duplicate blocks
    assert all(b.ref_cnt == 1 for b in blocks)   # Ensure all ref_cnt = 1 after first allocation 
    assert _free(pool) == 93 
    pool.free_blocks(blocks) 
    assert all(b.ref_cnt == 0 for b in blocks)  # Ensure all ref_cnt = 0 after blocks are free.
    assert _free(pool) == 100 

def test_pool_out_of_block_raises(): 
    pool = BlockPool(num_gpu_blocks=5)
    with pytest.raises(ValueError, match="Cannot get"):
        pool.get_new_blocks(6) 
    
def test_pool_freed_block_is_reusable():
    pool = BlockPool(num_gpu_blocks=3) 
    pool.free_blocks(pool.get_new_blocks(3)) 
    assert len(pool.get_new_blocks(3)) == 3 
    assert _free(pool) == 0 

# --- KVCacheManager ---
def _mgr(num_blocks=100):
    return KVCacheManager(BlockPool(num_blocks), BLOCK_SIZE) 

def test_prefill_100_tokens_needs_7_blocks(): 
    mgr = _mgr(); req = _req()
    mgr.allocate_slots(req, 100, BLOCK_SIZE)
    assert len(mgr.req_to_blocks[req.id]) == 7   # ceil(100/7) = 7 
    assert _free(mgr.block_pool) == 93 

def test_decode_113th_token_triggers_8th_block(): 
    mgr = _mgr(); req = _req()
    mgr.allocate_slots(req, 100, BLOCK_SIZE) 
    assert len(mgr.req_to_blocks[req.id]) == 7 
    req.num_computed_tokens = 111
    mgr.allocate_slots(req, 1, BLOCK_SIZE) # Edge: Total = 111 + 1 = 112; 112 / 16 = 7 blocks 
    assert len(mgr.req_to_blocks[req.id]) == 7 
    req.num_computed_tokens = 112 
    mgr.allocate_slots(req, 1, BLOCK_SIZE) # ceil(113 / 7) = 8 blocks 
    assert len(mgr.req_to_blocks[req.id]) == 8 
    assert _free(mgr.block_pool) == 92 

def test_free_returns_all_blocks(): 
    mgr = _mgr(); req = _req()
    mgr.allocate_slots(req, 100, BLOCK_SIZE) 
    mgr.free(req) 
    assert req.id not in mgr.req_to_blocks 
    assert _free(mgr.block_pool) == 100   


def test_free_unknown_request_is_noop(): 
    mgr = _mgr(); req = _req()
    mgr.free(req)                   # The req has not been allocated. Test None circumstance protection. 
    assert _free(mgr.block_pool) == 100   











