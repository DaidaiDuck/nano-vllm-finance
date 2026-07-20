
import torch
from nano_vllm.paged.kv_cache_utils import KVCacheBlock
from nano_vllm.paged.block_pool import BlockPool
from nano_vllm.core.types import Request

class KVCacheManager:
    def __init__(
        self, 
        block_pool:BlockPool,
        block_size:int,
    ):
        self.req_to_blocks: dict[str, list[KVCacheBlock]] = {} 
        self.block_pool = block_pool
        self.block_size = block_size 
    
    def allocate_slots(
        self,
        request: Request, 
        num_new_tokens: int,
    ) -> list[KVCacheBlock]:
        total_tokens = request.num_computed_tokens + num_new_tokens 
        blocks_needed = (total_tokens + self.block_size - 1) // self.block_size
        blocks_allocated:list[KVCacheBlock] = self.req_to_blocks.get(request.request_id, []) # When new request comes, initialize its block table to []. 
        num_new_blocks_needed = blocks_needed - len(blocks_allocated)

        if num_new_blocks_needed <= 0: 
            # No need to allocate a new block because the last existing one has not been filled. 
            return [] 
        
        if num_new_blocks_needed > self.block_pool.free_block_queue.get_num_free_blocks():
            # No enough free blocks to allocate for this request. 
            # Return None. 
            return None 
        
        new_blocks = self.block_pool.get_new_blocks(blocks_needed - len(blocks_allocated))
        self.req_to_blocks[request.request_id] = blocks_allocated + new_blocks
        
        return new_blocks
        

    def free(
        self, 
        request:Request
    ):
        """ Free all blocks in the request.
            Args:
                request:Request
        """
        blocks_allocated = self.req_to_blocks.pop(request.request_id, None) 
        if blocks_allocated:
            # Free blocks of the request in the block pool.
            self.block_pool.free_blocks(blocks_allocated)
    

    