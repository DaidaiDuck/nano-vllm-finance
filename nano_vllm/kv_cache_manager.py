
import torch
from nano_vllm.kv_cache_utils import KVCacheBlock
from nano_vllm.block_pool import BlockPool
from nano_vllm.types import Request

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
        block_size: int
    ):
        total_tokens = request.num_computed_tokens + num_new_tokens 
        blocks_needed = (total_tokens + block_size - 1) // block_size
        blocks_allocated:list[KVCacheBlock] = self.req_to_blocks.get(request.id, []) # When new request comes, initialize its block table to []. 
        
        if len(blocks_allocated) < blocks_needed: 
            new_blocks = self.block_pool.get_new_blocks(blocks_needed - len(blocks_allocated))
            self.req_to_blocks[request.id] = blocks_allocated + new_blocks
        

    def free(
        self, 
        request:Request
    ):
        """ Free all blocks in the request.
            Args:
                request:Request
        """
        blocks_allocated = self.req_to_blocks.pop(request.id, None) 
        if blocks_allocated:
            # Free blocks of the request in the block pool.
            self.block_pool.free_blocks(blocks_allocated)
    
    def get_block_table(
        self,
        request:Request
    ):
        """
        [derek.sun] FlashAttention needs block_table's shape to be [batch_size, num_blocks]. 
        FlashAttention also needs data type to be int32. 
        get_block_table converts [num_blocks] to [batch_size, num_blocks] with dtype=int32. 
        """
        block_ids = [block.block_id for block in self.req_to_blocks[request.id]]
        return torch.tensor([block_ids], dtype=torch.int32, device="cuda")
    

    