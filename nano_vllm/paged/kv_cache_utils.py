from dataclasses import dataclass 
from typing import Optional 

@dataclass(slots=True)
class KVCacheBlock:
    """KV-cache block metadata.""" 
    # block id, ranging from 0 to num_blocks-1
    block_id:int
    # Reference count
    ref_cnt: int = 0
    prev_free_block: Optional["KVCacheBlock"] = None 
    next_free_block: Optional["KVCacheBlock"] = None 
