from nano_vllm.kv_cache_utils import KVCacheBlock

class BlockPool:
    """
    Block Pool manages which blocks are free or used. 
    """
    def __init__(self, num_gpu_blocks: int):
        self.num_gpu_blocks = num_gpu_blocks
        # All kv-cache blocks 
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks) 
        ]
        # Free Block Pool 
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

    
    def get_new_blocks(self, num_blocks:int) -> list[KVCacheBlock]:
        """ 
            Get new blocks from free block pool
                Note we do not check block cache in this function. 
                
                Args:
                    num_blocks: number of blocks to allocate. 
                
                Returns:
                    A list of new blocks. 
        """
        if num_blocks > self.free_block_queue.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool.")
        
        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
        
        for block in ret: 
            assert block.ref_cnt == 0
            block.ref_cnt += 1
        return ret

    def free_blocks(self, ordered_blocks: list[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their eviction priority, where the first block will be evicted first. 
        
        Args:
        ordered_blocks: A list of blocks to free ordered by their eviction priority. 
        """ 
        free_blocks = [] 
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                free_blocks.append(block)
        
        self.free_block_queue.append_n(free_blocks)
        
class FreeKVCacheBlockQueue: 
    
    def __init__(self, blocks: list[KVCacheBlock]) -> None: 
        self.num_free_blocks = len(blocks)
        
        # Initialize doubly links of consequtive blocks 
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i-1] 
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i+1] 
        
        # Create a dummy head and tail block for the doubly linked list to reduce branching in the code. 
        # Our implementation guranteed that the dummy head and tail are NEVER got popped, so we could safely assume each real blocks in the queue has prev and next blocks. 
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1) 
        
        if self.num_free_blocks > 0: 
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head


    def popleft_n(self, n:int) -> list[KVCacheBlock]:
        if n == 0:
            return [] 
        assert self.num_free_blocks >= n
        
        self.num_free_blocks -= n
        
        # Pop n blocks from the head of the list 
        curr_block = self.fake_free_list_head.next_free_block
        ret = []
        for _ in range(n):
            assert curr_block is not None
            ret.append(curr_block)
            
            curr_block = curr_block.next_free_block
            
        if curr_block is not None:
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        
        return ret
    
    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        
        """Put a list of blocks back at the free list
        
        Args:
            blocks: List of blocks to append 
        """
        if len(blocks) == 0:
            return 
        
        prev_block = self.fake_free_list_tail.prev_free_block
        
        assert prev_block is not None, (
            "prev_free_block of fake_free_list_tail should always exist."
        )
        
        for block in blocks: 
            prev_block.next_free_block = block
            block.prev_free_block = prev_block
            
            prev_block = block 
        
        # Connect fake_free_list_tail after prev_block
        prev_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = prev_block 
            
        self.num_free_blocks += len(blocks) 


    def get_num_free_blocks(self):
        return self.num_free_blocks