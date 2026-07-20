

from nano_vllm.core.sampler import Sampler
from nano_vllm.paged.kv_cache_manager import KVCacheManager
from nano_vllm.paged.paged_attention import ENGINE_CTX
from nano_vllm.core.types import Request, SamplingParams,SchedulerOutput, ModelRunnerOutput
import torch


class GPUModelRunner:

    def __init__(self, model, sampler:Sampler, kv_cache_manager:KVCacheManager, block_size):
        self.model = model
        self.sampler = sampler
        self.kv_cache_manager = kv_cache_manager
        self.block_size = block_size

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput: 
        # 1. Prepare inputs
        req_ids, input_ids, positions, logits_indices = self._prepare_inputs(scheduler_output) 

        # 2. GPU Forward
        with torch.no_grad(): 
            hidden = self.model.model(input_ids=input_ids, position_ids=positions).last_hidden_state # [1, total, H]
        
        # 3. Get last token's hidden states of each request 
        last_hidden = hidden[0, logits_indices]     # [num_reqs, H] 
        logits = self.model.lm_head(last_hidden)    # [num_reqs, vocab_size]
        
        # 4. Sample 
        sampling_params_list = [self.requests[req_id].sampling_params for req_id in req_ids]
        sampled = self.sampler.sample(logits, sampling_params_list)       # [num_reqs] 

        return ModelRunnerOutput(
            sampled_token_ids = [[t] for t in sampled], # list[list[int]]
            req_to_index= {req_id: i for i, req_id in enumerate(req_ids)}
        )
        

    def _prepare_inputs(self, scheduler_output:SchedulerOutput): 
        """
        Assume 3 requests. block_size = 4
        req1: Prefill. prompt_len: 4. num_computed = 0. num_scheduled = 4.
        req2: Decode. num_computed = 5. num_scheduled = 1. 
        req3: Decode. num_computed = 9. num_scheduled = 1.
        
        num_scheduled_tokens = [4, 1, 1]
        seqlens_q = [4, 1, 1]  The seq_len of q = num_scheduled_tokens. 
        cu_seqlens_q = [0, 0+4, 4+1, 5+1] = [0, 4, 5, 6]  
        seq_lens_k = [0 + 4, 5 + 1, 9 + 1] = [4, 6, 10] The seq_len of k = num_computed_tokens + num_scheduled_tokens. 
        positions = [0,1,2,3, 5, 9] 
        input_ids = [req1's 4 prompt token ids, req2's new token id, req3's new token id] 

        slot_mapping. Assume block_table: req1 = [2], req2=[7, 3] req3 = [5, 8, 1]
        req1 pos0..3 -> block_id: 2. offset: 0..3. -> slot = block_id * block_size + offset = 2 * 4 + 0..3 = 8,9,10,11
        req2 pos5 -> block_id: 3 offset: 1. -> slot = 3 * 4 + 1 = 13
        req3 pos9 -> block_id: 1 offset: 1. -> slot = 1 * 4 + 1 = 5 
        slot_mapping = [8,9,10,11, 13, 5] 

        hidden: shape = [6, H].
        logits_indices = the line number of the last token of each request in the hidden states = cu_seqlens_q[1:] - 1 = [3,4,5] 
        """
        reqs: list[Request] = (scheduler_output.scheduled_new_reqs + scheduler_output.scheduled_resumed_reqs + scheduler_output.scheduled_running_reqs)
        self.requests: dict[str, Request] = {req.request_id: req for req in reqs}
        
        req_ids = list(scheduler_output.num_scheduled_tokens)   # Get all request ids in this step
        # input_ids: the batch's new tokens' ids
        # positions: tokens' positions/index in its request
        # slot_mapping: the position to put the token's kv in KVCache
        input_ids, positions, slot_mapping = [], [], []
        # cu_seqlens_q: cumulative seq_lens of q
        # seq_lens_k: the history keys each forward will look at = num_computed + num_scheduled tokens of each request in batch 
        # block_tables: block_ids of each request in batch
        cu_seqlens_q, seq_lens_k, block_tables = [0], [], []

        for rid in req_ids:
            req:Request = self.requests[rid] 
            num_scheduled = scheduler_output.num_scheduled_tokens[rid]
            start = req.num_computed_tokens
            block_ids = [block.block_id for block in self.kv_cache_manager.req_to_blocks[rid]]

            for j in range(num_scheduled): 
                pos = start + j
                input_ids.append(req._all_token_ids[pos]) 
                positions.append(pos)
                block_id = block_ids[pos // self.block_size]
                offset = pos % self.block_size 
                kvcache_index = block_id * self.block_size + offset
                slot_mapping.append(kvcache_index)
            cu_seqlens_q.append(cu_seqlens_q[-1] + num_scheduled) 
            seq_lens_k.append(start + num_scheduled) 
            block_tables.append(block_ids) 
        
        logits_indices =  [c - 1 for c in cu_seqlens_q[1:]]  # the line number of the last token of each request in the hidden states

        # Convert to tensor
        input_ids = torch.tensor([input_ids], dtype=torch.int64, device="cuda") 
        positions = torch.tensor([positions], dtype=torch.int64, device="cuda") 
        ENGINE_CTX.cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda") 
        ENGINE_CTX.seq_lens_k = torch.tensor(seq_lens_k, dtype=torch.int32, device="cuda") 
        # Pad the block_table to shape [batch, max_blocks] 
        max_blocks = max(len(block_ids) for block_ids in block_tables)
        # Convert block_tables [[block0], [block1, block2], [block3, block5]] to [[block0, 0], [block1, block2], [block3, block5]]
        block_table = [block_ids + [0] * (max_blocks - len(block_ids)) for block_ids in block_tables] # Use 0 for padding. 
        ENGINE_CTX.block_table = torch.tensor(block_table, dtype=torch.int32, device="cuda") 
        ENGINE_CTX.slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64, device="cuda")
        ENGINE_CTX.max_seqlen_q = max(scheduler_output.num_scheduled_tokens.values()) 
        ENGINE_CTX.max_seqlen_k = max(seq_lens_k)

        logits_indices = torch.tensor(logits_indices, dtype=torch.int64, device="cuda") 
        return req_ids, input_ids, positions, logits_indices





