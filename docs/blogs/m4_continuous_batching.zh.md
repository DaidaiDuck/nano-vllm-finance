# 系统设计
```
LLM (用户入口)
    └─ PagedEngine
        └─ Tokenizer (HF)
        └─ Model (HF) -> GPU Forward: batch_attn_forward (custom) 
        └─ BlockPool (custom)
        └─ KVCacheManager (custom) 
        └─ Sampler (custom)
        └─ GPUModelRunner (custom)
        └─ Scheduler (custom)
```

# 核心代码讲解
## PagedEngine改动: 
``` python
def step(self):
        # 1. Schedule num_new_tokens for each request. 
        scheduler_output = self.scheduler.schedule() 

        # 2. Execute GPU Forward according to scheduler_output. 
        model_output = self.model_runner.execute_model(scheduler_output)

        # 3. Update from scheduler_output and model_output
        steps_output = self.scheduler.update_from_output(scheduler_output, model_output)

        return steps_output
```
核心改动:PagedEngine放弃M3的单个请求的生成，改用scheduler规划每一步该GPU Forward哪些请求，每个请求计算多少个token. `step()`一共分为3步。
1. `self.scheduler.schedule()`的目的就是决定每个请求这一步要计算多少个新tokens. 
2. `self.model_runner.execute_model(scheduler_output)`就是去执行GPU Forward，得到这一步每一个请求sample出来的新token. 
3. 根据scheduler的调度结果scheduler_output和GPU Forward的结果model_output来更新每个请求的状态，包括更新num_computed_tokens，检查请求是否finish，如果finish则free请求的KV块等等。 

## Scheduler实现
### __init__
``` python    
    def __init__(
            self, 
            block_size: int, 
            kv_cache_manager:KVCacheManager, 
            model_config,
            max_num_scheduled_tokens:int,
    ):
        self.running: list[Request] = []                            # Running queue
        self.waiting: deque = deque()                               # Waiting queue: Use FIFO policy 
        self.block_size = block_size                                # block size
        self.kv_cache_manager = kv_cache_manager                    # KVCache Manager 
        self.max_num_running_seqs = 10                              # Max number of concurrent running requests 
        self.current_step = 0                       
        self.long_prefill_token_threshold = 4096                    # Max number of prefill tokens at a step 
        self.max_model_len = model_config.max_model_len             # Max context length the model can support: Max prompt + output length the model can support for a request. 
        self.max_num_scheduled_tokens = max_num_scheduled_tokens    # Token budget / Max number of scheduled tokens at a step  
        self.requests: dict[str, Request] = {}                      
```
初始化running, waiting队列。waiting这里我们用Python自带的deque，在需要抢占请求的时候我们优先抢占running队列的最后一个请求而不是第一个请求，因为最前面的请求完成度最高，而靠后面的请求完成度较低，被抢占导致之后需要重算的代价也较低。
我在M4使用了Chunked Prefill,设置long_prefill_token_threshold为4096,表示每一步一个请求最长能被Prefill的长度是4096。

```python
    def schedule(self) -> SchedulerOutput:
        self.current_step += 1 

        scheduled_new_reqs: list[Request] = []                      # New requests that are scheduled in this step from self.waiting. 
        scheduled_running_reqs: list[Request] = []                  # Requests that are already running (in self.running) in this step 
        scheduled_resumed_reqs: list[Request] = []                  # Requests that are resumed in this step from self.waiting. 
        preempted_reqs: set[Request] = set()                        # Requests that are preempted in this step 
        num_scheduled_tokens: dict[str, int] = {}                   # request_id -> number of new tokens scheduled for this request in this step
        token_budget = self.max_num_scheduled_tokens 


        # Step 1: Scheduled the RUNNING requests first. 
        req_index = 0 
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = request.num_tokens - request.num_computed_tokens
            num_new_tokens = min(num_new_tokens, token_budget, self.max_model_len - request.num_computed_tokens)

            if num_new_tokens == 0: 
                # Skip this request
                req_index += 1 
                continue 

            # Allocate newly blocks (if needed) for the request.
            while True: 
                # NOTE (derek.sun) allocate_slots return newly allocated blocks for the request.
                # Since we only need a new block when the last old block is filled, hence most of the time 
                # allocate_slots would return an empty list []. 
                new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens) 

                if new_blocks is not None:
                    # This request can be scheduled. 
                    # [] also count as a success since it means no need to allocate a block.
                    break 
                
                # There are no free blocks to schedule this request. 
                # Preempt the last running request. 
                # Use Recompute policy: Recompute num_tokens in the request. 
                # NOTE(derek.sun) Why preempt from the last instead of the begin?
                # The last running request is the running request with least progress. 
                # Hence the cost of recompute is minimal compared to other running requests in front. 
                # Moreover, it may happen that you just schedule the first request but find no enough free blocks.
                # Then you just preempt yourself, which does not make sense.
                # It makes more sense to preempt requests that are in the tail which have not been scheduled yet. 

                preempt_req = self.running[-1]

                if preempt_req is request:
                    # There is no more request to be preempted.
                    break  
                
                self.kv_cache_manager.free(preempt_req)
                self.running.pop() 
                preempt_req.status = RequestStatus.PREEMPTED
                preempted_reqs.add(preempt_req)
                self.waiting.append(preempt_req) 

            if new_blocks is None: 
                # This request cannot be scheduled. There are no free blocks and requests to be preempted. 
                # There is no need to consider other running requests. 
                break  
            
            # Schedule this request. 
            scheduled_running_reqs.append(request)
            num_scheduled_tokens[request.request_id] = num_new_tokens 
            token_budget -= num_new_tokens 
            
            # Prepare to schedule the next request. 
            req_index += 1 

        # Step 2. Schedule the WAITING requests.
        # Only schdule WAITING requests when there is no preempted request.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                request:Request = self.waiting[0] # O(1) 
                num_new_tokens = request.num_tokens - request.num_computed_tokens
                num_new_tokens = min(num_new_tokens, token_budget, self.long_prefill_token_threshold) 
                # NOTE(derek.sun) The only way num_new_tokens <= 0 is empty prompt. We reject empty prompt before add_request. 
                assert num_new_tokens > 0 

                new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)

                if new_blocks is None: 
                    # This request cannot been scheduled. No need to schedule the next request.
                    # NOTE(derek.sun) What if the first request is too big? If KVCacheManager fails to allocate slots, would it block following smaller requests? 
                    # 1. We use Chunked Prefill. We set long_prefill_token_threshold. 
                    # We use break instead of continue here:
                    #   1. because we need to preserve the FIFO policy.
                    #   2. because we do not want to skip large requests. 
                    break 

                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                self.waiting.popleft() # Do not use remove. 

                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                # Add request to the running queue 
                request.status = RequestStatus.RUNNING 
                self.running.append(request) 


        # 3. Summarize the output 
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs = scheduled_new_reqs, 
            scheduled_resumed_reqs = scheduled_resumed_reqs, 
            scheduled_running_reqs = scheduled_running_reqs,
            preempted_req_ids = {req.request_id for req in preempted_reqs},
            num_scheduled_tokens = num_scheduled_tokens,
            total_num_scheduled_tokens = total_num_scheduled_tokens,
        )

        return scheduler_output
```    
概括:
在schedule方法中，我们优先处理正在running的请求，如果没有足够的free blocks时我们抢占running队列的最后一个请求的blocks。如果我们抢占的请求刚好是正在处理的这个请求，则直接结束，M4我们只实现最简单的Recompute抢占策略。
接下来如果token_budget足够并且在处理running请求时没有发生抢占，我们会处理waiting队列中的请求。直到token_budget不够或者无法分到足够的free blocks则结束这一步。
最后我们总结这一步的结果并且输出scheduler_output。

核心知识:
`schedule`过程中我们没有Prefill和Decode这一概念，每个请求这一步要计算的新token数是该请求的当前的总token数减去这个请求已经计算的token数(`num_new_tokens = request.num_tokens - request.num_computed_tokens`). 对于Decode来说，一般num_new_tokens = 1, 对于Prefill来说，一般num_new_tokens取决于Prompt的token数，token_budget和chunked prefill的长度限制. 


```python
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput
    ): 
        sampled_token_ids = model_runner_output.sampled_token_ids 
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        outputs = []
        stopped_reqs = set()

        for req_id, num_token_scheduled in num_scheduled_tokens.items():
            request:Request = self.requests[req_id] 

            # 1. update num_computed_tokens
            request.num_computed_tokens += num_token_scheduled

            # 2. get current request's new token
            req_index = model_runner_output.req_to_index[req_id]
            new_tokens = sampled_token_ids[req_index]

            # 3. Check if request is doing chunked prefill.
            if request.num_computed_tokens < request.num_prompt_tokens:
                # The request has not finish prefill.
                continue 

            # 4. Append token to the request
            request._output_token_ids.extend(new_tokens)
            request._all_token_ids.extend(new_tokens)

            # 5. Check stop condition
            finish_reason:FinishReason = self._check_stop(request) # Check EOS or max tokens 

            # 6. If request is stopped, clean the request.
            if finish_reason is not None:
                request.status = RequestStatus.FINISHED
                request.finish_reason = finish_reason
                stopped_reqs.add(request)
                self.kv_cache_manager.free(request) 
            
            outputs.append(RequestOutput(
                request_id = req_id, 
                token_ids = list(request._output_token_ids), # Do not pass by reference here. 
                finished = finish_reason is not None,
                finish_reason = finish_reason,
            ))

        # 7.Remove stopped request from running.
        self.running = [r for r in self.running if r not in stopped_reqs] 

        return outputs 


    def add_request(self, request:Request):
        self.waiting.append(request) 
        self.requests[request.request_id] = request 
    
    def _check_stop(self, request:Request) -> FinishReason | None: 
        if request._output_token_ids[-1] == request.eos_token_id:
            return FinishReason.STOP
        elif len(request._output_token_ids) >= request.max_tokens:
            return FinishReason.LENGTH
        return None 
```
概括:

## GPUModelRunner实现

## batch_attn_forward的实现: 
```python
def batch_attn_forward(
    self,
    hidden_states: torch.Tensor,         # shape: [batch, seq, hidden_size]  
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    **kwargs 
):
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
```
举例:
使用(Qwen2.5-3B): hidden_size=2048, num_heads=16, num_kv_heads=2(GQA), head_dim=128. 假设block_size = 4. 

并发跑3个请求:
req1: Prefill. prompt:4 tokens. num_new_tokens:4
req2: Decode. num_computed_tokens: 5. num_new_tokens:1
req3: Decode. num_computed_tokens: 9. num_new_tokens:1 

1. hidden_states 和 position_embeddings 是什么? 
hidden_states 表示这一层的输入，shape是[batch, seq_len, hidden_size] = [1, 6, 2048].
M4把三个请求铺平，打包成一条长度为6 (4+1+1)的序列，所以batch=1, seq_len = total_tokens = 6. 至于kernel如何分辨哪些输入属于哪个请求则是根据cu_seqlens_q。其实对于多请求场景我们也可以采用多个batch，seq_len用batch里最大的请求的长度，多余的位置用padding替代。但因为请求之间长度差异较大，这种方法容易造成大量空padding，造成大量浪费，故不采纳。

position_embeddings是由外层Qwen2Model根据我传入的position_ids进行RoPE后的结果？shape和position_ids一样是[batch, seq_len,]


什么是Packed? 



# 其他知识
self._check_stop必须加上self才能成功调用. 


