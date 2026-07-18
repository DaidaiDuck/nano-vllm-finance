
from collections import deque
from nano_vllm.paged.kv_cache_manager import KVCacheManager
from nano_vllm.core.types import RequestOutput, SchedulerOutput, Request, RequestStatus, ModelRunnerOutput, FinishReason

class Scheduler():
    
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
                token_ids = request._output_token_ids,
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

