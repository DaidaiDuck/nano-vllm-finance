# nano_vllm/paged/engine.py -- M3 and later (PagedAttention). Supersedes simple/engine.py from M3 on.
from typing import Iterator
import types
from typing_extensions import deprecated
from nano_vllm.paged.block_pool import BlockPool
from nano_vllm.paged.gpu_model_runner import GPUModelRunner
from nano_vllm.paged.scheduler import ModelRunnerOutput, Scheduler
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nano_vllm.paged.kv_cache_manager import KVCacheManager
from nano_vllm.paged.paged_attention import ENGINE_CTX, paged_attn_forward, batch_attn_forward
from nano_vllm.core.sampler import Sampler
from nano_vllm.core.types import SamplingParams, RequestOutput, Request

class PagedEngine:
    """
    M4: Multi Request Inference Engine
    Achieve Continous Batching. 
    """
    def __init__(
            self, 
            model_name: str,
            block_size: int = 256, # flash_attn_with_kvcache requires block_size % 256 == 0 
            device: str = "cuda" 
            ):
        # Load tokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load model 
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        self.model.eval()
        # Load sampler 
        self.sampler = Sampler()

        # Load model configuration
        cfg = self.model.config
        num_layers = cfg.num_hidden_layers
        self.block_size = block_size
        # [derek.sun] In M3, set num_blocks to 10000. 10000 is enough for M3 scenario.
        num_blocks = 1000
        num_kv_heads = cfg.num_key_value_heads 
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads) 
        dtype= self.model.dtype    # = bf16


        # Initialize KV Cache and Block Pool 
        self.k_cache = torch.zeros(num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)  
        self.v_cache = torch.zeros_like(self.k_cache) 
        self.block_pool = BlockPool(num_blocks)
        self.kv_cache_manager = KVCacheManager(self.block_pool, block_size)

        # Replace
        for i, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            # Inject physical KV Cache for each layer
            attn.k_cache = self.k_cache[i]
            attn.v_cache = self.v_cache[i] 
            # Replace HuggingFaces forward method into custom method paged_attn_forward 
            attn.forward = types.MethodType(batch_attn_forward, attn) 

        max_num_scheduled_tokens = 4096 # TODO: Fix it for now. 
        self.scheduler = Scheduler(block_size, self.kv_cache_manager, self.model.config, max_num_scheduled_tokens)
        self.model_runner = GPUModelRunner(self.model, self.sampler, self.kv_cache_manager, self.block_size, )

    def _format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(self._format_prompt(prompt)))

    def step(self):
        # 1. Schedule num_new_tokens for each request. 
        scheduler_output = self.scheduler.schedule() 
        if scheduler_output.total_num_scheduled_tokens == 0:
            return [] 

        # 2. Execute GPU Forward according to scheduler_output. 
        model_output = self.model_runner.execute_model(scheduler_output)
        
        # 3. Update from scheduler_output and model_output
        steps_output = self.scheduler.update_from_output(scheduler_output, model_output)

        return steps_output


    def add_request(self, request:Request):
        request.eos_token_id = self.tokenizer.eos_token_id 
        self.scheduler.add_request(request)

    def has_unfinished_requests(self):
        return bool(self.scheduler.waiting or self.scheduler.running)
    
class LLM:
    """
    User Entrance
    """
    def __init__(self, model_name: str):
        self.engine = PagedEngine(model_name=model_name)
        self.counter = 0

    def generate(
        self,
        prompts: str | list[str],
        samplingParams: SamplingParams | None = None
    ) -> list[RequestOutput]:
        """
            Wait until all requests are generated. 
        """
        if isinstance(prompts,str):
            prompts = [prompts] 
        if samplingParams is None:
            samplingParams = SamplingParams() 
        

        # 1. Add all requests to the waiting queue first. 
        for prompt in prompts:
            self.counter += 1
            prompt_token_ids = self.engine.tokenizer.encode(self.engine._format_prompt(prompt))
            request = Request(str(self.counter), prompt_token_ids, samplingParams)
            self.engine.add_request(request) 
            
        outputs = []
        while self.engine.has_unfinished_requests():
            # 2. One GPU Forward to process all requests. 
            steps_output: list[RequestOutput] = self.engine.step()

            for output in steps_output:
                if output.finished:
                    # 3. Add finished output to return outputs.
                    output.text = self.engine.tokenizer.decode(output.token_ids)
                    outputs.append(output)

        return outputs


    def count_prompt_tokens(self, prompt: str) -> int:
        """Count prompt prefill token number"""
        return self.engine.count_prompt_tokens(prompt)
    

    
    


        
        
    
