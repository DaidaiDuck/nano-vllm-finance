# nano_vllm/paged_engine.py
from typing import Iterator
import types 
from nano_vllm.block_pool import BlockPool
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nano_vllm.kv_cache_manager import KVCacheManager
from nano_vllm.paged_attention import ENGINE_CTX, paged_attn_forward
from nano_vllm.sampler import Sampler
from nano_vllm.types import SamplingParams, RequestOutput, Request

class PagedEngine:
    """
    M3 Single Request Inference Engine.

    Achieve Paged Attention 
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
        # TODO: Real and max gpu blocks will be calculated in M4 to maximize concurrency and throughput. 
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
            attn.forward = types.MethodType(paged_attn_forward, attn) 


    def _format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(self._format_prompt(prompt)))

    def generate(
            self,
            params: SamplingParams,
            request: Request
    ) -> RequestOutput:
        # Apply chat template (see count_prompt_tokens / _format_prompt)
        text = self._format_prompt(request.prompt)
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to("cuda") #  # shape [1, seq]
        prompt_len = input_ids.shape[1] 

        # Prefill
        # KVCache Manager allocate slots in prefill stage
        self.kv_cache_manager.allocate_slots(request, prompt_len, self.block_size)
        ENGINE_CTX.block_table = self.kv_cache_manager.get_block_table(request)
        ENGINE_CTX.cache_seqlens = torch.tensor([0], dtype=torch.int32, device="cuda")

        with torch.no_grad():
            outputs = self.model(
                input_ids = input_ids,
                position_ids = torch.arange(prompt_len, device="cuda").unsqueeze(0), 
            )
        request.num_computed_tokens += prompt_len
        
        # Sample next token
        logits = outputs.logits[0,-1,:]
        next_token = self.sampler.sample(logits, params)
        output_ids = [next_token]

        # Decode loop
        finished_reason = "length"
        finished = True
        for _ in range(params.max_tokens - 1):
            if next_token == self.tokenizer.eos_token_id:
                finished_reason = "stop"
                break
            
            self.kv_cache_manager.allocate_slots(request, 1, self.block_size)
            ENGINE_CTX.block_table = self.kv_cache_manager.get_block_table(request)
            ENGINE_CTX.cache_seqlens = torch.tensor([request.num_computed_tokens], dtype=torch.int32, device="cuda") 

            with torch.no_grad():
                outputs = self.model(
                    input_ids = torch.tensor([[next_token]], device="cuda"), # Convert to tensor shape [batch_size, seq_len]
                    position_ids = torch.tensor([[request.num_computed_tokens]],device="cuda"),
                )
    
            logits = outputs.logits[0, -1, :]
            next_token = self.sampler.sample(logits, params)
            output_ids.append(next_token)
            request.num_computed_tokens += 1 

        # Decode
        text = self.tokenizer.decode(output_ids)

        # Free blocks in the request for future use 
        self.kv_cache_manager.free(request) 

        return RequestOutput(
            request_id=request.id,
            prompt=request.prompt,
            text=text,
            token_ids=output_ids,
            finished=finished,
            finish_reason=finished_reason 
        )
    
    def generate_stream(
            self,
            params: SamplingParams, 
            request: Request,
    ) -> Iterator[int]: 
        """
        Streaming generation. Yield one token at a time.

        Usage: 
            for token in engine.generate_stream(prompt, params):
                print(token)
        """
        try:
            text = self._format_prompt(request.prompt)
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to("cuda")
            prompt_len = input_ids.shape[1]

            # Prefill
            # KVCache Manager allocate slots in prefill stage
            self.kv_cache_manager.allocate_slots(request, prompt_len, self.block_size)
            ENGINE_CTX.block_table = self.kv_cache_manager.get_block_table(request) 
            # [derek.sun] Before prefill, there is no token in cache. Hence we set [0] here. 
            # When prefill, KV Cache would write new K/Vs into [0: prompt_len]. 
            ENGINE_CTX.cache_seqlens = torch.tensor([0], dtype=torch.int32, device="cuda") 

            # === Prefill ===
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    # [derek.sun] positions_ids: [[ 0, 1, 2, ..., prompt_len-1 ]]
                    # position_ids need dtype to be int64/Long. 
                    position_ids= torch.arange(prompt_len, device="cuda").unsqueeze(0)
                )

            # update num_computed_tokens
            request.num_computed_tokens += prompt_len 
            logits = outputs.logits[0, -1, :]

            # Sample first token
            next_token = self.sampler.sample(logits, params)

            torch.cuda.synchronize()
            yield next_token # First token out. 

            # === Decode loop ===
            for _ in range(params.max_tokens - 1):
                if next_token == self.tokenizer.eos_token_id: 
                    break

                inp = torch.tensor([[next_token]], device="cuda")
                self.kv_cache_manager.allocate_slots(request, 1, self.block_size)
                ENGINE_CTX.block_table = self.kv_cache_manager.get_block_table(request) 
                ENGINE_CTX.cache_seqlens = torch.tensor([request.num_computed_tokens], dtype=torch.int32, device="cuda")

                with torch.no_grad():
                    outputs = self.model(
                        input_ids = inp,
                        # [derek.sun] position_ids should be the position of the new token, which is num_computed_tokens. 
                        position_ids = torch.tensor([[request.num_computed_tokens]], device="cuda") 
                    )
                # update num_computed_tokens
                request.num_computed_tokens += 1 
                # calcukate next token 
                logits = outputs.logits[0,-1,:]
                next_token = self.sampler.sample(logits, params)

                torch.cuda.synchronize()
                yield next_token
        finally:
            # Free blocks in the request for future use 
            self.kv_cache_manager.free(request) 




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
        
        outputs = []
        for prompt in prompts:
            self.counter += 1
            request = Request(str(self.counter), prompt)
            
            output = self.engine.generate(samplingParams, request)
            outputs.append(output)

        return outputs

    def generate_stream(
        self,
        prompt: str,
        samplingParams: SamplingParams | None = None,
    ) -> Iterator[int]:
        """SSE Generate: Yield one token each time. 
        """
        if samplingParams is None:
            samplingParams = SamplingParams()
        self.counter += 1
        request = Request(str(self.counter), prompt)
        yield from self.engine.generate_stream(samplingParams, request)

    def count_prompt_tokens(self, prompt: str) -> int:
        """Count prompt prefill token number"""
        return self.engine.count_prompt_tokens(prompt)
    

    
    


        
        
    
