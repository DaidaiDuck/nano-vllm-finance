# nano_vllm/simple/engine.py -- M1 & M2 ONLY (HF DynamicCache / MyKVCache). M3+ live in nano_vllm/paged/.
from typing import Iterator

from nano_vllm.simple.kv_cache import MyKVCache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nano_vllm.core.sampler import Sampler
from nano_vllm.core.types import SamplingParams, RequestOutput

class SimpleEngine:
    """
    单请求 generation 引擎 —— **M1 + M2 专用**。
    M3 起改用 nano_vllm/paged/engine.py 的 PagedEngine(PagedAttention),本文件不再演进。

    use_custom_cache 选择 KV cache 后端, 用于 M1 vs M2 的公平对比:
      True  (M2) -> 自研的预分配 MyKVCache
      False (M1) -> HuggingFace 自带的 DynamicCache (torch.cat 增长)
    """
    def __init__(self, model_name: str, use_custom_cache: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        self.model.eval()
        self.sampler = Sampler()
        self.use_custom_cache = use_custom_cache
        if use_custom_cache:
            config = self.model.config
            self.my_kv_cache = MyKVCache(
                num_layers = config.num_hidden_layers,
                max_seq_len = 8192,
                num_kv_heads = config.num_key_value_heads,
                head_dim = getattr(
                    config, "head_dim", config.hidden_size // config.num_attention_heads # Must use integer division //.
                ),
                dtype=self.model.dtype,
                device=self.model.device,
            )
        else:
            self.my_kv_cache = None  # M1 mode: HF creates its own DynamicCache

    def _format_prompt(self, prompt: str) -> str:
        """套上 chat template, 返回真正喂给模型的字符串。"""
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def count_prompt_tokens(self, prompt: str) -> int:
        """返回套上 chat template 后真实的 prefill token 数。

        注意: 这比 tokenizer.encode(prompt) 多出 template 的特殊标记
        (<|im_start|>user ... <|im_end|><|im_start|>assistant), 才是模型
        实际处理的输入长度。
        """
        return len(self.tokenizer.encode(self._format_prompt(prompt)))

    def generate(
            self,
            prompt: str, 
            params: SamplingParams,
            request_id: str = "0"
    ) -> RequestOutput:
        """生成一个请求的输出"""

        # Select KV cache backend: custom MyKVCache (M2) or HF DynamicCache (M1).
        # M2: reset our pre-allocated cache and hand it to the model.
        # M1: pass None so HF creates a DynamicCache, then re-read it each step.
        if self.use_custom_cache:
            self.my_kv_cache.reset()
            past_key_values = self.my_kv_cache
        else:
            past_key_values = None

        # Apply chat template (see count_prompt_tokens / _format_prompt)
        text = self._format_prompt(prompt)
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to("cuda")

        # Prefill
        with torch.no_grad():
            outputs = self.model(
                input_ids = input_ids,
                past_key_values = past_key_values,
                use_cache = True,
            )
        if not self.use_custom_cache:
            past_key_values = outputs.past_key_values

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

            with torch.no_grad():
                outputs = self.model(
                    input_ids = torch.tensor([[next_token]], device="cuda"), # 转化成[batch_size,seq_len]
                    past_key_values = past_key_values,
                    use_cache=True
                )
            if not self.use_custom_cache:
                past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1, :]
            next_token = self.sampler.sample(logits, params)
            output_ids.append(next_token)

        # Decode
        text = self.tokenizer.decode(output_ids)

        return RequestOutput(
            request_id=request_id,
            prompt=prompt,
            text=text,
            token_ids=output_ids,
            finished=finished,
            finish_reason=finished_reason 
        )
    
    def generate_stream(
            self,
            prompt: str, 
            params: SamplingParams, 
            request_id: str = "0", 
    ) -> Iterator[int]: 
        """
        Streaming generation. Yield one token at a time.

        Usage: 
            for token in engine.generate_stream(prompt, params):
                print(token)
            
        For benchmarking TTFT/TPOT:
            start = time.perf_counter()
            first_token_time = None 
            for token in engine.generate_stream(prompt, params): 
                if first_token_time is None:
                    first_token_time = time.perf_counter()
            ttft = first_token_time - start 
        """

        # Select KV cache backend (see generate() for the M1/M2 rationale).
        if self.use_custom_cache:
            self.my_kv_cache.reset()
            past_key_values = self.my_kv_cache
        else:
            past_key_values = None

        # Apply chat template.
        # Why apply_chat_template instead of encode(messages):
        #   instruct models are trained on a specific conversation format with
        #   special tokens (e.g. <|im_start|>user ... <|im_end|>). encode() takes
        #   raw text only and would omit those markers, so the model would "continue"
        #   the text instead of "answering" it. apply_chat_template wraps messages
        #   into the model's own format (stored in the tokenizer config).
        # tokenize=False:
        #   return the formatted string only (don't encode yet), so the next line
        #   can encode it with our own options (return_tensors="pt", .to("cuda")).
        # add_generation_prompt=True:
        #   append the assistant turn opener (e.g. <|im_start|>assistant\n) so the
        #   model knows it's its turn to speak. Required for inference; only set
        #   False when building training data.
        text = self._format_prompt(prompt)
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to("cuda")

        # === Prefill ===
        # no_grad: inference never calls .backward(), so disable autograd to skip
        # building the computation graph. Saves a lot of memory and is slightly
        # faster. (torch.inference_mode() is an even more aggressive equivalent.)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        if not self.use_custom_cache:
            past_key_values = outputs.past_key_values
        # logits shape is [batch, seq_len, vocab_size].
        #   0  -> first (and only) sequence in the batch
        #   -1 -> the last position; the model emits a next-token distribution for
        #         every position, but we only want the token that follows the whole
        #         prompt, i.e. the last one
        #   :  -> the full vocab vector (a score per token) for the sampler
        logits = outputs.logits[0, -1, :]

        # Sample first token
        next_token = self.sampler.sample(logits, params)

        # CUDA kernels run asynchronously: model(...) returns once the work is
        # queued, not when the GPU finishes. synchronize() blocks the CPU until the
        # GPU is actually done, so the TTFT timestamp taken right after yield is
        # accurate. Only needed for latency benchmarking; skip it in production
        # since it gives up CPU/GPU overlap.
        torch.cuda.synchronize()
        yield next_token # First token out. 

        # === Decode loop ===
        for _ in range(params.max_tokens - 1):
            if next_token == self.tokenizer.eos_token_id: 
                break

            inp = torch.tensor([[next_token]], device="cuda")
            with torch.no_grad():
                outputs = self.model(
                    input_ids = inp,
                    past_key_values = past_key_values,
                    use_cache=True,
                )
                if not self.use_custom_cache:
                    past_key_values = outputs.past_key_values
                logits = outputs.logits[0,-1,:]
                next_token = self.sampler.sample(logits, params)

                torch.cuda.synchronize()
                yield next_token
        




class LLM:
    """
    用户入口
    """
    def __init__(self, model: str, use_custom_cache: bool = True):
        self.engine = SimpleEngine(model, use_custom_cache=use_custom_cache)
        self.counter = 0

    def generate(
        self,
        prompts: str | list[str],
        samplingParams: SamplingParams | None = None
    ) -> list[RequestOutput]:
        """同步生成"""
        if isinstance(prompts,str):
            prompts = [prompts] 
        if samplingParams is None:
            samplingParams = SamplingParams() 
        
        outputs = []
        for prompt in prompts:
            self.counter += 1
            request_id = str(self.counter)
            output = self.engine.generate(prompt, samplingParams, request_id)
            outputs.append(output)

        return outputs

    def generate_stream(
        self,
        prompt: str,
        samplingParams: SamplingParams | None = None,
    ) -> Iterator[int]:
        """流式生成: 逐个 yield token id (单 prompt)。

        与 generate 不同, 这里只支持单个 prompt (流式无法交错多个请求),
        供 benchmark 测 TTFT/TPOT 使用。
        """
        if samplingParams is None:
            samplingParams = SamplingParams()
        self.counter += 1
        request_id = str(self.counter)
        yield from self.engine.generate_stream(prompt, samplingParams, request_id)

    def count_prompt_tokens(self, prompt: str) -> int:
        """真实 prefill token 数 (含 chat template 标记)。"""
        return self.engine.count_prompt_tokens(prompt)
    
    
    


        
        
    
