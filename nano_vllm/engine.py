# nano_vllm/engine.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nano_vllm.sampler import Sampler
from nano_vllm.types import SamplingParams, RequestOutput

class SimpleEngine: 
    """
    M1 阶段的简单引擎: 单请求 generation
    用 HF 自带的 KV cache
    """
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )  
        self.model.eval() 
        self.sampler = Sampler() 
    
    def generate(
            self,
            prompt: str, 
            params: SamplingParams,
            request_id: str = "0"
    ) -> RequestOutput:
        """生成一个请求的输出"""

        # Apply chat template
        messages = [{"role":"user", "message":prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(text) 

        # Prefill 
        with torch.no_grad():
            outputs = self.model(
                input_ids = input_ids,
                use_cache=True 
            )
        past_key_values = outputs.past_key_values
        
        # Sample next token 
        logits = outputs[0,-1,:]
        next_token = self.sampler.sample(logits, params)
        output_ids = [next_token] 

        # Decode loop 
        finished_reason = "length"
        finished = false
        for _ in range(params.max_tokens - 1):
            if next_token == self.tokenizer.eos_token_id:
                finished_reason = "stop"
                finished = True
                break

            with torch.no_grad():
                outputs = self.model(
                    input_ids = [[next_token]], # 转化成[batch_size,seq_len] 
                    past_key_values = past_key_values,
                    use_cache=True #TODO:这里如果不加这行会怎么样? 
                )
            past_key_values = outputs.past_key_values
            logits = outputs[0, -1, :] 
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
            finished_reason=finished_reason 
        )
        

class LLM:
    """
    用户入口
    """
    def __init__(self, model: str):
        self.engine = SimpleEngine(model)
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
        
    
