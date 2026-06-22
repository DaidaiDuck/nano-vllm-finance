# nano_vllm/generation.py
import token

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nano_vllm import Sampler, SamplingParams 

def generate(
        model, 
        tokenizer, 
        prompt: str, 
        params: SamplingParams
):
    """
    自己实现的 generation:
    - prefill 一次
    - decode 循环
    - 用 HF 自带的 KV cache (use_cache=True)
    """
    sampler = Sampler()
    # 1. Tokenize 
    messages = [{"role":"user", "content": prompt}] 
    # apply_chat_template 把 messages dict 格式化成模型训练时用的特殊文本（如 <|im_start|>user...），
    # tokenizer.encode 只接受字符串，所以必须先转成 text 再 encode
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,           # 只返回字符串，不做 tokenization（下一行手动 encode，避免重复）
        add_generation_prompt=False  # 不自动追加 <|im_start|>assistant\n（此处手动控制）
    )
    # return_tensors="pt": 返回 PyTorch tensor 而非 list；.to("cuda"): 移到 GPU，与模型同设备
    input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")

    # 2. Prefill (一次 forward 处理整个 prompt)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True) # 用HF自己的cache

    past_key_values = outputs.past_key_values # KV cache 
    # logits shape: [batch, seq_len, vocab_size]；[0] 取第0条，[-1] 取最后一个 token 位置，[:] 取全部词表分数
    next_token_logits = outputs.logits[0, -1, :]
    
    # 3. Sample first token
    next_token = sampler.sample(next_token_logits, params)
    output_ids = [next_token]

    # 4. Decode loop
    for _ in range(params.max_tokens - 1):
        # 检查 EOS
        if next_token == tokenizer.eos_token_id:
            break; 

        # 生成下一个token的logits
        # next_token 是 int（token id），需包装成 [1,1] 的 tensor（batch=1, seq_len=1）才能喂给模型
        inputs = torch.tensor([[next_token]], device="cuda")
        # no_grad: 推理不需要反向传播，关闭梯度记录，节省约一半显存并加速
        with torch.no_grad():
            outputs = model(
                input_ids = inputs,
                past_key_values=past_key_values,
                use_cache=True
            )
        
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[0,-1,:]

        # 用logits sample出下一个token
        next_token = sampler.sample(next_token_logits, params)
        output_ids.append(next_token)


    # 5. Return: Decode tokens to string
    return tokenizer.decode(output_ids, skip_special_tokens=True) 

def main():
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    print("Generating...")
    params = SamplingParams(temperature=0.0, max_tokens=50) # Greedy

    prompts = [
        "Hello, who are you?",
        "What is 2 + 2?",
    ]

    for prompt in prompts:
        response = generate(model, tokenizer, prompt, params)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")

    if __name__ == "__main__":
        main()
