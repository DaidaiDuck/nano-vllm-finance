import json
from pathlib import Path
import random

def generate_synthetic_prompts(
        num_prompts: int,
        prompt_len_range: tuple[int, int],
        tokenizer, 
        seed: int = 42,
) -> list[str]:
    """生成 synthetic prompts (M1-M2 用)"""
    random.seed(seed)

    base_prompts = [
        "Tell me about artificial intelligence and its impact on society.",
        "Explain quantum computing in simple terms.",
        "What are the benefits of regular exercise?",
        "How does photosynthesis work?",
        "Describe the history of the internet.",
    ]

    # 通过重复和截断生成不同长度
    prompts = []

    for i in range(num_prompts):
        base = base_prompts[i % len(base_prompts)] 
        target_len = random.randint(*prompt_len_range)  #TODO: 这行代码啥意思? 

        # 重复 base 直到达到目标长度
        current = base 
        while len(tokenizer.encode(current)) < target_len: 
            current = current + " " + base 
        
        # 截断到目标长度
        tokens = tokenizer.encode(current)[:target_len] 
        prompt = tokenizer.decode(tokens)
        prompts.append(prompt)
    
    return prompts 

def load_sharegpt(num_prompts: int = 100) -> list[str]:
    """加载 ShareGPT 真实对话 (M3+ 用)"""
    # 实际项目中下载 ShareGPT 数据
    # 这里简化
    try:
        from datasets import load_dataset
        ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
        prompts = []
        for item in ds.select(range(num_prompts)):
            if 'conversations' in item and len(item['conversations']) > 0:
                prompts.append(item['conversations'][0]['value'])
        return prompts
    except Exception as e:
        print(f"Failed to load ShareGPT: {e}, using synthetic")
        return None