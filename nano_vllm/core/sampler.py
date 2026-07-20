# nano_vllm/sampler.py
import torch
from dataclasses import dataclass
from nano_vllm.core.types import SamplingParams

class Sampler:
    def sample(self, logits: torch.Tensor, sampling_params_list: list[SamplingParams]) -> list[int]: 
        """
        Sample the next token for each request in batch. 
        Args: 
            logits: [batch, vacab_size] 
        """
        return [self.sample_one(logits[i], sampling_params_list[i]) for i in range(logits.shape[0])]

    def sample_one(self, logits: torch.Tensor, params: SamplingParams) -> int: 
        """
        从 logits sample 出一个 token
        Args:
            logits: [vocab_size] 一维 tensor
            params: 采样参数
        Returns:
            next_token_id: int
        """
        # Greedy 
        if params.temperature == 0.0:
            return logits.argmax().item() # 找到概率最高的 
        
        # 温度采样
        logits = logits / params.temperature

        # top-k
        if params.top_k > 0:
            # 1. 安全检查。如果词表一共才5个词，你设置top_k=100，那就取5。
            top_k = min(params.top_k, logits.size(-1)) # -1 表示最后一个维度（词表维），对任意 batch shape 均适用
            # 2. 寻找“生死线（Threshold）”
            kth_value = torch.topk(logits, top_k)[0][-1]
            # 3. 将logits小于kth_value的进行Masking
            logits[logits < kth_value] = -float('inf') # 负无穷经过 softmax 后概率=0；正无穷则相反，会把全部概率集中到该 token

        # top-p
        if params.top_p < 1.0:
            # 1. 降序排列。把 logits 从高到低排好。
            # sorted_logits 是排好序的分数，sorted_indices 记住了它们原本在词表里的位置。
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            
            # 2. 计算累计概率 (Cumulative Sum)
            # softmax 将 logits 转成概率分布，cumsum 做前缀累加（如 [0.5,0.3,0.15,0.05] → [0.5,0.8,0.95,1.0]）
            # dim=-1 同上，沿词表维操作
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

            # 3. 找出哪些词越界 (累积和超过top_p)
            sorted_indices_to_remove = cumulative_probs > params.top_p 

            # 4. 按照top-p的定义刚刚超过top_p的值我们也要保留。所以向右平移一格 保留刚刚超过top_p的位置. 
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone() 

            # 5. 兜底永远保留第一名作为选择 （万一第一名直接大于top_p了）
            sorted_indices_to_remove[0] = False 

            # 6. 找到要执行masking的index并执行masking 
            indices_to_remove = sorted_indices[sorted_indices_to_remove] 
            logits[indices_to_remove] = -float('inf')

        # 进行Sample
        # 将logits转化为概率, -float('inf') softmax后概率为0，不会被选中. 
        probs = torch.softmax(logits, dim=-1)
        # multinomial 按概率分布随机抽 1 个 token（高概率被选中概率大，但非必然）；.item() 把结果从 tensor 转成 Python int
        return torch.multinomial(probs, 1).item()
            


        