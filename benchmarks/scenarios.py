from dataclasses import dataclass
from typing import Callable

@dataclass 
class Scenario: 
    name: str
    description: str 
    prompt_len_range: tuple[int, int] # (min, max) tokens 
    output_len: int 
    concurrency: int 
    num_requests: int 

    def get_sampling_params(self): 
        from nano_vllm import SamplingParams 
        return SamplingParams(
            temperature=0.0, # greedy for reproducibility 
            max_tokens=self.output_len
        )
    

# 预定义场景
SCENARIOS = {
    "short_chat": Scenario(
        name="short_chat",
        description="Short chat: ~100 token prompt, ~100 token output",
        prompt_len_range=(50, 150), 
        output_len=100, 
        concurrency=1,  # M1 单请求
        num_requests=50,
    ),

    "medium_chat": Scenario(
        name="medium_chat",
        description="Medium chat: ~500 token prompt, ~200 token output",
        prompt_len_range=(400, 600),
        output_len=200,
        concurrency=1,
        num_requests=30,
    ),

    "long_context": Scenario(
        name="long_context",
        description="Long context: ~2000 token prompt, ~100 token output",
        prompt_len_range=(1800, 2200),
        output_len=100,
        concurrency=1,
        num_requests=20,
    ),

    # M4 后才能测
    "high_concurrency": Scenario(
        name="high_concurrency", 
        description="High concurrency: ~100 token prompt, ~100 token output, 16 concurrent",
        prompt_len_range=(50, 150),
        output_len=100,
        concurrency=16,
        num_requests=64,
    )
}