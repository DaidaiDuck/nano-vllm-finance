# nano_vllm/types.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class SamplingParams:
    temperature: float = 0.0 
    top_k: int = -1
    top_p: float = 1.0 
    max_tokens: int = 100 

@dataclass
class RequestOutput: 
    request_id: str 
    prompt: str 
    text: str 
    token_ids: list[int]
    finished: bool
    finish_reason: Optional[str] = None  # "stop" | "length"

    
@dataclass
class Request:
    id: str
    prompt: str
    num_computed_tokens: int = 0  # tokens already in KV Cache


@dataclass
class KVCacheConfig:
    """
    The KV cache configurato
    """