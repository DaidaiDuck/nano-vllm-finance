# nano_vllm/types.py
from dataclasses import dataclass, field
from typing import Optional
from xmlrpc.client import boolean

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
    finished: boolean
    finish_reason: Optional[str] = None  # "stop" | "length"

    
