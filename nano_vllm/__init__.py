# nano_vllm/__init__.py
# Public API. Milestone code lives in subpackages (which file belongs to which milestone
# is answered by the folder):
#   nano_vllm.core    -> shared across all milestones: types, sampler
#   nano_vllm.simple  -> M1 (HF DynamicCache) + M2 (MyKVCache)   — SimpleEngine
#   nano_vllm.paged   -> M3 and later (PagedAttention)           — PagedEngine
#
# The top-level `LLM` is the import-safe M1/M2 engine (no flash-attn dependency, works on
# CPU). M3 needs flash-attn + CUDA, so import it explicitly only where you have them:
#     from nano_vllm.paged.engine import LLM
from nano_vllm.core.types import RequestOutput, SamplingParams, Request
from nano_vllm.core.sampler import Sampler
from nano_vllm.simple.engine import LLM, SimpleEngine

__all__ = ["LLM", "SimpleEngine", "SamplingParams", "RequestOutput", "Request", "Sampler"]
