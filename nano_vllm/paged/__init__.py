# nano_vllm/paged/__init__.py — M3 and later (PagedAttention). M4 continuous batching grows here.
#
# Deliberately does NOT import `.engine` (or `.paged_attention`) at package import time:
# those pull in flash-attn (GPU-only), whereas block_pool / kv_cache_manager / kv_cache_utils
# are pure-Python and must stay importable on CPU (e.g. tests/test_block_pool.py on a laptop).
# Import what you need explicitly:
#     from nano_vllm.paged.engine import LLM            # needs flash-attn + CUDA
#     from nano_vllm.paged.block_pool import BlockPool  # CPU-safe
