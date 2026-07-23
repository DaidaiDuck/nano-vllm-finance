"""Confirm nano's forward is not idempotent and show how the paged cache drifts across calls.

debug_combined's consistency check proved nano's layer-0 attention changes between two
identical forwards (the only persistent state is the paged k_cache). This runs the SAME
single-request prefill several times with a fresh ENGINE_CTX each time and reports, per call:

  - layer-0 self_attn output diff vs the first call   (0 = idempotent)
  - the flat KV-cache content abs-mean over the slots flash reads

If the attention output changes call to call, the cache read/write is the bug. Watching the
cache abs-mean grow or shift shows whether writes accumulate or land in the wrong place.

Run on the pod:
    NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct python scripts/debug_idempotent.py
"""
import os

import torch

from nano_vllm.paged.engine import LLM
from nano_vllm.paged.paged_attention import ENGINE_CTX

MODEL = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-3B-Instruct")
PROMPT = "Hello"
N = 4


def main():
    llm = LLM(MODEL)
    engine = llm.engine
    tok = engine.tokenizer
    prompt_ids = tok.encode(engine._format_prompt(PROMPT))
    P = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device="cuda")
    pos_ids = torch.tensor([list(range(P))], device="cuda")

    attn0 = engine.model.model.layers[0].self_attn
    layer0_kcache = attn0.k_cache  # [num_blocks, block_size, n_kv, head_dim] for layer 0

    def set_ctx():
        ENGINE_CTX.cu_seqlens_q = torch.tensor([0, P], dtype=torch.int32, device="cuda")
        ENGINE_CTX.cu_seqlens_k = torch.tensor([0, P], dtype=torch.int32, device="cuda")
        ENGINE_CTX.slot_mapping = torch.tensor(list(range(P)), dtype=torch.int64, device="cuda")
        ENGINE_CTX.block_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
        ENGINE_CTX.max_seqlen_q = P
        ENGINE_CTX.max_seqlen_k = P

    def run():
        store = {}

        def hook(_m, _i, out):
            store["out"] = (out[0] if isinstance(out, tuple) else out).detach().float()
        h = attn0.register_forward_hook(hook)
        try:
            with torch.no_grad():
                engine.model.model(input_ids=input_ids, position_ids=pos_ids)
        finally:
            h.remove()
        return store["out"]

    first = None
    print(f"P={P}   (block 0, slots 0..{P-1})\n")
    for i in range(N):
        set_ctx()
        out = run()
        if first is None:
            first = out
        # abs-mean of the KV cache over the slots flash actually reads for this request
        cache_slots = layer0_kcache[0, :P].float()
        print(f"call {i}: layer0 self_attn out Δ vs call0 = "
              f"{(out - first).abs().max().item():.4e}   "
              f"k_cache[0,:P] abs-mean = {cache_slots.abs().mean().item():.4e}")

    print("\nIf Δ is 0 every call -> idempotent (look elsewhere). If Δ grows -> the paged "
          "cache read/write across forwards is the bug.")


if __name__ == "__main__":
    main()
