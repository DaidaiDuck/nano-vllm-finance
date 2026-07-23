"""Decide whether layer-0's flash output is consistent with the q/k/v it was actually given.

debug_forward.py proved the bug is inside batch_attn_forward at layer 0, and debug_varlen.py
proved the kernel call is correct on clean data. The remaining question: are the q/k/v (after
projection + RoPE) and the KV written into the cache correct, or is something in the real
forward feeding flash bad inputs?

This spies on flash_attn_varlen_func, captures the FIRST call's exact inputs (q + the paged
k/v cache + cu_seqlens_k + block_table) and its output, then recomputes the same attention
with a plain PyTorch SDPA reference over the keys reconstructed from the cache. If they match,
flash is doing exactly what SDPA does -- so the KV write and flash call are fine and the bug is
purely in q/k/v/RoPE (compare those to HF next). If they diverge, the KV write or the flash
call in the real context is wrong.

Run on the pod:
    NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct python scripts/debug_attn_layer0.py
"""
import os

import torch
import torch.nn.functional as F

import nano_vllm.paged.paged_attention as pa
from nano_vllm.paged.engine import LLM
from nano_vllm.paged.paged_attention import ENGINE_CTX

MODEL = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-3B-Instruct")
PROMPT = "Hello"


def main():
    llm = LLM(MODEL)
    engine = llm.engine
    tok = engine.tokenizer
    prompt_ids = tok.encode(engine._format_prompt(PROMPT))
    P = len(prompt_ids)

    # Single-request prefill into a clean block 0 (same setup as debug_forward.py).
    positions = list(range(P))
    slot_mapping = [p for p in positions]  # block 0 -> slot == offset == position
    ENGINE_CTX.cu_seqlens_q = torch.tensor([0, P], dtype=torch.int32, device="cuda")
    ENGINE_CTX.cu_seqlens_k = torch.tensor([0, P], dtype=torch.int32, device="cuda")
    ENGINE_CTX.slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64, device="cuda")
    ENGINE_CTX.block_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
    ENGINE_CTX.max_seqlen_q = P
    ENGINE_CTX.max_seqlen_k = P

    # Spy on the kernel: record the first call's inputs and output, then delegate.
    captured = {}
    real_flash = pa.flash_attn_varlen_func

    def spy(**kw):
        out = real_flash(**kw)
        if not captured:
            captured.update(kw)
            captured["out"] = out
        return out

    pa.flash_attn_varlen_func = spy
    try:
        input_ids = torch.tensor([prompt_ids], device="cuda")
        pos_ids = torch.tensor([positions], device="cuda")
        with torch.no_grad():
            engine.model.model(input_ids=input_ids, position_ids=pos_ids)
    finally:
        pa.flash_attn_varlen_func = real_flash

    q = captured["q"]              # [P, HQ, D]  (post-RoPE)
    k_cache = captured["k"]        # [num_blocks, block_size, HKV, D]
    v_cache = captured["v"]
    HQ, D = q.shape[1], q.shape[2]
    HKV = k_cache.shape[2]
    print(f"P={P}  HQ={HQ}  HKV={HKV}  D={D}")

    # Reconstruct this sequence's keys/values from the paged cache (block 0, offsets 0..P-1).
    keys = k_cache[0, :P]          # [P, HKV, D]
    values = v_cache[0, :P]

    # Plain SDPA reference with GQA expansion + causal mask.
    q_s = q.transpose(0, 1)[None].float()                       # [1, HQ, P, D]
    k_s = keys.transpose(0, 1)[None].float().repeat_interleave(HQ // HKV, dim=1)
    v_s = values.transpose(0, 1)[None].float().repeat_interleave(HQ // HKV, dim=1)
    ref = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True)[0].transpose(0, 1)

    flash = captured["out"].float()   # [P, HQ, D]
    diff = (flash - ref).abs().max().item()
    print(f"\nflash vs SDPA on the SAME q/k/v:  max|Δ| = {diff:.4e}")
    if diff < 2e-2:
        print("MATCH -> flash + KV write are correct; the q/k/v (projection or RoPE) is what "
              "differs from HF. Compare nano's post-RoPE q/k against HF next.")
    else:
        print("MISMATCH -> flash gets bad inputs in the real context: the KV written into the "
              "cache does not match k/v, or the block_table/cu_seqlens_k the runner set is off.")

    # Extra: does the KV write round-trip? (cache content vs the k that flash also received)
    # Here keys are read straight back from the cache; compare against what q attends to is
    # implicit above. Also sanity-check the cache is non-zero where it should be.
    print(f"cache[0,:P] abs-mean = {keys.abs().mean().item():.4e} "
          f"(should be >0; zero means the KV write did not land)")


if __name__ == "__main__":
    main()
