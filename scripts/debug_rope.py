"""Pinpoint why nano's q/k differ from HF's: compare the inputs fed to apply_rotary_pos_emb.

debug_attn_layer0.py proved flash + the KV write are correct, so nano's post-RoPE q/k must
differ from HF's. Both paths call the SAME apply_rotary_pos_emb, so identical inputs would
give identical outputs -- the divergence is in what each feeds it: the projected/reshaped q
(q_in) or the rotary tables (cos/sin, i.e. position_embeddings).

This spies on apply_rotary_pos_emb in both runs (nano imported it into paged_attention;
HF calls the one in transformers' qwen2 module) and captures the FIRST call's q_in, cos, sin.
Whichever of those differs between nano and HF is the bug:

  - cos/sin differ  -> batch_attn_forward is receiving the wrong `position_embeddings`
                       (arg-order / capture bug, or a shape/convention change in transformers).
  - q_in differs    -> the projection reshape/transpose before RoPE is wrong.

Run on the pod:
    NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct python scripts/debug_rope.py
"""
import os

import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
from transformers import AutoModelForCausalLM

import nano_vllm.paged.paged_attention as pa
from nano_vllm.paged.engine import LLM
from nano_vllm.paged.paged_attention import ENGINE_CTX

MODEL = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-3B-Instruct")
PROMPT = "Hello"

_REAL_ROPE = qwen2_mod.apply_rotary_pos_emb


def _make_spy(store):
    def spy(q, k, cos, sin, *args, **kwargs):
        if not store:
            store["q_in"] = q.detach().float()
            store["cos"] = cos.detach().float()
            store["sin"] = sin.detach().float()
        return _REAL_ROPE(q, k, cos, sin, *args, **kwargs)
    return spy


def main():
    llm = LLM(MODEL)
    engine = llm.engine
    tok = engine.tokenizer
    prompt_ids = tok.encode(engine._format_prompt(PROMPT))
    P = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device="cuda")
    pos_ids = torch.tensor([list(range(P))], device="cuda")

    # Single-request prefill setup for the patched model.
    ENGINE_CTX.cu_seqlens_q = torch.tensor([0, P], dtype=torch.int32, device="cuda")
    ENGINE_CTX.cu_seqlens_k = torch.tensor([0, P], dtype=torch.int32, device="cuda")
    ENGINE_CTX.slot_mapping = torch.tensor(list(range(P)), dtype=torch.int64, device="cuda")
    ENGINE_CTX.block_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
    ENGINE_CTX.max_seqlen_q = P
    ENGINE_CTX.max_seqlen_k = P

    # NANO run: batch_attn_forward calls paged_attention's imported apply_rotary_pos_emb,
    # so patch it THERE (patching the transformers module would not affect the name already
    # bound inside paged_attention).
    nano = {}
    pa.apply_rotary_pos_emb = _make_spy(nano)
    try:
        with torch.no_grad():
            engine.model.model(input_ids=input_ids, position_ids=pos_ids)
    finally:
        pa.apply_rotary_pos_emb = _REAL_ROPE

    # HF run: the stock attention calls the module-level apply_rotary_pos_emb.
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    hf.eval()
    ref = {}
    qwen2_mod.apply_rotary_pos_emb = _make_spy(ref)
    try:
        with torch.no_grad():
            hf(input_ids)
    finally:
        qwen2_mod.apply_rotary_pos_emb = _REAL_ROPE

    def cmp(name):
        a, b = nano[name], ref[name]
        if a.shape != b.shape:
            print(f"  {name:6} SHAPE differs: nano {tuple(a.shape)}  hf {tuple(b.shape)}")
            return
        print(f"  {name:6} shape {tuple(a.shape)}  max|Δ| = {(a - b).abs().max().item():.4e}")

    print("\ninputs fed to apply_rotary_pos_emb (layer 0):")
    cmp("cos")
    cmp("sin")
    cmp("q_in")

    print("\nInterpretation:")
    print("  cos/sin differ -> wrong position_embeddings reaching batch_attn_forward.")
    print("  q_in differs    -> projection reshape/transpose before RoPE is wrong.")
    print("  all match       -> divergence is downstream of RoPE (unexpected; revisit).")


if __name__ == "__main__":
    main()
