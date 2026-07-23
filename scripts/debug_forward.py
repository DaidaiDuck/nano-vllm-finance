"""Localize the M4 forward bug by comparing the patched model against clean HF, layer by layer.

The kernel call is proven correct (debug_varlen.py all-pass), yet test_m4_vs_hf produces
garbage from token 0 even at batch=1. So the bug is in batch_attn_forward's non-kernel parts
(q/k/v projection, RoPE, KV write) or the runner. This script runs ONE single-request prefill
through the patched model with ENGINE_CTX set up by hand, compares the final logits to a fresh
HF model on the same input_ids, and -- crucially -- hooks every decoder layer on both models to
report the first layer whose hidden states diverge. That first divergent layer is where the bug
lives.

Run on the pod:
    NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct python scripts/debug_forward.py
"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_vllm.paged.engine import LLM
from nano_vllm.paged.paged_attention import ENGINE_CTX

MODEL = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-3B-Instruct")
PROMPT = "Hello"


def _capture_layer_outputs(backbone):
    """Hook each decoder layer of a Qwen2Model backbone; dict is filled on the next forward."""
    captured = {}

    def make_hook(i):
        def hook(_module, _inputs, output):
            # decoder layer returns a tuple; hidden states are output[0]
            captured[i] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    handles = [layer.register_forward_hook(make_hook(i))
               for i, layer in enumerate(backbone.layers)]
    return captured, handles


def main():
    torch.manual_seed(0)

    # --- patched model (nano M4) ------------------------------------------------------
    llm = LLM(MODEL)
    engine = llm.engine
    tok = engine.tokenizer
    prompt_ids = tok.encode(engine._format_prompt(PROMPT))
    P = len(prompt_ids)
    print(f"prompt tokens: {P}")

    # Set up ENGINE_CTX exactly as _prepare_inputs would for a single-request prefill into
    # a fresh block 0. k_cache starts as zeros, so block 0 is clean.
    block_id = 0
    positions = list(range(P))
    slot_mapping = [block_id * engine.block_size + p for p in positions]
    ENGINE_CTX.cu_seqlens_q = torch.tensor([0, P], dtype=torch.int32, device="cuda")
    ENGINE_CTX.cu_seqlens_k = torch.tensor([0, P], dtype=torch.int32, device="cuda")
    ENGINE_CTX.slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64, device="cuda")
    ENGINE_CTX.block_table = torch.tensor([[block_id]], dtype=torch.int32, device="cuda")
    ENGINE_CTX.max_seqlen_q = P
    ENGINE_CTX.max_seqlen_k = P

    input_ids = torch.tensor([prompt_ids], device="cuda")
    pos_ids = torch.tensor([positions], device="cuda")

    nano_caps, nano_handles = _capture_layer_outputs(engine.model.model)
    with torch.no_grad():
        nano_hidden = engine.model.model(input_ids=input_ids, position_ids=pos_ids).last_hidden_state
        nano_logits = engine.model.lm_head(nano_hidden[0, -1])
    for h in nano_handles:
        h.remove()

    # --- clean HF reference -----------------------------------------------------------
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    hf.eval()
    hf_caps, hf_handles = _capture_layer_outputs(hf.model)
    with torch.no_grad():
        hf_out = hf(input_ids)  # HF builds its own causal mask + KV internally
        hf_logits = hf_out.logits[0, -1]
    for h in hf_handles:
        h.remove()

    # --- compare ----------------------------------------------------------------------
    print("\nper-layer max|Δ| of hidden states (patched vs HF):")
    first_bad = None
    for i in range(len(nano_caps)):
        a = nano_caps[i][0].float()   # [P, H]
        b = hf_caps[i][0].float()
        diff = (a - b).abs().max().item()
        flag = ""
        if diff > 1e-1 and first_bad is None:
            first_bad = i
            flag = "  <-- first divergence"
        print(f"  layer {i:2d}: {diff:.4e}{flag}")

    print(f"\nfinal logits argmax  nano={nano_logits.argmax().item()}  hf={hf_logits.argmax().item()}")
    print(f"final logits max|Δ|  {(nano_logits.float() - hf_logits.float()).abs().max().item():.4e}")

    if first_bad is None:
        print("\nAll layers match -> attention is fine; bug is in logits/sampling/runner glue.")
    elif first_bad == 0:
        print("\nLayer 0 already diverges -> bug is inside batch_attn_forward "
              "(q/k/v projection, RoPE, or the KV write), not accumulated across layers.")
    else:
        print(f"\nLayers 0..{first_bad-1} match, layer {first_bad} diverges -> look at what is "
              f"state-dependent across layers (most likely the KV cache write/read, since each "
              f"layer has its own k_cache slice).")


if __name__ == "__main__":
    main()
