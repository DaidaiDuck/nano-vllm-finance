"""Resolve the contradiction: o_proj INPUT matches (debug_oproj) but self_attn OUTPUT differs
(debug_attn_output). Same input + same weights must give the same output -- so one of those,
or the o_proj weights, is not what we think. Measure all three in ONE process, same models.

Prints:
  (1) o_proj weight diff        nano vs HF   -> should be 0 (same checkpoint)
  (2) o_proj INPUT diff         nano vs HF   -> ~0 per debug_oproj
  (3) self_attn OUTPUT diff     nano vs HF   -> the 6.7 to explain

If (1)==0 and (2)==0 but (3) large, the two captures are inconsistent (a measurement
artifact); the script then recomputes o_proj(input) by hand for both to see which side's
real self_attn output disagrees with its own o_proj(input).

Run on the pod:
    NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct python scripts/debug_combined.py
"""
import os

import torch
from transformers import AutoModelForCausalLM

from nano_vllm.paged.engine import LLM
from nano_vllm.paged.paged_attention import ENGINE_CTX

MODEL = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-3B-Instruct")
PROMPT = "Hello"


def _capture(backbone, input_ids, pos_ids):
    """Capture layer-0 self_attn's o_proj INPUT and OUTPUT in a single forward.

    A forward_hook receives (module, input, output), so one hook grabs both -- no flaky
    forward_pre_hook needed.
    """
    attn0 = backbone.layers[0].self_attn
    store = {}

    def hook(_m, inp, out):
        store["in"] = inp[0].detach().float()
        store["out"] = out.detach().float()

    h = attn0.o_proj.register_forward_hook(hook)
    kwargs = {"input_ids": input_ids}
    if pos_ids is not None:
        kwargs["position_ids"] = pos_ids
    try:
        with torch.no_grad():
            backbone(**kwargs)
    finally:
        h.remove()
    return attn0.o_proj, store


def main():
    llm = LLM(MODEL)
    engine = llm.engine
    tok = engine.tokenizer
    prompt_ids = tok.encode(engine._format_prompt(PROMPT))
    P = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device="cuda")
    pos_ids = torch.tensor([list(range(P))], device="cuda")

    ENGINE_CTX.cu_seqlens_q = torch.tensor([0, P], dtype=torch.int32, device="cuda")
    ENGINE_CTX.cu_seqlens_k = torch.tensor([0, P], dtype=torch.int32, device="cuda")
    ENGINE_CTX.slot_mapping = torch.tensor(list(range(P)), dtype=torch.int64, device="cuda")
    ENGINE_CTX.block_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
    ENGINE_CTX.max_seqlen_q = P
    ENGINE_CTX.max_seqlen_k = P

    nano_oproj, nano = _capture(engine.model.model, input_ids, pos_ids)

    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    hf.eval()
    hf_oproj, hf_s = _capture(hf.model, input_ids, None)

    wdiff = (nano_oproj.weight.float() - hf_oproj.weight.float()).abs().max().item()
    print(f"\n(1) o_proj.weight  nano vs hf   max|Δ| = {wdiff:.4e}")

    print(f"(2) o_proj INPUT   nano vs hf   max|Δ| = "
          f"{(nano['in'] - hf_s['in']).abs().max().item():.4e}")
    print(f"(3) o_proj OUTPUT  nano vs hf   max|Δ| = "
          f"{(nano['out'] - hf_s['out']).abs().max().item():.4e}")

    # Sanity: does each side's own o_proj(input) reproduce its captured output?
    def recompute(oproj, s):
        y = torch.nn.functional.linear(s["in"].to(oproj.weight.dtype), oproj.weight, oproj.bias)
        return (y.float() - s["out"]).abs().max().item()

    print(f"\nconsistency  nano o_proj(input)->output Δ = {recompute(nano_oproj, nano):.4e}"
          f"   hf Δ = {recompute(hf_oproj, hf_s):.4e}  (both ~0 expected)")


if __name__ == "__main__":
    main()
