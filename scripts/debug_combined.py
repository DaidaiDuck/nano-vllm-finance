"""Resolve the contradiction with ONLY the self_attn forward_hook (o_proj hooks don't fire
in this environment). Four forwards in one process:

  nano real o_proj      -> self_attn output           (post o_proj)
  nano Identity o_proj   -> self_attn output           (== o_proj INPUT)
  hf   real o_proj       -> self_attn output
  hf   Identity o_proj   -> self_attn output           (== o_proj INPUT)

Then: o_proj weight diff, o_proj INPUT diff (nano vs hf), o_proj OUTPUT diff (nano vs hf),
and a consistency check that linear(input) reproduces output on each side. If INPUT matches
and weights match but OUTPUT differs, the OUTPUT capture is the artifact.

Run on the pod:
    NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct python scripts/debug_combined.py
"""
import os

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from nano_vllm.paged.engine import LLM
from nano_vllm.paged.paged_attention import ENGINE_CTX

MODEL = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-3B-Instruct")
PROMPT = "Hello"


def _run(backbone, input_ids, pos_ids, identity_oproj):
    """Return layer-0 self_attn output; optionally with o_proj swapped for Identity."""
    attn0 = backbone.layers[0].self_attn
    real = attn0.o_proj
    if identity_oproj:
        attn0.o_proj = nn.Identity()

    store = {}

    def hook(_m, _inp, out):
        store["out"] = (out[0] if isinstance(out, tuple) else out).detach().float()

    h = attn0.register_forward_hook(hook)
    kwargs = {"input_ids": input_ids}
    if pos_ids is not None:
        kwargs["position_ids"] = pos_ids
    try:
        with torch.no_grad():
            backbone(**kwargs)
    finally:
        h.remove()
        attn0.o_proj = real
    return store["out"]


def main():
    llm = LLM(MODEL)
    engine = llm.engine
    tok = engine.tokenizer
    prompt_ids = tok.encode(engine._format_prompt(PROMPT))
    P = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device="cuda")
    pos_ids = torch.tensor([list(range(P))], device="cuda")

    def set_ctx():
        ENGINE_CTX.cu_seqlens_q = torch.tensor([0, P], dtype=torch.int32, device="cuda")
        ENGINE_CTX.cu_seqlens_k = torch.tensor([0, P], dtype=torch.int32, device="cuda")
        ENGINE_CTX.slot_mapping = torch.tensor(list(range(P)), dtype=torch.int64, device="cuda")
        ENGINE_CTX.block_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
        ENGINE_CTX.max_seqlen_q = P
        ENGINE_CTX.max_seqlen_k = P

    nb = engine.model.model
    set_ctx(); nano_out = _run(nb, input_ids, pos_ids, identity_oproj=False)
    set_ctx(); nano_in = _run(nb, input_ids, pos_ids, identity_oproj=True)

    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    hf.eval()
    hb = hf.model
    hf_out = _run(hb, input_ids, None, identity_oproj=False)
    hf_in = _run(hb, input_ids, None, identity_oproj=True)

    nano_oproj = nb.layers[0].self_attn.o_proj
    hf_oproj = hb.layers[0].self_attn.o_proj

    print(f"\n(1) o_proj.weight   nano vs hf   max|Δ| = "
          f"{(nano_oproj.weight.float() - hf_oproj.weight.float()).abs().max().item():.4e}")
    print(f"(2) o_proj INPUT    nano vs hf   max|Δ| = {(nano_in - hf_in).abs().max().item():.4e}")
    print(f"(3) o_proj OUTPUT   nano vs hf   max|Δ| = {(nano_out - hf_out).abs().max().item():.4e}")

    def recompute(oproj, x):
        return torch.nn.functional.linear(x.to(oproj.weight.dtype), oproj.weight, oproj.bias).float()

    print(f"\nconsistency  nano linear(input) vs output  Δ = "
          f"{(recompute(nano_oproj, nano_in) - nano_out).abs().max().item():.4e}")
    print(f"             hf   linear(input) vs output  Δ = "
          f"{(recompute(hf_oproj, hf_in) - hf_out).abs().max().item():.4e}   (both ~0 expected)")


if __name__ == "__main__":
    main()
