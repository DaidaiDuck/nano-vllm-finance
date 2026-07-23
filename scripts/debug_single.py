"""One real forward, no Identity swap, no cross-run comparison. Verify nano's own pipeline is
self-consistent, then compare its attention output to HF.

Captured in a SINGLE nano forward (real o_proj):
  raw       = flash output [P, HQ, D]                    (spy)
  nano_out  = layer-0 self_attn output [1, P, hidden]    (post o_proj, hook)
  nano_in   = raw.reshape(1, P, HQ*D)                    (what the code feeds o_proj)

Checks:
  (C) linear(nano_in, W) vs nano_out       -> 0 means the reshape really is the o_proj input
  compare nano_out to HF's self_attn output (a separate HF forward)

If (C)==0 and nano_out != hf_out with identical weights, then nano_in != HF's o_proj input,
i.e. the raw attention or the reshape genuinely differs from HF -- and we chase that. If (C)
is large, batch_attn_forward does something between the reshape and o_proj we have missed.

Run on the pod:
    NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct python scripts/debug_single.py
"""
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import nano_vllm.paged.paged_attention as pa
from nano_vllm.paged.engine import LLM
from nano_vllm.paged.paged_attention import ENGINE_CTX

MODEL = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-3B-Instruct")
PROMPT = "Hello"


def _self_attn_out(backbone, input_ids, pos_ids):
    store = {}
    attn0 = backbone.layers[0].self_attn

    def hook(_m, _i, out):
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
    return store["out"]


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

    # Single nano forward: spy flash + hook self_attn together.
    raw = {}
    real_flash = pa.flash_attn_varlen_func

    def spy(**kw):
        out = real_flash(**kw)
        raw.setdefault("out", out.detach().float())
        return out

    pa.flash_attn_varlen_func = spy
    try:
        nano_out = _self_attn_out(engine.model.model, input_ids, pos_ids)
    finally:
        pa.flash_attn_varlen_func = real_flash

    r = raw["out"]                       # [P, HQ, D]
    HQ, D = r.shape[1], r.shape[2]
    nano_in = r.reshape(1, P, HQ * D)    # what batch_attn_forward feeds o_proj

    oproj = engine.model.model.layers[0].self_attn.o_proj
    recomputed = F.linear(nano_in.to(oproj.weight.dtype), oproj.weight, oproj.bias).float()
    c = (recomputed - nano_out).abs().max().item()
    print(f"\nP={P}  raw {tuple(r.shape)}")
    print(f"(C) linear(raw.reshape) vs nano self_attn out   max|Δ| = {c:.4e}")
    if c < 5e-2:
        print("    -> consistent: raw.reshape IS the o_proj input.")
    else:
        print("    -> INCONSISTENT: batch_attn_forward alters attn_output between the reshape "
              "and o_proj (or the spy grabbed a different call). This is the smoking gun.")

    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    hf.eval()
    hf_out = _self_attn_out(hf.model, input_ids, None)
    print(f"\nnano self_attn out vs HF   max|Δ| = {(nano_out - hf_out).abs().max().item():.4e}")


if __name__ == "__main__":
    main()
