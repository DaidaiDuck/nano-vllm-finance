"""Find the exact reshape bug by comparing the o_proj INPUT (nano vs HF).

self_attn output diverges (debug_attn_output) but the flash output was correct
(debug_attn_layer0), so `out.reshape(*input_shape, -1)` feeds o_proj a wrongly-laid-out
tensor. o_proj weights are shared, so its input is what differs.

To read the o_proj input reliably, we temporarily swap layer-0's o_proj for nn.Identity, so
the self_attn module output (a hook we know fires) *becomes* the reshaped pre-o_proj tensor.
We also spy the raw flash output and brute-force which reshape of it reproduces HF's, which
names the correct fix.

Run on the pod:
    NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct python scripts/debug_oproj.py
"""
import os

import torch
from torch import nn
from transformers import AutoModelForCausalLM

import nano_vllm.paged.paged_attention as pa
from nano_vllm.paged.engine import LLM
from nano_vllm.paged.paged_attention import ENGINE_CTX

MODEL = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-3B-Instruct")
PROMPT = "Hello"


def _capture_attn0_with_identity_oproj(backbone, input_ids, pos_ids):
    """Run a forward with layer-0's o_proj replaced by Identity; return the pre-o_proj tensor
    that self_attn emits (== whatever the code fed o_proj), shape [1, seq, hidden]."""
    attn0 = backbone.layers[0].self_attn
    real_oproj = attn0.o_proj
    attn0.o_proj = nn.Identity()

    store = {}

    def hook(_m, _inp, output):
        store["out"] = (output[0] if isinstance(output, tuple) else output).detach().float()

    h = attn0.register_forward_hook(hook)
    kwargs = {"input_ids": input_ids}
    if pos_ids is not None:
        kwargs["position_ids"] = pos_ids
    try:
        with torch.no_grad():
            backbone(**kwargs)
    finally:
        h.remove()
        attn0.o_proj = real_oproj
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

    # Spy the raw flash output while capturing nano's pre-o_proj tensor.
    nano_raw = {}
    real_flash = pa.flash_attn_varlen_func

    def spy(**kw):
        out = real_flash(**kw)
        nano_raw.setdefault("out", out.detach().float())
        return out

    pa.flash_attn_varlen_func = spy
    try:
        nano_in = _capture_attn0_with_identity_oproj(engine.model.model, input_ids, pos_ids)
    finally:
        pa.flash_attn_varlen_func = real_flash

    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    hf.eval()
    hf_in = _capture_attn0_with_identity_oproj(hf.model, input_ids, None)

    raw = nano_raw["out"]                 # [P, HQ, D]
    HQ, D = raw.shape[1], raw.shape[2]
    print(f"\nflash raw out {tuple(raw.shape)}   o_proj input: nano {tuple(nano_in.shape)} "
          f"hf {tuple(hf_in.shape)}")
    print(f"nano o_proj input vs hf o_proj input   max|Δ| = {(nano_in - hf_in).abs().max().item():.4e}")

    hf_flat = hf_in[0]                    # [P, HQ*D]
    print("\nwhich layout of the raw flash output reproduces HF's o_proj input:")
    candidates = {
        "raw.reshape(P, HQ*D)          [head-major, current]": raw.reshape(P, HQ * D),
        "raw.transpose(1,2).reshape    [dim-major]":           raw.transpose(1, 2).reshape(P, HQ * D),
    }
    for name, cand in candidates.items():
        diff = (cand - hf_flat).abs().max().item()
        print(f"  {'MATCH' if diff < 5e-2 else 'no   '}  {name}  max|Δ| = {diff:.4e}")


if __name__ == "__main__":
    main()
