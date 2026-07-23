"""Find the exact reshape bug: compare the o_proj INPUT (nano vs HF) and brute-force the layout.

self_attn output diverges (debug_attn_output) but the flash output was correct
(debug_attn_layer0), so `out.reshape(*input_shape, -1)` feeds o_proj a wrongly-laid-out
tensor. o_proj weights are shared, so its input is what differs. This captures:

  - nano_raw    : the flash output [total, heads, head_dim] (via a spy)
  - nano_oproj  : the tensor nano actually feeds o_proj [1, seq, hidden] (pre-forward hook)
  - hf_oproj    : the tensor HF feeds o_proj [1, seq, hidden] (pre-forward hook)

Then it reshapes nano_raw several ways and reports which one reproduces hf_oproj. The layout
that matches is the correct reshape; nano's current one is the bug.

Run on the pod:
    NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct python scripts/debug_oproj.py
"""
import os

import torch
from transformers import AutoModelForCausalLM

import nano_vllm.paged.paged_attention as pa
from nano_vllm.paged.engine import LLM
from nano_vllm.paged.paged_attention import ENGINE_CTX

MODEL = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-3B-Instruct")
PROMPT = "Hello"


def _hook_oproj_input(backbone, store):
    def pre_hook(_m, inputs):
        store["in"] = inputs[0].detach().float()
    return backbone.layers[0].self_attn.o_proj.register_forward_pre_hook(pre_hook)


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

    # nano: spy flash to grab the raw output, and hook o_proj to grab its input.
    nano_raw = {}
    real_flash = pa.flash_attn_varlen_func

    def spy(**kw):
        out = real_flash(**kw)
        nano_raw.setdefault("out", out.detach().float())
        return out

    pa.flash_attn_varlen_func = spy
    nano_oproj = {}
    h = _hook_oproj_input(engine.model.model, nano_oproj)
    with torch.no_grad():
        engine.model.model(input_ids=input_ids, position_ids=pos_ids)
    h.remove()
    pa.flash_attn_varlen_func = real_flash

    # HF: hook o_proj input.
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    hf.eval()
    hf_oproj = {}
    h = _hook_oproj_input(hf.model, hf_oproj)
    with torch.no_grad():
        hf(input_ids)
    h.remove()

    raw = nano_raw["out"]        # [P, HQ, D]
    HQ, D = raw.shape[1], raw.shape[2]
    nano_in = nano_oproj["in"]   # [1, P, HQ*D]
    hf_in = hf_oproj["in"]       # [1, P, HQ*D]

    print(f"\nflash raw out: {tuple(raw.shape)}   o_proj input: nano {tuple(nano_in.shape)} "
          f"hf {tuple(hf_in.shape)}")
    print(f"nano_oproj_in vs hf_oproj_in   max|Δ| = {(nano_in - hf_in).abs().max().item():.4e}")

    # Brute-force which reshape of the raw flash output reproduces HF's o_proj input.
    hf_flat = hf_in[0]  # [P, HQ*D]
    candidates = {
        "reshape(P, HQ*D)  [head-major, nano's current]": raw.reshape(P, HQ * D),
        "transpose(0,1).reshape then back? [interleaved]":
            raw.transpose(1, 2).reshape(P, HQ * D),  # dim-major instead of head-major
    }
    print("\nwhich layout of the raw flash output matches HF's o_proj input:")
    for name, cand in candidates.items():
        diff = (cand - hf_flat).abs().max().item()
        print(f"  {'MATCH' if diff < 5e-2 else 'no   '}  {name:48} max|Δ| = {diff:.4e}")


if __name__ == "__main__":
    main()
