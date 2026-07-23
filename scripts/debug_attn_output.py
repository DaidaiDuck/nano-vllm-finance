"""Compare the FULL self-attention module output (post reshape + o_proj), nano vs HF.

The chain so far: RoPE inputs are bit-identical (debug_rope), the flash kernel matches SDPA
on those q/k/v (debug_attn_layer0), yet the decoder layer output diverges hugely at layer 0
(debug_forward). The one segment never measured is what batch_attn_forward does AFTER flash:
`out.reshape(*input_shape, -1)` then `o_proj`. This hooks layer-0's self_attn on both models
and compares its returned attention output. If it diverges, the reshape or o_proj is the bug;
if it matches, the attention is fine and the divergence must come from elsewhere in the layer.

Run on the pod:
    NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct python scripts/debug_attn_output.py
"""
import os

import torch
from transformers import AutoModelForCausalLM

from nano_vllm.paged.engine import LLM
from nano_vllm.paged.paged_attention import ENGINE_CTX

MODEL = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-3B-Instruct")
PROMPT = "Hello"


def _hook_attn0(backbone, store):
    def hook(_m, _inp, output):
        store["out"] = (output[0] if isinstance(output, tuple) else output).detach().float()
    return backbone.layers[0].self_attn.register_forward_hook(hook)


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

    nano = {}
    h = _hook_attn0(engine.model.model, nano)
    with torch.no_grad():
        engine.model.model(input_ids=input_ids, position_ids=pos_ids)
    h.remove()

    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    hf.eval()
    ref = {}
    h = _hook_attn0(hf.model, ref)
    with torch.no_grad():
        hf(input_ids)
    h.remove()

    a, b = nano["out"], ref["out"]
    print(f"\nself_attn(layer 0) output   nano {tuple(a.shape)}  hf {tuple(b.shape)}")
    if a.shape != b.shape:
        print("SHAPE MISMATCH -> the reshape after flash is wrong.")
        return
    diff = (a - b).abs().max().item()
    print(f"max|Δ| = {diff:.4e}")
    if diff < 5e-2:
        print("MATCH -> attention module is correct; divergence is elsewhere in the layer "
              "(revisit debug_forward's setup).")
    else:
        print("MISMATCH -> flash output was correct but the module output is not, so the bug "
              "is in `out.reshape(*input_shape, -1)` or `o_proj` (head interleaving / layout).")
        # Localise: does a permutation of heads fix it? Compare token 0 head layouts.
        print(f"  nano[0,0,:8] = {a[0,0,:8].tolist()}")
        print(f"  hf  [0,0,:8] = {b[0,0,:8].tolist()}")


if __name__ == "__main__":
    main()
