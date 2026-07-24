"""Confirm the late batch>=4 divergence is a near-tie argmax flip (bf16/flash numerics),
not a logic bug.

test_m4_vs_hf now matches HF token-for-token at batch 1/2, and 'Name three primary colors.'
diverges only late (index 20-27) at batch >= 4, on coherent phrasing. If that divergence is a
genuine near-tie, the top-2 HF logits at that position are almost equal -- so which token wins
is decided by rounding, and flash's different accumulation legitimately flips it. This prints
the HF top-2 margin at the first divergence point; a tiny margin (<~1.0) confirms numerics.

Run on the pod:
    NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct python scripts/debug_numerics.py
"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_vllm import SamplingParams
from nano_vllm.paged.engine import LLM

MODEL = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-3B-Instruct")
MAX_TOKENS = 32
POOL = [
    "Hello", "What is 2 + 2?", "Explain photosynthesis in one sentence.",
    "Name three primary colors.", "Write a haiku about the sea.",
    "Why is the sky blue? Answer briefly.", "List the first five prime numbers.",
    "Translate 'good morning' into French.",
]
PROBE = "Name three primary colors."


def _first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    hf.eval()

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROBE}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = hf.generate(prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                          do_sample=False, max_new_tokens=MAX_TOKENS,
                          pad_token_id=tokenizer.eos_token_id)
    hf_ids = gen[0][prompt_ids.shape[1]:].tolist()

    # nano in a batch of 8 (where it diverges)
    llm = LLM(MODEL)
    base = llm.counter
    outs = llm.generate(POOL, SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS))
    by_index = {int(o.request_id) - base - 1: o for o in outs}
    nano_ids = by_index[POOL.index(PROBE)].token_ids

    idx = _first_divergence(nano_ids, hf_ids)
    print(f"\nfirst divergence at index {idx}")
    if idx is None:
        print("no divergence -- nothing to check")
        return
    print(f"  nano token {nano_ids[idx]} = {tokenizer.decode([nano_ids[idx]])!r}")
    print(f"  hf   token {hf_ids[idx]} = {tokenizer.decode([hf_ids[idx]])!r}")

    # Re-run HF up to the divergence point and inspect the logit margin at that step.
    full = prompt_ids[0].tolist() + hf_ids[:idx]
    with torch.no_grad():
        logits = hf(torch.tensor([full], device="cuda")).logits[0, -1].float()
    top = torch.topk(logits, 2)
    margin = (top.values[0] - top.values[1]).item()
    print(f"\nHF top-2 logits at the divergence step: "
          f"{tokenizer.decode([top.indices[0]])!r} {top.values[0]:.3f}  vs  "
          f"{tokenizer.decode([top.indices[1]])!r} {top.values[1]:.3f}")
    print(f"margin = {margin:.4f}")
    print("  margin < ~1.0 -> near-tie: the divergence is bf16/flash numerics, not a bug."
          if margin < 1.0 else
          "  margin large -> NOT a near-tie; investigate as a real difference.")


if __name__ == "__main__":
    main()
