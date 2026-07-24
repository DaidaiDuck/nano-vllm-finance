"""Confirm a vs-HF divergence is a near-tie argmax flip (bf16/flash numerics), not a bug --
for ANY prompt and at ANY index (creative prompts hit near-ties early, so index alone can't
tell bug from numerics; the logit margin can).

For the probe prompt it prints:
  1. first divergence (nano-in-batch vs HF) and HF's top-2 logit margin there
  2. batch invariance: nano ALONE (batch=1) vs nano IN THE BATCH -- if identical, nano is
     self-consistent and the divergence is purely vs-HF numerics; if not, the batch changes
     the result (still numerics if the margin is a near-tie, since flash-varlen's float order
     depends on batch composition).

A margin below ~1.0 confirms numerics regardless of index.

Run on the pod (set the probe if you want a different prompt):
    NANO_VLLM_PROBE="Write a haiku about the sea." \
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
PROBE = os.getenv("NANO_VLLM_PROBE", "Name three primary colors.")


def _first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _nano_ids(llm, prompts, probe):
    base = llm.counter
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS))
    by_index = {int(o.request_id) - base - 1: o for o in outs}
    return by_index[prompts.index(probe)].token_ids


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

    llm = LLM(MODEL)
    nano_batch = _nano_ids(llm, POOL, PROBE)          # in the batch of 8
    nano_alone = _nano_ids(llm, [PROBE], PROBE)       # batch = 1

    print(f"\nprobe: {PROBE!r}")

    # (2) batch invariance
    bi = _first_divergence(nano_alone, nano_batch)
    if bi is None:
        print("batch invariance: nano ALONE == nano IN BATCH  (self-consistent)")
    else:
        print(f"batch invariance: nano alone vs in-batch diverge at index {bi} "
              f"(flash-varlen float order depends on batch -- check margin below)")

    # (1) vs HF + margin
    idx = _first_divergence(nano_batch, hf_ids)
    if idx is None:
        print("vs HF: no divergence")
        return
    print(f"\nvs HF: first divergence at index {idx}")
    print(f"  nano {nano_batch[idx]} = {tokenizer.decode([nano_batch[idx]])!r}   "
          f"hf {hf_ids[idx]} = {tokenizer.decode([hf_ids[idx]])!r}")

    full = prompt_ids[0].tolist() + hf_ids[:idx]
    with torch.no_grad():
        logits = hf(torch.tensor([full], device="cuda")).logits[0, -1].float()
    top = torch.topk(logits, 2)
    margin = (top.values[0] - top.values[1]).item()
    print(f"  HF top-2: {tokenizer.decode([top.indices[0]])!r} {top.values[0]:.3f}  vs  "
          f"{tokenizer.decode([top.indices[1]])!r} {top.values[1]:.3f}   margin = {margin:.4f}")
    print("  -> near-tie: numerics, not a bug." if margin < 1.0
          else "  -> NOT a near-tie: investigate as a real difference.")


if __name__ == "__main__":
    main()
