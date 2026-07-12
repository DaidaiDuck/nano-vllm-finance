# tests/test_m3_vs_hf.py
"""M3 correctness regression: PagedAttention greedy output must match HuggingFace
generate() token-for-token.

This is M3's acceptance gate — it proves that swapping HF's attention for the
monkey-patched paged path (flash_attn_with_kvcache over a block-based KV cache) does
not change the result. Same contract M1 and M2 passed.

Gated (needs a real model + CUDA + flash-attn):
    NANO_VLLM_INTEGRATION=1 NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct \
        python -m pytest tests/test_m3_vs_hf.py -v

Notes
-----
- Default model is **3B**, not 0.5B. flash_attn accumulates in a different float order
  than HF's SDPA, so bf16 rounding can flip a *near-tie* greedy argmax. At 0.5B the
  argmax margins are small and late-token flips are common (numerics, not a bug); at 3B
  the margins are wide enough to match token-for-token. Override with NANO_VLLM_TEST_MODEL.
- The reference model is a **separate, unpatched** HF instance. `llm.engine.model` has its
  attention monkey-patched to the paged path, so calling `.generate()` on it would run M3
  again (M3-vs-M3), not a real HF reference.
"""
import os

import pytest
import torch

pytest.importorskip("flash_attn")  # M3 needs the paged FlashAttention kernel

pytestmark = pytest.mark.skipif(
    not os.getenv("NANO_VLLM_INTEGRATION") or not torch.cuda.is_available(),
    reason="set NANO_VLLM_INTEGRATION=1 and run on a CUDA GPU with flash-attn",
)

MODEL_NAME = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-3B-Instruct")
MAX_TOKENS = 32
PROMPTS = [
    "Hello",
    "What is 2 + 2?",
    "Explain photosynthesis in one sentence.",
]


@pytest.fixture(scope="module")
def llm():
    """The M3 engine. Constructing it monkey-patches its own model's attention."""
    from nano_vllm.paged.engine import LLM
    return LLM(MODEL_NAME)


@pytest.fixture(scope="module")
def hf_ref():
    """A SEPARATE, UNPATCHED HF model for the reference.

    Must not reuse llm.engine.model: M3 patches that instance's attention, so it is no
    longer plain HuggingFace. A fresh AutoModelForCausalLM keeps the reference honest.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return model, tokenizer


def _hf_greedy_ids(model, tokenizer, prompt):
    """Reproduce nano's greedy decode with HF generate(); return only the new tokens.

    Must mirror nano's input construction and decode settings 1:1 for a fair comparison:
    - same chat template + add_generation_prompt
    - do_sample=False to force greedy (Qwen's generation_config defaults to sampling)
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=MAX_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
    return out[0][input_ids.shape[1]:].tolist()


def _first_divergence(a, b):
    """Index of the first mismatch (or None). Helps tell a real bug from a late flip:
    divergence at index 0 => logic bug; a late index on a close call => bf16 numerics."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_matches_hf(llm, hf_ref, prompt):
    """M3 greedy token sequence must equal HF generate() token-for-token."""
    from nano_vllm import SamplingParams

    hf_model, hf_tokenizer = hf_ref

    nano = llm.generate(prompt, SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS))[0]
    hf_ids = _hf_greedy_ids(hf_model, hf_tokenizer, prompt)

    idx = _first_divergence(nano.token_ids, hf_ids)
    assert nano.token_ids == hf_ids, (
        f"token mismatch for {prompt!r} at index {idx} "
        f"(index 0 => logic bug; late index on a close call => bf16/kernel numerics):\n"
        f"  nano={nano.token_ids}\n  hf  ={hf_ids}"
    )
    assert nano.text == hf_tokenizer.decode(hf_ids)
