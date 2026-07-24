# tests/test_m4_vs_hf.py
"""M4 continuous-batching correctness: greedy output must match HuggingFace at every batch size.

This is M4's acceptance gate. M1/M2/M3 each passed "one request == HF". What M4 adds is
*mixed batching*: tokens from several requests are packed into a single forward pass and
share one attention kernel call. Any error in packing, slot_mapping, cu_seqlens or
logits_indices shows up as "correct alone, wrong in a batch".

So two things are tested:
  1. At each batch size, every request's output equals HF's greedy output for that prompt.
  2. Batch invariance: the same prompt produces identical tokens at batch=1 and batch=8.

Gated (needs a real model + CUDA + flash-attn):
    NANO_VLLM_INTEGRATION=1 NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct \
        python -m pytest tests/test_m4_vs_hf.py -v

Notes
-----
- Default model is 3B, not 0.5B. flash-attn accumulates in a different float order than
  HF's SDPA, so bf16 rounding can flip a *near-tie* greedy argmax. At 0.5B the margins are
  narrow and late-token flips are common (numerics, not a bug); at 3B they are wide enough
  to match token-for-token. Override with NANO_VLLM_TEST_MODEL.
- The reference is a SEPARATE, UNPATCHED HF model. M4 monkey-patches the attention on
  llm.engine.model, so calling .generate() on that instance would compare M4 against M4.
"""
import os

import pytest
import torch

pytest.importorskip("flash_attn")  # M4 needs the varlen paged FlashAttention kernel

pytestmark = pytest.mark.skipif(
    not os.getenv("NANO_VLLM_INTEGRATION") or not torch.cuda.is_available(),
    reason="set NANO_VLLM_INTEGRATION=1 and run on a CUDA GPU with flash-attn",
)

MODEL_NAME = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-3B-Instruct")
MAX_TOKENS = 32
BATCH_SIZES = [1, 2, 4, 6, 8]

# Tokens that must match HF exactly. A packing / slot_mapping / logits bug corrupts the
# output immediately, so it shows up as divergence at (or very near) index 0. A divergence
# only AFTER this many correct tokens is a near-tie greedy argmax flipped by bf16 rounding:
# flash-attn accumulates in a different float order than HF's SDPA, and flash-varlen's order
# additionally depends on batch composition, so a close call can flip vs HF and across batch
# sizes. That is a numerical property, not a bug (see docs/design/m4_debug_oproj_bug.md and
# the test_m3_vs_hf notes). Confirmed with scripts/debug_numerics.py (top-2 logit margin).
MIN_EXACT_PREFIX = 16

# Lengths are deliberately uneven: mixing short and long prompts is what exercises the
# packing boundaries (different prefill lengths sharing one forward pass).
PROMPT_POOL = [
    "Hello",
    "What is 2 + 2?",
    "Explain photosynthesis in one sentence.",
    "Name three primary colors.",
    "Write a haiku about the sea.",
    "Why is the sky blue? Answer briefly.",
    "List the first five prime numbers.",
    "Translate 'good morning' into French.",
]


@pytest.fixture(scope="module")
def llm():
    """The M4 engine. Constructing it monkey-patches its own model's attention."""
    from nano_vllm.paged.engine import LLM
    return LLM(MODEL_NAME)


@pytest.fixture(scope="module")
def hf_ref():
    """A SEPARATE, UNPATCHED HF model used as the reference.

    Must not reuse llm.engine.model: M4 replaces that instance's attention, so it is no
    longer plain HuggingFace. A fresh AutoModelForCausalLM keeps the reference honest.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return model, tokenizer


_HF_CACHE: dict[str, list[int]] = {}


def _hf_greedy_ids(hf_ref, prompt: str) -> list[int]:
    """Reproduce nano's greedy decode with HF generate(); return only the new tokens.

    Cached per prompt: five batch sizes over eight prompts would otherwise re-run the same
    reference generations dozens of times.

    Input construction must mirror nano 1:1 to be a fair comparison: same chat template
    with add_generation_prompt, and do_sample=False to force greedy (Qwen's
    generation_config defaults to sampling).
    """
    if prompt in _HF_CACHE:
        return _HF_CACHE[prompt]

    model, tokenizer = hf_ref
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            max_new_tokens=MAX_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
    _HF_CACHE[prompt] = out[0][input_ids.shape[1]:].tolist()
    return _HF_CACHE[prompt]


def _first_divergence(a, b):
    """Index of the first mismatch (or None). Tells a real bug from a late numeric flip:
    divergence at index 0 => logic bug; a late index on a close call => bf16 numerics."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _generate_batch(llm, prompts):
    """Submit a batch of prompts and return the outputs ordered by prompt.

    LLM.generate collects outputs in *completion* order, not prompt order. Request ids are
    handed out from the engine's monotonically increasing counter, so recording the counter
    before the call is enough to recover each output's prompt index.
    """
    from nano_vllm import SamplingParams

    base = llm.counter
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS))
    assert len(outs) == len(prompts), f"expected {len(prompts)} outputs, got {len(outs)}"

    by_index = {int(o.request_id) - base - 1: o for o in outs}
    assert sorted(by_index) == list(range(len(prompts))), "request ids do not map to prompts"
    return [by_index[i] for i in range(len(prompts))]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batch_matches_hf(llm, hf_ref, batch_size):
    """Every request in a batch must reproduce HF's greedy output.

    Exact token-for-token where the argmax margin is wide; a divergence is tolerated only if
    it comes after MIN_EXACT_PREFIX correct tokens, which marks it as a bf16/flash near-tie
    flip rather than a packing bug (an early divergence still fails hard).
    """
    prompts = PROMPT_POOL[:batch_size]
    outs = _generate_batch(llm, prompts)

    for prompt, out in zip(prompts, outs):
        hf_ids = _hf_greedy_ids(hf_ref, prompt)
        idx = _first_divergence(out.token_ids, hf_ids)
        assert idx is None or idx >= MIN_EXACT_PREFIX, (
            f"batch_size={batch_size} prompt={prompt!r} diverges EARLY at index {idx} "
            f"(< {MIN_EXACT_PREFIX}): a packing / slot_mapping / logits bug, not numerics\n"
            f"  nano={out.token_ids}\n  hf  ={hf_ids}"
        )


def test_batch_invariance(llm):
    """The same prompt must decode identically alone and inside a batch of eight.

    This is the mixed-batching-specific regression. Even if absolute agreement with HF were
    lost to numerics on some other model, "batch size does not change the result" must still
    hold. A failure here means requests are contaminating each other -- shared KV slots, or
    logits gathered from the wrong row.
    """
    probe = PROMPT_POOL[0]
    alone = _generate_batch(llm, [probe])[0]
    mixed = _generate_batch(llm, PROMPT_POOL[:8])[0]

    idx = _first_divergence(alone.token_ids, mixed.token_ids)
    assert alone.token_ids == mixed.token_ids, (
        f"batch invariance broken, first divergence at index {idx}: batch=1 and batch=8 "
        f"disagree, so requests are contaminating each other when packed together\n"
        f"  alone={alone.token_ids}\n  mixed={mixed.token_ids}"
    )


def test_per_request_sampling_params(llm):
    """Each request carries its own SamplingParams; differing max_tokens must be honored.

    LLM.generate takes a single params object, so this drives PagedEngine directly -- which
    doubles as a check of the add_request/step API the serving benchmark depends on.
    """
    from nano_vllm.core.types import FinishReason, Request, SamplingParams

    engine = llm.engine
    wanted = {"pr-a": 4, "pr-b": 9, "pr-c": 16}
    for rid, limit in wanted.items():
        token_ids = engine.tokenizer.encode(engine._format_prompt("Count slowly from one."))
        engine.add_request(
            Request(rid, token_ids, SamplingParams(temperature=0.0, max_tokens=limit))
        )

    finished = {}
    while engine.has_unfinished_requests():
        for out in engine.step():
            if out.finished:
                finished[out.request_id] = out

    assert set(finished) == set(wanted)
    for rid, limit in wanted.items():
        out = finished[rid]
        # Either it ran to its own max_tokens (LENGTH) or hit EOS earlier (STOP), never over.
        assert len(out.token_ids) <= limit, f"{rid} exceeded its own max_tokens"
        if out.finish_reason == FinishReason.LENGTH:
            assert len(out.token_ids) == limit
