# Building nano-vllm from Scratch (M1): A Simple, Correct Single-Request Engine

**English** | [中文](m1_building_nano_vllm.zh.md)

> Series: reimplementing the core ideas of vLLM, one milestone at a time.
> **M1 — the baseline.** Get single-request LLM inference (prefill + decode)
> running correctly on top of HuggingFace.
> A clean, vLLM-style API, so every later optimization has a reference to beat.
>
> M1 design doc: [m1_design.md](../design/m1_design.md) · Code tag:
> [m1](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m1)

---

nano-vllm-finance rebuilds vLLM's core ideas from scratch to reveal how modern LLM
serving works. Part 1 (M1) is the baseline: a simple single-request engine that
runs Prefill + Decode using HuggingFace's model and KV cache, paired with a custom
Sampler and a clean `LLM` API. M1 has no custom KV cache, no PagedAttention, and no
Continuous Batching. Its single requirement is correctness: in Greedy mode, M1's
output must match HuggingFace **token-for-token**, providing a trustworthy reference
for every later optimization. This condition is enforced at every subsequent
milestone.

# Design rationale

## 1. Why rebuild vLLM?

The primary purpose is to understand how modern LLM inference works and why it is
structured the way it is, and to answer questions such as:

1. What is the structure of modern LLM inference, and why is it structured this way?
2. Why are there two distinct phases, Prefill and Decode?
3. What is a KV Cache, and what does it provide?
4. How should PagedAttention be understood?
5. What is Continuous Batching?

## 2. Why does M1 implement only correct single-request inference, not vLLM's core components?

M1's defining feature is simplicity. The components that make vLLM faster — a custom
KV Cache, PagedAttention, Continuous Batching, a Scheduler — are deliberately
excluded, because each is assigned to a different milestone. Introducing them in M1
would complicate its goal: correctness would then require validating both the
generation loop *and* the new component at once — debugging two hard things
simultaneously. Furthermore, the point is to compare the system with and without
each component; without that before/after contrast, the project loses its purpose.

## 3. What is M1's architecture?

```
LLM (user entry)
    └─ SimpleEngine
        └─ Tokenizer (HF)
        └─ Model (HF, use_cache=True) <- KV cache = HF DynamicCache
        └─ Sampler (custom, M1)
```

`LLM` is the user-facing entry point, wrapping a `SimpleEngine`. The engine
implements the Prefill + Decode loop and coordinates the tokenizer, the model, and
the Sampler. The KV cache is HuggingFace's `DynamicCache`.

## 4. How is the Prefill + Decode loop implemented?

Prefill is a single forward pass over the entire prompt; it produces the logits for
the next token and fills the KV cache with the prompt's keys/values. Decode then
proceeds token by token:

```python
# === Prefill ===
with torch.no_grad():
    outputs = self.model(
        input_ids = input_ids,
        use_cache = True,
    )
past_key_values = outputs.past_key_values
logits = outputs.logits[0, -1, :]
# Sample first token
next_token = self.sampler.sample(logits, params)

torch.cuda.synchronize()
yield next_token  # First token out.

# === Decode loop ===
for _ in range(params.max_tokens - 1):
    if next_token == self.tokenizer.eos_token_id:
        break

    inp = torch.tensor([[next_token]], device="cuda")
    with torch.no_grad():
        outputs = self.model(
            input_ids = inp,
            past_key_values = past_key_values,
            use_cache = True,
        )
    past_key_values = outputs.past_key_values  # Update KV cache
    logits = outputs.logits[0, -1, :]
    next_token = self.sampler.sample(logits, params)

    torch.cuda.synchronize()
    yield next_token
```

Each Decode step feeds only the newest token `next_token`, not the entire sequence.
`use_cache=True` retains all previous keys/values. The logits are read as
`outputs.logits[0, -1, :]`: `0` is the (single-request) batch index, `-1` is the
last position in the sequence, and `:` is the full vocabulary — because the model
emits a logits distribution at every position, but Decode only needs the logits of
the current text's last token.

## 5. Why a custom Sampler?

Mainly to follow vLLM's Sampler design and for learning purposes. The Sampler is a
short pipeline: when `temperature == 0` it takes the Greedy path; otherwise it
scales by temperature → applies top-k → applies top-p → softmax → multinomial.

## 6. How is single-request correctness verified?

A unit test in `tests/test_m1_vs_hf.py`: in Greedy mode, the engine's output token
ids must exactly match HuggingFace's output.

```python
# tests/test_m1_vs_hf.py — greedy must match HF exactly
assert nano.token_ids == hf_greedy_ids
```

One caveat: Qwen's `generation_config` defaults to sampling (`do_sample=True`,
`temperature=0.7`), so `do_sample=False` must be passed to force HF into Greedy —
otherwise the reference itself is stochastic and the test is meaningless. The full
M1 suite passes 23/23 (12 CPU unit + 11 GPU integration) on an A100 80GB SXM with
Qwen2.5-3B-Instruct, with `test_m1_vs_hf` matching on every prompt. This
token-for-token parity establishes the correctness of the generation loop and
serves as the non-regression contract every later milestone must keep green.

# M1 benchmarks

Three single-request scenarios were benchmarked (`Throughput` is output tokens/s):

| Scenario | Prompt | Output | Throughput | P99 latency | TTFT | TPOT |
|----------|--------|--------|------------|-------------|------|------|
| short_chat | ~125 | 100 | 29.4 tok/s | 3.62 s | 36 ms | 34.0 ms |
| medium_chat | ~526 | 200 | 29.2 tok/s | 7.38 s | 38 ms | 34.2 ms |
| long_context | ~1999 | 100 | 29.2 tok/s | 3.47 s | 89 ms | 33.7 ms |

## Analysis

Output throughput is constant at ~29 tokens/s; it approximately equals Decode speed
(1/TPOT).

TPOT stays ~34 ms, nearly unchanged from 125 to 2000 prompt tokens, because Decode
is memory-bound: almost all of each step's cost comes from loading the model weights
into the GPU (memory bandwidth), rather than from the compute of an attention pass
over the KV cache.

TTFT increases with prompt length (36 → 89 ms) because Prefill is compute-bound.
Note, however, that the 125-token and 526-token prompts have nearly identical TTFT,
which indicates that loading the model weights costs roughly 35 ms — so short-prompt
scenarios remain memory-bound. As the prompt grows, it shifts to compute-bound; at
2000 tokens the TTFT rises to 89 ms.

Core conclusion:

```
latency ≈ TTFT + output_len * TPOT
```

This formula separates the two physical phases — compute-bound Prefill (counted in
TTFT) and memory-bound Decode — and explains why `long_context` and `short_chat`
have nearly identical latency despite a 16× difference in prompt size: both emit 100
tokens, and the output length dominates wall-clock time. It also explains why
`medium_chat` takes the longest, since its output length is 200 tokens.

Final note: this ~29 tokens/s output throughput is a baseline, not a headline
figure. It is held down by roughly three factors — a per-token
`cuda.synchronize()` for timing accuracy, no batching, and no custom KV cache — all
of which are optimization opportunities for later milestones.

# M1's limitations

1. Single request only — `batch=1` is hardcoded; no concurrency.
2. Relies on HuggingFace's KV Cache rather than a custom implementation —
   HuggingFace's KV Cache uses `torch.cat`, which yields poor performance; this is
   covered in M2.
3. No Prefix Caching — repeated prompts are recomputed from scratch.

# Appendix / notes

- Design doc: [m1_design.md](../design/m1_design.md)
- Benchmark setup: [benchmark_environment.md](../design/benchmark_environment.md)
