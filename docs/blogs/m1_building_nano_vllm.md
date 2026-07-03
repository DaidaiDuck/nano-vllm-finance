# Building nano-vllm from Scratch, Part 1: A Correct Single-Request Engine

> Series: reimplementing the core ideas of vLLM, one milestone at a time.
> **M1 — the baseline.** Get single-request LLM inference (prefill + decode)
> running correctly on top of HuggingFace, with a clean vLLM-style API, so every
> later optimization has a verified reference to beat.
>
> Design doc: [m1_design.md](../design/m1_design.md) · Code tag:
> [m1](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m1)

---

## TL;DR

<!-- 3–5 sentences: what M1 is, what it deliberately is NOT, and the one
     takeaway (correctness first, HF-identical greedy output). Fill in last. -->

- TODO

---

## 1. Why build another vLLM?

<!-- Motivation. Learning by reconstruction; understand prefill/decode, KV cache,
     sampling, streaming from first principles. State the milestone roadmap
     (M1 baseline → M2 custom KV cache → M3 PagedAttention → M4 batching ...). -->

- TODO

## 2. The goal of M1: correctness, not speed

<!-- The design principle: simplicity is M1's virtue. Explain the non-goals
     (no custom KV cache, no paging, no batching) and why deferring them is the
     right call. Reference m1_design.md §2. -->

- TODO

## 3. Architecture in one diagram

<!-- Drop in the ASCII diagram from the design doc; walk through LLM →
     SimpleEngine → (Tokenizer, Model, Sampler) → GPU. Keep it short. -->

```
LLM (user entry)
  └─ SimpleEngine
       ├─ Tokenizer (HF)
       ├─ Model (HF, use_cache=True)   ← KV cache = HF DynamicCache
       └─ Sampler (custom)
```

- TODO

## 4. The generation loop: prefill + decode

<!-- The heart of the post. Explain:
     - prefill = one forward over the whole prompt, fills the KV cache
     - decode  = feed ONE token at a time, reuse the cache (use_cache=True)
     - why we only feed the new token (the O(1)-per-step trick the cache buys)
     Include the generate_stream code sketch and the "first token = TTFT" marker. -->

```python
# prefill once, then decode one token at a time reusing the KV cache
outputs = model(input_ids, use_cache=True)          # prefill
next_token = sampler.sample(outputs.logits[0, -1])  # first token
yield next_token
for _ in range(max_tokens - 1):
    outputs = model([[next_token]], past_key_values=past_kv, use_cache=True)
    next_token = sampler.sample(outputs.logits[0, -1])
    yield next_token
```

- TODO: explain `use_cache`, why `logits[0, -1, :]`, EOS handling.

## 5. Sampling, implemented by hand

<!-- Why not HF generate(): black box, and M4 needs batched sampling anyway.
     Walk the pipeline: greedy shortcut → temperature → top-k → top-p → softmax
     → multinomial. One or two gotchas (in-place masking, top-p shift-by-one). -->

- TODO

## 6. The chat-template trap

<!-- The single most common beginner bug: feeding the raw prompt to an
     instruction-tuned model. Show apply_chat_template + add_generation_prompt,
     and what garbled output looks like without it. -->

- TODO

## 7. Streaming for free (and why it matters for TTFT)

<!-- Streaming ≠ HTTP/SSE. A Python generator IS streaming. It's what lets us
     measure the exact instant the first token appears (TTFT). Mention the
     torch.cuda.synchronize() timing subtlety. -->

- TODO

## 8. Proving it's correct: nano-vllm == HuggingFace

<!-- The most important test. Greedy output must match HF token-for-token.
     Explain the do_sample=False gotcha (Qwen's generation_config defaults to
     sampling). This test becomes the non-regression contract for M2–M6. -->

```python
# tests/test_m1_vs_hf.py — greedy must match HF exactly
assert nano.token_ids == hf_greedy_ids
```

- TODO

## 9. Benchmarks

<!-- PENDING GPU RUN. Pull numbers from benchmarks/results/m1/ once available.
     Environment is fixed in design/benchmark_environment.md (A100 80GB SXM,
     Qwen2.5-3B bf16, PyTorch 2.4 / CUDA 12.4). Keep placeholders until then. -->

> **TBD** — benchmarks not yet run. Environment:
> [benchmark_environment.md](../design/benchmark_environment.md).

| Scenario | Throughput | P99 latency | TTFT | TPOT |
|----------|-----------|-------------|------|------|
| short_chat | — | — | — | — |
| medium_chat | — | — | — | — |
| long_context | — | — | — | — |

Expected qualitative shape (to confirm): TTFT grows ~linearly with prompt length
(prefill is compute-bound); TPOT is roughly flat across scenarios (decode is
memory-bandwidth-bound).

## 10. What M1 deliberately leaves broken (→ M2)

<!-- The honest limitations section. Lead with the KV cache: HF DynamicCache
     grows via torch.cat, O(n²) copy + memory jitter. This is the cliffhanger
     that sets up M2 (custom pre-allocated cache). Tease the M1-vs-M2 numbers. -->

- The KV cache is HF's `DynamicCache`: it grows by `torch.cat` every step —
  O(n²) copy traffic and memory jitter. **M2 replaces it.**
- Single request only; batch=1 hardcoded; no HTTP; no prefix caching.

- TODO

---

## Appendix / notes

- Design doc: [m1_design.md](../design/m1_design.md)
- Benchmark setup: [benchmark_environment.md](../design/benchmark_environment.md)
- Next: **Part 2 — a custom pre-allocated KV cache** ([m2_design.md](../design/m2_design.md))
