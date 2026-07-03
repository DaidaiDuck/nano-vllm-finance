# Building nano-vllm from Scratch, Part 1: A Correct Single-Request Engine

**English** | [中文](m1_building_nano_vllm.zh.md)

> Series: reimplementing the core ideas of vLLM, one milestone at a time.
> **M1 — the baseline.** Get single-request LLM inference (prefill + decode)
> running correctly on top of HuggingFace, with a clean vLLM-style API, so every
> later optimization has a verified reference to beat.
>
> Design doc: [m1_design.md](../design/m1_design.md) · Code tag:
> [m1](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m1)

---

## TL;DR

nano-vllm reconstructs the core of vLLM from scratch to expose how modern LLM
serving works. Part 1 (M1) is the baseline: a single-request engine that performs
prefill + decode on top of a HuggingFace model, paired with a custom sampler and a
clean `LLM` API. It deliberately omits every optimization — no custom KV cache, no
paging, no batching. Its single requirement is correctness: in greedy mode the
output matches HuggingFace **token-for-token**, giving every later optimization a
trustworthy reference to beat.

## 1. Why rebuild vLLM?

vLLM is fast, but speed alone does not convey understanding. The goal of this series
is to explain *why* LLM inference is structured the way it is — why there are
distinct prefill and decode phases, what a KV cache actually provides, how sampling
and streaming work — by reconstructing the system from first principles, one
milestone at a time. M1 is the foundation; later parts address the KV cache, then
paging, then batching. A flawed foundation propagates its bugs into everything built
on top of it, so M1's sole objective is correctness.

## 2. Correctness before speed

Simplicity is M1's defining virtue. The components that make vLLM fast — a custom KV
cache, PagedAttention, continuous batching, a scheduler — are intentionally left
out. Each is a milestone in its own right, and introducing one now would mean
debugging two hard problems simultaneously: the generation loop *and* the
optimization. Delegating everything except the parts under study to HuggingFace
allows the loop to be validated in isolation and gives each later optimization a
clean before/after comparison.

## 3. Architecture

The engine is three components behind a facade:

```
LLM (user entry)
  └─ SimpleEngine
       ├─ Tokenizer (HF)
       ├─ Model (HF, use_cache=True)   ← KV cache = HF DynamicCache
       └─ Sampler (custom)
```

`LLM` is the user-facing entry point (vLLM-style: `llm.generate(prompt, params)`).
It wraps a `SimpleEngine` that owns the prefill + decode loop and coordinates the
tokenizer, the model, and the sampler. The KV cache is HuggingFace's built-in
implementation, reused via `use_cache=True` rather than reimplemented.

## 4. The generation loop: prefill + decode

The generation loop is the core of M1, and it runs in two phases. **Prefill** is a
single forward pass over the entire prompt; it produces the logits for the next
token and populates the KV cache with the prompt's keys and values. **Decode** then
proceeds one token at a time:

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

The essential technique is that each decode step feeds **only the new token**, not
the entire sequence. `use_cache=True` retains every previous key/value, so step *n*
performs O(1) input work rather than re-reading all *n* tokens. The relevant logits
are read as `logits[0, -1, :]` — batch 0, the *last* position, the full vocabulary —
because the model emits a distribution at every position, but only the one following
the current text is needed. The loop terminates on the EOS token or upon reaching
`max_tokens`.

## 5. A hand-written sampler

HuggingFace's `generate()` is deliberately avoided: it is a black box that is
awkward to customize, and batched sampling (a later milestone) must be hand-written
regardless, so an in-house implementation is preferable to one that would later be
discarded. The `Sampler` is a short pipeline: a greedy shortcut when
`temperature == 0` (a plain `argmax`), otherwise temperature scaling → top-k → top-p
→ softmax → `multinomial`. The subtle parts are the masking details — filtering
writes `-inf` into rejected logits so that softmax zeros them, and top-p requires a
shift-by-one so the token that *crosses* the threshold is retained rather than
dropped.

## 6. The chat-template requirement

A frequent mistake is feeding a raw prompt directly to an instruction-tuned model.
Such models are trained on a specific conversation format with special tokens;
without it, the model *continues* the text instead of answering it. The fix is the
tokenizer's chat template:

```python
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
```

`add_generation_prompt=True` appends the assistant turn opener (e.g.
`<|im_start|>assistant\n`), signaling that it is the model's turn to respond.
Omitting it produces degenerate output — a common and easily overlooked failure
mode.

## 7. Streaming and TTFT measurement

Streaming is not synonymous with HTTP or SSE — a Python generator *is* streaming.
Converting the loop's `return` into `yield` is essentially free and exposes the one
quantity required to measure latency correctly: the exact instant the first token is
produced. This is TTFT (Time To First Token), which cannot be measured without a
token-by-token interface. One subtlety applies — CUDA kernels execute
asynchronously, so `torch.cuda.synchronize()` is called before the timestamp is
taken; otherwise the measurement would reflect when the work was *queued*, not when
the GPU completed it.

## 8. Correctness: nano-vllm == HuggingFace

The most important test asserts token-level parity:

```python
# tests/test_m1_vs_hf.py — greedy must match HF exactly
assert nano.token_ids == hf_greedy_ids
```

In greedy mode, the engine's output must match HuggingFace **token-for-token**. One
caveat applies: Qwen's `generation_config` defaults to sampling (`do_sample=True`,
`temperature=0.7`), so `do_sample=False` must be passed to force HF into greedy —
otherwise the reference is stochastic and the test is meaningless. The full M1 suite
passes **23/23** (12 CPU unit + 11 GPU integration) on an A100 80GB SXM with
Qwen2.5-3B-Instruct, with `test_m1_vs_hf` matching across every prompt. This
token-for-token parity establishes that the generation loop is correct, and it
serves as the non-regression contract every later milestone must keep green.

## 9. Benchmarks

Three single-request scenarios were benchmarked (`Throughput` is output tokens/s):

| Scenario | Prompt | Output | Throughput | P99 latency | TTFT | TPOT |
|----------|--------|--------|------------|-------------|------|------|
| short_chat | ~125 | 100 | 29.4 tok/s | 3.62 s | 36 ms | 34.0 ms |
| medium_chat | ~526 | 200 | 29.2 tok/s | 7.38 s | 38 ms | 34.2 ms |
| long_context | ~1999 | 100 | 29.2 tok/s | 3.47 s | 89 ms | 33.7 ms |

The results are consistent. **Throughput is flat at ~29 tok/s** irrespective of
prompt length — for a single request it equals decode speed (≈ 1/TPOT). **TPOT
remains ~34 ms** from 125 to 2000 prompt tokens, because decode is
memory-bandwidth-bound: each step is dominated by streaming the model weights rather
than by attention over the cache. **TTFT scales with prompt length** (36 → 89 ms)
because prefill is compute-bound, though a ~35 ms fixed overhead masks this until the
prompt grows large.

The central result is that end-to-end latency follows a simple law, verified to
within 1%:

```
latency ≈ TTFT + output_len × TPOT
```

This formula separates the two physical phases — compute-bound prefill (TTFT) and
memory-bound decode (TPOT) — and explains why `long_context` and `short_chat`
exhibit nearly identical latency despite a 16× difference in prompt size: both emit
100 tokens, and the *output* length dominates wall-clock time. One caveat: ~29 tok/s
is a baseline, not a headline figure — a per-token `cuda.synchronize()` is used for
timing accuracy, and no batching is present.

## 10. Deferred to later milestones

M1 is intentionally minimal. Its explicit limitations:

- **Single request only** — `batch=1` is hardcoded; no concurrency.
- **Relies on HuggingFace's built-in KV cache** rather than a custom one.
- **No HTTP serving** — streaming is in-process via a generator.
- **No prefix caching** — repeated prompts are recomputed from scratch.
- **EOS is the only stop condition** — no custom stop strings.

None of these is a defect in M1; they constitute the roadmap. With a correct,
measured baseline in place and a test that guarantees it cannot be silently broken,
subsequent parts replace these components one at a time — beginning with the KV
cache.

---

## Appendix / notes

- Design doc: [m1_design.md](../design/m1_design.md)
- Benchmark setup: [benchmark_environment.md](../design/benchmark_environment.md)
- Next: **Part 2 — the KV cache** ([m2_design.md](../design/m2_design.md))
