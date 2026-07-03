# M1 Design: Simple Single-Request Engine

> **Status**: Completed (2025-XX-XX)
> **Tag**: [m1](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m1)
> **Blog**: [Building nano-vllm Part 1](https://...)

## 1. Overview

M1 is the first milestone of nano-vllm-finance. The objective is to bring up an
**end-to-end single-request inference path (prefill + decode)** with the smallest
amount of code that is still correct, establishing a verified baseline for every
optimization that follows.

Core implementation:

- Built on a HuggingFace `transformers` causal LM
- Reuses HuggingFace `DynamicCache` for KV cache management
- Custom sampler (greedy / temperature / top-k / top-p)
- Custom generation loop (prefill + autoregressive decode)
- Token-level streaming via Python generators

Approximate footprint: ~350 lines including tests.

## 2. Goals & Non-Goals

### Goals

- ✅ Run single-request generation end to end, **bit-for-bit identical to
  HuggingFace** in greedy mode
- ✅ Implement sampling in-house rather than relying on HF `generate()`
- ✅ Support token-level streaming as the basis for accurate TTFT measurement
- ✅ Expose a clean, vLLM-style `LLM` API
- ✅ Keep the design modular so M2+ can swap internals without API churn

### Non-Goals (intentionally deferred)

- ❌ Custom KV cache management — **M2**
- ❌ PagedAttention — **M3**
- ❌ Continuous batching / concurrent requests — **M4**
- ❌ Scheduler abstraction — **M4**
- ❌ Quantization-specific paths — **M5**
- ❌ HTTP serving — **M6**
- ❌ Multimodal support — **V2**

**Design principle**: simplicity is M1's virtue. If the foundation is unsound,
M2–M6 inherit the damage.

## 3. Architecture

```
┌─────────────────────────────────────────┐
│              M1 Architecture             │
│                                          │
│  LLM (user entry)                        │
│   │                                      │
│   ↓                                      │
│  SimpleEngine                            │
│   ├─ Tokenizer (HF)                      │
│   ├─ Model (HF, use_cache=True)          │
│   └─ Sampler (custom)                    │
│   │                                      │
│   ↓                                      │
│  GPU                                     │
└─────────────────────────────────────────┘
```

### Components

#### `LLM` (`engine.py`)

User-facing entry point following the facade pattern; wraps a `SimpleEngine`.

```python
llm = LLM("Qwen/Qwen2.5-3B-Instruct")
output = llm.generate(prompt, params)
```

#### `SimpleEngine` (`engine.py`)

Coordinates tokenizer, model, and sampler, and owns the prefill + decode loop.
Two core methods:

- `generate_stream()` — runs prefill + decode and yields one token id at a time
  (streaming / benchmarking)
- `generate()` — runs prefill + decode and returns a complete `RequestOutput`
  (collected `token_ids` + decoded `text`)

The two methods are **separate implementations** of the same prefill/decode loop —
`generate()` does **not** call `generate_stream()`. Their outputs must stay
identical; this is enforced by a test
(`tests/test_engine.py::test_generate_stream_matches_generate`) rather than by
shared code. The duplication is a known cleanup item for a later milestone.

#### `Sampler` (`sampler.py`)

Selects the next token from a logits vector. Supports:

- Greedy (`temperature == 0`)
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) filtering

Application order: `temperature → top_k → top_p → softmax → multinomial`.

#### KV Cache

**Not implemented in M1.** HuggingFace `DynamicCache` is reused and threaded
through the loop via `past_key_values`. Replacing it with a custom cache is the
defining task of M2.

## 4. Key Design Decisions

### Decision 1: Reuse HF `DynamicCache` instead of a custom KV cache

**Why**:

- M1's goal is *correct bring-up*, not performance.
- A custom cache is the core deliverable of M2; building it now would dilute the
  milestone and conflate two sources of risk.
- Reusing the HF cache lets us validate the generation loop in isolation.
- It leaves M2 with a clear, measurable win: swap the cache and benchmark the
  delta.

**Trade-off**:

- M1 throughput does not reflect nano-vllm's potential.
- `DynamicCache` grows the cache by `torch.cat` on every decode step, which
  reallocates and copies the entire K/V tensor each time. This causes repeated
  allocator churn (memory "jitter") and O(n²) copy traffic over a sequence —
  exactly the inefficiency M2 removes with a pre-allocated contiguous buffer.

### Decision 2: Custom `Sampler` instead of HF `generate()`

**Why**:

- HF `generate()` is effectively a black box and hard to customize.
- Batched sampling in M4 must be implemented in-house regardless; writing the
  sampler now lets it *evolve* rather than be thrown away and rewritten.
- Educational value: explicit control over the sampling pipeline.

**Trade-off**:

- We own the boundary conditions (EOS handling, `max_tokens`), but the code is
  small (~50 lines).

### Decision 3: Token-level streaming via Python generators

**Why**:

- Accurate TTFT (Time To First Token) requires the precise instant the first
  token is produced; only a streaming interface exposes it.
- A Python generator `yield` is the simplest possible streaming mechanism — no
  need to wait for HTTP streaming in M6.
- Effectively zero cost: turning `return` into `yield` is the whole change.

**Trade-off**:

- Synchronous streaming only; no concurrent requests until M4.
- Requires `torch.cuda.synchronize()` before timing so the measured instant
  reflects completed GPU work, not just kernel enqueue.

**Insight**: streaming ≠ SSE/HTTP. A Python generator *is* streaming.

### Decision 4: No Scheduler / KVCacheManager / ModelRunner abstractions

**Why**:

- These abstractions only earn their complexity once multiple requests are
  batched together.
- For a single request they would be over-engineering for a requirement that
  does not yet exist.
- vLLM's elaborate architecture is driven by production-scale needs M1 has not
  reached. KISS.

**Future**: M4 introduces the `Scheduler` and `ModelRunner` abstractions.

### Decision 5: bfloat16, not float16

**Why**:

- bfloat16 is the dtype Qwen2.5 is officially recommended to run in.
- bf16 shares fp32's exponent range, so it is far less prone to overflow.
- fp16 frequently produces `inf` during LLM inference, corrupting output.
- bf16 is natively supported on RTX 4090 / A100 / H100.

**Trade-off**:

- Same memory footprint as fp16 (2 bytes/element).
- Slightly lower mantissa precision than fp16, but LLM inference is insensitive
  to it.

## 5. Implementation Details

### Generation loop

```python
def generate_stream(prompt, params):
    # 1. Tokenize with the model's chat template
    input_ids = tokenize_with_template(prompt)

    # 2. Prefill (single forward over the whole prompt)
    outputs = model(input_ids, use_cache=True)
    past_kv = outputs.past_key_values

    # 3. Sample the first token
    logits = outputs.logits[0, -1, :]
    next_token = sampler.sample(logits, params)
    yield next_token  # ← TTFT marker

    # 4. Decode loop
    for _ in range(max_tokens - 1):
        if next_token == EOS:
            break
        # Only the new token is fed; history lives in the KV cache
        outputs = model(
            input_ids=[[next_token]],
            past_key_values=past_kv,
            use_cache=True,
        )
        past_kv = outputs.past_key_values
        next_token = sampler.sample(outputs.logits[0, -1, :], params)
        yield next_token
```

### Sampler

```python
class Sampler:
    def sample(self, logits, params):
        # Greedy shortcut
        if params.temperature == 0.0:
            return logits.argmax().item()

        # Order: temperature → top_k → top_p → softmax → multinomial
        logits = logits / params.temperature
        logits = apply_top_k(logits, params.top_k)
        logits = apply_top_p(logits, params.top_p)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()
```

### Chat template handling

Instruction-tuned models **must** receive input wrapped in their chat template:

```python
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,            # return a string; encode separately for control
    add_generation_prompt=True,  # append the assistant turn opener
)
input_ids = tokenizer.encode(text, return_tensors="pt")
```

Skipping the template yields garbled output — a common first-timer mistake — because
the model is trained to respond only to its specific conversation format.

## 6. Testing Strategy

### Unit tests (`tests/test_sampler.py`, `tests/test_types.py`)

CPU-only, no model required:

- Greedy determinism (argmax)
- Top-k restricts candidates to the k highest-scoring tokens
- Top-p restricts to the nucleus
- `SamplingParams` / `RequestOutput` field semantics

### Correctness: nano-vllm vs HuggingFace (`tests/test_m1_vs_hf.py`)

**The most important test.** Greedy output must match HF **token-for-token**. The
test reuses the same loaded model/tokenizer, replicates nano's exact input
construction (chat template + `add_generation_prompt`), and forces HF greedy with
`do_sample=False` — required because Qwen's `generation_config` defaults to
sampling (`do_sample=True`, `temperature=0.7`):

```python
@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_matches_hf(llm, prompt):
    model, tokenizer = llm.engine.model, llm.engine.tokenizer
    nano = llm.generate(prompt, SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS))[0]
    hf_ids = _hf_greedy_ids(model, tokenizer, prompt)  # same template, HF greedy, prompt stripped
    assert nano.token_ids == hf_ids
    assert nano.text == tokenizer.decode(hf_ids)
```

Passing this proves the generation loop is bug-free. Every later optimization
must keep this test green (a non-regression contract).

### Streaming consistency (`tests/test_engine.py::test_generate_stream_matches_generate`)

```python
def test_generate_stream_matches_generate():
    streamed = list(llm.generate_stream(prompt, params))
    full = llm.generate(prompt, params)[0]
    assert streamed == full.token_ids
```

> Integration tests requiring a real model + CUDA are gated behind
> `NANO_VLLM_INTEGRATION=1` and run on a smaller model (Qwen2.5-0.5B-Instruct).

### Status — all green ✅

Full suite passes on the pinned environment
([benchmark_environment.md](benchmark_environment.md); Python 3.11.10,
PyTorch 2.4 / CUDA 12.4, A100 80GB SXM, model = Qwen2.5-3B-Instruct):

| Stage | Tests | Result |
|-------|-------|--------|
| CPU unit (`test_sampler`, `test_types`) | 12 | ✅ 12 passed |
| GPU integration (`test_m1_vs_hf`, `test_engine`, `test_engine_m1`, `test_generation`) | 11 | ✅ 11 passed |
| **Total** | **23** | **✅ all passed** |

Critically, `test_m1_vs_hf::test_greedy_matches_hf` passes for **all** prompts —
nano-vllm greedy output matches HuggingFace **token-for-token**, confirming the
prefill/decode loop is correct. This is the non-regression contract M2–M6 must
keep green.

Reproduce with `bash scripts/run_tests.sh` (see [tests/README.md](../../tests/README.md)).

## 7. Performance Characteristics

> Hardware, software, scenarios, and metric definitions are specified once in
> [benchmark_environment.md](benchmark_environment.md) and shared across M1–M3
> for comparability.

Environment: A100 80GB SXM, Qwen2.5-3B bf16, PyTorch 2.4 / CUDA 12.4
(see [benchmark_environment.md](benchmark_environment.md)). Run
`20260703_205705`.

### Benchmark data

`Throughput` is **output** tokens/s (the honest single-request decode metric —
see note below). TTFT/TPOT are averages.

| Scenario | Prompt | Output | Throughput | P99 latency | TTFT | TPOT |
|----------|--------|--------|------------|-------------|------|------|
| short_chat | ~125 | 100 | 29.4 tok/s | 3.62 s | 36 ms | 34.0 ms |
| medium_chat | ~526 | 200 | 29.2 tok/s | 7.38 s | 38 ms | 34.2 ms |
| long_context | ~1999 | 100 | 29.2 tok/s | 3.47 s | 89 ms | 33.7 ms |

Raw results: [benchmarks/results/nano_vllm_m1_20260703_205705.json](../../benchmarks/results/nano_vllm_m1_20260703_205705.json)

### Observations (measured)

1. **Output throughput is flat (~29 tok/s) across all scenarios.** Single-request
   throughput ≈ 1/TPOT ≈ 1/0.034 ≈ 29.4 — it is just decode speed, independent of
   prompt length.
2. **TPOT is stable (~34 ms) from 125 to ~2000 prompt tokens** — decode is
   memory-bandwidth-bound. At 3B on an A100, each step is dominated by streaming
   the model weights, not by attention over the KV cache, so context length barely
   moves per-token cost.
3. **TTFT grows with prompt length (36 → 38 → 89 ms)** — prefill is compute-bound.
   But a fixed ~35 ms overhead dominates short prompts (125 → 526 barely changes);
   only the ~2000-token prompt exposes real prefill compute. So TTFT ≈ fixed
   overhead + prefill(n), not a clean line.
4. **Latency ≈ TTFT + output_len × TPOT, confirmed to <1%.** End-to-end latency is
   driven by **output length**, not prompt length: `long_context` (16× the prompt
   of `short_chat`) has nearly identical latency because both emit 100 tokens.

**Caveats (read before comparing to M2):**

- `total_throughput` (66 / 106 / 613 tok/s) counts prefilled **input** tokens, so
  it balloons for long prompts and is misleading for a single request. Use
  `output_throughput`.
- ~29 tok/s single-request is a **baseline, not a headline**: a per-token
  `torch.cuda.synchronize()` (for accurate TPOT) adds fixed overhead, and there is
  no batching. Real speedups come from the KV cache (M2) and batching (M4).
- These scenarios are ≤~2100 tokens, where HF `DynamicCache`'s `torch.cat`
  O(n²) cost is still small. **The M1→M2 win will be modest here** — the
  length-sweep experiment (128→2048, [m2_design.md](m2_design.md) §7) is where the
  quadratic cost actually shows up.

## 8. Known Limitations

1. **Single request only** — no concurrent requests.
2. **KV cache bound by HF** — `torch.cat` growth causes allocator jitter and
   O(n²) copy traffic.
3. **EOS is the only stop condition** — no custom stop strings.
4. **No HTTP streaming** — `yield` is in-process only.
5. **No prefix caching** — repeated prompts are recomputed from scratch.
6. **Per-token GPU sync** — accurate for benchmarking, removable in production.
7. **`bsz=1` hardcoded** — no batching.

Each limitation is addressed across M2–M6.

## 9. Future Work

### M2 — Custom KV Cache

- Replace HF `DynamicCache` with a custom `MyKVCache`.
- Pre-allocate a contiguous K/V buffer to eliminate `torch.cat` growth.
- Keep the streaming interface unchanged.
- Output must remain bit-for-bit identical to M1 in greedy mode.

### M3 — PagedAttention

- Move from a contiguous buffer to block-based storage.
- Introduce a `BlockPool` and per-request `block_table`.
- Integrate a FlashAttention paged kernel.
- Add memory-utilization and fragmentation metrics to quantify the gain over M2.

### Interface contract

The following surfaces remain **stable** across M2–M6 so internals can be
swapped without breaking user code:

- `LLM.generate(prompts, params) -> list[RequestOutput]`
- `LLM.generate_stream(prompt, params) -> Iterator[int]`
- `SamplingParams` fields
- `RequestOutput` fields

This contract is M1's lasting deliverable.

## 10. References

### Source code studied

- [HuggingFace Qwen2 modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)
- [vLLM LLM entrypoint](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py)
- [vLLM Sampler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py)

### Papers

- Attention is All You Need (Vaswani et al., 2017)
- The bfloat16 numerical format (Google, 2018)

### Reading notes

- [docs/reading_notes/hf_qwen2_attention.md](../reading_notes/hf_qwen2_attention.md)
- [docs/reading_notes/vllm_llm_engine.md](../reading_notes/vllm_llm_engine.md)
```
