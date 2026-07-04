# M2 Design: Custom Pre-allocated KV Cache (`MyKVCache`)

> **Status**: In progress (2025-XX-XX)
> **Tag**: [m2](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m2)
> **Blog**: [Building nano-vllm Part 2](https://...)

## 1. Overview

M2 is the second milestone of nano-vllm-finance. The objective is to **replace
HuggingFace `DynamicCache` with a custom, pre-allocated, contiguous KV cache
(`MyKVCache`)**, removing M1's main inefficiency: `DynamicCache` grows the cache
by `torch.cat` on every decode step, causing allocator jitter and O(n²) copy
traffic over a sequence (see [m1_design.md](m1_design.md) §4 Decision 1).

The generation loop, sampler, streaming interface, and public `LLM` API stay
unchanged. **Correctness is non-negotiable: greedy output must remain bit-for-bit
identical to M1.** The win is purely in memory behavior and per-step decode cost.

Core implementation:

- `MyKVCache`: one-time pre-allocated `[num_layers, max_seq_len, num_kv_heads, head_dim]`
  buffer for K and V
- O(n) per-step writes (slice assignment) instead of O(n) per-step copies (`torch.cat`)
- Constant, non-jittering memory footprint
- Duck-types the HF `Cache` interface so it drops into the model via
  `past_key_values` with no model-code changes

## 2. Goals & Non-Goals

### Goals

- ✅ Eliminate `torch.cat` cache growth; write each step into a pre-allocated slot
- ✅ O(n) total copy traffic over a sequence (down from O(n²))
- ✅ Constant GPU memory for the cache — no per-step alloc/free, no jitter
- ✅ Implement the HF `Cache` interface (`update` / `get_seq_length` /
  `get_max_length`) so `past_key_values=MyKVCache(...)` works unmodified
- ✅ **Bit-for-bit identical to M1 in greedy mode** (non-regression contract)
- ✅ Keep the `LLM` / `SimpleEngine` API and streaming behavior unchanged

### Non-Goals (intentionally deferred)

- ❌ Paged / block-based storage — **M3**
- ❌ Fragmentation handling & cross-request memory sharing — **M3**
- ❌ Batch > 1 / concurrent requests — **M4**
- ❌ Dynamic cache resize (grow beyond `max_seq_len`) — out of scope
- ❌ Prefix caching — later

**Design principle**: M2 changes *where K/V lives*, not *how generation works*.
A single, well-tested storage swap behind a stable interface.

## 3. Architecture

```
┌─────────────────────────────────────────────┐
│              M2 Architecture                 │
│                                              │
│  LLM (user entry)                            │
│   │                                          │
│   ↓                                          │
│  SimpleEngine                                │
│   ├─ Tokenizer (HF)                          │
│   ├─ Model (HF)                              │
│   │     past_key_values = MyKVCache  ◄── M2  │
│   └─ Sampler (custom)                        │
│   │                                          │
│   ↓                                          │
│  GPU                                         │
└─────────────────────────────────────────────┘
```

The only structural change from M1 is the cache: the HF model receives a
`MyKVCache` instance as `past_key_values`. Because `MyKVCache` **duck-types the
HF `Cache` interface** (`update`, `get_seq_length`, `get_max_length`), the model's
attention layers call it exactly as they would call `DynamicCache` — no model code
is touched.

## 4. Key Design Decisions

### Decision 1: Pre-allocate the entire buffer once (and why `DynamicCache` is O(n²))

**How `DynamicCache` grows.** On every decode step it concatenates the new token's
K/V onto the stored history, once per layer:

```python
# DynamicCache.update (simplified)
self.key_cache[layer_idx] = torch.cat(
    [self.key_cache[layer_idx], key_states], dim=-2   # append along the seq dim
)
```

**`torch.cat` rebuilds, it does not append.** A PyTorch tensor must occupy one
contiguous block, and the memory right after `A` is not guaranteed to be free. So
`torch.cat([A, B])` allocates a *fresh* `len(A)+len(B)` block, copies the **whole
of `A`** into it, then copies `B`, and lets the old `A` be freed. Every decode step
therefore re-copies the **entire history accumulated so far**.

**Cost 1 — O(n²) copy traffic.** Generating `n` tokens copies
`1 + 2 + ... + n = n(n+1)/2 ≈ O(n²)` token-slots of K/V, and this happens in
**every layer**. A 2000-token generation moves ~2M token-slots per layer instead
of 2000; on Qwen2.5-3B (36 layers) multiply by 36. The ideal — write one slot per
step — is O(n).

**Cost 2 — allocator churn / memory jitter.** Each step allocates a slightly
larger block and frees the previous one. Because the sizes grow monotonically,
they never match the CUDA caching allocator's pooled blocks well, so reuse is poor
and free memory fragments into pieces too small for the next allocation — the
sawtooth "memory jitter" that can OOM even when total free memory would suffice.

**Analogy.** `torch.cat` is like keeping a list by buying a bigger sheet of paper
for every new item, recopying everything onto it, and throwing the old sheet away
— by item 2000 you have recopied millions of lines. M2 buys one large sheet up
front and writes each item on the next blank line: never recopy, never reallocate.

**The fix.** `__init__` allocates the full
`[num_layers, max_seq_len, num_kv_heads, head_dim]` buffer for K and V once, then
never allocates again. Each decode step is a slice write into existing memory
(`[layer, start:end]`): O(seq_len) work, not O(history). Copy traffic drops to
O(n); cache memory is flat and predictable — no jitter, no fragmentation churn.

**Trade-off**:

- Reserves `max_seq_len` worth of memory even when the actual sequence is short
  (internal waste). A 100-token chat still pays for a 4096-slot buffer. This is
  precisely the inefficiency **M3 (PagedAttention)** removes with block-level
  allocation.

| | HF `DynamicCache` (`torch.cat`) | M2 pre-allocated |
|---|---|---|
| Per-step op | rebuild the whole K/V tensor | write one slice |
| Total copy traffic | O(n²) | O(n) |
| Allocation | alloc/free every step, growing size | once, constant |
| Memory curve | sawtooth jitter + fragmentation risk | flat, stable |
| Long sequences | degrades noticeably | stable |

### Decision 2: Seq/token-major storage layout

Storage is `[num_layers, max_seq_len, num_kv_heads, head_dim]` — sequence
dimension **before** heads — whereas HF passes K/V as `[1, num_kv_heads, seq_len,
head_dim]` (heads first).

**Why**:

- The hot path is "append along the sequence dimension"; putting `max_seq_len`
  outermost keeps each written token's data contiguous and makes `[layer, start:end]`
  a clean contiguous slice.
- It mirrors the **token-major layout M3's paged cache will use**, so the mental
  model and storage carry forward instead of being rewritten.

**Trade-off**:

- Every `update` must transpose between the HF layout and storage layout (the
  `squeeze(0) → transpose(0,1) → contiguous()` dance on write, reversed on read).
  This costs a copy per call. Acceptable at M2's altitude; a real paged kernel
  (M3) consumes the token-major layout directly and avoids the round-trip.

### Decision 3: `current_len` shared across layers, advanced only on the last layer

`update()` is called **once per layer per step**, and all layers advance in
lockstep, so a single `current_len` is shared. It is bumped **only** when
`layer_idx == num_layers - 1`.

**Why**:

- Bumping it on every layer call would over-count by `seq_len * num_layers`.
- Never bumping it leaves `start` stuck at 0, so every step overwrites position 0
  and the cache silently never grows.

### Decision 4: `batch=1` assumption

The decode path is single-sequence (inherited from M1). `update()` asserts
`key_states.shape[0] == 1`. Batching is an M4 concern and would change the storage
shape and indexing.

### Decision 5: Duck-type the HF `Cache` interface

`MyKVCache` implements the methods the HF model calls on a cache object, so it can
be passed as `past_key_values` directly.

**Why**:

- Zero model-code changes; the swap is invisible to attention layers.
- Keeps the M1→M2 diff confined to the cache class plus a few lines of engine
  wiring.

**Integration risk to watch**: different `transformers` versions may call extra
cache methods (e.g. `get_usable_length`, the `seen_tokens` property, or
`reorder_cache`). If the pinned version requires them, add thin shims to
`MyKVCache`. This is the most likely source of integration breakage and should be
verified against the exact `transformers` version in
[benchmark_environment.md](benchmark_environment.md).

## 5. Implementation Details

### `MyKVCache.__init__` — one-time allocation

```python
self.k_cache = torch.zeros(
    num_layers, max_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device,
)
self.v_cache = torch.zeros_like(self.k_cache)
self.current_len = 0

# Footprint: element_size() (bytes/elem) * numel() (total elems) * 2 (K and V)
size_mb = self.k_cache.element_size() * self.k_cache.numel() * 2 / 1e6
```

### `MyKVCache.update` — validate, transpose-write, slice-return

```python
def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
    assert key_states.dim() == 4              # NOT `.shape == 4`
    assert key_states.shape[0] == 1           # batch=1 in M2

    start = self.current_len
    end = start + key_states.shape[2]         # seq_len
    if end > self.max_seq_len:
        raise RuntimeError("KV cache overflow: ...")

    # HF [1, H, S, D] -> storage [S, H, D]
    self.k_cache[layer_idx, start:end] = key_states.squeeze(0).transpose(0, 1).contiguous()
    self.v_cache[layer_idx, start:end] = value_states.squeeze(0).transpose(0, 1).contiguous()

    if layer_idx == self.num_layers - 1:      # advance once per step
        self.current_len = end

    # storage [end, H, D] -> HF [1, H, end, D]; slice [:end] excludes unwritten zeros
    k_full = self.k_cache[layer_idx, :end].transpose(0, 1).unsqueeze(0).contiguous()
    v_full = self.v_cache[layer_idx, :end].transpose(0, 1).unsqueeze(0).contiguous()
    return k_full, v_full
```

Full implementation: [nano_vllm/kv_cache.py](../../nano_vllm/kv_cache.py).

### Engine integration (pending work)

`generate_stream()` / `generate()` in [nano_vllm/engine.py](../../nano_vllm/engine.py)
currently rely on HF's default cache (`past_key_values = outputs.past_key_values`
re-read each step). M2 replaces this with a persistent `MyKVCache` instance:

```python
# Build once from the model config
head_dim = model.config.hidden_size // model.config.num_attention_heads
cache = MyKVCache(
    num_layers=model.config.num_hidden_layers,
    max_seq_len=MAX_SEQ_LEN,
    num_kv_heads=model.config.num_key_value_heads,
    head_dim=head_dim,
    dtype=model.dtype,
    device=model.device,
)
cache.reset()  # per request

# Prefill + every decode step: pass the SAME cache object
outputs = model(input_ids=..., past_key_values=cache, use_cache=True)
# No need to read back outputs.past_key_values — the cache object persists.
```

Key differences from M1:

- The cache object is created from model config and **persists across steps**;
  M1 re-read `outputs.past_key_values` each step.
- `cache.reset()` is called at the start of each request (cheap: just resets
  `current_len`; memory is reused).

## 6. Testing Strategy

### Unit tests — [tests/test_kv_cache.py](../../tests/test_kv_cache.py)

Already implemented; pure tensor logic, **CPU by default and CUDA-parametrized
when available** (no model needed). Covers:

- Per-layer roundtrip with distinct data (catches wrong-layer / cross-layer writes)
- `current_len` advances only on the last layer
- Multi-step decode accumulation
- Exact `max_seq_len` boundary + single-write and incremental overflow
- `reset` then reuse (no stale data leaks through)
- dtype / device / contiguity of returned tensors
- **Equivalence with HF `DynamicCache`** (shape + values, prefill and decode)

### Non-regression: M2 vs M1 (most important)

Greedy output must be **byte-for-byte identical to M1** (which is in turn identical
to HF). This is the M1 correctness contract, now applied across the cache swap:

```python
def test_m2_greedy_matches_m1():
    params = SamplingParams(temperature=0.0, max_tokens=32)
    for prompt in test_prompts:
        assert m2_llm.generate(prompt, params)[0].text == m1_reference[prompt]
```

### Streaming consistency

`list(llm.generate_stream(prompt, params)) == llm.generate(prompt, params)[0].token_ids`
must still hold after the swap.

> Integration tests requiring a real model + CUDA stay gated behind
> `NANO_VLLM_INTEGRATION=1` (see M1 doc §6).

## 7. Performance Characteristics — M1 vs M2

> Hardware, software, scenarios, and metric definitions are specified once in
> [benchmark_environment.md](benchmark_environment.md) and shared across M1–M3.

> **TBD** — M2 benchmarks not yet run. All cells below are placeholders to be
> filled after the M2 suite executes on GPU, using the **same** environment as M1.

### Benchmark methodology — surfacing O(n²) vs O(n)

The default scenarios (fixed output lengths) **under-show** the effect: the
asymptotic gap only opens up as the *generated sequence* grows. The whole
M1-vs-M2 difference lives on the **decode length axis**, so the experiments must
sweep it. Both engines run the identical harness on the same GPU (warmup, locked
clocks, fixed seed per [benchmark_environment.md](benchmark_environment.md)); the
only variable is the cache backend (`DynamicCache` vs `MyKVCache`).

**Experiment 1 — Output-length sweep (asymptotics).**
Fix a short prompt (~32 tok); vary `max_tokens` over {128, 256, 512, 1024, 2048}
with greedy decoding. Record avg TPOT, total decode time, and peak memory per
length.
- Expected M1: per-step cost rises with length → avg **TPOT grows ∝ n**, total
  decode time **∝ n²**.
- Expected M2: avg **TPOT flat** → total decode time **∝ n**.
- The headline plot: *total decode time vs output length* — M1 curves upward
  (quadratic), M2 is a straight line.

**Experiment 2 — Inter-token latency trace (the smoking gun).**
One long generation (e.g. 1024 tok); record the timestamp of **every** yielded
token (`generate_stream` already exposes this) → plot inter-token latency vs token
index.
- Expected M1: an upward-sloping line (later tokens cost more) with **sawtooth**
  spikes from reallocation.
- Expected M2: a **flat** line.
- This single plot is the clearest visualization of O(n²) vs O(n) and of the
  memory jitter.

**Experiment 3 — Memory trace (optional).**
Sample `torch.cuda.memory_allocated()` each decode step.
- Expected M1: staircase growth + jitter. M2: flat at the pre-allocated size.
- **Honest caveat**: for short sequences M2's flat line sits *above* M1's (it
  reserves `max_seq_len`). M2's single-request memory advantage is
  **stability/predictability**, not lower peak; the raw-peak win comes later when
  many requests are packed (M4). M2's unambiguous single-request win is **speed**
  (TPOT on long sequences).

**Harness changes required (deferred to M2 — not part of M1 work):**
- A cache-backend switch so one harness runs both `DynamicCache` and `MyKVCache`.
- Per-token timestamp capture in the runner (full inter-token curve, not just TTFT).
- `peak_memory_mb` via `torch.cuda.max_memory_allocated()` (the Table B prerequisite).
- Optional per-step memory sampling for Experiment 3.

### Table A — Latency & throughput (M1 → M2)

| Scenario | Metric | M1 | M2 | Δ% |
|----------|--------|----|----|----|
| short_chat (~100 tok) | Throughput (tok/s) | — | — | — |
| short_chat | P99 latency (s) | — | — | — |
| short_chat | TTFT (s) | — | — | — |
| short_chat | TPOT (ms) | — | — | — |
| medium_chat (~200 tok) | Throughput (tok/s) | — | — | — |
| medium_chat | P99 latency (s) | — | — | — |
| medium_chat | TTFT (s) | — | — | — |
| medium_chat | TPOT (ms) | — | — | — |
| long_context (~1000 tok) | Throughput (tok/s) | — | — | — |
| long_context | P99 latency (s) | — | — | — |
| long_context | TTFT (s) | — | — | — |
| long_context | TPOT (ms) | — | — | — |

### Table B — Memory efficiency (the M2 headline)

Peak GPU memory via `torch.cuda.max_memory_allocated()`, per scenario.

| Scenario | M1 peak mem (MB) | M2 peak mem (MB) | Δ% | Notes |
|----------|------------------|------------------|----|-------|
| short_chat (~100 tok) | — | — | — | |
| medium_chat (~200 tok) | — | — | — | |
| long_context (~1000 tok) | — | — | — | M1 `torch.cat` growth most visible here |

Full report: [benchmarks/results/m2/report.md](../../benchmarks/results/m2/report.md)

### Expected qualitative results (to confirm)

1. **TPOT improves most on long sequences** — O(n²)→O(n) copy traffic; the longer
   the sequence, the bigger the M2 win.
2. **Peak memory is flat and non-jittering** — pre-allocation removes per-step
   alloc/free; M2 may use *more* memory for short sequences (reserves
   `max_seq_len`) but avoids M1's growth spikes on long ones.
3. **TTFT ≈ unchanged** — prefill is compute-bound and dominated by the same
   forward pass.
4. **Greedy output identical** — the non-regression contract.

> **Dependency for Table B**: `ScenarioMetrics` in
> [benchmarks/metrics.py](../../benchmarks/metrics.py) has **no memory field
> today**. Recording peak memory requires adding one (e.g. `peak_memory_mb`) and
> capturing `torch.cuda.max_memory_allocated()` in the runner. This is a
> prerequisite to filling Table B and is tracked as follow-up work.

## 8. Known Limitations

1. **Fixed `max_seq_len` wastes memory** — short sequences still reserve the full
   buffer (internal fragmentation). The core motivation for M3.
2. **Single contiguous buffer can't grow** — exceeding `max_seq_len` raises; no
   spill or resize.
3. **`batch=1` hardcoded** — no batched decode (M4).
4. **Transpose overhead per `update`** — layout round-trip costs a copy each call;
   removed by a paged kernel in M3.
5. **No cross-request sharing** — each request owns its full buffer; no prefix or
   block sharing across requests.

## 9. Future Work

### M3 — PagedAttention

- Replace the single contiguous buffer with **block-based storage**.
- Introduce a `BlockPool` and per-request `block_table`.
- Integrate a **FlashAttention paged kernel** that consumes the token-major layout
  directly (eliminating the M2 transpose round-trip).
- Add **KV cache utilization** and **fragmentation** metrics (already defined in
  [benchmark_environment.md](benchmark_environment.md)) to quantify the gain over
  M2's fixed pre-allocation.

### Interface contract (unchanged from M1)

These surfaces remain **stable** across M2–M6 so internals can be swapped without
breaking user code:

- `LLM.generate(prompts, params) -> list[RequestOutput]`
- `LLM.generate_stream(prompt, params) -> Iterator[int]`
- `SamplingParams` fields
- `RequestOutput` fields

M2 validates this contract: the cache changes completely, the API does not.

## 10. References

### Source code studied

- [HuggingFace Cache utils (`DynamicCache`)](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)
- [HuggingFace Qwen2 modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)
- [vLLM PagedAttention / block manager](https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager.py)

### Papers

- Efficient Memory Management for LLM Serving with PagedAttention (Kwon et al., SOSP 2023)
- Attention is All You Need (Vaswani et al., 2017)

### Reading notes

- [docs/reading_notes/hf_dynamic_cache.md](../reading_notes/hf_dynamic_cache.md)
- [docs/reading_notes/vllm_paged_attention.md](../reading_notes/vllm_paged_attention.md)
