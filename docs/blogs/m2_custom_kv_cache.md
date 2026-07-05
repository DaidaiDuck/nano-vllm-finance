# Building nano-vllm from Scratch (M2): A Custom KV Cache — the Optimization That Made Things *Slower*

**English** | [中文](m2_custom_kv_cache.zh.md)

> Series: reimplementing the core ideas of vLLM, one milestone at a time.
> **M2 — Custom KV Cache.** Replace HuggingFace's `DynamicCache` with a
> pre-allocated, fixed-size, contiguous tensor (`MyKVCache`). The goal was to kill
> the O(n²) copies caused by M1's `torch.cat` — but the measurements told the
> opposite story.
>
> M2 design doc: [m2_design.md](../design/m2_design.md) · Code tag:
> [m2](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m2)

---
M1's conclusion was: the engine is correct and works, but it leans on HuggingFace's
`DynamicCache`, which grows the KV with `torch.cat` every step — an O(n²) copy in
theory. M2's goal was to write a **pre-allocated, contiguous** `MyKVCache` to
replace it, and prove the output stays token-for-token identical to M1. Correctness
was achieved. But when I put M1 (HF cache) and M2 (MyKVCache) side by side in the
same benchmark, the result was **unexpected: M2 is slower and uses more memory, and
that famous O(n²) bottleneck barely shows up**. This post explains why — and why M2
is still worth doing.

# Design rationale

## 1. Why write a custom KV cache?

Two reasons. **The surface reason**: M1's `DynamicCache` does
`torch.cat([history, new_token])` every decode step — copying the entire history
each step, so generating n tokens costs O(n²) total copy traffic, plus memory
fragmentation from the repeated alloc/free. **The deeper reason**: the later
milestones — M3 (PagedAttention) and M4 (Continuous Batching) — **require owning
the KV cache's memory layout**; HF's `DynamicCache` can't do paging or efficient
concurrency. So writing my own cache is unavoidable.

## 2. The design of MyKVCache

```
LLM (user entry)
    └─ SimpleEngine
        └─ Tokenizer (HF)
        └─ Model (HF, past_key_values = self.my_kv_cache, use_cache=True) <- KV cache = self.my_kv_cache (M2)
        └─ Sampler (M1, custom)
```

The core idea: **allocate once, then only write into position**:

```python
# Allocated once at startup; never alloc/free again
self.k_cache = torch.zeros(num_layers, max_seq_len, num_kv_heads, head_dim, ...)
self.v_cache = torch.zeros_like(self.k_cache)
self.current_len = 0    # tokens written so far, shared across layers
```

The layout is **seq-major**: `[num_layers, max_seq_len, num_kv_heads, head_dim]` —
sequence dimension before the head dimension. This is chosen to **align with M3's
paged (token-major) layout**, saving a rewrite later. The cost: HF hands K/V in
heads-first `[1, num_kv_heads, seq_len, head_dim]`, so every `update` must
**transpose** — `squeeze→transpose→contiguous` on write, and
`transpose→unsqueeze→contiguous` on the read-back. Remember this "transpose tax";
it becomes the main character later.

Each `update` writes the new token into the `[current_len : current_len+seq_len]`
slice and never touches history — in theory turning each step's O(n) copy into an
O(1) write.

## 3. Wiring it into the engine

I added a `use_custom_cache` switch on the engine: `True` uses MyKVCache (M2),
`False` uses HF `DynamicCache` (M1), so **one benchmark harness can fairly compare
both backends**.

One gotcha during integration: the HF model's forward calls
`past_key_values.get_mask_sizes(query_length, layer_idx)` to build the attention
mask, and my first MyKVCache didn't have that method. Adding a thin shim fixed it:

```python
def get_mask_sizes(self, query_length, layer_idx):
    kv_length = self.get_seq_length(layer_idx) + query_length  # stored + new tokens
    kv_offset = 0                                              # 0 for a full (non-sliding) cache
    return kv_length, kv_offset
```

Note `query_length` is an **int** in transformers 5.13.0 — the signature varies
across versions, so the safest move is to `inspect` your installed
`DynamicLayer.get_mask_sizes` source and mirror it. This was the only method that
had to be added.

## 4. Correctness validation

`tests/test_m1_vs_hf.py` asserts that in greedy mode, MyKVCache's output matches HF
token-for-token.

```bash
# ① correctness
NANO_VLLM_INTEGRATION=1 python -m pytest tests/test_m1_vs_hf.py tests/test_kv_cache.py -v
```

```python
assert nano.token_ids == hf_greedy_ids
```

**3/3 pass on Qwen2.5-3B-Instruct** — MyKVCache is a correct drop-in replacement for
HF's `DynamicCache`.

**Correctness: done.** Now performance — where the story gets interesting.

# The M2 benchmark — a surprising result

Same A100 80GB SXM, Qwen2.5-3B bf16, run once with `--cache hf` and once with
`--cache custom`.

```bash
# ② Parity (M1 rerun vs M2, expect latency ≈ equal)
python -m benchmarks.m2_benchmark --cache hf --version m1_rerun \
    --scenarios short_chat medium_chat long_context
python -m benchmarks.m2_benchmark --cache custom --version m2 \
    --scenarios short_chat medium_chat long_context \
    --baseline benchmarks/results/nano_vllm_m1_rerun_<timestamp>.json

# ③ Cache-level O(n²)→O(n) (model-free, a few seconds)
python -m benchmarks.cache_microbench --max-steps 4096 --checkpoints 256
```

**Latency / throughput (M1 → M2):**

| Scenario | TPOT | Throughput (output tok/s) | Change |
|---|---|---|---|
| short_chat | 34.6 → 36.8 ms | 28.9 → 27.2 | **↓5.8%** |
| medium_chat | 34.7 → 36.9 ms | 28.8 → 27.1 | **↓6.0%** |
| long_context | 34.6 → 36.9 ms | 28.4 → 26.7 | **↓6.1%** |

**Peak memory:**

| Scenario | M1 | M2 | Δ |
|---|---|---|---|
| short | 6242 MB | 6538 MB | +296 |
| medium | 6397 MB | 6677 MB | +280 |
| long | 6953 MB | 7174 MB | +221 |

**Micro-benchmark (model removed, cache op only, 36 layers/step; warmup added to
remove cold start):**

| seq_len | DynamicCache | MyKVCache |
|---|---|---|
| 289 | 803 µs | 3236 µs |
| 1057 | 841 µs | 3246 µs |
| 2081 | 873 µs | 3189 µs |
| 4128 | 880 µs | 3406 µs |
| **total (4096 steps)** | **4.6 s** | **13.3 s** |

**One-line conclusion: M2 is consistently ~6% slower, uses ~300 MB more memory, and
the cache op itself is ~3.8× slower. The exact opposite of the "faster" intuition.**

## Analysis: so where did the predicted O(n²) go?

First, a mental model — **per-step time = fixed overhead + copy**:

- **Fixed overhead**: the GPU kernels launched per step (launch + allocation), whose
  count is **independent of sequence length**.
- **Copy**: the time to move the KV bytes, which **grows with sequence length**
  (O(n)).

Both findings fall out of this decomposition.

---

### Finding 1: the per-step O(n) copy of `torch.cat` — how small is it, really?

First the **copy**. At sequence length n, each `torch.cat` re-copies the history:

```
bytes per layer = n × 2 heads × 128 × 2 bytes (bf16) × 2 (K and V) ≈ 1024·n bytes
36 layers total ≈ 36 × 1024·n ≈ 37 KB × n
```

Plugging in the two ends (A100 HBM bandwidth ≈ 2 TB/s):

| seq_len n | copy bytes | copy time |
|---|---|---|
| 33 | ~1.2 MB | **~1 µs** |
| 4128 | ~150 MB | **~75 µs** |

Now the **fixed overhead**: 36 layers × `cat(K)`+`cat(V)` = 72 kernel launches, at
~11 µs each (reverse-derived from the data; the real range is ~5–20 µs) → **~800 µs**,
independent of n.

Put together:

```
per step = 800 µs (fixed) + 1 µs → 75 µs (copy, grows with n)
```

**So the copy does grow every step, but it's buried under the ~800 µs fixed
overhead — nearly invisible.** The measurement confirms it: 803 µs (n=289) → 880 µs
(n=4128), a rise of ~77 µs.

> **Conclusion**: for short prompts, DynamicCache is **overhead-bound**, not
> copy-bound. O(n²) is real, but it only dominates at tens of thousands of tokens
> where the copy exceeds 800 µs — practically unreachable in real inference.
> **"`torch.cat` is expensive" is an overrated urban legend.**

---

### Finding 2: MyKVCache is 3.8× slower — just count the kernels

MyKVCache is slower **not because it copies more bytes, but because each `update`
launches more GPU kernels**. Count them per layer per step:

| | Operations | # kernels |
|---|---|---|
| **DynamicCache** | `cat(K)` + `cat(V)` | **2** |
| **MyKVCache** | write: transpose+`contiguous`; slice-assign; read: transpose+`contiguous` — ×3 for K and V | **6** |

MyKVCache's seq-major layout (the "transpose tax" from §2) forces a few extra
copying `contiguous` calls each time. Estimating the fixed overhead:

```
DynamicCache:  36 layers × 2 kernels × ~11 µs ≈  800 µs   (measured ~860)
MyKVCache:     36 layers × 6 kernels × ~15 µs ≈ 3240 µs   (measured ~3260)
```

**3× the kernel count + each transpose kernel is a bit more expensive (strided
access) → ~3.8× the time.** (The per-kernel µs figures are reverse-derived, not
directly profiled — see the caveat below.)

---

### Reconciling with the end-to-end 6%

The cache op is only a small part of TPOT, but M2 made that part more expensive:

```
MyKVCache cache op     ≈ 3260 µs/step
DynamicCache cache op  ≈  860 µs/step
extra                  ≈ 2400 µs/step

2400 µs ÷ M1's TPOT (34600 µs) ≈ 6.9%
```

**≈ the ~6% end-to-end slowdown measured. It checks out** — the slowdown is exactly
that extra 2400 µs of kernel overhead. (`34600 µs` = M1's measured TPOT, i.e. the
time for one decode step; the division ties the isolated micro-benchmark back to the
end-to-end result, confirming the slowdown lives in the cache op.)

> **Honesty caveat**: only the *measured* numbers are solid — DynamicCache ~860 µs,
> MyKVCache ~3260 µs (3.8×), and the ~6% end-to-end regression. The exact
> "N kernels × M µs" breakdown is a *reverse-derived hypothesis*, consistent with
> the known ~5–20 µs kernel-launch cost but not directly profiled. To turn it into
> fact, profile with `torch.profiler` and count the real kernels.

# What M2 taught me

1. **"Custom means faster" is an illusion — you have to measure.** I started with the
   expectation of killing O(n²) and ended up building a *slower* cache. Had I only
   written theory and skipped the benchmark, I wouldn't know what actually happens.
   **Measurement beats intuition.**

2. **Asymptotic complexity ≠ the real bottleneck.** O(n²) is correct asymptotically,
   but at the small, real-world request sizes here, the fixed overhead (kernel
   launches) dominates.

3. **M2's value isn't single-request speed — it's ownership.** The transpose
   overhead is the "cost of control": once I own the cache, M3 can build
   PagedAttention on top of it and M4 can do batching. And the transpose tax
   **disappears in M3, when a paged attention kernel consumes the seq-major layout
   directly** (no more converting back to HF's heads-first). So M2 is "pay the tax
   now, get the refund at M3."

# Limitations / next

- **~6% slower and ~300 MB more memory at single request** — the seq-major layout's
  transpose overhead, plus the fixed pre-allocation.
- **The O(n²) / fragmentation payoff didn't materialize** — at single request with
  moderate lengths, both caches are overhead-bound. Those gains only show up at
  **M4 (concurrent, variable-length requests)**.
- **Next: M3 (PagedAttention)** — replace the contiguous buffer with block storage +
  a block_table, and write a paged kernel that consumes the seq-major layout
  directly. That's when the transpose tax vanishes and on-demand allocation +
  cross-request sharing get unlocked. M2 laid the road; M3 starts collecting rent.

# Appendix / notes

- Design doc: [m2_design.md](../design/m2_design.md)
- Benchmark setup: [benchmark_environment.md](../design/benchmark_environment.md)
- Previous: [M1 — A Simple, Correct Single-Request Engine](m1_building_nano_vllm.md)
