# Building nano-vllm from scratch (M3): Implementing PagedAttention

**English** | [中文](m3_paged_attention.zh.md)

> M2 pre-allocated a **contiguous KV cache sized to the model's max sequence length**. M3
> replaces it with **Block Pool management**: all blocks are owned by the Block Pool, one
> block holds the KV of a configurable number of tokens, and each request is allocated only
> the blocks it needs for its actual context length. This **on-demand allocation** yields a
> huge memory saving over M2 for short-context requests (shown in the benchmark below). On
> top of that, a request's KV blocks need not be physically contiguous, so — unlike M2 — M3
> can use fragmented memory and maximize utilization.

> M3 design doc: [m3_design.md](../design/m3_design.md) · code tag:
> [m3](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m3)

---

M2 proved a hand-rolled contiguous cache is feasible and correct, but because it allocates
by the model's max sequence length it wastes a lot of memory on short requests — its real
value is **laying the groundwork for M3/M4 to own the memory layout**. M3 replaces M2's
contiguous buffer with **block-based KV management**: each request holds only
`ceil(seq_len/block_size)` blocks, and when the request finishes its blocks are freed back
to the Block Pool for reuse. M3 is also the first milestone to use FlashAttention's paged
kernel. Below I walk through every core concept of implementing PagedAttention from scratch.


# Core concepts

## 0. Milestone positioning
- **M1**: the simplest single-request engine, using HF `DynamicCache`.
- **M2**: a hand-rolled contiguous KV cache (`MyKVCache`), pre-allocating memory to the
  model's max context length.
- **M3**: PagedAttention — the KV cache is managed in blocks; a Block Pool tracks which
  blocks are free or in use, and a KV Cache Manager maps each request to the KV blocks
  assigned to it.

## 2. M3 architecture
```
LLM (user entry)
    └─ PagedEngine
        └─ Tokenizer (HF)
        └─ Model (HF) -> GPU forward: paged_attn_forward (custom)
        └─ BlockPool (custom)
        └─ KVCacheManager (custom)
        └─ Sampler (custom)
```


## 3. Core source files
| Name | What it is | Where |
|---|---|---|
| Physical KV | the big tensor storing K/V, sliced into blocks | engine's `self.k_cache/v_cache` |
| BlockPool | tracks which blocks are free/used | `BlockPool` |
| Block Table | maps a request to the KV blocks assigned to it | `KVCacheManager.req_to_blocks` |

## 4. Implementing a custom paged_attn_forward
Because M3 keeps the HF model body (embedding/MLP/norm/RoPE), I replace each layer's
`Qwen2Attention.forward` with my own `paged_attn_forward` so it can use my custom KV cache.

### 4.1 Why can M1/M2 do self.model(past_key_values=my_kv_cache) but M3 can't?
|   | Cache shape | Can HF's attention read it? |
|---|---|---|
| M1 HF DynamicCache | contiguous `[1, num_kv_heads, seq_len, head_dim]` | Yes |
| M2 MyKVCache | custom `[1, seq_len, num_kv_heads, head_dim]`, but converted back to HF's `[1, num_kv_heads, seq_len, head_dim]` before attention | Yes |
| M3 Paged Cache | blocked + BlockTable indirection, physically scattered in memory | No |

Because the KV blocks are scattered and cannot be gathered into one big contiguous tensor,
you can't use `self.model(past_key_values=my_kv_cache)` — a custom path is required.


## 4.2 Why flash_attn_with_kvcache?
`flash_attn_with_kvcache` supports the paged layout. Hand it `k_cache`/`v_cache`,
`block_table`, `cache_seqlens` and it will, on its own:

1. write the new K/V into the KV blocks the block_table points to;
2. when computing attention, gather K/V from the scattered blocks according to `block_table`.


### 4.3 How `types.MethodType` works
`attn.forward = types.MethodType(paged_attn_forward, attn)`
- `MethodType(func, obj)`: builds a **bound method**, fixing `obj` as `func`'s first argument
  `self`. So `attn.forward(x)` == `paged_attn_forward(attn, x)`.
- Assigned to the **instance** attribute `forward`, it shadows the class method.
- `nn.Module.__call__` runs `self.forward(...)`; Python attribute lookup **hits the instance
  attribute first** → it runs my `paged_attn_forward`.
- Duck typing — **no need to subclass** `Qwen2Attention`.

### 4.4 Call chain (one `self.model()`)
`self.model()` → `Qwen2ForCausalLM.forward` → `Qwen2Model.forward` → `for layer` →
`Qwen2DecoderLayer.forward` → `self.self_attn(...)` → `nn.Module.__call__` → `self.forward` →
**`paged_attn_forward`** (reads ENGINE_CTX → flash_attn_with_kvcache).

## 4.5 ENGINE_CTX (= M3's version of vLLM's attn_metadata)
- A module-level **singleton** holding this step's `block_table` / `cache_seqlens`.
- **Why it's needed**:
  The paged kernel must live where attention actually runs, i.e. `Qwen2Attention.forward`. To
  make HF's model use our paged kernel you either rewrite the whole model (vLLM's approach,
  large effort) or replace just that one `forward` (monkey-patch) — I chose monkey-patch
  because it's far less work.
- **How**: HF gives no API hook, so I can only monkey-patch into `forward`. Every layer of the
  GPU forward now uses my `paged_attn_forward`, which understands the blocked KV layout, and
  `ENGINE_CTX` passes in the request's latest block table and cache sequence length.

## 5. flash_attn_with_kvcache
**One call = write paged cache (append new K/V) + compute attention.**
### Shapes
| Arg | Shape |
|---|---|
| `q` | `[batch, seq, n_heads, head_dim]` |
| `k_cache`/`v_cache` | `[num_blocks, block_size, n_kv, head_dim]` |
| `k`/`v` (new tokens) | `[batch, seq, n_kv, head_dim]` |
| `block_table` | `[batch, max_blocks]` **int32 cuda** |
| `cache_seqlens` | `[batch]` **int32 cuda** |
| returns `out` | `[batch, seq, n_heads, head_dim]` |
### `cache_seqlens` semantics
= the number of tokens already in the cache **before** this forward. **prefill = 0; decode =
num_computed_tokens**. flash_attn writes the new K/V into `[cache_seqlens : cache_seqlens+seq]`.
### Why it must be int32
The kernel's ABI reads the index tensors as int32; torch defaults to int64 → error / misreads.

## 6. BlockPool / KVCacheManager
- **BlockPool**: `FreeKVCacheBlockQueue` implements a doubly-linked list + dummy head/tail;
  allocate blocks via `get_new_blocks` (ref_cnt+1); free blocks via `free_blocks` (ref_cnt-1,
  returned to the pool when it hits 0).
- **KVCacheManager**: `req_to_blocks: dict[str, list[KVCacheBlock]]` maps a request to its KV
  blocks by `request_id`; `allocate_slots` tops up by
  `ceil(all request tokens / block_size) - blocks already held`.

---

## M3 Benchmark: two independent lines

Ran M1/M2/M3 on A100 80GB (Qwen2.5-3B bf16, single request).

### **Regression test**: does M3 greedy output match HF greedy output?
Result: token-for-token match (`test_m3_vs_hf.py` 3/3), passed ✅.

Command: `NANO_VLLM_INTEGRATION=1 NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct python -m pytest tests/test_m3_vs_hf.py -v`
```
============================================================= test session starts =============================================================
platform linux -- Python 3.11.10, pytest-9.1.1, pluggy-1.6.0 -- /usr/bin/python
cachedir: .pytest_cache
rootdir: /workspace/nano-vllm-finance
configfile: pyproject.toml
plugins: asyncio-1.4.0, anyio-4.6.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 3 items

tests/test_m3_vs_hf.py::test_greedy_matches_hf[Hello] PASSED                                                                            [ 33%]
tests/test_m3_vs_hf.py::test_greedy_matches_hf[What is 2 + 2?] PASSED                                                                   [ 66%]
tests/test_m3_vs_hf.py::test_greedy_matches_hf[Explain photosynthesis in one sentence.] PASSED                                          [100%]

============================================================= 3 passed in 21.75s ==============================================================
```

### Latency: the speedup comes from the flash-attn kernel, not paging

| Scenario | TPOT M1 | TPOT M2 | **TPOT M3** | M3 vs M1 | **M3 throughput** | **M3 TTFT** |
|---|---|---|---|---|---|---|
| short_chat | 30.2 | 31.0 | **26.2** ms | −13% | **38.2** tok/s | 26.9 ms |
| medium_chat | 30.3 | 31.2 | **26.4** ms | −13% | **37.9** tok/s | 30.6 ms |
| long_context | 30.0 | 31.1 | **26.3** ms | −12% | **37.0** tok/s | 91.7 ms |

> **Data source** (all in [`benchmarks/results/m3/`](../../benchmarks/results/m3/), same session):
> M1 `nano_vllm_m1_20260711_012918.json`, M2 `nano_vllm_m2_20260711_013719.json`,
> M3 `nano_vllm_m3_20260711_014557.json` — fields `avg_tpot` / `avg_ttft` / `output_throughput`
> (M3 TTFT in seconds, short→long: 0.0269 / 0.0306 / 0.0917). The long prompt's TTFT is much
> higher because prefill processes the whole long prompt in one shot.

M3 decode is ~13% faster (vs M1), ~15% (vs M2). **The key: the lower latency comes from the
fused `flash_attn_with_kvcache` kernel + dropping M2's transpose overhead — not paging itself**
(paging is memory management; it doesn't change single-request compute speed). M2's line "pay
the tax now, refund at M3" came true: M2 was ~6% slower than M1, and M3 refunds that overhead
and adds the kernel win on top.


### Memory: huge savings from PagedAttention — M3 goal achieved ✅

Per-request KV footprint (M2 pre-reserves `max_seq_len` = 302 MB; M3 holds only
`ceil(actual_seq_len / 256)` blocks).

**Where 302 MB comes from**: first, the KV bytes per token (all layers, K and V):
```
KV per token = num_layers × num_kv_heads × head_dim × 2 (bf16 bytes) × 2 (K,V)
             = 36 × 2 × 128 × 2 × 2 = 36864 bytes ≈ 36 KB / token
```
M2 reserves **unconditionally** by the model's max sequence length `max_seq_len = 8192`
(independent of how much is actually used):
```
M2 per request = 36 KB × 8192 ≈ 301,989,888 bytes ≈ 302 MB
```
M3 rounds the actual length up to whole blocks: `M3 = 36 KB × ceil(seq_len/256) × 256`. Short
requests only use tens–hundreds of tokens, and that gap is the saving below.


| Scenario | avg len | M2 (fixed) | M3 (paged) | saved |
|---|---|---|---|---|
| short_chat | 225 | 302 MB | 9.4 MB | **−96.9%** |
| medium_chat | 726 | 302 MB | 28.3 MB | **−90.6%** |
| long_context | 2099 | 302 MB | 84.9 MB | **−71.9%** |

> **Data source**: avg lengths from M3 [`nano_vllm_m3_20260711_014557.json`](../../benchmarks/results/m3/nano_vllm_m3_20260711_014557.json)
> (`avg_prompt_len + avg_output_len`); MB computed by `benchmarks/kv_footprint.py --block-size 256`
> (M2 = fixed `max_seq_len` 8192).

The shorter the request, the more memory M3 saves — as expected.

# M3 limitations / next steps

- M3 is still fundamentally a single-request engine; it does not handle concurrent requests.
  M4 will implement **Continuous Batching** — a Scheduler that runs inference for multiple
  requests at each step — which should lower latency and raise throughput.

---
# Appendix / notes

- Design doc: [m3_design.md](../design/m3_design.md)
- Benchmark setup: [benchmark_environment.md](../design/benchmark_environment.md)
- Previous post: [Building nano-vllm from scratch (M2): Custom KV Cache — an optimization that "looked faster, measured slower"](m2_custom_kv_cache.md)
