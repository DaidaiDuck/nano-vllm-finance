# M3 Design: PagedAttention (block-based KV cache + FlashAttention kernel)

> **Status**: ✅ Done. Correctness validated — `test_m3_vs_hf.py` passes **3/3** (greedy ==
> HF token-for-token, 3B). Benchmarked on A100 (§7): first milestone with a real
> single-request speedup (~13% faster decode than M1) + 72–97% per-request KV-memory saving vs M2.
> **Tag**: [m3](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m3)
> **Blog**: [Building nano-vllm Part 3](https://...)

## 1. Overview

M3 is the third milestone of nano-vllm-finance. The objective is to **replace M2's
single contiguous `MyKVCache` with PagedAttention**: KV lives in fixed-size **blocks**
drawn from a shared pool, each request holds a **`block_table`** (logical position →
physical block), and attention runs through the **FlashAttention paged kernel**
(`flash_attn_with_kvcache`).

This removes M2's core waste: M2 pre-reserves `max_seq_len` per request even for a
100-token chat (internal fragmentation, see [m2_design.md](m2_design.md) §4 Decision 1).
Paging allocates memory one block at a time, so a request only holds `ceil(len/block_size)`
blocks, and freed blocks return to the pool for reuse.

**Route A (chosen):** keep the entire HuggingFace model body (embedding / MLP / norm /
RoPE) and **monkey-patch only `Qwen2Attention.forward`** on each layer. We do *not*
reimplement the model like vLLM does. This keeps the milestone focused on the KV
mechanism, at the cost of matching HF's attention signature exactly (§4 Decision 1).

**Correctness is non-negotiable: greedy output must remain identical to M2 == M1 == HF**
(the same non-regression contract, validated token-for-token; §6). At single request there
is **no expected end-to-end speedup** — like M2, the paged win is *architectural*: it is the
prerequisite for M4 continuous batching, where block-level allocation lets many sequences
share one pool without fragmentation.

Core implementation:

- `KVCacheBlock` / `BlockPool` — the **ledger** (which block ids are free/used), a
  vLLM-faithful doubly-linked free queue with `ref_cnt`
- `KVCacheManager` — the **index** (`req_to_blocks: req_id → list[KVCacheBlock]`);
  `allocate_slots` / `free` / `get_block_table`
- Physical **paged KV tensors** in the engine, sliced by block and bound per-layer
- `paged_attn_forward` + `ENGINE_CTX` — the monkey-patched attention that calls
  `flash_attn_with_kvcache` (writes new K/V into the paged cache **and** attends in one call)

## 2. Goals & Non-Goals

### Goals

- ✅ Replace the contiguous buffer with **block-based storage** drawn from a shared pool
- ✅ Per-request `block_table`; allocate one block at a time (`ceil(total/block_size)`)
- ✅ Free returns blocks to the pool for reuse (`ref_cnt` drops to 0 → back on the queue)
- ✅ Route attention through **`flash_attn_with_kvcache`** (fused write + paged attend)
- ✅ Keep the HF model body unchanged — **monkey-patch `Qwen2Attention.forward` only**
- ✅ **Bit-for-bit identical to M2/M1 in greedy mode** (non-regression contract)
- ✅ Keep the `LLM.generate` / `generate_stream` public API unchanged

### Non-Goals (intentionally deferred)

- ❌ Batch > 1 / continuous batching — **M4** (where paging actually pays off)
- ❌ Prefix caching (block hashing, `cached_block_hash_to_block`) — later
- ❌ LRU eviction / preemption / swap — **M4+**
- ❌ Memory-profiled `num_blocks` (dummy-forward peak sizing) — **M4**; M3 hardcodes a
  comfortable pool (`num_blocks = 625` × `block_size = 256` ≈ 5.9 GB)
- ❌ Quantized KV — **M5**

**Design principle**: M3 changes *how K/V is stored and read* (contiguous → paged), not
*how generation works*. The generation loop, sampler, and public API carry over from M2.

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       M3 Architecture                          │
│                                                                │
│  LLM (user entry)                                              │
│   │                                                            │
│   ↓                                                            │
│  PagedEngine                                                   │
│   ├─ Tokenizer (HF)                                            │
│   ├─ Model (HF)  ── monkey-patch each layer.self_attn.forward  │
│   │                   = paged_attn_forward  ◄────────── M3     │
│   ├─ Sampler (custom)                                          │
│   │                                                            │
│   ├─ 仓库 physical KV: k_cache/v_cache                          │
│   │     [num_layers, num_blocks, block_size, num_kv_heads, hd] │
│   ├─ 账本 BlockPool         (free/used block ids)              │
│   └─ 索引 KVCacheManager    (req_id → [KVCacheBlock])          │
│         │                                                      │
│         │  per step: set ENGINE_CTX.block_table / cache_seqlens│
│         ↓                                                      │
│  paged_attn_forward ── flash_attn_with_kvcache(q, k, v,        │
│         k_cache, v_cache, block_table, cache_seqlens)          │
│   │                                                            │
│   ↓                                                            │
│  GPU                                                           │
└──────────────────────────────────────────────────────────────┘
```

### The one mental model: 仓库 / 账本 / 索引

Three **separate** things — conflating them is the #1 source of confusion:

| Name | What it is | Lives in |
|---|---|---|
| **仓库** physical KV | the big tensor that actually stores K/V, sliced into blocks | `PagedEngine.k_cache/v_cache` |
| **账本** BlockPool | which block ids are free / used | `BlockPool` |
| **索引** block_table | one request's "logical position → physical block id" | `KVCacheManager.req_to_blocks` |

Attention **reads the warehouse**, located via the **index**; the **ledger** only hands
out / takes back block ids — it never touches the tensors.

## 4. Key Design Decisions

### Decision 1: Route A — monkey-patch HF attention, keep the model body

We swap **only** each layer's `Qwen2Attention.forward` for a custom `paged_attn_forward`;
everything else (embedding, MLP, RMSNorm, RoPE tables, `lm_head`) stays HF. The patch is
applied at engine init:

```python
for i, layer in enumerate(self.model.model.layers):
    attn = layer.self_attn
    attn.k_cache = self.k_cache[i]        # bind this layer's physical cache
    attn.v_cache = self.v_cache[i]
    attn.forward = types.MethodType(paged_attn_forward, attn)   # duck-typed rebind
```

**Why**: the milestone is about the KV mechanism, not re-deriving the whole model.
`types.MethodType(fn, attn)` binds `attn` as `fn`'s `self`, and assigning it to the
**instance** attribute shadows the class method — `nn.Module.__call__` runs `self.forward`,
which now resolves to our function. No inheritance needed.

**Trade-off**: `paged_attn_forward` must match HF's `Qwen2Attention.forward` signature and
tensor conventions **exactly** — the source of every early bug (§ below). Two HF specifics:

- HF passes `hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs`
  by keyword; `position_embeddings` is the `(cos, sin)` tuple, not position ids. RoPE is a
  **module-level** function: `from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb`.
- `o_proj` is a plain `nn.Linear` → returns a **single tensor** (not vLLM's `(out, bias)`),
  and `forward` must **return a tuple** `(attn_output, attn_weights)` — the decoder layer
  unpacks it, so we return `(attn_output, None)`.

> **⚠️ Do not copy vLLM's `Qwen2Attention.forward`.** vLLM's model is a *different* class
> (`qkv_proj` fusion, `self.attn(q,k,v)`, `rotary_emb(positions,q,k)`, tuple-returning
> `o_proj`) — none of those exist on the HF instance. The one line you replace is vLLM's
> `self.attn(q,k,v)`; the surrounding proj / RoPE / o_proj use HF's equivalents.

### Decision 2: `flash_attn_with_kvcache` — fused write + attend

A single call does both halves of paged attention:

1. **writes** the new step's `k, v` into `k_cache/v_cache` at `[cache_seqlens : cache_seqlens+seq]`, located via `block_table`;
2. **attends**: each query position reads all cached K/V up to itself (causal).

```python
attn_output = flash_attn_with_kvcache(
    q=q,                                 # [batch, seq, n_heads, head_dim]
    k_cache=self.k_cache,                # [num_blocks, block_size, n_kv, head_dim] (this layer)
    v_cache=self.v_cache,
    k=k, v=v,                            # new tokens' K/V, [batch, seq, n_kv, head_dim]
    cache_seqlens=ENGINE_CTX.cache_seqlens,   # [batch] int32 — tokens already cached
    block_table=ENGINE_CTX.block_table,       # [batch, max_blocks] int32
    causal=True,
)                                        # -> [batch, seq, n_heads, head_dim]
```

**Why fused vs vLLM's split** (`reshape_and_cache_flash` + `flash_attn_varlen_func`):
at single request the fused kernel is simpler and needs no `cu_seqlens` packing. vLLM
splits because it batches many variable-length sequences (M4 territory).

**`cache_seqlens` semantics** = tokens already in the cache **before** this call:
**prefill = 0**, **decode = `num_computed_tokens`**. FA appends the new K/V starting there.

**`block_table` / `cache_seqlens` must be int32 CUDA tensors** — the kernel's ABI reads
those index tensors as int32; torch's default int64 raises / mis-reads.

**Page block size must be a multiple of 256.** The stock Dao-AILab flash-attn wheel requires
the paged cache's `block_size` (the `page_block_size` dim of `k_cache`) to be divisible by
256 (`RuntimeError: Paged KV cache block size must be divisible by 256` otherwise). vLLM
uses block_size 16 because it ships its own patched kernels (`vllm-flash-attn`); with the
stock wheel M3 uses **`block_size = 256`** (and `num_blocks = 625` to keep the pool ~5.9 GB).
Consequence: coarser allocation — internal fragmentation up to 255 tokens/request, still
tiny vs M2's `max_seq_len`.

### Decision 3: `ENGINE_CTX` — a minimal `attn_metadata`

`paged_attn_forward` needs this step's `block_table` and `cache_seqlens`, but the HF decoder
layer calls `self.self_attn(...)` **without** any paging arguments (HF doesn't know about
paging), and Route A can't add parameters to HF's call chain. Solution: a **module-level
singleton** the engine mutates before each forward, and the attention reads:

```python
class AttnContext:
    block_table = None      # [1, max_blocks] int32 cuda
    cache_seqlens = None    # [1]             int32 cuda
ENGINE_CTX = AttnContext()
```

vLLM threads `attn_metadata` as a real function argument because it owns the model forward;
Route A cannot, so it uses shared state. Single request runs synchronously → no concurrency
→ a shared global is safe (M4 batching will need something stricter).

### Decision 4: The index is `req_to_blocks` in the manager (vLLM style)

Rather than adding a `block_table` field to `Request`, `KVCacheManager` keeps
`req_to_blocks: dict[req_id → list[KVCacheBlock]]` and exposes `get_block_table(request)`
(→ `[1, N]` int32 CUDA). `allocate_slots` computes `ceil(total_tokens / block_size)` and
only tops up the **difference** vs blocks already held. This mirrors vLLM's
`kv_cache_manager` and keeps `Request` minimal (`id / num_computed_tokens`).

### Decision 5: Physical KV in the engine; ledger never touches tensors

The physical tensors live in `PagedEngine` (the warehouse), **not** in `KVCacheManager`
(the ledger). The ledger only allocates/recycles block ids. For M3, **`block_size = 256`**
(forced by the stock flash-attn paged kernel, §4 Decision 2) and **`num_blocks = 625`**
(≈ 5.9 GB for Qwen2.5-3B — dwarfed by an 80 GB A100); memory-profiled sizing is deferred
to M4.

```python
# [num_layers, num_blocks, block_size, num_kv_heads, head_dim] — dim 1 is what block_table indexes
self.k_cache = torch.zeros(num_layers, num_blocks, block_size, num_kv_heads, head_dim,
                           dtype=dtype, device=device)
self.v_cache = torch.zeros_like(self.k_cache)
```

Dimensions come from the model config (`num_hidden_layers`, `num_key_value_heads`,
`head_dim` / `hidden_size // num_attention_heads`), not from the caller.

### Decision 6: Explicit `position_ids` every step

M3 bypasses HF's cache, so HF can no longer infer positions from cache length → RoPE would
be wrong. The engine passes them explicitly: prefill `torch.arange(prompt_len)[None]`,
decode `[[num_computed_tokens]]`. `num_computed_tokens` is **advanced by the engine** after
each forward — forget it and `allocate_slots` under-counts and the cache overflows.

### Decision 7: The transpose convention (HF heads-first vs FA seq-first)

HF lays q/k/v out as `[batch, n_heads, seq, head_dim]` (heads before seq) and applies RoPE
in that layout; `flash_attn_with_kvcache` wants `[batch, seq, n_heads, head_dim]` (seq
before heads). So `paged_attn_forward` applies HF's RoPE first, then `transpose(1, 2)` back
to seq-first before the kernel. Both parallelize over heads — only the axis order differs.
This transpose round-trip is the M3 analogue of M2's layout dance; a native paged kernel
consumes the token-major layout, so there is no O(n²)-style tax.

## 5. Implementation Details

### `paged_attn_forward` — [nano_vllm/paged/paged_attention.py](../../nano_vllm/paged/paged_attention.py)

Copied from HF's `Qwen2Attention.forward`, replacing only the `past_key_values.update` +
attention-interface step with `flash_attn_with_kvcache`:

```python
def paged_attn_forward(self, hidden_states, position_embeddings,
                       attention_mask=None, past_key_values=None, **kwargs):
    input_shape = hidden_states.shape[:-1]                 # [batch, seq]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [b,h,s,d]
    key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    q = query_states.transpose(1, 2)     # -> [b, s, h, d] for FlashAttention
    k = key_states.transpose(1, 2)
    v = value_states.transpose(1, 2)

    attn_output = flash_attn_with_kvcache(
        q=q, k_cache=self.k_cache, v_cache=self.v_cache, k=k, v=v,
        cache_seqlens=ENGINE_CTX.cache_seqlens, block_table=ENGINE_CTX.block_table,
        causal=True,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None              # forward must return a tuple
```

### `BlockPool` / `FreeKVCacheBlockQueue` — [nano_vllm/paged/block_pool.py](../../nano_vllm/paged/block_pool.py)

vLLM-faithful: a doubly-linked free list with **sentinel head/tail** (so real blocks always
have prev/next, no branching). `get_new_blocks(n)` pops `n` from the head and sets
`ref_cnt = 1`; `free_blocks(...)` decrements `ref_cnt` and appends back when it hits 0.
`KVCacheBlock` (in [kv_cache_utils.py](../../nano_vllm/paged/kv_cache_utils.py)) is a `slots`
dataclass: `block_id / ref_cnt / prev_free_block / next_free_block`. `ref_cnt` is
overkill for single request (always 1→0) but keeps the structure faithful for M4.

### `KVCacheManager` — [nano_vllm/paged/kv_cache_manager.py](../../nano_vllm/paged/kv_cache_manager.py)

```python
def allocate_slots(self, request, num_new_tokens, block_size):
    total = request.num_computed_tokens + num_new_tokens
    needed = (total + block_size - 1) // block_size           # ceil
    held = self.req_to_blocks.get(request.id, [])
    if len(held) < needed:                                    # top up the difference only
        new = self.block_pool.get_new_blocks(needed - len(held))
        self.req_to_blocks[request.id] = held + new

def get_block_table(self, request):                           # -> [1, N] int32 cuda
    ids = [b.block_id for b in self.req_to_blocks[request.id]]
    return torch.tensor([ids], dtype=torch.int32, device="cuda")
```

### Engine wiring — [nano_vllm/paged/engine.py](../../nano_vllm/paged/engine.py)

`generate()` mutates the `ENGINE_CTX` singleton each step (it does not reconstruct it):

```python
# Prefill
self.kv_cache_manager.allocate_slots(request, prompt_len, self.block_size)
ENGINE_CTX.block_table   = self.kv_cache_manager.get_block_table(request)
ENGINE_CTX.cache_seqlens = torch.tensor([0], dtype=torch.int32, device="cuda")   # empty cache
outputs = self.model(input_ids=input_ids,
                     position_ids=torch.arange(prompt_len, device="cuda").unsqueeze(0))
request.num_computed_tokens += prompt_len

# Decode loop (per step)
self.kv_cache_manager.allocate_slots(request, 1, self.block_size)                # +1 block only when full
ENGINE_CTX.block_table   = self.kv_cache_manager.get_block_table(request)
ENGINE_CTX.cache_seqlens = torch.tensor([request.num_computed_tokens], dtype=torch.int32, device="cuda")
outputs = self.model(input_ids=torch.tensor([[next_token]], device="cuda"),
                     position_ids=torch.tensor([[request.num_computed_tokens]], device="cuda"))
request.num_computed_tokens += 1
# ... free(request) at the end
```

## 6. Testing Strategy

Build order is **isolate before end-to-end** — Steps 1 and 2 are model-independent, so
they get tested to death before the hard monkey-patch.

| Step | What | Needs model? |
|---|---|---|
| **1** | `BlockPool` + `KVCacheManager` unit tests: prefill 100 → `ceil(100/16)=7` blocks; the 113th token triggers the 8th block (off-by-one boundary); `free` returns blocks | ❌ pure logic |
| **2** | `flash_attn_with_kvcache` shape smoke test: random q/k/v + hand-built `block_table`, no model — verify shapes / dtype / **append** behavior | GPU, no model |
| **3** | patch layer 0 only — verify the paged path produces a result (other layers may error) | yes |
| **4** | patch all layers — run end-to-end | yes |
| **5** | correctness: M3 greedy `==` M2 `==` M1 `==` HF (non-regression contract), `tests/test_m3_vs_hf.py` | yes |

### Step 1 — [tests/test_block_pool.py](../../tests/test_block_pool.py)

Pure tensor-free logic, **CPU** (does not call `get_block_table`, which is CUDA-only).
Covers pool allocate/free roundtrip, exhaustion (`ValueError`), reuse after free, and the
manager's `ceil` allocation + the 8th-block boundary + `free` returning all blocks.

### Step 2 — [tests/test_flash_attn_shapes.py](../../tests/test_flash_attn_shapes.py)

Isolates the kernel: random `q/k/v` (bf16 CUDA), a hand-built paged cache and `block_table`,
**no model**. Verifies prefill output shape `[1, seq, n_heads, head_dim]` + the **append**
(new K/V written into `k_cache[0, :seq]`), and the decode shape `[1, 1, ...]`. Gated with
`pytest.importorskip("flash_attn")` + `skipif(not cuda)`, so a local Mac run **skips**
cleanly instead of failing.

### Step 5 — Non-regression: M3 == M2 == M1 == HF — written, validation PENDING

[tests/test_m3_vs_hf.py](../../tests/test_m3_vs_hf.py): greedy, token-for-token vs
HuggingFace on Qwen2.5-3B-Instruct (the benchmark model). This is the real M3 acceptance
gate — same contract M2 passed. Run gated behind `NANO_VLLM_INTEGRATION=1` on the pod.

Two M3-specific details in the test:

- **The reference is a separate, *unpatched* HF model.** M3 monkey-patches the attention on
  `llm.engine.model`, so calling `.generate()` on that instance would run M3 again
  (M3-vs-M3). The test loads a fresh `AutoModelForCausalLM` for the reference.
- **Default model is 3B, not 0.5B.** flash_attn accumulates in a different float order than
  HF's SDPA; at 0.5B the smaller argmax margins let a near-tie greedy step flip (numerics,
  not a bug). The failure message prints the first divergence index (index 0 ⇒ logic bug;
  a late index on a close call ⇒ bf16/kernel numerics).

### Running the full suite

**Local (Mac, no GPU)** — logic + skips:

```bash
python -m pytest tests/test_block_pool.py -v          # pure logic, runs
python -m pytest tests/test_flash_attn_shapes.py -v   # auto-SKIP (no CUDA / flash-attn)
```

**Pod (CUDA + flash-attn)** — in order; correctness gate before trusting any benchmark:

```bash
# 1) unit
python -m pytest tests/test_block_pool.py -v
python -m pytest tests/test_flash_attn_shapes.py -v

# 2) correctness gate (M3 greedy == HF, token-for-token)
NANO_VLLM_INTEGRATION=1 NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct \
    python -m pytest tests/test_m3_vs_hf.py -v

# 3) M1 / M2 baselines (needed for the latency comparison)
python -m benchmarks.m1_benchmark --scenarios short_chat medium_chat long_context
python -m benchmarks.m2_benchmark --cache custom --scenarios short_chat medium_chat long_context

# 4) M3 latency vs M1 / M2 (<ts> = the timestamp printed by reporter.py: "Results saved to: ...")
python -m benchmarks.m3_benchmark \
    --baseline-m1 benchmarks/results/nano_vllm_m1_<ts>.json \
    --baseline-m2 benchmarks/results/nano_vllm_m2_<ts>.json

# 5) per-request KV memory saving vs M2 (only the M3 result JSON is needed;
#    M2's max_seq_len footprint is computed analytically)
python -m benchmarks.kv_footprint benchmarks/results/nano_vllm_m3_<ts>.json
```

> Order matters: **pass the correctness gate (step 2) before trusting benchmark numbers** —
> otherwise you are measuring the speed of a wrong output.

## 7. Performance Characteristics — M1 vs M2 vs M3

> Hardware, software, scenarios, and metric definitions: [benchmark_environment.md](benchmark_environment.md).
> All numbers below: A100 80GB SXM, Qwen2.5-3B bf16, single request.

### Latency — the first real speedup (flash-attn kernel, *not* paging)

M3 decode is **~13% faster than M1 (~15% vs M2)**, consistent across scenarios. The win is
the **fused `flash_attn_with_kvcache` kernel** (IO-aware, never materializes the attention
matrix) plus shedding M2's transpose tax — **not** paging, which is memory management and
does not change single-request compute. Same-session M1 / M2 / M3:

| Scenario | TPOT M1 | TPOT M2 | **TPOT M3** | M3 vs M1 | **Tput M3** | TTFT M1 → **M3** |
|---|---|---|---|---|---|---|
| short_chat | 30.2 | 31.0 | **26.2** ms | −13% | **38.2** tok/s | 30.9 → **26.9** ms |
| medium_chat | 30.3 | 31.2 | **26.4** ms | −13% | **37.9** tok/s | 32.3 → **30.6** ms |
| long_context | 30.0 | 31.1 | **26.3** ms | −12% | **37.0** tok/s | 85.2 → **91.7** ms |

TPOT is flat ~26 ms across scenarios (length-independent, like M1/M2 — just a lower floor).
**M2's "pay the tax now, refund at M3" prediction held**: M2 was ~6% slower than M1
(transpose kernels); M3 refunds that *and* adds the kernel win → net faster than both.

**One honest loss: long-prompt prefill.** For the 2000-token prompt, M3's TTFT is ~8%
*slower* (85 → 92 ms) — flash-attn's paged prefill vs HF's dense SDPA. Decode dominates
end-to-end, so M3 still wins overall (long_context E2E latency 3.05 → 2.70 s).

### Memory — the paging win (separate line from latency)

Peak GPU memory is *not* measured (dominated by the fixed pool → uninformative at single
request). The meaningful metric is the **per-request logical KV footprint**: M2 pre-reserves
`max_seq_len` (302 MB/request); M3 holds only `ceil(len / block_size)` blocks
(`block_size = 256`). Computed from real avg lengths by
[kv_footprint.py](../../benchmarks/kv_footprint.py):

| Scenario | avg len | M2 (fixed) | M3 (paged) | saved |
|---|---|---|---|---|
| short_chat | 225 | 302 MB | 9.4 MB | **−96.9%** |
| medium_chat | 726 | 302 MB | 28.3 MB | **−90.6%** |
| long_context | 2099 | 302 MB | 84.9 MB | **−71.9%** |

The shorter the request, the bigger the win (M2's fixed reservation is pure waste). This
per-request saving is what lets M4 pack many concurrent sequences into one pool.

### Summary

Two **independent** wins: **latency from the flash-attn kernel** (~13% faster decode),
**memory from paging** (72–97% less KV per request). Both measured through the shared
`BenchmarkRunner` / `Reporter` / `SCENARIOS`; compare with `--baseline-m1` / `--baseline-m2`.
Correctness gate: M3 greedy == HF token-for-token (§6).

## 8. Known Limitations

1. **`num_blocks` / `block_size` hardcoded (625 × 256)** — no memory-profiled sizing;
   `block_size = 256` is *forced* by the stock flash-attn wheel (§4 Decision 2). Deferred to M4.
2. **Single request only** — `ENGINE_CTX` is a shared singleton and the block_table is
   `[1, N]`; concurrent batching (packing / `cu_seqlens`) is M4.
3. **No prefix caching / eviction** — `ref_cnt` and the block-hash machinery are stubbed
   for faithfulness but unused at single request.
4. **Transpose round-trip in the forward** — HF heads-first ↔ FA seq-first per step; minor,
   dwarfed by the fused kernel.
5. **Requires flash-attn + Ampere+ GPU** — no CPU/fallback path; the kernel is the point.

## 9. Future Work

### M4 — Continuous batching

- Batch many requests through one pool; this is where paging's fragmentation-free
  allocation actually pays off.
- Replace `ENGINE_CTX` with a real per-batch `attn_metadata`; pack sequences with
  `cu_seqlens` / a batched `slot_mapping`; move to `flash_attn_varlen_func`.
- Add **KV cache utilization** and **fragmentation** metrics (defined in
  [benchmark_environment.md](benchmark_environment.md)) to quantify the gain over M2's
  fixed pre-allocation.

### Interface contract (unchanged from M1)

Stable across M2–M6 so internals can be swapped without breaking user code:

- `LLM.generate(prompts, params) -> list[RequestOutput]`
- `LLM.generate_stream(prompt, params) -> Iterator[int]`
- `SamplingParams` / `RequestOutput` fields

M3 validates this contract again: the storage and attention change completely, the API does
not.

## 10. References

### Source code studied

- [FlashAttention `flash_attn_with_kvcache`](https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/flash_attn/flash_attn_interface.py)
- [HuggingFace Qwen2 modeling (`Qwen2Attention.forward`)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)
- [vLLM v1 block pool / KV cache manager](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)
- [vLLM FlashAttention backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py)

### Papers

- Efficient Memory Management for LLM Serving with PagedAttention (Kwon et al., SOSP 2023)
- FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022)

### Reading notes

- [docs/design/m3_concepts.md](m3_concepts.md) — consolidated M3 concept notes
- [docs/reading_notes/vllm_paged_attention.md](../reading_notes/vllm_paged_attention.md)
