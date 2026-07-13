# nano-vllm-finance

> A from-scratch LLM inference engine, rebuilt milestone by milestone to
> understand and reconstruct the core ideas behind vLLM — with an eye toward
> low-latency quantitative-trading workloads.

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org)

## Status

🚧 **In progress. M1–M3 complete**, all validated token-for-token vs HuggingFace and
benchmarked on an A100.

- **M1** — single-request prefill + decode engine on HuggingFace, custom sampler,
  token-level streaming. Greedy output matches HF **token-for-token**.
- **M2** — custom pre-allocated contiguous KV cache (`MyKVCache`). Correct, but a
  *successful failure*: at single request it is **~6% slower and uses ~250 MB more**
  (the `torch.cat` bottleneck it targeted turned out negligible). The value is
  architectural — groundwork for paging.
- **M3** — **PagedAttention**: block-based KV cache + the FlashAttention paged kernel
  (`flash_attn_with_kvcache`), wired by monkey-patching HF's attention. Greedy output
  matches HF token-for-token (`test_m3_vs_hf` 3/3). **The first real single-request
  speedup — ~13% faster decode than M1 (~15% vs M2)** from the fused kernel, plus **72–97%
  less KV memory per request** from block allocation.

M4–M6 are the roadmap below.

## Overview

nano-vllm-finance reconstructs a vLLM-style inference engine from scratch, one
optimization at a time, benchmarking at every step so each improvement is measured
against the previous milestone.

- ✅ **Custom KV cache** — replace HF's `DynamicCache` with a pre-allocated buffer (M2)
- ✅ **PagedAttention** — block-based KV cache + paged kernel (M3)
- ⬜ **Continuous batching** — concurrent requests (M4)
- ⬜ **Quant specialization** — fixed system prompt + short query patterns (M5)
- ⬜ **HTTP serving** (M6)

## Milestones

| Milestone | Feature | Status | Tag | Design | Blog |
|-----------|---------|--------|-----|--------|------|
| M1 | Single-request engine (prefill + decode) | ✅ Done | [m1](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m1) | [design](docs/design/m1_design.md) | [EN](docs/blogs/m1_building_nano_vllm.md) · [中文](docs/blogs/m1_building_nano_vllm.zh.md) |
| M2 | Custom KV cache | ✅ Done | [m2](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m2) | [design](docs/design/m2_design.md) | [EN](docs/blogs/m2_custom_kv_cache.md) · [中文](docs/blogs/m2_custom_kv_cache.zh.md) |
| M3 | PagedAttention | ✅ Done | [m3](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m3) | [design](docs/design/m3_design.md) | [EN](docs/blogs/m3_paged_attention.md) · [中文](docs/blogs/m3_paged_attention.zh.md) |
| M4 | Continuous batching | ⬜ Planned | — | — | — |
| M5 | Quant specialization | ⬜ Planned | — | — | — |
| M6 | HTTP service | ⬜ Planned | — | — | — |

To explore a completed milestone: `git checkout m1`.

## M1 benchmark (baseline)

Single request, A100 80GB SXM, Qwen2.5-3B bf16 — full setup in
[benchmark_environment.md](docs/design/benchmark_environment.md). `Throughput` is
output tokens/s.

| Scenario | Prompt | Output | Throughput | P99 latency | TTFT | TPOT |
|----------|--------|--------|------------|-------------|------|------|
| short_chat | ~125 | 100 | 29.4 tok/s | 3.62 s | 36 ms | 34.0 ms |
| medium_chat | ~526 | 200 | 29.2 tok/s | 7.38 s | 38 ms | 34.2 ms |
| long_context | ~1999 | 100 | 29.2 tok/s | 3.47 s | 89 ms | 33.7 ms |

End-to-end latency follows `latency ≈ TTFT + output_len × TPOT` (verified to <1%),
cleanly separating compute-bound prefill (TTFT) from memory-bound decode (TPOT).

## M2 benchmark — a "successful failure"

M2 swaps HF's `DynamicCache` for a pre-allocated contiguous `MyKVCache`. Output is
**identical to M1 token-for-token**, but the expected `torch.cat` (O(n²)) win never
materializes — at single request the copy is ~0.2% of a decode step. M2 is in fact
**~6% slower** (extra transpose kernels) and uses **~250 MB more** (it pre-reserves
`max_seq_len`). Same-session vs M1:

| Scenario | TPOT (M1 → M2) | Throughput (M1 → M2) | Peak mem (M1 → M2) |
|----------|----------------|----------------------|--------------------|
| short_chat | 34.6 → 36.8 ms | 28.9 → 27.2 tok/s | 6242 → 6538 MB |
| medium_chat | 34.7 → 36.9 ms | 28.8 → 27.1 tok/s | 6397 → 6677 MB |
| long_context | 34.6 → 36.9 ms | 28.4 → 26.7 tok/s | 6953 → 7174 MB |

The value is architectural: owning the cache is the prerequisite for M3. See
[m2_design.md](docs/design/m2_design.md) §7 for the full analysis.

## M3 benchmark — the first real speedup

M3 replaces the contiguous cache with a block-based paged cache read by the
FlashAttention paged kernel. **Decode (TPOT) is ~13% faster than M1 (~15% vs M2)**,
consistently across scenarios — the win is the **fused flash-attn kernel** (plus shedding
M2's transpose tax), *not* paging itself. Same-session M1 / M2 / M3:

| Scenario | TPOT (M1 / M2 / **M3**) | M3 vs M1 | Throughput **M3** | TTFT (M1 / **M3**) |
|----------|-------------------------|----------|-------------------|--------------------|
| short_chat | 30.2 / 31.0 / **26.2** ms | **−13%** | **38.2** tok/s | 30.9 / **26.9** ms |
| medium_chat | 30.3 / 31.2 / **26.4** ms | **−13%** | **37.9** tok/s | 32.3 / **30.6** ms |
| long_context | 30.0 / 31.1 / **26.3** ms | **−12%** | **37.0** tok/s | 85.2 / **91.7** ms |

> **Source** (all in [`benchmarks/results/m3/`](benchmarks/results/m3/), same session):
> M1 `nano_vllm_m1_20260711_012918.json`, M2 `nano_vllm_m2_20260711_013719.json`,
> M3 `nano_vllm_m3_20260711_014557.json` — fields `avg_tpot` / `avg_ttft` / `output_throughput`.

Two honest caveats: (1) M1/M2 here are re-baselined in the same session as M3, so their
absolute numbers differ from the M2 table above (run-to-run machine variance — what
matters is the within-session delta). (2) **Long-prompt prefill (TTFT) is ~8% slower**
(long_context: 85 → 92 ms) — flash-attn's one loss — but decode dominates, so M3 still
wins end-to-end.

**The separate memory win — paging, not flash-attn.** M2 pre-reserves `max_seq_len`
(302 MB/request for Qwen2.5-3B); M3 holds only `ceil(len / block_size)` blocks. Per-request
KV footprint (`python -m benchmarks.kv_footprint --block-size 256`):

| Scenario | avg len | M2 (fixed) | M3 (paged) | KV memory saved |
|----------|---------|------------|------------|-----------------|
| short_chat | 225 | 302 MB | 9.4 MB | **−96.9%** |
| medium_chat | 726 | 302 MB | 28.3 MB | **−90.6%** |
| long_context | 2099 | 302 MB | 84.9 MB | **−71.9%** |

> **Source**: avg lengths from M3 [`nano_vllm_m3_20260711_014557.json`](benchmarks/results/m3/nano_vllm_m3_20260711_014557.json)
> (`avg_prompt_len + avg_output_len`); MB computed by `benchmarks/kv_footprint.py --block-size 256`
> (M2 = fixed `max_seq_len` 8192).

The shorter the request, the bigger the win (M2's fixed reservation is pure waste). This
per-request saving is what lets M4 pack many concurrent sequences into one pool. Full
analysis: [m3_design.md](docs/design/m3_design.md) §7.

## Quick start

```bash
cd nano-vllm-finance
pip install -e .                    # ① 项目+ 运行依赖(transformers==5.13.0)
pip install -r requirements.txt     # ② 补 dev 工具(pytest 等,跑测试要)
pip show transformers               # 确认 5.13.0

# run the tests (CPU tests always; GPU integration on a CUDA machine)
bash scripts/run_tests.sh

# run the M1 benchmark (from the repo root, as a module)
python -m benchmarks.m1_benchmark --version m1 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --scenarios short_chat medium_chat long_context
```

See [tests/README.md](tests/README.md) and [benchmarks/README.md](benchmarks/README.md)
for details.

## Project structure

Milestone code is split into subpackages **by architecture generation** — which file
belongs to which milestone is answered by the folder:

```
nano_vllm/
  core/          shared across all milestones: types, sampler
  simple/        M1 (HF DynamicCache) + M2 (MyKVCache) — SimpleEngine
  paged/         M3+ PagedAttention — engine, paged_attention, block_pool,
                 kv_cache_manager, kv_cache_utils (M4 continuous batching grows here)
benchmarks/      benchmark harness + results
tests/           unit (CPU) + integration (GPU) tests
docs/design/     per-milestone design docs + benchmark_environment.md
docs/blogs/      milestone write-ups (EN / 中文)
scripts/         run_tests.sh and other helpers
```

`from nano_vllm import LLM` gives the import-safe M1/M2 engine (no flash-attn dependency).
M3 needs flash-attn + CUDA, so import it explicitly: `from nano_vllm.paged.engine import LLM`.
To browse a finished milestone's exact code, `git checkout <tag>` — a tag is a full snapshot,
so at `m2` there is no `paged/` yet, and at `m3` `paged/` is the PagedAttention version.

| Tag | Read | What |
|-----|------|------|
| m1 / m2 | `nano_vllm/simple/` | HF cache → custom contiguous cache |
| m3 | `nano_vllm/paged/` | PagedAttention |
| m4 (planned) | `nano_vllm/paged/` (extended) | continuous batching |
