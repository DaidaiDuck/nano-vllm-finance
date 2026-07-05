# nano-vllm-finance

> A from-scratch LLM inference engine, rebuilt milestone by milestone to
> understand and reconstruct the core ideas behind vLLM — with an eye toward
> low-latency quantitative-trading workloads.

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org)

## Status

🚧 **In progress.** **M1 (single-request baseline) is complete**: a correct
prefill + decode engine on top of HuggingFace, a custom sampler, and token-level
streaming. Greedy output matches HuggingFace **token-for-token**, 23/23 tests pass,
and it is benchmarked on an A100. M2–M6 are the roadmap below.

## Overview

nano-vllm-finance reconstructs a vLLM-style inference engine from scratch, one
optimization at a time, benchmarking at every step so each improvement is measured
against the previous milestone. The advanced pieces are **goals on the roadmap, not
yet built**:

- **Custom KV cache** — replace HF's `DynamicCache` with a pre-allocated buffer (M2)
- **PagedAttention** — block-based KV cache + paged kernel (M3)
- **Continuous batching** — concurrent requests (M4)
- **Quant specialization** — fixed system prompt + short query patterns (M5)
- **HTTP serving** (M6)

## Milestones

| Milestone | Feature | Status | Tag | Design | Blog |
|-----------|---------|--------|-----|--------|------|
| M1 | Single-request engine (prefill + decode) | ✅ Done | [m1](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m1) | [design](docs/design/m1_design.md) | [EN](docs/blogs/m1_building_nano_vllm.md) · [中文](docs/blogs/m1_building_nano_vllm.zh.md) |
| M2 | Custom KV cache | ✅ Done | [m2](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m2) | [design](docs/design/m2_design.md) | [EN](docs/blogs/m2_custom_kv_cache.md) · [中文](docs/blogs/m2_custom_kv_cache.zh.md) |
| M3 | PagedAttention | ⬜ Planned | — | — | — |
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

```
nano_vllm/       core engine — engine, sampler, types (and kv_cache for M2)
benchmarks/      benchmark harness + results
tests/           unit (CPU) + integration (GPU) tests
docs/design/     per-milestone design docs + benchmark_environment.md
docs/blogs/      milestone write-ups (EN / 中文)
scripts/         run_tests.sh and other helpers
```
