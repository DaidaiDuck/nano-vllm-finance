# Architecture

> Last updated: 2025-XX-XX (after M2 completion)
> Current version: M2

## Current Architecture (M2)

[最新架构图]

LLM → SimpleEngine → (Tokenizer + Model + MyKVCache + Sampler) → GPU

### Components

#### LLM (entry point)
...

#### SimpleEngine (orchestrator)
...

#### MyKVCache (M2 new)
Pre-allocated continuous KV tensor management.
See [m2_kv_cache.md](design/m2_kv_cache.md) for details.

## Evolution History

This architecture evolved over milestones:

### M1: SimpleEngine + HF DynamicCache
[M1 架构图]
[简短描述]

### M2: SimpleEngine + MyKVCache  ← Current
[M2 架构图]
[与 M1 的差异]

### M3 (planned): SimpleEngine + PagedAttention
[M3 计划图]

### M4 (planned): Engine + Scheduler + ModelRunner
[M4 计划图]

## Key Design Decisions

### Why custom KV cache (M2)?
HuggingFace's DynamicCache uses torch.cat, causing memory fragmentation.
Self-managed cache enables future optimizations like PagedAttention.

### Why no Scheduler in M1-M3?
M1-M3 focus on single-request correctness. Scheduler is introduced in M4
when multi-request batching is needed.

...