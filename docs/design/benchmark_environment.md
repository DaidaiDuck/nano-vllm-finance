# Benchmark Environment (shared across M1–M3)

> Single source of truth for the benchmarking setup. Every milestone design doc
> (M1, M2, M3, …) references this file so that all reported numbers are measured
> on the **same hardware and configuration** and remain directly comparable.
>
> The whole point of the milestone benchmarks is the *relative* delta
> (M2 vs M1, M3 vs M2). Changing the GPU, model, dtype, or driver between
> milestones invalidates the comparison — keep this fixed.

## Hardware

| Item | Value |
|------|-------|
| GPU | NVIDIA A100-80GB (single) <!-- TODO: confirm actual card --> |
| ECC | off |
| Clocks | locked (fixed SM/memory clocks for reproducibility) |
| Interconnect | N/A (single GPU) |

> If the final runs use a different card (e.g. RTX 4090 24GB), update this table
> **once** here and re-state it nowhere else.

## Software

| Item | Value |
|------|-------|
| Model | Qwen/Qwen2.5-3B-Instruct |
| dtype | bfloat16 |
| CUDA | <TODO> |
| PyTorch | <TODO> |
| transformers | <TODO> |
| Driver | <TODO> |
| OS | <TODO> |

## Measurement protocol

- **Warmup**: 3 requests per scenario before timing (discarded).
- **Sampling**: greedy (`temperature=0.0`) for reproducibility.
- **Timing**: `torch.cuda.synchronize()` before each timestamp so measurements
  reflect completed GPU work, not kernel enqueue.
- **Seed**: fixed (`seed=42`) for synthetic prompt generation.
- **Repeats**: report aggregate over all requests in a scenario (avg / p50 / p95 / p99).

## Scenarios

| Scenario | Prompt len (tokens) | Output len | Concurrency | #Requests |
|----------|---------------------|------------|-------------|-----------|
| short_chat | 50–150 | 100 | 1 | 50 |
| medium_chat | 400–600 | 200 | 1 | 30 |
| long_context | 1800–2200 | 100 | 1 | 20 |
| high_concurrency | 50–150 | 100 | 16 | 64 (M4+) |

## Metrics glossary

| Metric | Definition |
|--------|------------|
| Throughput | output (or total) tokens / total wall-clock duration |
| TTFT | Time To First Token — request submit → first token emitted |
| TPOT | Time Per Output Token — `(total_time - ttft) / (output_len - 1)` |
| Latency | end-to-end time per request (avg / p50 / p95 / p99 / max) |
| Peak memory | `torch.cuda.max_memory_allocated()` over the run (M2+) |
| KV cache utilization | used KV slots / allocated KV slots (M3, paged) |
| Fragmentation | 1 − (largest free block / total free) or wasted-slot ratio (M3) |
