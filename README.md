# nano-vllm-finance

> A production-grade LLM inference engine built from scratch, 
> optimized for quantitative trading scenarios.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org)

## Overview

nano-vllm-finance is a from-scratch LLM inference engine designed for low-latency
quantitative trading workloads. Key differentiators:

- **Custom PagedAttention**: ~85% performance of vLLM
- **Triton Attention Kernel**: ~80% of FlashAttention-2
- **Persistent System Prompt Cache**: 5x P99 latency reduction
- **Quant-First Design**: Optimized for fixed system prompt + short query patterns

## Performance

| Scenario | Throughput | P99 Latency | vs vLLM |
|----------|-----------|-------------|---------|
| Quant typical | XXX tok/s | XXX ms | 0.85x |
| Long context | XXX tok/s | XXX ms | 1.10x |

[Full benchmark report](benchmarks/results/final/)

## Architecture

[architecture diagram]

See [architecture.md](docs/architecture.md) for details.

## Quick Start

```bash
pip install -e .
python examples/basic_generate.py
```

## Development History

This project was developed iteratively across 6 milestones:

| Milestone | Feature | Tag | Blog |
|-----------|---------|-----|------|
| M1 | Single-request inference | [m1](https://github.com/you/repo/tree/m1) | [Blog 1](https://...) |
| M2 | Custom KV cache | [m2](https://github.com/you/repo/tree/m2) | [Blog 2](https://...) |
| M3 | PagedAttention | [m3](https://github.com/you/repo/tree/m3) | [Blog 3](https://...) |
| M4 | Continuous Batching | [m4](https://github.com/you/repo/tree/m4) | [Blog 4](https://...) |
| M5 | Quant specialization | [m5](https://github.com/you/repo/tree/m5) | [Blog 5](https://...) |
| M6 | HTTP service | [m6](https://github.com/you/repo/tree/m6) | [Blog 6](https://...) |

To explore any version: `git checkout m{N}`

## Project Structure