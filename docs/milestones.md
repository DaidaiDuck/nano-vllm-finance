# Milestone Progress

## ✅ M1: Simple Single-Request Engine (Oct 2025)

**Goal**: Build minimum viable inference engine

**Features**:
- HuggingFace KV cache integration
- Custom sampler (greedy, top-k, top-p)
- Prefill + decode loop
- LLM API wrapper

**Code**: [tag/m1](https://github.com/you/repo/tree/m1)
**Blog**: [Building nano-vllm Part 1](https://...)
**Benchmark**: [m1 results](benchmarks/results/m1/report.md)

**Stats**:
- 350 LOC
- 7 tests
- Output matches HuggingFace 100% (greedy mode)

## 🚧 M2: Custom KV Cache Management (Nov 2025)

**Goal**: Replace HF DynamicCache with self-managed cache

**Features**:
- Pre-allocated continuous KV tensor
- HF Cache interface compatibility
- Eliminate torch.cat overhead

**Status**: In progress

## 📋 M3: PagedAttention (Dec 2025)

**Goal**: Implement paged KV cache

**Planned features**:
- Block pool
- Block table per request
- FlashAttention paged kernel integration

## 📋 M4-M6: Future

...