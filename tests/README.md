# Tests

Pinned environment: [docs/design/benchmark_environment.md](../docs/design/benchmark_environment.md).

## Quick start (recommended)

```bash
# from repo root — runs CPU tests, then GPU integration tests if CUDA is present
bash scripts/run_tests.sh

# use the actual benchmark model instead of the fast 0.5B default
MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/run_tests.sh
```

## What runs, in order

**Stage 1 — CPU only (no GPU, no model download), always runs:**

| File | Covers |
|------|--------|
| `test_sampler.py` | greedy / temperature / top-k / top-p |
| `test_types.py` | `SamplingParams` / `RequestOutput` fields |

```bash
python -m pytest tests/test_sampler.py tests/test_types.py -v
```

**Stage 2 — GPU integration (needs CUDA + HF model), gated by `NANO_VLLM_INTEGRATION=1`:**

| File | Covers |
|------|--------|
| `test_m1_vs_hf.py` | **most important** — greedy output matches HuggingFace token-for-token |
| `test_engine.py` | `LLM.generate` returns `RequestOutput`; stream == generate |
| `test_engine_m1.py` | basic generation, batch prompts, `max_tokens`, determinism |
| `test_generation.py` | standalone `generation.generate()` |

```bash
NANO_VLLM_INTEGRATION=1 NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
    python -m pytest tests/test_m1_vs_hf.py tests/test_engine.py \
                     tests/test_engine_m1.py tests/test_generation.py -v
```

Without `NANO_VLLM_INTEGRATION=1` (or on a machine without CUDA), Stage 2 is
skipped automatically — so a plain `pytest` only runs the fast CPU suite.

## M2 (not part of M1)

`test_kv_cache.py` tests `MyKVCache` (M2). It is pure tensor logic and runs on
CPU (and CUDA if available), but belongs to the M2 milestone:

```bash
python -m pytest tests/test_kv_cache.py -v
```
