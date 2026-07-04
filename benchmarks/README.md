# Benchmarks

Pinned environment (GPU, model, dtype, CUDA/PyTorch, protocol):
[docs/design/benchmark_environment.md](../docs/design/benchmark_environment.md).

**Run tests first** ([tests/README.md](../tests/README.md)) — a benchmark on a
buggy engine produces meaningless numbers. Once `test_m1_vs_hf.py` is green,
proceed here.

## How to run (M1)

Run as a **module from the repo root** (the entry point uses absolute imports, so
`python benchmarks/m1_benchmark.py` will not resolve the packages):

```bash
# from repo root, on the A100 pod
python -m benchmarks.m1_benchmark \
    --version m1 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --scenarios short_chat medium_chat long_context
```

- `--scenarios`: **list the M1 scenarios explicitly.** The default is *all*
  scenarios, which includes `high_concurrency` — that one is an M4 target and
  would just run 64 requests serially here (slow, and not meaningful for M1).
- `--model`: the benchmark model, fixed to Qwen2.5-3B-Instruct per the environment
  doc.
- `--version`: tags the output file (`m1`).

## Output

Results are written to `benchmarks/results/<engine>_<version>_<timestamp>.json`
(e.g. `benchmarks/results/nano_vllm_m1_20260703_101500.json`) and a per-scenario
report is printed to stdout (throughput, latency percentiles, TTFT, TPOT).

Copy the headline numbers into the design doc / blog tables once the run finishes:
- [docs/design/m1_design.md](../docs/design/m1_design.md) §7
- [docs/blogs/m1_building_nano_vllm.md](../docs/blogs/m1_building_nano_vllm.md) §9

## How to run (M2) — keep it minimal

M2 replaces HF's `DynamicCache` with the custom pre-allocated `MyKVCache`. At single
request the copy cost of `torch.cat` is ~0.2% of a decode step (dominated by weight
loading), so **there is no end-to-end speedup to chase.** Three things matter:

**1. Correctness (the milestone itself).** MyKVCache must produce output identical
to HF, token-for-token:

```bash
NANO_VLLM_INTEGRATION=1 python -m pytest tests/test_m1_vs_hf.py tests/test_kv_cache.py -v
```

**2. Parity — M2 must not regress vs M1.** Run both cache backends through the same
harness and compare; expect the latency/TPOT numbers to be ≈ equal.

```bash
# M1 path (HF DynamicCache)
python -m benchmarks.m2_benchmark --cache hf --version m1_rerun \
    --scenarios short_chat medium_chat long_context

# M2 path (MyKVCache) + comparison against the M1 run
python -m benchmarks.m2_benchmark --cache custom --version m2 \
    --scenarios short_chat medium_chat long_context \
    --baseline benchmarks/results/nano_vllm_m1_rerun_<timestamp>.json
```

**3. The cache O(n²) → O(n), shown honestly.** The structural win lives in the cache
op itself, not end-to-end. `cache_microbench.py` times `DynamicCache.update` vs
`MyKVCache.update` with the model removed, so the scaling is visible:

```bash
python -m benchmarks.cache_microbench --max-steps 4096 --checkpoints 256
```

DynamicCache's per-step time rises with sequence length (total O(n²)); MyKVCache is
flat (total O(n)). Pair this with a note that it does **not** move end-to-end TPOT.

**On memory:** `m2_benchmark.py` prints peak allocated memory. Expect **M2 ≥ M1** —
MyKVCache pre-reserves `max_seq_len`, so at single request it uses *more*, not less.
That is the intended tradeoff (pre-allocation enables M3/M4), not a regression.
Fragmentation is an M4 (batching) phenomenon; don't build the M2 story on it.

> `max_seq_len` (in `SimpleEngine.__init__`) must exceed the longest prompt +
> output. 8192 costs ~302 MB on the A100 (negligible) and does not affect TPOT.

## Reproducibility checklist

- Use **Secure Cloud** and the pinned A100 80GB SXM (not Community).
- Same environment for M1/M2/M3 — do not change GPU/model/dtype between milestones.
- Warmup (3 requests/scenario) and `torch.cuda.synchronize()` timing are built into
  the runner; greedy decoding (`temperature=0.0`) and `seed=42` prompt generation
  keep runs deterministic.

## Scope note

- `m1_benchmark.py` — the original M1 harness; latency / throughput / TTFT / TPOT.
- `m2_benchmark.py` — adds the `--cache {hf,custom}` backend switch + peak memory.
  Use it for the M1-vs-M2 parity check above.
- `cache_microbench.py` — model-free; times the cache op alone to show O(n²) vs O(n).
- Fragmentation shown fairly only under concurrency — deferred to **M4**
  (see [m2_design.md](../docs/design/m2_design.md) §7).
