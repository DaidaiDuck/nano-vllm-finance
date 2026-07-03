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

## Reproducibility checklist

- Use **Secure Cloud** and the pinned A100 80GB SXM (not Community).
- Same environment for M1/M2/M3 — do not change GPU/model/dtype between milestones.
- Warmup (3 requests/scenario) and `torch.cuda.synchronize()` timing are built into
  the runner; greedy decoding (`temperature=0.0`) and `seed=42` prompt generation
  keep runs deterministic.

## Scope note

`peak_memory_mb`, per-token latency traces, and the M1-vs-M2 comparison harness
are **M2 work** (see [m2_design.md](../docs/design/m2_design.md) §7). The M1 run
above only measures throughput / latency / TTFT / TPOT.
