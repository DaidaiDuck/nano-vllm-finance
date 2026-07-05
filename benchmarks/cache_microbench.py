# benchmarks/cache_microbench.py
# Run as a module from the repo root:
#   python -m benchmarks.cache_microbench
#   python -m benchmarks.cache_microbench --max-steps 4096 --checkpoints 256
#
# Micro-benchmark: HF DynamicCache vs MyKVCache, timing the KV-cache op in isolation
# (model removed) so each cache's per-step scaling is visible. One "step" = writing
# one new token to every layer (num_layers update() calls) = one decode step minus
# the model.
#
# FINDINGS (A100, Qwen2.5-3B config: 36 layers, 2 kv heads, 128 head_dim) — these
# contradict the naive "custom cache is faster" intuition:
#   - DynamicCache (torch.cat): the copy IS O(n), but per-step time only rises gently
#     (~800 -> ~880 us over seq 33 -> 4128). It is OVERHEAD-BOUND — a ~800 us fixed
#     floor (per-layer kernel launch + alloc) dwarfs the O(n) copy. O(n^2) is real but
#     needs tens of thousands of tokens to dominate; unreachable in practice.
#   - MyKVCache (pre-alloc, seq-major): flat but ~3.8x SLOWER (~3260 vs ~860 us). Not
#     because it copies more bytes, but because it issues ~3x more GPU kernels per
#     update (transpose + contiguous, on both write and read) vs DynamicCache's single
#     cat. The slowdown is kernel-launch count, not bytes.
# So at single request the custom cache is SLOWER. Its payoff is architectural
# (owning the layout enables M3 PagedAttention / M4 batching), not raw speed.
#
# MEASUREMENT NOTE — ALWAYS WARM UP. Without warmup the first timed step eats one-time
# cold-start costs (CUDA kernel JIT/caching + the allocator's first cudaMalloc). In an
# early run the first point (seq=33) measured 936 us — the HIGHEST of all — purely from
# cold start; after adding warmup it dropped to 833 us and the curve became clean.
# _time_decode() below warms a throwaway cache before the timed loop.

import argparse
import json
import time

import torch

from nano_vllm.kv_cache import MyKVCache


def _kv(num_kv_heads, seq_len, head_dim, device, dtype):
    return torch.randn(1, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)


def _time_decode(make_cache, num_layers, num_kv_heads, head_dim, device, dtype,
                 prefill_len, max_steps, checkpoints):
    """Prefill + max_steps single-token decodes; record per-step time at checkpoints.

    A "step" writes one new token to every layer (num_layers cache.update calls),
    which is exactly what one decode step of the real engine does.
    """
    k_pre = _kv(num_kv_heads, prefill_len, head_dim, device, dtype)
    # Reuse one 1-token tensor for every decode step (update copies it, never mutates).
    k1 = _kv(num_kv_heads, 1, head_dim, device, dtype)

    # Warmup on a THROWAWAY cache: the CUDA kernel cache and caching allocator are
    # process-global, so heating them here means the first *timed* step below isn't
    # inflated by one-time cold-start costs (kernel JIT + first cudaMalloc).
    warmup = make_cache()
    for i in range(num_layers):
        warmup.update(k_pre, k_pre, i)
    for _ in range(5):
        for i in range(num_layers):
            warmup.update(k1, k1, i)
    if device == "cuda":
        torch.cuda.synchronize()
    del warmup

    # The real, timed cache — starts fresh so seq_len labels begin at prefill_len+1.
    cache = make_cache()
    for i in range(num_layers):
        cache.update(k_pre, k_pre, i)
    if device == "cuda":
        torch.cuda.synchronize()

    per_step_us = {}          # seq_len -> microseconds for that decode step (all layers)
    total_start = time.perf_counter()
    for step in range(max_steps):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(num_layers):
            cache.update(k1, k1, i)
        if device == "cuda":
            torch.cuda.synchronize()
        dt_us = (time.perf_counter() - t0) * 1e6

        seq_len = prefill_len + step + 1
        if step % checkpoints == 0 or step == max_steps - 1:
            per_step_us[seq_len] = dt_us
    total_s = time.perf_counter() - total_start
    return per_step_us, total_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=36)       # Qwen2.5-3B
    parser.add_argument("--num-kv-heads", type=int, default=2)      # GQA
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--prefill-len", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--checkpoints", type=int, default=128,
                        help="record the per-step time every N steps")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    max_seq_len = args.prefill_len + args.max_steps + 8

    print(
        f"device={device} dtype={args.dtype} layers={args.num_layers} "
        f"kv_heads={args.num_kv_heads} head_dim={args.head_dim} "
        f"prefill={args.prefill_len} steps={args.max_steps}"
    )

    from transformers import DynamicCache
    backends = {
        "DynamicCache": lambda: DynamicCache(),
        "MyKVCache": lambda: MyKVCache(
            args.num_layers, max_seq_len, args.num_kv_heads, args.head_dim, dtype, device
        ),
    }

    results = {}
    for name, make in backends.items():
        try:
            per_step, total = _time_decode(
                make, args.num_layers, args.num_kv_heads, args.head_dim,
                device, dtype, args.prefill_len, args.max_steps, args.checkpoints,
            )
        except Exception as e:  # e.g. DynamicCache API differs in this transformers
            print(f"\n{name}: FAILED ({type(e).__name__}: {e})")
            continue
        results[name] = {"per_step_us": per_step, "total_s": total}

    if not results:
        print("No backend ran successfully.")
        return

    # Per-step time vs seq_len: DynamicCache rises only gently (overhead-bound);
    # MyKVCache is flat but several x higher (more kernel launches per update).
    print("\n=== per-step update time (us) vs seq_len ===")
    names = list(results.keys())
    lens = sorted(next(iter(results.values()))["per_step_us"].keys())
    print(f"{'seq_len':>8}" + "".join(f"{n:>16}" for n in names))
    for sl in lens:
        row = "".join(
            f"{results[n]['per_step_us'].get(sl, float('nan')):>16.1f}" for n in names
        )
        print(f"{sl:>8}{row}")

    print("\n=== total time to run all decode steps ===")
    for n in names:
        print(f"  {n:<14} {results[n]['total_s']:.3f} s")
    if "DynamicCache" in results and "MyKVCache" in results:
        d = results["DynamicCache"]["total_s"]
        m = results["MyKVCache"]["total_s"]
        if m <= d:
            print(f"  MyKVCache is {d / max(m, 1e-9):.1f}x FASTER on the cache op alone")
        else:
            print(f"  MyKVCache is {m / max(d, 1e-9):.1f}x SLOWER on the cache op alone")

    out = "benchmarks/results/cache_microbench.json"
    with open(out, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "device": device,
                "results": {
                    n: {
                        "total_s": r["total_s"],
                        "per_step_us": {str(k): v for k, v in r["per_step_us"].items()},
                    }
                    for n, r in results.items()
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
