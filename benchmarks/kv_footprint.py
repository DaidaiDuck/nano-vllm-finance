# benchmarks/kv_footprint.py
# Per-request KV memory footprint: M2 (max_seq_len pre-alloc) vs M3 (block-based).
#
# This is ANALYTICAL, not a GPU measurement. M3 pre-allocates the whole paged pool up
# front, so torch.cuda.max_memory_allocated() hides the per-request saving. The logical
# footprint a request actually needs is fully determined by length + config, so we compute
# it. Real lengths come from a benchmark result JSON (per-scenario avg prompt/output len).
#
# Runs on CPU, no GPU / no model weights (only the config is fetched).
#
#   python -m benchmarks.kv_footprint benchmarks/results/nano_vllm_m3_<ts>.json
#   python -m benchmarks.kv_footprint <json> --max-seq-len 8192 --block-size 16
#
# M2 reserves max_seq_len regardless of actual length; M3 holds ceil(len/block_size) blocks.
# Saving = 1 - M3_bytes / M2_bytes. The shorter the request, the bigger the win.
import argparse
import json
import math


def _bytes_per_token(num_layers, num_kv_heads, head_dim, dtype_bytes):
    # K and V, summed over all layers, for ONE token slot.
    return num_layers * num_kv_heads * head_dim * dtype_bytes * 2


def _resolve_config(args):
    """Config dims: explicit CLI overrides win; otherwise read from the HF config (no weights)."""
    if args.num_layers and args.num_kv_heads and args.head_dim:
        return args.num_layers, args.num_kv_heads, args.head_dim
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return cfg.num_hidden_layers, cfg.num_key_value_heads, head_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="benchmark result JSON (for per-scenario avg lengths)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max-seq-len", type=int, default=8192,
                    help="M2's pre-allocated max_seq_len (nano_vllm/engine.py default: 8192)")
    ap.add_argument("--block-size", type=int, default=16, help="M3 paged block size")
    ap.add_argument("--dtype-bytes", type=int, default=2, help="bf16 = 2")
    # Optional offline overrides (skip AutoConfig download):
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--num-kv-heads", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=None)
    args = ap.parse_args()

    num_layers, num_kv_heads, head_dim = _resolve_config(args)
    bpt = _bytes_per_token(num_layers, num_kv_heads, head_dim, args.dtype_bytes)
    MB = 1e6

    with open(args.result_json) as f:
        data = json.load(f)

    print(f"Model: {args.model}  "
          f"(layers={num_layers}, kv_heads={num_kv_heads}, head_dim={head_dim}, "
          f"dtype={args.dtype_bytes}B)")
    print(f"bytes/token (all layers, K+V): {bpt/1024:.1f} KB")
    print(f"M2 max_seq_len={args.max_seq_len}  |  M3 block_size={args.block_size}\n")

    # M2 is length-independent: it always reserves max_seq_len.
    m2_mb = bpt * args.max_seq_len / MB

    header = f"{'scenario':<16}{'avg_len':>9}{'M2_MB':>10}{'M3_MB':>10}{'M3/M2':>9}{'saved':>9}"
    print(header)
    print("-" * len(header))

    for name, m in data["scenarios"].items():
        avg_len = m["avg_prompt_len"] + m["avg_output_len"]     # tokens actually held
        blocks = math.ceil(avg_len / args.block_size)
        m3_slots = blocks * args.block_size                     # rounded up to a full block
        m3_mb = bpt * m3_slots / MB
        ratio = m3_mb / m2_mb
        print(f"{name:<16}{avg_len:>9.0f}{m2_mb:>10.1f}{m3_mb:>10.1f}"
              f"{ratio:>8.1%}{1 - ratio:>9.1%}")

    print(f"\nNote: uses per-scenario AVG length (per-request lengths are not serialized).")
    print(f"M2 pre-reserves max_seq_len (length-independent); M3 rounds up to block_size "
          f"(internal frag <= {args.block_size - 1} tokens/request).")


if __name__ == "__main__":
    main()
