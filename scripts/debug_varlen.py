"""Isolate the paged flash_attn_varlen_func call from the whole model.

test_m4_vs_hf fails at token 0 even for batch=1, which means batch_attn_forward's forward
is wrong independently of any multi-request packing. This script removes the model entirely:
it feeds random q/k/v through the *exact same* paged varlen call pattern batch_attn_forward
uses, and compares against flash_attn_func (the non-paged batched API) as the golden
reference. If a case mismatches, the paged calling convention is the bug -- not the runner,
not the scheduler, not the sampler.

Run on the pod:
    python scripts/debug_varlen.py
"""
import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func

torch.manual_seed(0)
DEVICE = "cuda"
DTYPE = torch.bfloat16
HQ, HKV, D = 16, 2, 128          # Qwen2.5-3B: 16 query heads, 2 KV heads (GQA), head_dim 128
BLOCK = 256                       # stock flash-attn requires paged block_size % 256 == 0
NUM_BLOCKS = 8


def _paged_attn(q, k, v, cu_q, cu_k, max_q, max_k, block_table):
    """Mirror batch_attn_forward step 2+3: scatter KV into a paged cache, then varlen attend.

    q/k/v are packed [total_tokens, heads, head_dim]. block_table says which physical blocks
    each sequence owns. The KV is written via the same block_id*BLOCK+offset flattening the
    runner computes into slot_mapping.
    """
    k_cache = torch.zeros(NUM_BLOCKS, BLOCK, HKV, D, device=DEVICE, dtype=DTYPE)
    v_cache = torch.zeros_like(k_cache)

    # Write each sequence's KV into its blocks at the right offsets (what slot_mapping encodes)
    for s in range(len(cu_k) - 1):
        seq_len = cu_k[s + 1] - cu_k[s]
        for pos in range(seq_len):
            phys_block = block_table[s][pos // BLOCK].item()
            offset = pos % BLOCK
            src = cu_k[s] + pos
            k_cache[phys_block, offset] = k[src]
            v_cache[phys_block, offset] = v[src]

    return flash_attn_varlen_func(
        q=q, k=k_cache, v=v_cache,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=max_q, max_seqlen_k=max_k,
        block_table=block_table, causal=True,
    )


def _report(name, ref, out):
    diff = (out.float() - ref.float()).abs().max().item()
    ok = diff < 2e-2  # bf16 tolerance
    print(f"{'PASS' if ok else 'FAIL'}  {name:28} max|Δ| = {diff:.4e}")
    return ok


def case_prefill_single_block():
    """Prefill: q_len == k_len == S, one block, non-zero block id (tests block_table offset)."""
    S = 10
    q = torch.randn(S, HQ, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(S, HKV, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(S, HKV, D, device=DEVICE, dtype=DTYPE)

    # Golden: standard causal self-attention over the whole sequence
    ref = flash_attn_func(q[None], k[None], v[None], causal=True)[0]

    cu = torch.tensor([0, S], dtype=torch.int32, device=DEVICE)
    block_table = torch.tensor([[2]], dtype=torch.int32, device=DEVICE)  # block 2, not 0
    out = _paged_attn(q, k, v, cu, cu, S, S, block_table)
    return _report("prefill single block", ref, out)


def case_decode_single_block():
    """Decode: q_len == 1, k_len == S (all history). Bottom-right causal -> query sees all S."""
    S = 12
    k = torch.randn(S, HKV, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(S, HKV, D, device=DEVICE, dtype=DTYPE)
    q1 = torch.randn(1, HQ, D, device=DEVICE, dtype=DTYPE)  # the single new-token query

    # Golden: 1 query against S keys, causal -> the last position attends to everything
    ref = flash_attn_func(q1[None], k[None], v[None], causal=True)[0]

    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    cu_k = torch.tensor([0, S], dtype=torch.int32, device=DEVICE)
    block_table = torch.tensor([[3]], dtype=torch.int32, device=DEVICE)
    out = _paged_attn(q1, k, v, cu_q, cu_k, 1, S, block_table)
    return _report("decode single block", ref, out)


def case_prefill_two_blocks():
    """Prefill spanning two NON-CONTIGUOUS physical blocks -- tests block_table indexing
    across a page boundary (the whole point of paging)."""
    S = BLOCK + 4  # 260 -> needs 2 blocks
    q = torch.randn(S, HQ, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(S, HKV, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(S, HKV, D, device=DEVICE, dtype=DTYPE)

    ref = flash_attn_func(q[None], k[None], v[None], causal=True)[0]

    cu = torch.tensor([0, S], dtype=torch.int32, device=DEVICE)
    block_table = torch.tensor([[5, 1]], dtype=torch.int32, device=DEVICE)  # scattered blocks
    out = _paged_attn(q, k, v, cu, cu, S, S, block_table)
    return _report("prefill two blocks", ref, out)


def case_mixed_batch():
    """One prefill + two decodes packed together -- the actual M4 step shape.

    Each request's k_cache holds its FULL history (k_len); the packed q holds only this
    step's new queries (q_len). Golden is computed per-request and compared segment by
    segment, which is exactly what varlen should reproduce.
    """
    specs = [(8, 8, 2), (1, 6, 4), (1, 9, 7)]  # (q_len, k_len, physical_block)

    q_packed, refs = [], []
    cu_q, cu_k, block_ids = [0], [0], []
    # Full-history K/V per request, written straight into the cache below.
    k_cache = torch.zeros(NUM_BLOCKS, BLOCK, HKV, D, device=DEVICE, dtype=DTYPE)
    v_cache = torch.zeros_like(k_cache)

    for q_len, k_len, block in specs:
        k = torch.randn(k_len, HKV, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(k_len, HKV, D, device=DEVICE, dtype=DTYPE)
        q = torch.randn(q_len, HQ, D, device=DEVICE, dtype=DTYPE)  # this step's queries
        # Golden: q_len queries are the LAST q_len positions of a length-k_len causal seq
        refs.append(flash_attn_func(q[None], k[None], v[None], causal=True)[0])

        k_cache[block, :k_len] = k     # write full history into this request's block
        v_cache[block, :k_len] = v
        q_packed.append(q)
        cu_q.append(cu_q[-1] + q_len)
        cu_k.append(cu_k[-1] + k_len)
        block_ids.append(block)

    out = flash_attn_varlen_func(
        q=torch.cat(q_packed), k=k_cache, v=v_cache,
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=DEVICE),
        cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=DEVICE),
        max_seqlen_q=max(s[0] for s in specs),
        max_seqlen_k=max(s[1] for s in specs),
        block_table=torch.tensor([[b] for b in block_ids], dtype=torch.int32, device=DEVICE),
        causal=True,
    )

    ok = True
    for i, (q_len, _, _) in enumerate(specs):
        seg = out[cu_q[i]:cu_q[i] + q_len]
        ok &= _report(f"  mixed req {i}", refs[i], seg)
    return ok


if __name__ == "__main__":
    print(f"flash-attn paged varlen isolation | HQ={HQ} HKV={HKV} D={D} BLOCK={BLOCK}\n")
    results = [
        case_prefill_single_block(),
        case_decode_single_block(),
        case_prefill_two_blocks(),
        case_mixed_batch(),
    ]
    print("\n" + ("ALL PASS -> kernel call is correct; bug is elsewhere (runner/KV write ordering)"
                  if all(results) else
                  "SOME FAIL -> the paged varlen calling convention is wrong; that is the bug"))
