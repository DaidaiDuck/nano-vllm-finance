# tests/test_gpu_model_runner.py
"""Unit tests for GPUModelRunner._prepare_inputs -- pure CPU, real Request/KVCacheManager/BlockPool.

This is the acceptance gate for M4's *packing* logic: flattening the tokens that several
requests need computed this step into one 1-D sequence, plus every piece of metadata the
flash-attn varlen kernel needs. What is really under test is the mapping between two
coordinate systems:

  - packed / query space: input_ids, positions, cu_seqlens_q, logits_indices  (len == total_tokens)
  - full-sequence / KV space: seq_lens_k, block_table, slot_mapping           (one entry per request)

No GPU, no model, no flash-attn kernel -- this is arithmetic only.

Run: python -m pytest tests/test_gpu_model_runner.py -q
"""
import importlib.machinery
import importlib.util
import sys
import types as pytypes

import pytest
import torch

# --- Make this file runnable on machines without flash-attn (e.g. a Mac laptop) --------
# nano_vllm.paged.paged_attention does `from flash_attn import ...` at module scope, and
# gpu_model_runner imports it to get ENGINE_CTX. _prepare_inputs never *calls* anything
# from flash_attn, so a stub module is enough. Where flash-attn is really installed (the
# GPU pod) this is skipped and the real package is used.
if importlib.util.find_spec("flash_attn") is None:  # pragma: no cover - environment dependent
    _stub = pytypes.ModuleType("flash_attn")
    _stub.flash_attn_with_kvcache = None
    _stub.flash_attn_varlen_func = None
    # transformers probes flash_attn via importlib.util.find_spec, which raises on a module
    # whose __spec__ is None. Give the stub a real spec; transformers then falls back to
    # importlib.metadata for the version, that lookup fails, and it correctly concludes
    # flash-attn is unavailable instead of trying to import its internals.
    _stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    sys.modules["flash_attn"] = _stub

from nano_vllm.core.types import Request, SamplingParams, SchedulerOutput  # noqa: E402
from nano_vllm.paged.block_pool import BlockPool  # noqa: E402
from nano_vllm.paged.gpu_model_runner import GPUModelRunner  # noqa: E402
from nano_vllm.paged.kv_cache_manager import KVCacheManager  # noqa: E402
from nano_vllm.paged.paged_attention import ENGINE_CTX  # noqa: E402

BLOCK = 4  # small blocks keep slot arithmetic hand-checkable; the real engine uses 256


@pytest.fixture(autouse=True)
def _tensors_on_cpu(monkeypatch):
    """_prepare_inputs hardcodes device="cuda"; downgrade to CPU when there is no GPU.

    Only active on machines without CUDA. On the pod these are real cuda tensors and every
    assertion below is unchanged.
    """
    if torch.cuda.is_available():
        return
    real_tensor = torch.tensor

    def cpu_tensor(data, *args, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(data, *args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
_OFFSET = {"r1": 1000, "r2": 2000, "r3": 3000}


def _tok(rid: str, pos: int) -> int:
    """Encode (request, position) into the token id so expected input_ids stay readable."""
    return _OFFSET[rid] + pos


def _make_request(rid: str, num_tokens: int, num_computed: int) -> Request:
    req = Request(
        rid,
        [_tok(rid, i) for i in range(num_tokens)],
        SamplingParams(temperature=0.0),
    )
    req.num_computed_tokens = num_computed
    return req


def _bind_blocks(kv: KVCacheManager, pool: BlockPool, rid: str, block_ids: list[int]) -> None:
    """Pin a request's block table to an exact sequence of block ids.

    This deliberately bypasses allocate_slots' natural ordering: we want the
    *non-contiguous* ids from the docstring example (req3 = [5, 8, 1]) so the test proves
    slots are still computed correctly when physical blocks are scattered.
    """
    kv.req_to_blocks[rid] = [pool.blocks[b] for b in block_ids]


def _sched_out(num_scheduled: dict[str, int], new=(), running=(), resumed=()) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=list(new),
        scheduled_running_reqs=list(running),
        scheduled_resumed_reqs=list(resumed),
        preempted_req_ids=set(),
        num_scheduled_tokens=dict(num_scheduled),
        total_num_scheduled_tokens=sum(num_scheduled.values()),
    )


@pytest.fixture
def three_requests():
    """The 3-request example from _prepare_inputs' docstring, block_size = 4.

    req1: prefill, num_computed=0, num_scheduled=4, blocks=[2]
    req2: decode,  num_computed=5, num_scheduled=1, blocks=[7, 3]
    req3: decode,  num_computed=9, num_scheduled=1, blocks=[5, 8, 1]
    """
    pool = BlockPool(num_gpu_blocks=16)
    kv = KVCacheManager(pool, block_size=BLOCK)

    r1 = _make_request("r1", num_tokens=4, num_computed=0)
    r2 = _make_request("r2", num_tokens=6, num_computed=5)
    r3 = _make_request("r3", num_tokens=10, num_computed=9)

    _bind_blocks(kv, pool, "r1", [2])
    _bind_blocks(kv, pool, "r2", [7, 3])
    _bind_blocks(kv, pool, "r3", [5, 8, 1])

    runner = GPUModelRunner(model=None, sampler=None, kv_cache_manager=kv, block_size=BLOCK)
    sched = _sched_out({"r1": 4, "r2": 1, "r3": 1}, new=[r1], running=[r2, r3])
    return runner, sched


# --------------------------------------------------------------------------------------
# Main case: the docstring example, field by field
# --------------------------------------------------------------------------------------
def test_prepare_inputs_matches_docstring_example(three_requests):
    runner, sched = three_requests
    req_ids, input_ids, positions, logits_indices = runner._prepare_inputs(sched)

    # req_ids order == insertion order of num_scheduled_tokens. execute_model uses it to
    # build req_to_index and to gather each request's sampling_params, so it must line up
    # with the row order of the logits.
    assert req_ids == ["r1", "r2", "r3"]

    # --- packed / query space ---------------------------------------------------------
    assert input_ids.shape == (1, 6)  # six tokens flattened into one sequence
    assert input_ids[0].tolist() == [
        _tok("r1", 0), _tok("r1", 1), _tok("r1", 2), _tok("r1", 3),  # prefill: all four
        _tok("r2", 5),                                               # decode: newest only
        _tok("r3", 9),
    ]
    # positions are absolute *within a request* (not packed offsets) -- RoPE depends on this
    assert positions[0].tolist() == [0, 1, 2, 3, 5, 9]
    assert ENGINE_CTX.cu_seqlens_q.tolist() == [0, 4, 5, 6]

    # logits_indices = cu_seqlens_q[1:] - 1: the row of each request's last token in the
    # packed hidden states. Not seq_lens_k - 1: hidden only holds this step's tokens, not
    # the history.
    assert logits_indices.tolist() == [3, 4, 5]

    # --- full-sequence / KV space -----------------------------------------------------
    # seq_lens_k = num_computed + num_scheduled = history each request attends over
    assert ENGINE_CTX.seq_lens_k.tolist() == [4, 6, 10]

    # slot = block_id * block_size + offset
    #   r1 pos 0..3 -> block 2, offsets 0..3      -> 2*4 + 0..3 = 8, 9, 10, 11
    #   r2 pos 5    -> block_table[5//4=1] = 3, offset 5%4 = 1 -> 3*4 + 1 = 13
    #   r3 pos 9    -> block_table[9//4=2] = 1, offset 9%4 = 1 -> 1*4 + 1 = 5
    assert ENGINE_CTX.slot_mapping.tolist() == [8, 9, 10, 11, 13, 5]

    # block_table is right-padded to max_blocks = 3. Padding with 0 is safe: seq_lens_k
    # bounds the kernel to the first ceil(seq_lens_k / block_size) blocks, so pad entries
    # are never read.
    assert ENGINE_CTX.block_table.tolist() == [[2, 0, 0], [7, 3, 0], [5, 8, 1]]

    assert ENGINE_CTX.max_seqlen_q == 4   # longest query = req1's prefill
    assert ENGINE_CTX.max_seqlen_k == 10  # longest history = req3


def test_prepare_inputs_dtypes(three_requests):
    """Wrong dtypes make flash-attn assert or silently miscompute; pin them separately."""
    runner, sched = three_requests
    _, input_ids, positions, logits_indices = runner._prepare_inputs(sched)

    # Anything indexing into embeddings / hidden states must be int64
    assert input_ids.dtype == torch.int64
    assert positions.dtype == torch.int64
    assert logits_indices.dtype == torch.int64
    # flash-attn wants int32 for cumulative boundaries and block tables
    assert ENGINE_CTX.cu_seqlens_q.dtype == torch.int32
    assert ENGINE_CTX.seq_lens_k.dtype == torch.int32
    assert ENGINE_CTX.block_table.dtype == torch.int32
    # scatter index tensors must be int64
    assert ENGINE_CTX.slot_mapping.dtype == torch.int64
    # max_seqlen must be a python int, not a tensor
    assert isinstance(ENGINE_CTX.max_seqlen_q, int)
    assert isinstance(ENGINE_CTX.max_seqlen_k, int)


def test_prepare_inputs_internal_consistency(three_requests):
    """Cross-field invariants that must hold for any batch, not just this example."""
    runner, sched = three_requests
    req_ids, input_ids, positions, logits_indices = runner._prepare_inputs(sched)

    total = sched.total_num_scheduled_tokens
    cu = ENGINE_CTX.cu_seqlens_q.tolist()

    assert cu[0] == 0 and cu[-1] == total   # boundaries run from 0 to total
    assert len(cu) == len(req_ids) + 1      # batch + 1
    assert input_ids.shape[1] == total
    assert positions.shape[1] == total
    assert ENGINE_CTX.slot_mapping.numel() == total
    assert len(logits_indices) == len(req_ids)  # exactly one logits row per request

    # Each query segment length == that request's scheduled token count
    seg = [cu[i + 1] - cu[i] for i in range(len(req_ids))]
    assert seg == [sched.num_scheduled_tokens[r] for r in req_ids]
    # Each logits index points at the last row of its own segment
    assert logits_indices.tolist() == [cu[i + 1] - 1 for i in range(len(req_ids))]


# --------------------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------------------
def test_pure_decode_batch():
    """All-decode step (one token per request) -- the steady state of continuous batching.

    Here cu_seqlens_q == [0,1,2,3] and logits_indices == [0,1,2]: every row of hidden is
    some request's last token.
    """
    pool = BlockPool(num_gpu_blocks=16)
    kv = KVCacheManager(pool, block_size=BLOCK)
    r1 = _make_request("r1", 4, 3)   # pos 3 -> block[0]=2, offset 3 -> 11
    r2 = _make_request("r2", 6, 5)   # pos 5 -> block[1]=3, offset 1 -> 13
    r3 = _make_request("r3", 10, 9)  # pos 9 -> block[2]=1, offset 1 -> 5
    _bind_blocks(kv, pool, "r1", [2])
    _bind_blocks(kv, pool, "r2", [7, 3])
    _bind_blocks(kv, pool, "r3", [5, 8, 1])

    runner = GPUModelRunner(None, None, kv, BLOCK)
    sched = _sched_out({"r1": 1, "r2": 1, "r3": 1}, running=[r1, r2, r3])
    _, input_ids, positions, logits_indices = runner._prepare_inputs(sched)

    assert input_ids.shape == (1, 3)
    assert positions[0].tolist() == [3, 5, 9]
    assert ENGINE_CTX.cu_seqlens_q.tolist() == [0, 1, 2, 3]
    assert logits_indices.tolist() == [0, 1, 2]
    assert ENGINE_CTX.seq_lens_k.tolist() == [4, 6, 10]
    assert ENGINE_CTX.slot_mapping.tolist() == [11, 13, 5]
    assert ENGINE_CTX.max_seqlen_q == 1


def test_single_request_prefill():
    """batch == 1 pure prefill: the degenerate case must match M3's single-request semantics."""
    pool = BlockPool(num_gpu_blocks=16)
    kv = KVCacheManager(pool, block_size=BLOCK)
    r = _make_request("r1", 6, 0)
    _bind_blocks(kv, pool, "r1", [2, 7])  # six tokens span two blocks

    runner = GPUModelRunner(None, None, kv, BLOCK)
    sched = _sched_out({"r1": 6}, new=[r])
    req_ids, _, positions, logits_indices = runner._prepare_inputs(sched)

    assert req_ids == ["r1"]
    assert positions[0].tolist() == [0, 1, 2, 3, 4, 5]
    assert ENGINE_CTX.cu_seqlens_q.tolist() == [0, 6]
    assert ENGINE_CTX.seq_lens_k.tolist() == [6]
    assert logits_indices.tolist() == [5]  # only the final token produces logits
    # pos 0..3 -> block 2 (8..11); pos 4,5 -> block 7 (28, 29)
    assert ENGINE_CTX.slot_mapping.tolist() == [8, 9, 10, 11, 28, 29]
    assert ENGINE_CTX.block_table.tolist() == [[2, 7]]  # single row, no padding needed


def test_chunked_prefill_second_chunk():
    """Second chunk of a chunked prefill: num_computed > 0 but still inside the prompt.

    The point: positions continue from num_computed (not from 0) and slots continue from
    there too -- otherwise this chunk would overwrite the KV the first chunk wrote.
    """
    pool = BlockPool(num_gpu_blocks=16)
    kv = KVCacheManager(pool, block_size=BLOCK)
    r = _make_request("r1", 10, 4)  # first four tokens were computed in an earlier step
    _bind_blocks(kv, pool, "r1", [2, 7, 5])

    runner = GPUModelRunner(None, None, kv, BLOCK)
    sched = _sched_out({"r1": 4}, running=[r])  # this step computes positions 4..7
    _, input_ids, positions, logits_indices = runner._prepare_inputs(sched)

    assert positions[0].tolist() == [4, 5, 6, 7]
    assert input_ids[0].tolist() == [_tok("r1", i) for i in (4, 5, 6, 7)]
    assert ENGINE_CTX.seq_lens_k.tolist() == [8]  # eight tokens of history once this lands
    assert ENGINE_CTX.slot_mapping.tolist() == [28, 29, 30, 31]  # all inside block 7
    assert logits_indices.tolist() == [3]


def test_block_table_padding_widths():
    """block_table is padded to the widest request in the batch; shorter rows get 0s."""
    pool = BlockPool(num_gpu_blocks=16)
    kv = KVCacheManager(pool, block_size=BLOCK)
    r1 = _make_request("r1", 2, 1)
    r2 = _make_request("r2", 14, 13)
    _bind_blocks(kv, pool, "r1", [6])           # one block
    _bind_blocks(kv, pool, "r2", [1, 2, 3, 4])  # four blocks

    runner = GPUModelRunner(None, None, kv, BLOCK)
    runner._prepare_inputs(_sched_out({"r1": 1, "r2": 1}, running=[r1, r2]))

    assert ENGINE_CTX.block_table.tolist() == [[6, 0, 0, 0], [1, 2, 3, 4]]
    assert ENGINE_CTX.block_table.shape == (2, 4)
