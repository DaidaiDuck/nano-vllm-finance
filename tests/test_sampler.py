# tests/test_sampler.py
"""Pure-CPU unit tests for Sampler. No model, no GPU.

The Sampler has two entry points since M4:
  - sample_one(logits[vocab], params) -> int      : the single-row policy (greedy/top-k/top-p)
  - sample(logits[batch, vocab], params_list)     : one row per request, each with its *own*
                                                    SamplingParams, returning list[int]

Note that sample_one masks logits in place (top-k / top-p write -inf), and sample passes row
views straight through, so the caller's tensor is modified. Every case below builds a fresh
tensor for that reason.
"""
import pytest
import torch

# Import from the submodule directly to avoid nano_vllm/__init__.py pulling in engine.py,
# which needs a GPU and a model.
from nano_vllm.core.sampler import Sampler
from nano_vllm.core.types import SamplingParams


def _logits(values):
    """A single row of logits, shape [vocab]."""
    return torch.tensor(values, dtype=torch.float32)


def _batch(rows):
    """A batch of logits, shape [batch, vocab]."""
    return torch.tensor(rows, dtype=torch.float32)


# --------------------------------------------------------------------------------------
# sample_one: the per-row sampling policy
# --------------------------------------------------------------------------------------
def test_greedy_returns_argmax():
    sampler = Sampler()
    params = SamplingParams(temperature=0.0)
    for _ in range(5):  # deterministic: same answer every time
        assert sampler.sample_one(_logits([0.1, 0.2, 5.0, 0.3, -1.0]), params) == 2


def test_returns_python_int():
    sampler = Sampler()
    out = sampler.sample_one(_logits([1.0, 2.0, 3.0]), SamplingParams(temperature=0.0))
    assert type(out) is int  # from .item(), not a 0-d tensor


def test_temperature_near_onehot():
    """One logit is enormous, so even at temperature=1.0 that token is always drawn."""
    sampler = Sampler()
    params = SamplingParams(temperature=1.0)
    torch.manual_seed(0)
    for _ in range(200):
        assert sampler.sample_one(_logits([0.0, 0.0, 50.0, 0.0]), params) == 2


def test_top_k_one_equals_greedy():
    """top_k=1 leaves only the argmax in the candidate set, so sampling collapses to greedy."""
    sampler = Sampler()
    params = SamplingParams(temperature=1.0, top_k=1)
    torch.manual_seed(0)
    for _ in range(200):
        assert sampler.sample_one(_logits([1.0, 4.0, 2.0, 3.0]), params) == 1


def test_top_k_restricts_candidates():
    """top_k=2 means every draw must land in the top-2 indices (here 3 and 1)."""
    sampler = Sampler()
    params = SamplingParams(temperature=1.0, top_k=2)
    torch.manual_seed(0)
    for _ in range(200):
        assert sampler.sample_one(_logits([0.5, 3.0, 1.0, 4.0, 0.1]), params) in {1, 3}


def test_top_k_larger_than_vocab():
    """top_k far beyond the vocab size must clamp instead of crashing."""
    sampler = Sampler()
    params = SamplingParams(temperature=1.0, top_k=1000)
    assert 0 <= sampler.sample_one(_logits([1.0, 2.0, 3.0, 4.0, 5.0]), params) <= 4


def test_top_p_restricts_to_nucleus():
    """logits=[10,0,0,0] gives the top token ~0.9999 probability, so a top_p=0.5 nucleus
    contains only that token."""
    sampler = Sampler()
    params = SamplingParams(temperature=1.0, top_p=0.5)
    torch.manual_seed(0)
    for _ in range(200):
        assert sampler.sample_one(_logits([10.0, 0.0, 0.0, 0.0]), params) == 0


def test_top_p_one_no_truncation():
    """top_p=1.0 disables truncation; any valid index may come back."""
    sampler = Sampler()
    params = SamplingParams(temperature=1.0, top_p=1.0)
    torch.manual_seed(0)
    assert 0 <= sampler.sample_one(_logits([1.0, 2.0, 3.0]), params) <= 2


# --------------------------------------------------------------------------------------
# sample: batched, one row and one SamplingParams per request (M4)
# --------------------------------------------------------------------------------------
def test_sample_returns_one_token_per_row():
    """The batched entry point returns a list[int] aligned with the rows of logits."""
    sampler = Sampler()
    logits = _batch([
        [0.0, 0.0, 9.0, 0.0],  # argmax 2
        [5.0, 0.0, 0.0, 0.0],  # argmax 0
        [0.0, 7.0, 0.0, 0.0],  # argmax 1
    ])
    greedy = SamplingParams(temperature=0.0)
    out = sampler.sample(logits, [greedy, greedy, greedy])

    assert out == [2, 0, 1]
    assert all(type(t) is int for t in out)


def test_sample_uses_each_requests_own_params():
    """Requests in one batch may carry different SamplingParams, and each row must be
    sampled with its own.

    Row 0 is greedy. Row 1 uses top_k=1, which collapses to that row's own argmax -- so both
    are deterministic and a mix-up (applying row 0's params to row 1) would be visible.
    """
    sampler = Sampler()
    logits = _batch([
        [0.0, 0.0, 9.0, 0.0],  # greedy -> 2
        [1.0, 4.0, 2.0, 3.0],  # top_k=1 -> 1
    ])
    params = [
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.0, top_k=1),
    ]
    torch.manual_seed(0)
    assert sampler.sample(logits, params) == [2, 1]


def test_sample_batch_of_one():
    """batch=1 is the degenerate case and must still return a list, not a bare int."""
    sampler = Sampler()
    out = sampler.sample(_batch([[0.0, 3.0, 0.0]]), [SamplingParams(temperature=0.0)])
    assert out == [1]


def test_sample_row_order_is_preserved():
    """Row i of the logits must map to element i of the result: the runner relies on this to
    line sampled tokens up with req_ids via req_to_index."""
    sampler = Sampler()
    vocab = 8
    # Row i peaks at index i, so the expected output is simply [0, 1, ..., n-1].
    rows = [[9.0 if j == i else 0.0 for j in range(vocab)] for i in range(6)]
    greedy = SamplingParams(temperature=0.0)
    assert sampler.sample(_batch(rows), [greedy] * 6) == list(range(6))


def test_sample_mismatched_params_length_raises():
    """Fewer params than rows is a programming error in the runner; it must not silently
    sample the wrong rows."""
    sampler = Sampler()
    logits = _batch([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    with pytest.raises(IndexError):
        sampler.sample(logits, [SamplingParams(temperature=0.0)])
