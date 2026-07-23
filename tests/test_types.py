# tests/test_types.py
"""Unit tests for the core data types: SamplingParams, RequestOutput, Request, FinishReason."""
# Import from the submodule directly to avoid nano_vllm/__init__.py pulling in engine.py,
# which needs a GPU and a model.
from nano_vllm.core.types import (
    FinishReason,
    Request,
    RequestOutput,
    RequestStatus,
    SamplingParams,
)


# --------------------------------------------------------------------------------------
# SamplingParams
# --------------------------------------------------------------------------------------
def test_sampling_params_defaults():
    p = SamplingParams()
    assert p.temperature == 0.0  # greedy by default, so benchmarks are reproducible
    assert p.top_k == -1
    assert p.top_p == 1.0
    assert p.max_tokens == 100


def test_sampling_params_override():
    p = SamplingParams(temperature=0.7, top_k=50, top_p=0.9, max_tokens=10)
    assert p.temperature == 0.7
    assert p.top_k == 50
    assert p.top_p == 0.9
    assert p.max_tokens == 10


# --------------------------------------------------------------------------------------
# RequestOutput
# --------------------------------------------------------------------------------------
def test_request_output_fields():
    out = RequestOutput(
        request_id="1",
        token_ids=[1, 2, 3],
        finished=True,
        text="hello",
    )
    assert out.request_id == "1"
    assert out.text == "hello"
    assert out.token_ids == [1, 2, 3]
    assert out.finished is True
    assert out.finish_reason is None  # unset until the request actually stops


def test_request_output_defaults():
    """finish_reason and text are optional: an unfinished request emits neither."""
    out = RequestOutput(request_id="1", token_ids=[7], finished=False)
    assert out.finish_reason is None
    assert out.text == ""


def test_request_output_finish_reason():
    out = RequestOutput(
        request_id="1",
        token_ids=[1],
        finished=True,
        finish_reason=FinishReason.STOP,
    )
    assert out.finish_reason == FinishReason.STOP


# --------------------------------------------------------------------------------------
# FinishReason
# --------------------------------------------------------------------------------------
def test_finish_reason_is_str_enum():
    """FinishReason subclasses str so it serialises straight into JSON reports and still
    compares equal to the plain strings the OpenAI-style API uses."""
    assert FinishReason.STOP == "stop"
    assert FinishReason.LENGTH == "length"
    assert isinstance(FinishReason.STOP, str)


# --------------------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------------------
def _params(**kw):
    return SamplingParams(**kw)


def test_request_initial_state():
    req = Request("r1", [1, 2, 3], _params(max_tokens=20))
    assert req.request_id == "r1"
    assert req.status == RequestStatus.WAITING
    assert req.num_prompt_tokens == 3
    assert req.num_computed_tokens == 0
    assert req.num_preemptions == 0
    assert req._output_token_ids == []
    assert req.max_tokens == 20  # lifted from sampling_params for the stop check
    assert req.finish_reason is None


def test_request_num_tokens_tracks_output():
    """num_tokens is prompt + generated so far -- the scheduler derives num_new_tokens from
    it, so it must grow as tokens are appended."""
    req = Request("r1", [1, 2, 3], _params())
    assert req.num_tokens == 3
    req._output_token_ids.extend([4, 5])
    req._all_token_ids.extend([4, 5])
    assert req.num_tokens == 5


def test_request_copies_prompt_token_ids():
    """_all_token_ids must be a copy: appending generated tokens to it must not mutate the
    caller's prompt list (lists are passed by reference in Python)."""
    prompt = [1, 2, 3]
    req = Request("r1", prompt, _params())

    req._all_token_ids.append(99)

    assert prompt == [1, 2, 3]
    assert req.prompt_token_ids == [1, 2, 3]
    assert req.num_tokens == 4


def test_request_is_hashable():
    """The scheduler puts requests in sets (preempted_reqs, stopped_reqs) and removes them
    from lists by identity, so Request must stay hashable -- this breaks if it is ever
    turned into a plain @dataclass."""
    a = Request("r1", [1], _params())
    b = Request("r1", [1], _params())
    assert len({a, b}) == 2  # identity, not value equality
    assert a in {a}


def test_request_eos_token_id_defaults_to_none():
    """eos_token_id is filled in by the engine at add_request time from the tokenizer."""
    assert Request("r1", [1], _params()).eos_token_id is None
    assert Request("r1", [1], _params(), eos_token_id=151645).eos_token_id == 151645
