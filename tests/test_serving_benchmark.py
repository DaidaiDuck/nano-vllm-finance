# tests/test_serving_benchmark.py
"""CPU tests for the open-loop serving benchmark: arrival generation and serving metrics.

No GPU, no model, no engine -- these cover the parts of the M4 benchmark that are pure
arithmetic, so the metric definitions can be trusted before any expensive run happens on
the pod.

Run: python -m pytest tests/test_serving_benchmark.py -q
"""
import math

import pytest

from benchmarks.arrival import generate_arrival_times
from benchmarks.metrics import ServingRecord, compute_serving_metrics
from benchmarks.serving_benchmark import run_serving_benchmark
from nano_vllm.core.types import RequestOutput


# --------------------------------------------------------------------------------------
# Arrival process
# --------------------------------------------------------------------------------------
def test_infinite_qps_arrives_at_once():
    """qps=inf means every request is submitted immediately: the saturated / offline mode
    (equivalent to vLLM's --request-rate inf)."""
    assert generate_arrival_times(10, float("inf")) == [0.0] * 10


def test_uniform_distribution_has_fixed_gap():
    assert generate_arrival_times(5, qps=2.0, distribution="uniform") == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_poisson_mean_gap_matches_rate():
    """Exponential inter-arrival gaps must average 1/lambda. Large sample, 10% tolerance."""
    qps = 5.0
    times = generate_arrival_times(20000, qps=qps, seed=0)
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert math.isclose(sum(gaps) / len(gaps), 1 / qps, rel_tol=0.1)


def test_arrivals_are_monotonic_and_reproducible():
    times = generate_arrival_times(100, qps=3.0, seed=7)
    assert times == generate_arrival_times(100, qps=3.0, seed=7)  # same seed, same schedule
    assert all(a <= b for a, b in zip(times, times[1:]))          # non-decreasing
    assert times[0] == 0.0


def test_empty_and_invalid_inputs():
    assert generate_arrival_times(0, qps=1.0) == []
    with pytest.raises(ValueError):
        generate_arrival_times(5, qps=0)
    with pytest.raises(ValueError):
        generate_arrival_times(5, qps=-1)
    with pytest.raises(ValueError):
        generate_arrival_times(5, qps=1.0, distribution="gaussian")


# --------------------------------------------------------------------------------------
# Serving metrics
# --------------------------------------------------------------------------------------
def _record(rid, arrival, first, finish, output_len):
    return ServingRecord(rid, arrival, first, finish, output_len)


def test_ttft_includes_queueing_delay():
    """TTFT is measured from arrival, not from when the engine starts work, so time spent
    waiting in the queue counts. This is the whole point of the open-loop harness."""
    r = _record("a", arrival=1.0, first=3.0, finish=5.0, output_len=5)
    assert r.ttft == 2.0  # queued for two seconds before the first token
    assert r.e2e == 4.0
    assert r.tpot == pytest.approx((5.0 - 3.0) / 4)  # divided by output_len - 1


def test_tpot_is_zero_for_single_token_output():
    """With one output token there is no inter-token gap to measure; report 0, not a
    division by zero."""
    assert _record("a", 0.0, 1.0, 1.0, output_len=1).tpot == 0.0


def test_goodput_counts_only_fully_slo_compliant_requests():
    """Goodput requires *every* SLO to hold, not just one of them."""
    records = [
        _record("slow-tokens", 0.0, 0.10, 1.10, 11),  # ttft 0.10 ok, tpot 0.10 too slow
        _record("good",        0.0, 0.10, 0.30, 11),  # ttft 0.10 ok, tpot 0.02 ok
        _record("queued",      0.0, 0.50, 0.70, 11),  # ttft 0.50 too slow
    ]
    m = compute_serving_metrics(
        "t", records, total_duration=2.0, qps=4.0, slo_ttft=0.2, slo_tpot=0.05
    )
    assert m.num_slo_met == 1  # only "good" satisfies both
    assert m.goodput == pytest.approx(0.5)
    assert m.request_throughput == pytest.approx(1.5)  # throughput counts all completions
    assert m.goodput <= m.request_throughput           # goodput can never exceed throughput


def test_throughput_uses_wall_clock_not_summed_latency():
    """Under concurrency, per-request latencies overlap; summing them would inflate
    throughput. The denominator must be wall-clock duration."""
    records = [_record(str(i), 0.0, 0.1, 1.0, 10) for i in range(4)]
    m = compute_serving_metrics("t", records, total_duration=1.0, qps=float("inf"))
    assert m.output_throughput == pytest.approx(40.0)  # 4 requests x 10 tokens in 1s
    assert m.request_throughput == pytest.approx(4.0)


def test_percentiles_track_the_tail():
    """P99 must follow the slow tail, not the median -- the reason tail percentiles are
    reported at all is that averages hide queueing.

    Two stragglers out of 100, not one: numpy interpolates between order statistics, so a
    single outlier at the very top only drags P99 a fraction of the way toward it.
    """
    records = [_record(str(i), 0.0, 0.01, 0.5, 11) for i in range(98)]
    records += [_record(f"straggler{i}", 0.0, 5.0, 6.0, 11) for i in range(2)]

    m = compute_serving_metrics("t", records, total_duration=6.0, qps=10.0)
    assert m.p50_ttft == pytest.approx(0.01)   # the median is untouched by the tail
    assert m.p99_ttft == pytest.approx(5.0)    # P99 lands on the queued requests
    assert m.avg_ttft < 0.2                    # while the average hides them entirely


def test_empty_records_rejected():
    with pytest.raises(ValueError):
        compute_serving_metrics("t", [], total_duration=1.0, qps=1.0)


# --------------------------------------------------------------------------------------
# Spec persistence: nano and vLLM must replay byte-identical prompts, so the round-trip
# through disk must preserve both content and order exactly.
# --------------------------------------------------------------------------------------
def test_specs_roundtrip_preserves_content_and_order(tmp_path):
    from benchmarks.datasets import dump_specs, load_specs

    specs = [
        ("Explain photosynthesis.", 42),
        ("Hello", 7),
        ("Write a haiku about the sea.", 100),
    ]
    path = str(tmp_path / "specs.json")
    dump_specs(specs, path)

    assert load_specs(path) == specs  # same prompts, same output_len, same order


def test_specs_file_is_plain_json(tmp_path):
    """The file is inspectable and stable, not a pickle -- so a human can diff the exact
    prompt set that produced a benchmark run."""
    import json

    from benchmarks.datasets import dump_specs

    path = str(tmp_path / "specs.json")
    dump_specs([("hi", 5)], path)
    with open(path) as f:
        assert json.load(f) == [{"prompt": "hi", "output_len": 5}]


# --------------------------------------------------------------------------------------
# Driver loop, on a fake in-process engine (no GPU, no model).
#
# The metric maths above is covered in isolation; this exercises run_serving_benchmark's
# control flow -- arrival submission, first-token / finish timestamping, termination --
# which otherwise only ever runs on the pod.
# --------------------------------------------------------------------------------------
class _FakeTokenizer:
    def encode(self, text):
        return list(range(max(1, len(text.split()))))  # >=1 token, deterministic


class _FakeEngine:
    """Continuous-batching stand-in for PagedEngine.

    Every in-flight request emits exactly one token per step and finishes once it has
    emitted its own max_tokens. `empty_steps` injects leading steps that schedule nothing
    (the block-starved case) so the driver is proven to tolerate step() -> [] without
    dropping requests or spinning forever.
    """
    def __init__(self, empty_steps: int = 0):
        self.tokenizer = _FakeTokenizer()
        self.running = []
        self._empty_steps = empty_steps

    def _format_prompt(self, prompt):
        return prompt

    def add_request(self, request):
        request._emitted = 0
        self.running.append(request)

    def has_unfinished_requests(self):
        return bool(self.running)

    def step(self):
        if self._empty_steps > 0:
            self._empty_steps -= 1
            return []  # scheduled nothing this step, but work remains
        outputs, still_running = [], []
        for req in self.running:
            req._emitted += 1
            finished = req._emitted >= req.sampling_params.max_tokens
            outputs.append(RequestOutput(
                request_id=req.request_id,
                token_ids=list(range(req._emitted)),
                finished=finished,
            ))
            if not finished:
                still_running.append(req)
        self.running = still_running
        return outputs


def test_driver_saturated_processes_every_request():
    """qps=inf submits the whole batch at t=0; the driver must run all of them to
    completion and record each one's own output_len."""
    specs = [("one word", 1), ("two words here", 2), ("three words here now", 3)]
    m = run_serving_benchmark(_FakeEngine(), specs, qps=float("inf"))

    assert m.num_requests == 3           # every request finished and was recorded
    assert m.output_throughput > 0
    assert m.goodput <= m.request_throughput


def test_driver_records_each_requests_own_output_len():
    """Per-request output_len must be honored, not a shared value."""
    specs = [("a", 1), ("b", 4), ("c", 7)]
    m = run_serving_benchmark(_FakeEngine(), specs, qps=float("inf"))
    # 1 + 4 + 7 = 12 tokens over the wall clock
    assert m.output_throughput == pytest.approx(12 / m.total_duration, rel=1e-6)


def test_driver_survives_empty_scheduling_steps():
    """step() returning [] (block-starved) must not crash the driver, lose the request, or
    loop forever -- it just advances the clock until real work comes out."""
    specs = [("hello world", 2)]
    m = run_serving_benchmark(_FakeEngine(empty_steps=3), specs, qps=float("inf"))

    assert m.num_requests == 1
    assert m.p50_ttft >= 0.0  # first token arrived after the empty steps, still recorded


def test_driver_ttft_is_nonnegative_and_finite():
    specs = [("prompt text here", 3) for _ in range(4)]
    m = run_serving_benchmark(_FakeEngine(), specs, qps=float("inf"))
    for value in (m.p50_ttft, m.p99_ttft, m.avg_tpot, m.p99_e2e):
        assert value >= 0.0 and math.isfinite(value)
