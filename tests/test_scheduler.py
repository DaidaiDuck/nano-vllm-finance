# tests/test_scheduler.py
from collections import deque

import pytest
from nano_vllm.core.types import Request, RequestStatus, SamplingParams, SchedulerOutput, ModelRunnerOutput, FinishReason
from nano_vllm.paged.block_pool import BlockPool
from nano_vllm.paged.kv_cache_manager import KVCacheManager
from nano_vllm.paged.scheduler import Scheduler

BLOCK = 16 

class Cfg: 
    max_model_len = 4096 

def _kv(num_blocks):
    return KVCacheManager(block_pool=BlockPool(num_blocks), block_size= BLOCK) 

def _sched(kv, budget=4096):
    return Scheduler(block_size=BLOCK, kv_cache_manager=kv, model_config=Cfg(), max_num_scheduled_tokens=budget)

def _req(rid, prompt_len, max_tokens=64):
    return Request(rid, [1] * prompt_len, SamplingParams(max_tokens=max_tokens))

def _make_running(kv: KVCacheManager, rid, prompt_len): 
    """
    Make a running request. 
    Make sure the request has been prefilled in this function. 
    """
    r:Request = _req(rid, prompt_len)
    kv.allocate_slots(r, prompt_len) 
    r.num_computed_tokens += prompt_len
    r.status = RequestStatus.RUNNING 
    r._all_token_ids.append(0) # After prefill stage is finished, one output token is produced. 
    return r 

# ---------- Test RUNNING ----------

def test_decode_running_within_budget():
    kv = _kv(100)               # KVCacheManager
    s = _sched(kv)              # Scheduler 
    s.running = [_make_running(kv, "A", 5), _make_running(kv, "B", 5)] 
    out = s.schedule()
    assert {r.request_id for r in out.scheduled_running_reqs} == {"A" , "B"} 
    assert out.num_scheduled_tokens == {"A": 1, "B": 1} # decode one token per step 
    assert len(s.waiting) == 0 # All requests are running. 

def test_token_budget_caps(): 
    kv = _kv(100)               # KVCacheManager with 100 blocks
    s = _sched(kv, budget=1)    # Only one token budget 
    s.running = [_make_running(kv, "A", 5), _make_running(kv, "B", 5)] 
    out = s.schedule()
    assert {r.request_id for r in out.scheduled_running_reqs} == {"A"} 
    assert out.num_scheduled_tokens == {"A": 1}


def test_num_new_tokens_zero_skipped():
    kv = _kv(100)    
    s = _sched(kv)  
    r = _req("X", prompt_len=5) 
    r.num_computed_tokens = 5 
    kv.allocate_slots(r, 5)
    r.status = RequestStatus.RUNNING      
    s.running = [r]
    out = s.schedule() # num_new_tokens = 0. Hence nothing will do. 
    assert out.scheduled_running_reqs == [] 

# ---------- Test WAITING ----------

def test_waiting_admitted():
    kv = _kv(100)
    s = _sched(kv)  
    w = _req("W", prompt_len=10) 
    s.waiting = deque([w])
    out = s.schedule()
    assert [r.request_id for r in out.scheduled_new_reqs] == ["W"]
    assert w.status == RequestStatus.RUNNING 
    assert w in s.running and len(s.waiting) == 0 
    assert out.num_scheduled_tokens[w.request_id] == 10 

def test_resumed_preempted_classified():
    kv = _kv(100)
    s = _sched(kv)
    p = _req("P", prompt_len=8)
    p.status = RequestStatus.PREEMPTED         
    s.waiting = deque([p])
    out = s.schedule()
    assert [r.request_id for r in out.scheduled_resumed_reqs] == ["P"]
    assert out.scheduled_new_reqs == []

# ---------- Test Preemption ----------

def test_preemption_when_out_of_blocks():
    # Set only 2 blocks. A and B take one each at first. 
    # Make B try to take a new one -> preempt B. -> break. 
    kv = _kv(2)               # KVCacheManager
    s = _sched(kv)              # Scheduler 
    A = _make_running(kv, "A", 5)
    B = _make_running(kv, "B", 16)
    s.running = [A, B] 
    out = s.schedule() 
    # A: total: computed 5 + 1 = 6 tokens -> 1 block
    # B: total: computed 16 + 1 = 17 tokens -> 2 blocks. -> trigger preemption. -> cannot preempt B itself. 
    assert out.scheduled_running_reqs == [A]
    assert out.preempted_req_ids == set() 
    assert B.status == RequestStatus.RUNNING
    assert B in s.running

def test_no_waiting_admit_after_preemption():
    kv = _kv(2)
    A = _make_running(kv, "A", 16)
    B = _make_running(kv, "B", 5)
    s = _sched(kv)
    s.running = [A, B]
    # Add a waiting request C
    C = _req("C", prompt_len=4)
    s.waiting = deque([C]) 
    out = s.schedule()
    assert out.preempted_req_ids == {"B"}   # B is preempted. 
    assert C.status == RequestStatus.WAITING
    assert C not in s.running 


def _register_running(s:Scheduler, kv:KVCacheManager, rid, prompt_len, max_tokens=64, eos_token_id=999):
    """Assume prefill is finished. Make a request that is in Decode stage."""
    r = _req(rid, prompt_len, max_tokens) 
    r.eos_token_id = eos_token_id
    # Finish prefill 
    kv.allocate_slots(r, prompt_len) 
    r.num_computed_tokens += prompt_len
    r.status = RequestStatus.RUNNING
    s.running.append(r) 
    s.requests[rid] = r 
    return r 

def _sched_out(num_scheduled: dict): 
    return SchedulerOutput(
        scheduled_new_reqs=[], num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        preempted_req_ids=set(), scheduled_running_reqs=[], scheduled_resumed_reqs=[],
    )

def _model_out(sampled:dict):
    """sampled: req_id -> sampled new token ids in this step.

    NOTE(derek.sun) 
    sampled:{"A": [42], "B": [99]}
    req_ids = ["A", "B"] dict use key in iteration here. 
    sampled_token_ids = [sampled["A"], sampled["B"]] = [[42], [99]] 
    req_to_index = {"A":0, "B":1} 
    """
    req_ids = list(sampled)
    return ModelRunnerOutput(
        sampled_token_ids=[sampled[r] for r in req_ids],
        req_to_index={r:i for i, r in enumerate(req_ids)}
    )

# ---------- Test update_from_output ----------
def test_update_appends_token(): 
    kv = _kv(100); s = _sched(kv) 
    r = _register_running(s, kv, "A", prompt_len=5) 
    s.update_from_output(_sched_out({"A":1}), _model_out({"A": [42]})) 
    assert r._output_token_ids == [42]
    assert r._all_token_ids[-1] == 42 
    assert r.num_computed_tokens == 6
    assert r.status == RequestStatus.RUNNING 
    assert r in s.running

def test_update_eos_stops_frees_removes():
    kv = _kv(100); s = _sched(kv) 
    r = _register_running(s, kv, "A", prompt_len=5, eos_token_id=999)
    num_free_blocks_before = kv.block_pool.free_block_queue.get_num_free_blocks()
    blocks_held_by_r = len(kv.req_to_blocks["A"]) 
    s.update_from_output(_sched_out({"A":1}), _model_out({"A":[999]})) # Insert EOS into A
    assert r.status == RequestStatus.FINISHED  
    assert r.finish_reason == FinishReason.STOP
    assert r not in s.running 
    assert "A" not in kv.req_to_blocks # A's blocks are freed. 
    assert kv.block_pool.free_block_queue.get_num_free_blocks() == num_free_blocks_before + blocks_held_by_r
    
def test_update_max_tokens_stops():
    kv = _kv(100); s = _sched(kv)
    r = _register_running(s, kv, "A", prompt_len=5, max_tokens=1)   # Allow only 1 token output
    s.update_from_output(_sched_out({"A":1}), _model_out({"A":[42]})) # Must be Non EOS
    assert r.status == RequestStatus.FINISHED
    assert r.finish_reason == FinishReason.LENGTH
    assert r not in s.running 


def test_update_prefill_chunk_not_appended(): 
    kv = _kv(100); s = _sched(kv)
    r = _req("A", prompt_len=10); r.eos_token_id = 999
    kv.allocate_slots(r, 4) # Assume chunked prefill: 4 tokens 
    r.status = RequestStatus.RUNNING
    s.running.append(r)
    s.requests["A"] = r
    # Prefill not finished.
    s.update_from_output(_sched_out({"A":4}), _model_out({"A":[42]}))    # [42] is a placeholder to test if r._output_token_ids == []. In actual prefill stage, there is no output token id.
    assert r.num_computed_tokens == 4
    assert r._output_token_ids == []   # Still in prefill stage. No outpu tokens. 
    assert r in s.running 









