# tests/test_scheduler.py
from collections import deque

import pytest
from nano_vllm.core.types import Request, RequestStatus, SamplingParams
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





