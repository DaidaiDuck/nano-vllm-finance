# M4 Scheduler — TODO 解答 & 设计决策

> 对 `nano_vllm/paged/scheduler.py`(及相关 `nano_vllm/core/types.py` 的 `Request` /
> `SchedulerOutput`)里留的 TODO 的意见,外加顺手发现的、会挡住编译/运行的 bug。
> 参照系:vLLM v1 的 `vllm/v1/core/sched/scheduler.py`。

## 0. 先修:会直接崩的 bug(和 TODO 无关,但不修跑不起来)

1. **`scheduler.py` L39 `&&` → `and`**。Python 没有 `&&`,SyntaxError。
2. **`SchedulerOutput(...)` 构造不全 + 语法错**(L124-130):
   - L128 `finished_req_ids = copy(...)` 后**缺逗号** → SyntaxError;且 `copy` 未 import(见 TODO⑤)。
   - `SchedulerOutput` 在 types.py 里有 7 个必填字段,但这里只传了 5 个,**漏了 `num_scheduled_tokens`
     和 `total_num_scheduled_tokens`**(两者都无默认值)→ TypeError。7 个都要传。
3. **`Request` 是"无字段 dataclass",导致不可 hash + 所有实例相等**(types.py):字段全写在 **docstring**
   里(不是真注解),又手写了 `__init__`。于是 `@dataclass` 认为它**没有字段**,而 `eq=True/frozen=False`
   会把 `__hash__` 设成 `None`(**不可 hash**),生成的 `__eq__` 比较"空" → **任意两个 Request 相等**。后果:
   - `preempted_reqs: set[Request]` 的 `.add(request)` → `TypeError: unhashable type`。
   - `self.running.remove(x)` 用 `==` 找,因为全相等 → **永远删 `running[0]`**(潜在错删)。
   - **解法:去掉 `@dataclass`**,保留手写 `__init__` → 变普通类(按对象身份 hash/eq)→ `set[Request]`、
     `list.remove()`、`is` 全部正确,也和 vLLM 的 `Request`(普通类)一致。
4. **`types.py` L48 `self.status = Request.WAITING` → `RequestStatus.WAITING`**(`Request` 没有 `WAITING`)。
5. **`request.num_tokens` 从没在 `__init__` 里赋值**,但 `scheduler.py` L42 用它 → AttributeError。见 TODO⑥。
6. **`scheduler.py` L43 `self.max_model_len - request.max_tokens` 语义错**:`max_tokens` 是"最大**输出**长度"
   (sampling 参数)。这里要的是"这个请求还能再算多少 token"= **`self.max_model_len - request.num_computed_tokens`**。
7. **`update_from_output` 还是空的**:这是唯一推进 `num_computed_tokens`、判断 EOS/长度、把 finished 请求
   移出 running 并 `free` 掉、以及 populate `finished_req_ids` 的地方。不实现 → running 永不退出、
   `finished_req_ids` 永远空。见 TODO⑤。

---

## TODO 逐条意见

### ① running 用 list,preempt 时 `remove()` O(n) 太贵?(L15)
**保持 list。** 这正是 vLLM v1 的做法(`self.running: list[Request]`)。并发 running 数(几十~上百)很小,
`remove()` 的 O(n) 可忽略,而且只在 **preempt 这个例外路径**发生,不在每步的热路径。真到"几千并发 + 频繁
preempt"再换结构(有序 dict / 侵入式双向链表做 O(1) 删除)——对 M4 是过早优化。
- 小注:`self.running.remove(self.running[0])` 删头要搬后面所有元素(也 O(n));n 小无所谓。修了 bug#3 后
  这个 remove 才是按身份删、语义正确的。

### ② deque 怎么 peek?(L94)
`self.waiting[0]` **本身就是 O(1) peek**(deque 支持两端索引),你的写法已经对。但 **L108
`self.waiting.remove(request)` 是 O(n)**,而你只会删你 peek 的那个(队头)→ 改成 **`self.waiting.popleft()`**(O(1))。
配合:分配失败 `break`(不删,请求留在队头等下一步);成功才 `popleft()`。

### ③ `assert num_new_tokens > 0` 会不会不成立?(L97)
进 `while` 时循环条件保证 `token_budget > 0`(≥1),`long_prefill_token_threshold=4096>0`,所以只要
`num_prompt_tokens ≥ 1`,`min(...)` 就 ≥1。**唯一能触发的是"空 prompt"或 token_budget 记账 bug**——这两种
都该当场炸出来。所以:**保留这个 assert 当"不变量文档"**(vLLM 同处也有 `assert num_new_tokens > 0`),
并在 `add_request` 阶段拒掉空 prompt。

### ④ waiting 里 prompt 太大,阻塞后面的小请求(队头阻塞 HOL)?(L103)
真问题。当前是 FIFO + "整条 prompt 一次装":队头一个巨型 prompt 装不下 → `allocate_slots` 返回 None →
`break` → 后面的小请求全被堵。
- **正解 = chunked prefill**(vLLM 的答案),而你已经有 `long_prefill_token_threshold` 这个"分块上限",把它
  用起来:
  - waiting 请求的 `num_new_tokens = request.num_tokens - request.num_computed_tokens`(**不是**
    `num_prompt_tokens`),再 `min(token_budget, chunk)`;
  - 每步只给这一"块"分配 block;`update_from_output` 推进 `num_computed_tokens`;
  - 超长 prompt 就被摊到多步,不用一次占满所有 block,也就不堵后面。
- **廉价过渡版**(M4 v1 先不做 chunked prefill):保留"整条一次装 + break",但 (a) 在 `add_request` 拒掉
  `prompt_len > max_model_len`,(b) 文档里写明 HOL 是已知限制。
- **别用 `continue` 跳过队头去调后面的**——那会破坏 FIFO 公平、可能饿死大请求。

### ⑤ `copy(self.finished_req_ids)` 怎么拷贝?(L128)
- set 直接 `self.finished_req_ids.copy()` 或 `set(self.finished_req_ids)`,**别 import `copy` 模块**。
- **更关键:发出去之后要 `clear()`**(`self.finished_req_ids.clear()`),否则每步都把同一批 finished id
  重复上报给 runner。
- populate 在 `update_from_output`:请求命中 EOS 或达到 `max_tokens` 时 `self.finished_req_ids.add(req.request_id)`,
  同时把它移出 `self.running` 并 `kv_cache_manager.free(req)`。

### ⑥ `num_tokens` 干啥用的?(types.py L42)
= 这个请求**目前的总 token 数** = prompt + 已生成 = `len(self._all_token_ids)`。用途:
`num_new_tokens = num_tokens - num_computed_tokens` = **本步还需要算多少 token**(prefill 时 = 整条 prompt;
decode 时,每 append 一个新 token 后 = 1)。建议做成 **`@property`** 自动算,别手动维护(容易忘同步):
```python
@property
def num_tokens(self) -> int:
    return len(self._all_token_ids)
```

---

## 顺带:types.py 里的 TODO

- **L57 "list 在 python 里是 Pass By Reference 吗?还有哪些?"** → Python 是**"按对象引用传递"(call by
  sharing)**。`list / dict / set / 自定义对象`是**可变**的,传的是同一对象的引用,函数里改它会影响调用方——
  所以你 `list(prompt_token_ids)` 做一份拷贝是对的。`int / str / tuple / frozenset / bool`是**不可变**的,
  没法原地改,也就不存在这个"被别人改掉"的问题。判断口诀:**可变类型才要小心共享引用**。


```
# tests/test_scheduler.py
"""Scheduler.schedule() 单测 —— 纯 CPU,用真实 Request + 真实 KVCacheManager + BlockPool。
schedule() 路径不碰 GPU:只有 KVCacheManager.get_block_table 用 cuda,而 schedule 不调它。

⚠️ 需要先修好这些(否则 import 或运行就崩;详见 docs/design/m4_scheduler_notes.md):
  scheduler.py:
    - `&&` → `and`
    - SchedulerOutput(...) 补 `num_scheduled_tokens` / `total_num_scheduled_tokens` + 补逗号
      + `copy(...)` → `self.finished_req_ids.copy()`
    - L43 `- request.max_tokens` → `- request.num_computed_tokens`
  kv_cache_manager.py:
    - 加 `from typing import List`(或把注解改成小写 `list`)—— 否则 import 就 NameError
    - allocate_slots: `block_size` → `self.block_size`;无新块时 `new_blocks` 未定义 → `return []`
    - **allocate_slots 块不够时返回 None(不要 raise)** —— 抢占逻辑靠 `if new_blocks is not None`,
      而现在 `block_pool.get_new_blocks` 会 raise ValueError
    - free: `request.id` → `request.request_id`
  types.py Request:
    - 去掉 `@dataclass`(否则 `set[Request]` 不可 hash,抢占那步会 TypeError)
    - 加 `num_tokens` property:`return len(self._all_token_ids)`
    - `self.status = Request.WAITING` → `RequestStatus.WAITING`
"""
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
    return KVCacheManager(BlockPool(num_blocks), block_size=BLOCK)


def _sched(kv, budget=4096):
    return Scheduler(block_size=BLOCK, kv_cache_manager=kv,
                     model_config=Cfg(), max_num_scheduled_tokens=budget)


def _req(rid, prompt_len, max_tokens=64):
    return Request(rid, [1] * prompt_len, SamplingParams(max_tokens=max_tokens))


def _make_running(kv, rid, prompt_len):
    """把请求置成 running、prompt 已算完、还有 1 个待解码 token(num_new = 1)。"""
    r = _req(rid, prompt_len)
    kv.allocate_slots(r, prompt_len)          # 占好 prompt 的块
    r.num_computed_tokens = prompt_len
    r.status = RequestStatus.RUNNING
    r._all_token_ids.append(0)                # 追加 1 个生成 token → num_tokens = prompt_len + 1
    return r


# ---------- Step 1: RUNNING 请求 ----------

def test_decode_running_within_budget():
    kv = _kv(100)
    s = _sched(kv)
    s.running = [_make_running(kv, "A", 5), _make_running(kv, "B", 5)]
    out = s.schedule()
    assert {r.request_id for r in out.scheduled_running_reqs} == {"A", "B"}
    assert out.num_scheduled_tokens == {"A": 1, "B": 1}     # decode 每步 1 个 token
    assert len(s.waiting) == 0


def test_token_budget_caps():
    kv = _kv(100)
    s = _sched(kv, budget=1)                                # 预算只够 1 个 token
    s.running = [_make_running(kv, "A", 5), _make_running(kv, "B", 5)]
    out = s.schedule()
    assert len(out.scheduled_running_reqs) == 1             # 只调度得下 1 个


def test_num_new_tokens_zero_skipped():
    kv = _kv(100)
    s = _sched(kv)
    r = _req("X", prompt_len=5)
    kv.allocate_slots(r, 5)
    r.num_computed_tokens = 5
    r.status = RequestStatus.RUNNING                        # 没追加生成 token → num_tokens == computed → num_new = 0
    s.running = [r]
    out = s.schedule()
    assert out.scheduled_running_reqs == []                 # 跳过,且不死循环


# ---------- Step 2: WAITING 请求 ----------

def test_waiting_admitted():
    kv = _kv(100)
    s = _sched(kv)
    w = _req("W", prompt_len=10)                            # 默认 status = WAITING
    s.waiting = deque([w])
    out = s.schedule()
    assert [r.request_id for r in out.scheduled_new_reqs] == ["W"]
    assert w.status == RequestStatus.RUNNING
    assert w in s.running and len(s.waiting) == 0
    assert out.num_scheduled_tokens["W"] == 10             # 整条 prompt 一次 prefill


def test_resumed_preempted_classified():
    kv = _kv(100)
    s = _sched(kv)
    p = _req("P", prompt_len=8)
    p.status = RequestStatus.PREEMPTED                      # 被抢占过的请求恢复
    s.waiting = deque([p])
    out = s.schedule()
    assert [r.request_id for r in out.scheduled_resumed_reqs] == ["P"]   # 归为 resumed,不是 new
    assert out.scheduled_new_reqs == []


# ---------- 抢占 ----------

def test_preemption_when_out_of_blocks():
    # 池刚好 2 块:A、B 各占 1(prompt),池空。B 解码要第 2 块 → 抢占 A。
    kv = _kv(2)
    A = _make_running(kv, "A", 5)      # 占 1 块(prompt 5),池 2→1
    B = _make_running(kv, "B", 16)     # 占 1 块(prompt 16),池 1→0
    s = _sched(kv)
    s.running = [A, B]
    out = s.schedule()
    # A: total 6 → 仍 1 块 → 成功(先被调度)
    # B: computed 16 +1 = 17 → 需第 2 块,池空 → None → 抢占 running[0]=A
    assert "A" in out.preempted_req_ids
    assert A.status == RequestStatus.PREEMPTED
    assert A in s.waiting and A not in s.running
    # 注:A 本步"先被调度又被抢占"是当前实现的疑点(vLLM 从队尾 pop 抢最新的),
    #     见 m4_scheduler_notes.md;这里只断言抢占结果,不断言 A 是否在 scheduled_running_reqs。


def test_no_waiting_admit_after_preemption():
    kv = _kv(2)
    A = _make_running(kv, "A", 5)
    B = _make_running(kv, "B", 16)
    s = _sched(kv)
    s.running = [A, B]
    C = _req("C", prompt_len=4)                             # 一个等待中的新请求
    s.waiting = deque([C])
    out = s.schedule()
    assert out.preempted_req_ids                            # 本步发生了抢占
    assert C.status == RequestStatus.WAITING                # → 本步不准入新 waiting
    assert C not in s.running

```