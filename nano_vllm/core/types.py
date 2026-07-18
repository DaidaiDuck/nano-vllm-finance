# nano_vllm/types.py
from dataclasses import dataclass
from enum import IntEnum,Enum
from typing import Optional

@dataclass
class SamplingParams:
    temperature: float = 0.0 
    top_k: int = -1
    top_p: float = 1.0 
    max_tokens: int = 100  # Max number of output tokens in a request. (prompt not included)

@dataclass
class RequestOutput: 
    request_id: str 
    token_ids: list[int]        # Output token_ids 
    finished: bool              # True / False
    finish_reason: Optional[str] = None  # "stop" (eos) | "length" (max_tokens)
    text: str = ""              # Output text 


    
class Request:
    """
    request_id: str                     # The UNIQUE id of the request
    status: RequestStatus               # The status of the request 

    # === Input === 
    prompt_token_ids: list[int]| None   # list of token ids of prompt
    num_prompt_tokens: int              # length of prompt_token_ids
    sampling_params: SamplingParams     # Sampling tactics

    # === Output ===  
    max_tokens: int                     # Max number of tokens this request can output 
    _output_token_ids: list[int]        # Output token ids 
    _all_token_ids: list[int]           # Prompt + Output token ids
    
    # === Core Status === 
    num_computed_tokens: int = 0        # tokens already computed in the KV Cache
    num_preemptions: int = 0            # Number of times this request has been preempted by the scheduler. 
    num_tokens: int                     # current length of prompt + output token ids
    """
    

    def __init__(self, request_id, prompt_token_ids, sampling_params:SamplingParams, eos_token_id=None):  
        self.request_id = request_id
        self.status = RequestStatus.WAITING
        self.prompt_token_ids = prompt_token_ids
        self.sampling_params = sampling_params
        self.max_tokens = sampling_params.max_tokens # Max number of output tokens

        self._output_token_ids: list[int] = []
    
        # NOTE(derek.sun) list/dict/set/object are mutable. They are pass by reference. 
        # Here, if we use  self._all_token_ids: list[int] = prompt_token_ids, 
        # changing the value of _all_token_ids would also change prompt_token_ids.
        # We must make a copy of prompt_token_ids and pass the copy to _all_token_ids. 
        # int / str / tuple / frozenset / bool are immutable. Hence this issue would not appear. 
        self._all_token_ids: list[int] = list(prompt_token_ids) 
        self.num_computed_tokens = 0
        self.num_prompt_tokens = len(prompt_token_ids)
        self.num_preemptions = 0 
        self.eos_token_id = eos_token_id            # End token_id of the request. Used to check if the request should stop. 
        self.finish_reason = None 

    @property
    def num_tokens(self) -> int:
        """
        num_tokens = total number of tokens in the request at present = prompt + current output token number
        """
        return len(self._all_token_ids) 

class RequestStatus(IntEnum):
    WAITING = 1
    RUNNING = 2
    PREEMPTED = 3
    FINISHED = 4 

class FinishReason(str, Enum): 
    STOP = "stop" 
    LENGTH = "length" 

@dataclass
class KVCacheConfig:
    """
    The KV cache configurato
    """


@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[Request]
    num_scheduled_tokens: dict[str, int] 
    total_num_scheduled_tokens: int 
    preempted_req_ids: set 
    scheduled_running_reqs: list[Request]
    scheduled_resumed_reqs: list[Request]


@dataclass 
class ModelRunnerOutput:
    sampled_token_ids : list[list[int]] # new token ids produced for the requests in batch at each step 
    req_to_index : dict[str, int]       # key: request_id   value: line number of request in model runner's batch   Use for getting new tokens from sampled_token_ids

