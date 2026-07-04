# nano_vllm/kv_cache.py
# M2新增
import torch
# typing helpers used in annotations below; importing torch alone is not enough —
# Optional / Dict / Tuple must be imported or the annotations raise NameError.
from typing import Optional, Dict, Tuple

class MyKVCache:
    """
    M2 KV Cache: 预分配，连续存储
    实现HuggingFace Cache的接口, 可以无缝替换Dynamic Cache
    """

    def __init__(
            self,
            num_layers: int, 
            max_seq_len: int, # 按照max_seq_len预分配内存
            num_kv_heads: int, 
            head_dim: int, 
            dtype: torch.dtype, 
            device: str = "cuda",
            ):
        """预分配 cache tensor"""
        # 初始化变量 
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim 
        self.dtype = dtype
        self.device = device

        # 预分配 (一次性)
        # NOTE: the function is torch.zeros / torch.zeros_like (not "zeroes").
        self.k_cache = torch.zeros(
            num_layers, max_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)

        # 当前token长度为0
        self.current_len = 0

        # 打印显存占用 (debug)
        # element_size(): bytes per element (bfloat16 -> 2, float32 -> 4).
        # numel(): total number of elements
        #          (= num_layers * max_seq_len * num_kv_heads * head_dim).
        # element_size() * numel() = bytes of ONE cache; * 2 covers both K and V;
        # / 1e6 converts to MB.
        size_mb = self.k_cache.element_size() * self.k_cache.numel() * 2 / 1e6
        print(f"[MyKVCache] Allocated {size_mb:.1f} MB on {device}") 
    

    def update(
            self, 
            key_states: torch.Tensor, 
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict] = None
    # Return type uses SQUARE brackets: Tuple[A, B]. Tuple(A, B) with parentheses
    # would *call* Tuple and raise TypeError at class-definition time.
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Write new K,V and Return new K,V. 

        Args:
            key_states: [batch_size=1, num_kv_heads, seq_len, head_dim]
            value_states: same as key_states 
            layer_idx: layer index (0 to num_layers - 1)

        Returns: 
            (k_full, v_full): [batch_size=1, num_kv_heads, total_len, head_dim]
        """ 
        # 1. Validate shape
        # Use dim() (number of dimensions) to check for 4D. Do NOT write
        # `key_states.shape == 4`: .shape is a torch.Size, never equal to an int,
        # so that assertion would always fail.
        assert key_states.dim() == 4, f"Expected 4D, got {key_states.shape}"
        assert key_states.shape[0] == 1, "Batch size must be 1 in M2"
        
        bsz, num_kv_heads, seq_len, head_dim = key_states.shape
        assert num_kv_heads == self.num_kv_heads
        assert head_dim == self.head_dim 

        start = self.current_len
        end = start + seq_len 

        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: current={self.current_len}, seq_len={seq_len}, max={self.max_seq_len}"
            )
        
        # 2. Shape transformation: key_states.shape -> k_cache.shape 
        # [batch_size, num_kv_heads, seq_len, head_dim] -> [seq_len, num_kv_heads, head_dim]
        k_to_write = key_states.squeeze(0).transpose(0,1).contiguous()
        v_to_write = value_states.squeeze(0).transpose(0,1).contiguous()

        # 3. Write new key and value back to k_cache and v_cache 
        self.k_cache[layer_idx, start:end] = k_to_write
        self.v_cache[layer_idx, start:end] = v_to_write

        # Advance current_len ONLY after the last layer. update() is called once
        # per layer and all layers share current_len, so bumping it every call
        # would over-count by seq_len * num_layers. Forgetting to update it at all
        # leaves start stuck at 0, so every step overwrites position 0 and the
        # cache never actually grows.
        if layer_idx == self.num_layers - 1:
            self.current_len = end

        # 4. Shape transformation: k_cache.shape -> key_states.shape
        # [end, num_kv_heads, head_dim] -> [batch_size, num_kv_heads, seq_len, head_dim]
        # Slice [:end] — return ONLY the written tokens. Returning the whole
        # k_cache[layer_idx] would include unwritten zero rows up to max_seq_len,
        # and attention would attend to those zero K/V vectors -> wrong output.
        k_full = self.k_cache[layer_idx, :end].transpose(0,1).unsqueeze(0).contiguous()
        v_full = self.v_cache[layer_idx, :end].transpose(0,1).unsqueeze(0).contiguous() 

        # 5. return full key and value for next attention calculation
        return k_full, v_full 

    def get_seq_length(self, layer_idx: int = 0) -> int: 
        """当前cache中有多少token"""
        return self.current_len

    def get_max_length(self) -> Optional[int]:
        """cache容量""" 
        return self.max_seq_len

    def reset(self):
        """重置 cache (新请求开始)"""
        self.current_len = 0