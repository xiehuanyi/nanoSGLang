"""
KV Cache management.

Phase 1 (Naive): Pre-allocate a contiguous (num_layers, batch, num_kv_heads, max_seq_len, head_dim)
buffer per layer. Each sequence writes at its current position.

This is simple but wastes memory — Phase 2 will replace it with paged allocation.
"""

import torch


class NaiveKVCache:
    """
    Pre-allocated KV cache for a single sequence (batch=1 for now).

    Layout per layer: (max_batch, num_kv_heads, max_seq_len, head_dim)
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Pre-allocate: list of (k_cache, v_cache) per layer
        self.caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(num_layers):
            k = torch.zeros(max_batch_size, num_kv_heads, max_seq_len, head_dim,
                            dtype=dtype, device=device)
            v = torch.zeros(max_batch_size, num_kv_heads, max_seq_len, head_dim,
                            dtype=dtype, device=device)
            self.caches.append((k, v))

    def get_layer_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.caches[layer_idx]

    def get_all_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self.caches

    def clear(self):
        """Zero out all cache entries."""
        for k, v in self.caches:
            k.zero_()
            v.zero_()

    @property
    def memory_bytes(self) -> int:
        total = 0
        for k, v in self.caches:
            total += k.nelement() * k.element_size()
            total += v.nelement() * v.element_size()
        return total

    def __repr__(self) -> str:
        mb = self.memory_bytes / 1024 / 1024
        return (f"NaiveKVCache(layers={self.num_layers}, max_batch={self.max_batch_size}, "
                f"max_seq={self.max_seq_len}, memory={mb:.1f}MB)")
