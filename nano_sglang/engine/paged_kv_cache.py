"""
Paged KV Cache — Block-based memory management.

Instead of allocating one big contiguous buffer per sequence, we allocate
fixed-size "blocks" (each holds `block_size` tokens of KV cache for all layers).

BlockManager tracks free/used blocks and maps each request to its block list.

Layout per block:
  (num_layers, 2, num_kv_heads, block_size, head_dim)
  where 2 = [key, value]

This enables:
  - Memory sharing between sequences (prefix caching)
  - Dynamic memory allocation (no wasted space for short sequences)
  - Memory reclamation when sequences finish
"""

from dataclasses import dataclass, field

import torch


@dataclass
class BlockTable:
    """Mapping from request to its list of physical block indices."""
    # block_indices[i] = physical block index for the i-th logical block
    block_indices: list[int] = field(default_factory=list)

    @property
    def num_blocks(self) -> int:
        return len(self.block_indices)


class BlockManager:
    """
    Manages a pool of fixed-size KV cache blocks.

    Each block stores `block_size` tokens of KV data for ALL layers.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # FlashInfer-compatible KV cache layout (NHD)
        # Shape: (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k_cache = torch.zeros(
            num_layers, num_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )
        self.v_cache = torch.zeros(
            num_layers, num_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )

        # Free block tracking
        self.free_blocks: list[int] = list(range(num_blocks))
        self.ref_counts: dict[int, int] = {}  # block_id -> reference count

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    def can_allocate(self, num_blocks: int) -> bool:
        return self.num_free_blocks >= num_blocks

    def allocate(self) -> int:
        """Allocate a single block. Returns block index."""
        if not self.free_blocks:
            raise RuntimeError("Out of KV cache blocks!")
        block_id = self.free_blocks.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def free(self, block_id: int):
        """Decrement ref count; if zero, return block to free pool."""
        if block_id not in self.ref_counts:
            return
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] <= 0:
            del self.ref_counts[block_id]
            self.free_blocks.append(block_id)

    def increment_ref(self, block_id: int):
        """Increment reference count (for prefix cache sharing)."""
        self.ref_counts[block_id] = self.ref_counts.get(block_id, 0) + 1

    def free_block_table(self, block_table: BlockTable):
        """Free all blocks in a block table."""
        for block_id in block_table.block_indices:
            self.free(block_id)
        block_table.block_indices.clear()

    def allocate_blocks_for_tokens(self, num_tokens: int) -> list[int]:
        """Allocate enough blocks to hold num_tokens tokens."""
        num_needed = (num_tokens + self.block_size - 1) // self.block_size
        if not self.can_allocate(num_needed):
            raise RuntimeError(
                f"Need {num_needed} blocks but only {self.num_free_blocks} free"
            )
        return [self.allocate() for _ in range(num_needed)]

    def write_kv(
        self,
        block_id: int,
        layer_idx: int,
        slot_offset: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """
        Write key/value into a specific block at a given slot offset.
        k, v: (num_kv_heads, num_tokens, head_dim)
        """
        num_tokens = k.shape[1]
        # Transpose (kv_heads, tokens, dim) -> (tokens, kv_heads, dim) for NHD layout
        self.k_cache[layer_idx, block_id, slot_offset:slot_offset + num_tokens] = k.transpose(0, 1)
        self.v_cache[layer_idx, block_id, slot_offset:slot_offset + num_tokens] = v.transpose(0, 1)

    def read_kv(
        self,
        block_ids: list[int],
        layer_idx: int,
        total_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read concatenated K/V from a sequence of blocks.
        Returns: k, v each of shape (num_kv_heads, total_tokens, head_dim)
        """
        k_parts = []
        v_parts = []
        remaining = total_tokens
        for block_id in block_ids:
            n = min(remaining, self.block_size)
            # NHD layout: (block_size, kv_heads, head_dim)
            k_parts.append(self.k_cache[layer_idx, block_id, :n])
            v_parts.append(self.v_cache[layer_idx, block_id, :n])
            remaining -= n
            if remaining <= 0:
                break
        # (total_tokens, kv_heads, dim) -> (kv_heads, total_tokens, dim)
        return torch.cat(k_parts, dim=0).transpose(0, 1), \
               torch.cat(v_parts, dim=0).transpose(0, 1)

    @property
    def memory_bytes(self) -> int:
        return (self.k_cache.nelement() + self.v_cache.nelement()) * self.k_cache.element_size()

    def __repr__(self) -> str:
        mb = self.memory_bytes / 1024 / 1024
        return (f"BlockManager(blocks={self.num_blocks}, block_size={self.block_size}, "
                f"free={self.num_free_blocks}, memory={mb:.1f}MB)")


class PagedKVCache:
    """
    Paged KV cache that integrates with the model's forward pass.

    Wraps BlockManager to provide the same interface as NaiveKVCache
    but with block-based storage.
    """

    def __init__(self, block_manager: BlockManager):
        self.block_manager = block_manager

    def allocate_for_request(self, num_tokens: int) -> BlockTable:
        block_ids = self.block_manager.allocate_blocks_for_tokens(num_tokens)
        return BlockTable(block_indices=block_ids)

    def append_block(self, block_table: BlockTable) -> int:
        """Allocate and append one more block. Returns the new block id."""
        block_id = self.block_manager.allocate()
        block_table.block_indices.append(block_id)
        return block_id

    def free_request(self, block_table: BlockTable):
        self.block_manager.free_block_table(block_table)

    def get_kv_for_attention(
        self,
        block_table: BlockTable,
        layer_idx: int,
        total_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read all cached K/V for a request at a given layer."""
        return self.block_manager.read_kv(
            block_table.block_indices, layer_idx, total_tokens
        )

    def write_kv_token(
        self,
        block_table: BlockTable,
        layer_idx: int,
        seq_position: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Write K/V for token(s) at a given sequence position."""
        block_size = self.block_manager.block_size
        block_idx = seq_position // block_size
        slot_offset = seq_position % block_size

        # May need to allocate a new block
        while block_idx >= len(block_table.block_indices):
            self.append_block(block_table)

        self.block_manager.write_kv(
            block_table.block_indices[block_idx],
            layer_idx, slot_offset, k, v,
        )
