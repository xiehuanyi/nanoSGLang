"""
Radix Cache — Prefix-based KV cache sharing.

A radix tree stores token sequences and maps them to KV cache block indices.
When a new request arrives, we search the tree for the longest prefix match
and reuse those KV blocks, skipping their prefill.

Example:
  Request A: [1, 2, 3, 4, 5] → computed and stored
  Request B: [1, 2, 3, 6, 7] → prefix [1, 2, 3] already in cache, reuse!

Eviction uses LRU: each node tracks last access time. When memory is tight,
we evict the least recently used leaf nodes.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from nano_sglang.engine.paged_kv_cache import BlockManager


@dataclass
class RadixNode:
    """A node in the radix tree."""
    # Token sequence stored at this edge (from parent to this node)
    tokens: list[int] = field(default_factory=list)
    # Corresponding KV cache block indices for these tokens
    block_indices: list[int] = field(default_factory=list)
    # Children keyed by the first token of their edge
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    # Parent node
    parent: Optional["RadixNode"] = None
    # Reference count: how many active requests are using this node
    ref_count: int = 0
    # Last access time for LRU eviction
    last_access: float = field(default_factory=time.time)
    # Number of tokens from root to end of this node's edge
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)


class RadixCache:
    """
    Radix tree for prefix-aware KV cache management.

    The tree stores token sequences as edges. Each node maps to a set of
    KV cache block indices. When a new prompt comes in, we traverse the tree
    to find the longest matching prefix and reuse those KV blocks.
    """

    def __init__(self, block_manager: BlockManager):
        self.block_manager = block_manager
        self.root = RadixNode()
        self._num_cached_tokens = 0

    @property
    def num_cached_tokens(self) -> int:
        return self._num_cached_tokens

    def match_prefix(self, tokens: list[int]) -> tuple[list[int], int]:
        """
        Find the longest prefix match in the cache.

        Args:
            tokens: the full token sequence to match

        Returns:
            (block_indices, num_matched_tokens)
            block_indices: list of KV cache block IDs covering the matched prefix
            num_matched_tokens: how many tokens from `tokens` were matched
        """
        node = self.root
        matched = 0
        block_indices = []

        while matched < len(tokens):
            next_token = tokens[matched]
            if next_token not in node.children:
                break

            child = node.children[next_token]
            edge_tokens = child.tokens

            # Check how many tokens in this edge match
            edge_match = 0
            for i, t in enumerate(edge_tokens):
                if matched + i >= len(tokens) or tokens[matched + i] != t:
                    break
                edge_match += 1

            if edge_match == 0:
                break

            # Partial edge match: we can only use blocks up to the matched tokens
            # For simplicity, we only reuse blocks for fully matched edges
            if edge_match < len(edge_tokens):
                # Partial match — we could split the node, but for simplicity
                # we only take fully matched edges
                break

            # Full edge match
            block_indices.extend(child.block_indices)
            matched += edge_match
            child.last_access = time.time()
            child.ref_count += 1
            node = child

        return block_indices, matched

    def insert(
        self,
        tokens: list[int],
        block_indices: list[int],
        start_offset: int = 0,
    ):
        """
        Insert a token sequence and its KV cache blocks into the radix tree.

        Args:
            tokens: full token sequence
            block_indices: corresponding block IDs (one per block_size tokens)
            start_offset: how many tokens are already matched/cached
        """
        if start_offset >= len(tokens):
            return

        node = self.root
        pos = 0

        # Navigate to the insertion point (skip already-matched prefix)
        while pos < start_offset:
            next_token = tokens[pos]
            if next_token not in node.children:
                break
            child = node.children[next_token]
            if pos + child.num_tokens <= start_offset:
                pos += child.num_tokens
                node = child
            else:
                break

        # Insert remaining tokens
        remaining_tokens = tokens[start_offset:]
        # Figure out which blocks correspond to the new tokens
        block_size = self.block_manager.block_size
        start_block = start_offset // block_size
        remaining_blocks = block_indices[start_block:]

        if not remaining_tokens:
            return

        first_token = remaining_tokens[0]

        if first_token in node.children:
            # Edge already exists — check if we need to extend or split
            existing = node.children[first_token]
            # For simplicity, if there's a conflict we skip insertion
            # (a production system would split the node)
            return

        # Create new child node
        new_node = RadixNode(
            tokens=list(remaining_tokens),
            block_indices=list(remaining_blocks),
            parent=node,
            ref_count=0,
            last_access=time.time(),
            depth=node.depth + len(remaining_tokens),
        )
        node.children[first_token] = new_node

        # Increment ref counts on the blocks
        for bid in remaining_blocks:
            self.block_manager.increment_ref(bid)

        self._num_cached_tokens += len(remaining_tokens)

    def release(self, tokens: list[int], num_matched: int):
        """
        Decrement ref counts after a request finishes.

        Args:
            tokens: the token sequence that was matched
            num_matched: how many tokens were matched by match_prefix
        """
        node = self.root
        matched = 0

        while matched < num_matched:
            next_token = tokens[matched]
            if next_token not in node.children:
                break
            child = node.children[next_token]
            child.ref_count = max(0, child.ref_count - 1)
            matched += child.num_tokens
            node = child

    def evict_lru(self, num_blocks_needed: int) -> int:
        """
        Evict least-recently-used leaf nodes to free blocks.

        Only evicts nodes with ref_count == 0.

        Returns: number of blocks freed
        """
        freed = 0

        while freed < num_blocks_needed:
            # Find all evictable leaves (ref_count == 0)
            candidates = []
            self._collect_evictable_leaves(self.root, candidates)

            if not candidates:
                break  # Nothing to evict

            # Pick LRU
            candidates.sort(key=lambda n: n.last_access)
            victim = candidates[0]

            # Free its blocks
            for bid in victim.block_indices:
                self.block_manager.free(bid)
                freed += 1

            self._num_cached_tokens -= victim.num_tokens

            # Remove from parent
            if victim.parent is not None:
                first_token = victim.tokens[0]
                if first_token in victim.parent.children:
                    del victim.parent.children[first_token]

        return freed

    def _collect_evictable_leaves(
        self, node: RadixNode, result: list[RadixNode]
    ):
        """Recursively find leaf nodes with ref_count == 0."""
        if node.is_leaf and node.ref_count == 0 and node.parent is not None:
            result.append(node)
            return

        for child in node.children.values():
            self._collect_evictable_leaves(child, result)

    def clear(self):
        """Clear the entire cache."""
        self._clear_node(self.root)
        self.root.children.clear()
        self._num_cached_tokens = 0

    def _clear_node(self, node: RadixNode):
        for child in node.children.values():
            self._clear_node(child)
            for bid in child.block_indices:
                self.block_manager.free(bid)

    def stats(self) -> dict:
        num_nodes = self._count_nodes(self.root)
        return {
            "num_nodes": num_nodes,
            "num_cached_tokens": self._num_cached_tokens,
            "block_manager_free": self.block_manager.num_free_blocks,
            "block_manager_used": self.block_manager.num_used_blocks,
        }

    def _count_nodes(self, node: RadixNode) -> int:
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count
