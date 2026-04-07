"""
Batch Scheduler — Continuous Batching with Chunked Prefill.

Responsibilities:
  1. Select which requests from WAITING to admit into RUNNING
  2. Build a mixed batch of prefill + decode tokens
  3. Chunked prefill: split long prompts so prefill doesn't dominate
  4. Evict finished requests and reclaim their KV cache blocks

Key idea: every scheduling step produces a ScheduleBatch that the engine
forwards through the model in a single call.
"""

from dataclasses import dataclass, field
from typing import Optional

from nano_sglang.engine.request import Request, RequestQueue, RequestStatus
from nano_sglang.engine.paged_kv_cache import BlockManager, BlockTable, PagedKVCache


@dataclass
class ScheduledRequest:
    """One request's contribution to the current batch."""
    request: Request
    # Which token ids to feed into the model this step
    input_token_ids: list[int] = field(default_factory=list)
    # Whether this request is doing prefill in this step
    is_prefill: bool = False
    # Number of tokens being prefilled in this step (could be a chunk)
    num_prefill_tokens: int = 0


@dataclass
class ScheduleBatch:
    """The batch to execute in one forward pass."""
    scheduled: list[ScheduledRequest] = field(default_factory=list)
    # Total number of tokens in this batch (across all requests)
    total_tokens: int = 0

    @property
    def batch_size(self) -> int:
        return len(self.scheduled)

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled) == 0


class Scheduler:
    """
    Continuous batching scheduler with chunked prefill support.

    Parameters:
        max_batch_tokens: max total tokens per forward step
        max_running_requests: max concurrent requests in RUNNING state
        prefill_chunk_size: max tokens to prefill per request per step
                           (0 = no chunking, prefill entire prompt at once)
    """

    def __init__(
        self,
        request_queue: RequestQueue,
        paged_kv_cache: PagedKVCache,
        block_manager: BlockManager,
        max_batch_tokens: int = 4096,
        max_running_requests: int = 64,
        prefill_chunk_size: int = 512,
    ):
        self.queue = request_queue
        self.kv_cache = paged_kv_cache
        self.block_manager = block_manager
        self.max_batch_tokens = max_batch_tokens
        self.max_running_requests = max_running_requests
        self.prefill_chunk_size = prefill_chunk_size

        # Block tables for each request (request_id -> BlockTable)
        self.block_tables: dict[str, BlockTable] = {}

    def _blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_manager.block_size - 1) // self.block_manager.block_size

    def schedule(self) -> ScheduleBatch:
        """
        Build the next batch. Called once per forward step.

        Strategy:
          1. First, schedule all RUNNING requests that are decoding (1 token each)
          2. Then, try to admit WAITING requests (prefill), respecting budget
          3. For requests still prefilling (chunked), continue their prefill
        """
        batch = ScheduleBatch()
        budget = self.max_batch_tokens

        # ------------------------------------------------------------------
        # Step 1: Continue decode for running requests that finished prefill
        # ------------------------------------------------------------------
        finished_requests = []
        for req in list(self.queue.running):
            if req.status == RequestStatus.FINISHED:
                finished_requests.append(req)
                continue

            if not req.is_prefilling:
                # Decode: feed the last generated token
                if req.output_token_ids:
                    sr = ScheduledRequest(
                        request=req,
                        input_token_ids=[req.output_token_ids[-1]],
                        is_prefill=False,
                        num_prefill_tokens=0,
                    )
                    batch.scheduled.append(sr)
                    budget -= 1
                    batch.total_tokens += 1
            else:
                # Still prefilling (chunked): continue
                chunk_size = req.remaining_prefill
                if self.prefill_chunk_size > 0:
                    chunk_size = min(chunk_size, self.prefill_chunk_size)
                chunk_size = min(chunk_size, budget)

                if chunk_size > 0:
                    start = req.num_prefilled
                    tokens = req.prompt_tokens[start:start + chunk_size]

                    # Allocate blocks for the new chunk
                    blocks_needed = self._blocks_needed(start + chunk_size) - len(
                        self.block_tables.get(req.request_id, BlockTable()).block_indices
                    )
                    if blocks_needed > 0 and not self.block_manager.can_allocate(blocks_needed):
                        continue  # Skip, not enough memory

                    if blocks_needed > 0:
                        bt = self.block_tables.get(req.request_id, BlockTable())
                        for _ in range(blocks_needed):
                            bt.block_indices.append(self.block_manager.allocate())
                        self.block_tables[req.request_id] = bt

                    sr = ScheduledRequest(
                        request=req,
                        input_token_ids=tokens,
                        is_prefill=True,
                        num_prefill_tokens=chunk_size,
                    )
                    batch.scheduled.append(sr)
                    budget -= chunk_size
                    batch.total_tokens += chunk_size

        # Clean up finished
        for req in finished_requests:
            self._evict_request(req)

        # ------------------------------------------------------------------
        # Step 2: Admit new requests from WAITING
        # ------------------------------------------------------------------
        newly_admitted = []
        for req in list(self.queue.waiting):
            if len(self.queue.running) + len(newly_admitted) >= self.max_running_requests:
                break
            if budget <= 0:
                break

            # How much to prefill this step
            chunk_size = req.prompt_len
            if self.prefill_chunk_size > 0:
                chunk_size = min(chunk_size, self.prefill_chunk_size)
            chunk_size = min(chunk_size, budget)

            # Check if we can allocate blocks
            total_blocks_needed = self._blocks_needed(chunk_size)
            # Also need 1 block for the first decode token
            if not self.block_manager.can_allocate(total_blocks_needed):
                continue

            # Allocate
            block_ids = [self.block_manager.allocate() for _ in range(total_blocks_needed)]
            bt = BlockTable(block_indices=block_ids)
            self.block_tables[req.request_id] = bt

            tokens = req.prompt_tokens[:chunk_size]
            sr = ScheduledRequest(
                request=req,
                input_token_ids=tokens,
                is_prefill=True,
                num_prefill_tokens=chunk_size,
            )
            batch.scheduled.append(sr)
            budget -= chunk_size
            batch.total_tokens += chunk_size
            newly_admitted.append(req)

        # Move admitted to running (synchronous since we're in the engine loop)
        for req in newly_admitted:
            if req in self.queue.waiting:
                self.queue.waiting.remove(req)
            req.status = RequestStatus.RUNNING
            if req not in self.queue.running:
                self.queue.running.append(req)

        return batch

    def _evict_request(self, req: Request):
        """Free resources for a finished request."""
        if req.request_id in self.block_tables:
            self.block_manager.free_block_table(self.block_tables[req.request_id])
            del self.block_tables[req.request_id]
        if req in self.queue.running:
            self.queue.running.remove(req)
        req.status = RequestStatus.FINISHED
        if req not in self.queue.finished:
            self.queue.finished.append(req)

    def finish_request(self, req: Request):
        """Mark a request as finished and schedule eviction."""
        req.status = RequestStatus.FINISHED
        # Will be cleaned up in next schedule() call

    def get_block_table(self, request_id: str) -> Optional[BlockTable]:
        return self.block_tables.get(request_id)

    def ensure_decode_block(self, req: Request):
        """Ensure there's a block for the current decode position."""
        bt = self.block_tables.get(req.request_id)
        if bt is None:
            return
        # Need enough blocks to cover position (current_len - 1)
        needed = (req.current_len - 1) // self.block_manager.block_size + 1
        while len(bt.block_indices) < needed:
            if self.block_manager.can_allocate(1):
                bt.block_indices.append(self.block_manager.allocate())
            else:
                break
