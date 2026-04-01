"""
Request lifecycle management.

States:
  WAITING  → request queued, not yet scheduled
  RUNNING  → in the active batch, doing prefill or decode
  FINISHED → generation complete (EOS, max_tokens, or error)

Each request tracks its own token history, sampling params, and output queue.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop_token_ids: list[int] = field(default_factory=list)


@dataclass
class Request:
    request_id: str
    prompt_tokens: list[int]
    sampling_params: SamplingParams

    # Internal state
    status: RequestStatus = RequestStatus.WAITING
    output_token_ids: list[int] = field(default_factory=list)

    # For prefill chunking: how many prompt tokens have been prefilled so far
    num_prefilled: int = 0

    # KV cache block indices (for paged attention)
    block_indices: list[int] = field(default_factory=list)

    # Async output queue: engine pushes tokens, API reads them
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Timing
    arrival_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)

    @property
    def current_len(self) -> int:
        """Total sequence length so far (prompt + generated)."""
        return self.prompt_len + self.num_generated

    @property
    def remaining_prefill(self) -> int:
        """How many prompt tokens still need prefilling."""
        return self.prompt_len - self.num_prefilled

    @property
    def is_prefilling(self) -> bool:
        return self.num_prefilled < self.prompt_len

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_time is not None:
            return self.first_token_time - self.arrival_time
        return None


class RequestQueue:
    """Thread-safe request queue with three pools."""

    def __init__(self):
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self._lock = asyncio.Lock()

    async def add(self, request: Request):
        async with self._lock:
            request.status = RequestStatus.WAITING
            self.waiting.append(request)

    async def move_to_running(self, request: Request):
        async with self._lock:
            if request in self.waiting:
                self.waiting.remove(request)
            request.status = RequestStatus.RUNNING
            if request not in self.running:
                self.running.append(request)

    async def move_to_finished(self, request: Request):
        async with self._lock:
            if request in self.running:
                self.running.remove(request)
            request.status = RequestStatus.FINISHED
            request.finish_time = time.time()
            self.finished.append(request)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)
