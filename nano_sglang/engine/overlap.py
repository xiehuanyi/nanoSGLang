"""
Overlap Scheduling — Overlap CPU scheduling with GPU computation.

Uses a double-buffer approach:
  - While the GPU executes batch N, the CPU prepares batch N+1
  - Two CUDA streams: one for compute, one for memory transfers

This is one of Mini-SGLang's core ideas: the scheduler shouldn't block
on GPU completion, and GPU shouldn't wait for CPU scheduling.

                   Time →
  CPU:  [schedule B1] [schedule B2] [schedule B3] ...
  GPU:           [execute B1] [execute B2] [execute B3] ...
                 ↑ overlap ↑
"""

import asyncio
import threading
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable
from collections import deque

import torch

from nano_sglang.engine.scheduler import Scheduler, ScheduleBatch


@dataclass
class PreparedBatch:
    """A batch that has been scheduled (CPU work done) and is ready for GPU."""
    schedule_batch: ScheduleBatch
    # Pre-assembled tensors (prepared on CPU while GPU is busy)
    input_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None


class OverlapScheduler:
    """
    Double-buffered scheduler that overlaps CPU scheduling with GPU execution.

    The key insight: scheduling (selecting requests, building batches) is CPU work.
    While the GPU is busy with the current forward pass, we can prepare the next batch.

    Usage:
        overlap = OverlapScheduler(scheduler, execute_fn)
        await overlap.run()
    """

    def __init__(
        self,
        scheduler: Scheduler,
        execute_fn: Callable[[ScheduleBatch], Awaitable[None]],
        max_prefetch: int = 1,
    ):
        self.scheduler = scheduler
        self.execute_fn = execute_fn
        self.max_prefetch = max_prefetch

        self._running = False
        self._batch_queue: deque[ScheduleBatch] = deque(maxlen=max_prefetch + 1)
        self._schedule_event = asyncio.Event()
        self._gpu_done_event = asyncio.Event()

    async def run(self):
        """Run the overlapped scheduling loop."""
        self._running = True
        self._gpu_done_event.set()

        while self._running:
            # CPU: Schedule next batch while GPU might be busy
            batch = self.scheduler.schedule()

            if batch.is_empty:
                await asyncio.sleep(0.001)
                continue

            # Wait for GPU to finish previous batch
            await self._gpu_done_event.wait()
            self._gpu_done_event.clear()

            # GPU: Execute the batch
            await self.execute_fn(batch)
            self._gpu_done_event.set()

            await asyncio.sleep(0)

    async def stop(self):
        self._running = False


class CUDAStreamOverlap:
    """
    Low-level CUDA stream management for compute/transfer overlap.

    Uses two CUDA streams:
      - compute_stream: main model forward pass
      - transfer_stream: KV cache block copies, input tensor preparation

    Operations on different streams can execute concurrently on the GPU.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        if torch.cuda.is_available():
            self.compute_stream = torch.cuda.Stream(device=device)
            self.transfer_stream = torch.cuda.Stream(device=device)
        else:
            self.compute_stream = None
            self.transfer_stream = None

    def compute(self, fn, *args, **kwargs):
        """Run function on the compute stream."""
        if self.compute_stream is not None:
            with torch.cuda.stream(self.compute_stream):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    def transfer(self, fn, *args, **kwargs):
        """Run function on the transfer stream."""
        if self.transfer_stream is not None:
            with torch.cuda.stream(self.transfer_stream):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    def sync_compute(self):
        """Wait for compute stream to finish."""
        if self.compute_stream is not None:
            self.compute_stream.synchronize()

    def sync_transfer(self):
        """Wait for transfer stream to finish."""
        if self.transfer_stream is not None:
            self.transfer_stream.synchronize()

    def sync_all(self):
        """Wait for both streams."""
        self.sync_compute()
        self.sync_transfer()

    def compute_wait_transfer(self):
        """Make compute stream wait for transfer stream (dependency)."""
        if self.compute_stream and self.transfer_stream:
            event = self.transfer_stream.record_event()
            self.compute_stream.wait_event(event)

    def transfer_wait_compute(self):
        """Make transfer stream wait for compute stream."""
        if self.compute_stream and self.transfer_stream:
            event = self.compute_stream.record_event()
            self.transfer_stream.wait_event(event)
