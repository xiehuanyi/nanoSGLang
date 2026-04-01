"""
Inference Engine — Continuous Batching with Paged Attention.

This is the heart of nanoSGLang. It runs an async loop:
  1. Scheduler builds a mixed batch (prefill chunks + decode tokens)
  2. Engine executes one forward pass on the batch
  3. Sample tokens for decode requests
  4. Update KV cache, advance prefill positions
  5. Stream output tokens to waiting clients

Phase 1 (naive) mode is also kept for simplicity / debugging.
"""

import asyncio
import math
import time
import uuid
from typing import AsyncIterator, Optional

import torch

from nano_sglang.model.llama import CausalLM, ModelConfig, load_model_from_pretrained
from nano_sglang.model.tokenizer import Tokenizer
from nano_sglang.engine.kv_cache import NaiveKVCache
from nano_sglang.engine.paged_kv_cache import BlockManager, PagedKVCache
from nano_sglang.engine.request import Request, RequestQueue, SamplingParams, RequestStatus
from nano_sglang.engine.scheduler import Scheduler, ScheduleBatch, ScheduledRequest
from nano_sglang.engine.sampling import sample_token


# ---------------------------------------------------------------------------
# Output token wrapper
# ---------------------------------------------------------------------------


class TokenOutput:
    __slots__ = ("token_id", "text", "finished")

    def __init__(self, token_id: int, text: str, finished: bool):
        self.token_id = token_id
        self.text = text
        self.finished = finished


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """
    Continuous batching inference engine.

    Modes:
      - naive=True:  Phase 1 single-request mode (no scheduler, contiguous KV cache)
      - naive=False: Phase 2+ continuous batching with paged KV cache
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        max_seq_len: int = 4096,
        # Paged attention settings
        num_blocks: int = 256,
        block_size: int = 16,
        # Scheduler settings
        max_batch_tokens: int = 4096,
        max_running_requests: int = 64,
        prefill_chunk_size: int = 512,
        # Mode
        naive: bool = False,
    ):
        self.device = device
        self.max_seq_len = max_seq_len
        self.naive = naive

        # Load model + tokenizer
        self.model, self.config = load_model_from_pretrained(model_path, device, dtype)
        self.tokenizer = Tokenizer(model_path)
        self.dtype = next(self.model.parameters()).dtype

        if not naive:
            # Phase 2+: paged KV cache + scheduler
            self.block_manager = BlockManager(
                num_blocks=num_blocks,
                block_size=block_size,
                num_layers=self.config.num_hidden_layers,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            self.paged_kv_cache = PagedKVCache(self.block_manager)
            self.request_queue = RequestQueue()
            self.scheduler = Scheduler(
                request_queue=self.request_queue,
                paged_kv_cache=self.paged_kv_cache,
                block_manager=self.block_manager,
                max_batch_tokens=max_batch_tokens,
                max_running_requests=max_running_requests,
                prefill_chunk_size=prefill_chunk_size,
            )

            print(f"  Paged KV Cache: {self.block_manager}")
            print(f"  Scheduler: max_batch_tokens={max_batch_tokens}, "
                  f"chunk_size={prefill_chunk_size}")

            # Start the engine loop
            self._running = True
            self._loop_task: Optional[asyncio.Task] = None
        else:
            self._lock = asyncio.Lock()
            print("  Mode: naive (single request)")

    async def start(self):
        """Start the continuous batching loop (call after event loop is running)."""
        if not self.naive and self._loop_task is None:
            self._loop_task = asyncio.create_task(self._engine_loop())

    async def stop(self):
        """Stop the engine loop."""
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Public API: add request and get streaming output
    # ------------------------------------------------------------------

    async def add_request(
        self,
        prompt_tokens: list[int],
        sampling_params: Optional[SamplingParams] = None,
    ) -> Request:
        """Add a request and return the Request object (read its output_queue)."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        req = Request(
            request_id=uuid.uuid4().hex[:12],
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
        )

        if self.naive:
            # For naive mode, run generation directly
            asyncio.create_task(self._naive_generate(req))
        else:
            await self.request_queue.add(req)

        return req

    async def generate_stream(
        self,
        prompt_tokens: list[int],
        sampling_params: Optional[SamplingParams] = None,
    ) -> AsyncIterator[TokenOutput]:
        """Convenience wrapper: add request and yield outputs."""
        req = await self.add_request(prompt_tokens, sampling_params)
        while True:
            output = await req.output_queue.get()
            yield output
            if output.finished:
                break

    # ------------------------------------------------------------------
    # Phase 2+: Continuous batching engine loop
    # ------------------------------------------------------------------

    async def _engine_loop(self):
        """
        Main loop: schedule → forward → sample → update, repeat.

        This runs continuously while there are requests.
        Uses asyncio.sleep(0) to yield control between steps.
        """
        while self._running:
            # Check if there's work to do
            if self.request_queue.num_waiting == 0 and self.request_queue.num_running == 0:
                await asyncio.sleep(0.001)  # Idle
                continue

            # Schedule the next batch
            batch = self.scheduler.schedule()
            if batch.is_empty:
                await asyncio.sleep(0.001)
                continue

            # Execute forward pass
            await self._execute_batch(batch)

            # Yield control
            await asyncio.sleep(0)

    @torch.inference_mode()
    async def _execute_batch(self, batch: ScheduleBatch):
        """
        Execute one forward pass for a batch of mixed prefill + decode requests.

        For simplicity in this implementation, we process each request separately
        within the batch. A production system would pack them into a single
        padded tensor with proper attention masks.
        """
        for sr in batch.scheduled:
            req = sr.request
            if req.status == RequestStatus.FINISHED:
                continue

            bt = self.scheduler.get_block_table(req.request_id)
            if bt is None:
                continue

            input_ids = torch.tensor(
                [sr.input_token_ids], device=self.device
            )
            num_input = len(sr.input_token_ids)

            if sr.is_prefill:
                # Prefill (possibly chunked)
                start_pos = req.num_prefilled
                position_ids = torch.arange(
                    start_pos, start_pos + num_input, device=self.device
                ).unsqueeze(0)

                attn_mask = self._make_causal_mask(num_input, start_pos + num_input)

                # Forward pass using naive KV cache for this request
                # (We use a temporary contiguous cache and then copy into paged blocks)
                logits = self._forward_with_paged_cache(
                    input_ids, position_ids, attn_mask, req, bt, start_pos
                )

                req.num_prefilled += num_input

                # If prefill is complete, sample the first token
                if not req.is_prefilling:
                    next_token = sample_token(
                        logits[:, -1, :],
                        temperature=req.sampling_params.temperature,
                        top_p=req.sampling_params.top_p,
                        top_k=req.sampling_params.top_k,
                    ).item()

                    req.output_token_ids.append(next_token)
                    req.first_token_time = time.time()

                    text = self.tokenizer.decode([next_token])
                    finished = self._check_finished(req, next_token)

                    await req.output_queue.put(
                        TokenOutput(next_token, text, finished)
                    )

                    if finished:
                        self.scheduler.finish_request(req)

            else:
                # Decode: single token
                pos = req.current_len - 1
                position_ids = torch.tensor([[pos]], device=self.device)
                attn_mask = self._make_causal_mask(1, req.current_len)

                # Ensure we have block space
                self.scheduler.ensure_decode_block(req)

                logits = self._forward_with_paged_cache(
                    input_ids, position_ids, attn_mask, req, bt, pos
                )

                next_token = sample_token(
                    logits[:, -1, :],
                    temperature=req.sampling_params.temperature,
                    top_p=req.sampling_params.top_p,
                    top_k=req.sampling_params.top_k,
                ).item()

                req.output_token_ids.append(next_token)

                text = self.tokenizer.decode([next_token])
                finished = self._check_finished(req, next_token)

                await req.output_queue.put(
                    TokenOutput(next_token, text, finished)
                )

                if finished:
                    self.scheduler.finish_request(req)

    def _forward_with_paged_cache(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        req: Request,
        bt,
        start_pos: int,
    ) -> torch.Tensor:
        """
        Forward pass that reads/writes paged KV cache.

        We use a hybrid approach: the model's forward pass uses contiguous KV cache
        tensors (assembled from paged blocks), and we write back the new KV entries
        to the paged blocks after the forward pass.
        """
        bsz, seq_len = input_ids.shape
        num_layers = self.config.num_hidden_layers
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        # Build per-layer contiguous KV cache views from paged blocks
        total_len = start_pos + seq_len
        kv_caches = []
        for layer_idx in range(num_layers):
            if start_pos > 0:
                # Read existing KV from blocks
                k_existing, v_existing = self.block_manager.read_kv(
                    bt.block_indices, layer_idx, start_pos
                )
                # Allocate contiguous buffer
                k_buf = torch.zeros(
                    1, num_kv_heads, total_len, head_dim,
                    dtype=self.dtype, device=self.device,
                )
                v_buf = torch.zeros_like(k_buf)
                k_buf[0, :, :start_pos, :] = k_existing
                v_buf[0, :, :start_pos, :] = v_existing
            else:
                k_buf = torch.zeros(
                    1, num_kv_heads, total_len, head_dim,
                    dtype=self.dtype, device=self.device,
                )
                v_buf = torch.zeros_like(k_buf)

            kv_caches.append((k_buf, v_buf))

        # Forward pass
        logits = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_caches=kv_caches,
            cache_position=start_pos,
            attention_mask=attn_mask,
        )

        # Write new KV entries back to paged blocks
        for layer_idx in range(num_layers):
            k_new = kv_caches[layer_idx][0][0, :, start_pos:total_len, :]
            v_new = kv_caches[layer_idx][1][0, :, start_pos:total_len, :]

            # Write token by token into the correct block/slot
            for t in range(seq_len):
                abs_pos = start_pos + t
                block_idx = abs_pos // self.block_manager.block_size
                slot_offset = abs_pos % self.block_manager.block_size

                while block_idx >= len(bt.block_indices):
                    bt.block_indices.append(self.block_manager.allocate())

                self.block_manager.write_kv(
                    bt.block_indices[block_idx],
                    layer_idx,
                    slot_offset,
                    k_new[:, t:t + 1, :],
                    v_new[:, t:t + 1, :],
                )

        return logits

    def _check_finished(self, req: Request, token_id: int) -> bool:
        if token_id == self.tokenizer.eos_token_id:
            return True
        if token_id in req.sampling_params.stop_token_ids:
            return True
        if req.num_generated >= req.sampling_params.max_tokens:
            return True
        if req.current_len >= self.max_seq_len:
            return True
        return False

    # ------------------------------------------------------------------
    # Phase 1: Naive single-request mode
    # ------------------------------------------------------------------

    @torch.inference_mode()
    async def _naive_generate(self, req: Request):
        """Phase 1 naive generation: single request, contiguous KV cache."""
        async with self._lock:
            prompt_tokens = req.prompt_tokens
            prompt_len = len(prompt_tokens)

            kv_cache = NaiveKVCache(
                num_layers=self.config.num_hidden_layers,
                max_batch_size=1,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                max_seq_len=self.max_seq_len,
                dtype=self.dtype,
                device=self.device,
            )

            # Prefill
            input_ids = torch.tensor([prompt_tokens], device=self.device)
            position_ids = torch.arange(prompt_len, device=self.device).unsqueeze(0)
            attn_mask = self._make_causal_mask(prompt_len, prompt_len)

            logits = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                kv_caches=kv_cache.get_all_caches(),
                cache_position=0,
                attention_mask=attn_mask,
            )

            next_token = sample_token(
                logits[:, -1, :],
                temperature=req.sampling_params.temperature,
                top_p=req.sampling_params.top_p,
                top_k=req.sampling_params.top_k,
            ).item()

            req.output_token_ids.append(next_token)
            req.first_token_time = time.time()
            text = self.tokenizer.decode([next_token])
            finished = self._check_finished(req, next_token)

            await req.output_queue.put(TokenOutput(next_token, text, finished))
            if finished:
                return

            cur_pos = prompt_len

            # Decode loop
            for _ in range(req.sampling_params.max_tokens - 1):
                if cur_pos >= self.max_seq_len - 1:
                    await req.output_queue.put(TokenOutput(next_token, "", True))
                    return

                input_ids = torch.tensor([[next_token]], device=self.device)
                position_ids = torch.tensor([[cur_pos]], device=self.device)
                attn_mask = self._make_causal_mask(1, cur_pos + 1)

                logits = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    kv_caches=kv_cache.get_all_caches(),
                    cache_position=cur_pos,
                    attention_mask=attn_mask,
                )

                next_token = sample_token(
                    logits[:, -1, :],
                    temperature=req.sampling_params.temperature,
                    top_p=req.sampling_params.top_p,
                    top_k=req.sampling_params.top_k,
                ).item()

                req.output_token_ids.append(next_token)
                cur_pos += 1

                text = self.tokenizer.decode([next_token])
                finished = self._check_finished(req, next_token)

                await req.output_queue.put(TokenOutput(next_token, text, finished))
                if finished:
                    return

                await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _make_causal_mask(self, seq_len: int, full_len: int) -> torch.Tensor:
        if seq_len == full_len:
            mask = torch.full((seq_len, full_len), float("-inf"), device=self.device)
            mask = torch.triu(mask, diagonal=1)
        else:
            mask = torch.zeros(seq_len, full_len, device=self.device)
        return mask.unsqueeze(0).unsqueeze(0).to(self.dtype)
