"""
Inference Engine — Continuous Batching with Ragged Batch Forward.

Core loop:
  1. Scheduler builds a mixed batch (prefill chunks + decode tokens)
  2. Engine packs all requests into a single flat tensor (ragged batch)
  3. Single forward pass with cu_seqlens for sequence boundaries
  4. Scatter results back to individual requests, sample, update cache
  5. Stream output tokens to clients

Two modes:
  - naive=True:  Phase 1 single-request, contiguous KV cache
  - naive=False: Continuous batching, paged KV cache, ragged batch forward
"""

import asyncio
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


class TokenOutput:
    __slots__ = ("token_id", "text", "finished")

    def __init__(self, token_id: int, text: str, finished: bool):
        self.token_id = token_id
        self.text = text
        self.finished = finished


class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        max_seq_len: int = 4096,
        num_blocks: int = 256,
        block_size: int = 16,
        max_batch_tokens: int = 4096,
        max_running_requests: int = 64,
        prefill_chunk_size: int = 512,
        naive: bool = False,
    ):
        self.device = device
        self.max_seq_len = max_seq_len
        self.naive = naive

        self.model, self.config = load_model_from_pretrained(model_path, device, dtype)
        self.tokenizer = Tokenizer(model_path)
        self.dtype = next(self.model.parameters()).dtype

        if not naive:
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

            self._running = True
            self._loop_task: Optional[asyncio.Task] = None
        else:
            self._lock = asyncio.Lock()
            print("  Mode: naive (single request)")

    async def start(self):
        if not self.naive and self._loop_task is None:
            self._loop_task = asyncio.create_task(self._engine_loop())

    async def stop(self):
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_request(
        self,
        prompt_tokens: list[int],
        sampling_params: Optional[SamplingParams] = None,
    ) -> Request:
        if sampling_params is None:
            sampling_params = SamplingParams()

        req = Request(
            request_id=uuid.uuid4().hex[:12],
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
        )

        if self.naive:
            asyncio.create_task(self._naive_generate(req))
        else:
            await self.request_queue.add(req)

        return req

    async def generate_stream(
        self,
        prompt_tokens: list[int],
        sampling_params: Optional[SamplingParams] = None,
    ) -> AsyncIterator[TokenOutput]:
        req = await self.add_request(prompt_tokens, sampling_params)
        while True:
            output = await req.output_queue.get()
            yield output
            if output.finished:
                break

    # ------------------------------------------------------------------
    # Continuous batching engine loop
    # ------------------------------------------------------------------

    async def _engine_loop(self):
        while self._running:
            if self.request_queue.num_waiting == 0 and self.request_queue.num_running == 0:
                await asyncio.sleep(0.001)
                continue

            batch = self.scheduler.schedule()
            if batch.is_empty:
                await asyncio.sleep(0.001)
                continue

            await self._execute_batch_ragged(batch)
            await asyncio.sleep(0)

    @torch.inference_mode()
    async def _execute_batch_ragged(self, batch: ScheduleBatch):
        """
        Execute one forward pass for the entire batch using ragged/packed tensors.

        All requests' tokens are concatenated into a single flat tensor.
        cu_seqlens marks boundaries. One forward pass processes everything.
        """
        num_layers = self.config.num_hidden_layers
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        # Filter active requests
        active = [sr for sr in batch.scheduled
                  if sr.request.status != RequestStatus.FINISHED]
        if not active:
            return

        # ------------------------------------------------------------------
        # Build ragged batch tensors
        # ------------------------------------------------------------------
        all_input_ids = []
        all_positions = []
        q_lens = []        # number of new tokens per request (Q)
        k_lens = []        # total KV length per request (cached + new)
        cache_starts = []  # where cached KV starts for each request

        for sr in active:
            req = sr.request
            num_input = len(sr.input_token_ids)

            if sr.is_prefill:
                start_pos = req.num_prefilled
                positions = list(range(start_pos, start_pos + num_input))
                cached_len = start_pos  # tokens already cached from prior chunks
            else:
                pos = req.current_len - 1
                positions = [pos]
                cached_len = pos  # everything before this decode token

            all_input_ids.extend(sr.input_token_ids)
            all_positions.extend(positions)
            q_lens.append(num_input)
            k_lens.append(cached_len + num_input)  # total KV = cached + new
            cache_starts.append(cached_len)

            # Ensure blocks exist for decode
            if not sr.is_prefill:
                self.scheduler.ensure_decode_block(req)

        total_q = len(all_input_ids)
        batch_size = len(active)

        input_ids = torch.tensor(all_input_ids, device=self.device)
        positions = torch.tensor(all_positions, device=self.device)

        # Build cu_seqlens
        cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        for i in range(batch_size):
            cu_seqlens_q[i + 1] = cu_seqlens_q[i] + q_lens[i]
            cu_seqlens_k[i + 1] = cu_seqlens_k[i] + k_lens[i]

        max_seqlen_q = max(q_lens)
        max_seqlen_k = max(k_lens)

        # ------------------------------------------------------------------
        # Assemble cached KV per layer
        # ------------------------------------------------------------------
        cached_kvs: list[tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = []

        total_cached = sum(cache_starts)
        if total_cached > 0:
            for layer_idx in range(num_layers):
                k_parts = []
                v_parts = []
                for i, sr in enumerate(active):
                    req = sr.request
                    cached_len = cache_starts[i]
                    if cached_len > 0:
                        bt = self.scheduler.get_block_table(req.request_id)
                        if bt is not None:
                            k_cached, v_cached = self.block_manager.read_kv(
                                bt.block_indices, layer_idx, cached_len
                            )
                            # read_kv returns (num_kv_heads, cached_len, head_dim)
                            # transpose to (cached_len, num_kv_heads, head_dim)
                            k_parts.append(k_cached.transpose(0, 1))
                            v_parts.append(v_cached.transpose(0, 1))
                        else:
                            k_parts.append(torch.zeros(
                                cached_len, num_kv_heads, head_dim,
                                dtype=self.dtype, device=self.device))
                            v_parts.append(torch.zeros(
                                cached_len, num_kv_heads, head_dim,
                                dtype=self.dtype, device=self.device))
                    # If cached_len == 0, no KV to prepend for this request

                if k_parts:
                    cached_kvs.append((torch.cat(k_parts, dim=0),
                                       torch.cat(v_parts, dim=0)))
                else:
                    cached_kvs.append((None, None))
        else:
            cached_kvs = [(None, None)] * num_layers

        # ------------------------------------------------------------------
        # Single forward pass
        # ------------------------------------------------------------------
        logits, new_kvs = self.model.forward_packed(
            input_ids=input_ids,
            positions=positions,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            cached_kvs=cached_kvs,
        )

        # ------------------------------------------------------------------
        # Write new KV to paged blocks & sample tokens
        # ------------------------------------------------------------------
        q_offset = 0
        for i, sr in enumerate(active):
            req = sr.request
            num_input = q_lens[i]
            bt = self.scheduler.get_block_table(req.request_id)
            if bt is None:
                q_offset += num_input
                continue

            # Write new K/V into paged blocks
            start_pos = cache_starts[i]
            for layer_idx in range(num_layers):
                new_k = new_kvs[layer_idx][0][q_offset:q_offset + num_input]
                new_v = new_kvs[layer_idx][1][q_offset:q_offset + num_input]

                for t in range(num_input):
                    abs_pos = start_pos + t
                    block_idx = abs_pos // self.block_manager.block_size
                    slot_offset = abs_pos % self.block_manager.block_size

                    while block_idx >= len(bt.block_indices):
                        bt.block_indices.append(self.block_manager.allocate())

                    # new_k[t] is (num_kv_heads, head_dim), write_kv expects
                    # (num_kv_heads, num_tokens, head_dim)
                    self.block_manager.write_kv(
                        bt.block_indices[block_idx],
                        layer_idx,
                        slot_offset,
                        new_k[t:t + 1].transpose(0, 1),  # (kv_heads, 1, dim)
                        new_v[t:t + 1].transpose(0, 1),
                    )

            # Update prefill progress
            if sr.is_prefill:
                req.num_prefilled += num_input

            # Sample token if this request should produce output
            should_sample = (not sr.is_prefill) or (sr.is_prefill and not req.is_prefilling)

            if should_sample:
                # Get logits for the last token of this request
                last_logit = logits[q_offset + num_input - 1:q_offset + num_input]

                next_token = sample_token(
                    last_logit,
                    temperature=req.sampling_params.temperature,
                    top_p=req.sampling_params.top_p,
                    top_k=req.sampling_params.top_k,
                ).item()

                req.output_token_ids.append(next_token)
                if req.first_token_time is None:
                    req.first_token_time = time.time()

                text = self.tokenizer.decode([next_token])
                finished = self._check_finished(req, next_token)

                await req.output_queue.put(TokenOutput(next_token, text, finished))

                if finished:
                    self.scheduler.finish_request(req)

            q_offset += num_input

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

    def _make_causal_mask(self, seq_len: int, full_len: int) -> torch.Tensor:
        if seq_len == full_len:
            mask = torch.full((seq_len, full_len), float("-inf"), device=self.device)
            mask = torch.triu(mask, diagonal=1)
        else:
            mask = torch.zeros(seq_len, full_len, device=self.device)
        return mask.unsqueeze(0).unsqueeze(0).to(self.dtype)
