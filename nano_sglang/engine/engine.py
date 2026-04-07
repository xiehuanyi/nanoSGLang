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
import os
import time
import uuid
from typing import AsyncIterator, Optional

import torch


def _debug_mem_env() -> bool:
    return bool(os.environ.get("NANO_MEM_DEBUG"))

from nano_sglang.model.causal_lm import CausalLM, ModelConfig, load_model_from_pretrained
from nano_sglang.model.tokenizer import Tokenizer
from nano_sglang.engine.kv_cache import NaiveKVCache
from nano_sglang.engine.paged_kv_cache import BlockManager, PagedKVCache
from nano_sglang.engine.request import Request, RequestQueue, SamplingParams, RequestStatus
from nano_sglang.engine.scheduler import Scheduler, ScheduleBatch, ScheduledRequest
from nano_sglang.engine.sampling import sample_token

_FLASHINFER_AVAILABLE = False
try:
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper, BatchDecodeWithPagedKVCacheWrapper
    _FLASHINFER_AVAILABLE = True
except ImportError:
    pass


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

            # FlashInfer paged attention setup
            self.use_flashinfer = _FLASHINFER_AVAILABLE
            if self.use_flashinfer:
                self._workspace = torch.empty(
                    128 * 1024 * 1024, dtype=torch.uint8, device=self.device
                )
                self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self._workspace, kv_layout="NHD"
                )

                # --- CUDA Graph for decode ---
                max_bs = max_running_requests
                max_pages_per_req = (max_seq_len + block_size - 1) // block_size
                self._decode_workspace = torch.empty(
                    128 * 1024 * 1024, dtype=torch.uint8, device=self.device
                )
                # One CUDA-graph prefill wrapper per captured batch size
                self._cuda_graph_bs = sorted(set(
                    [1, 2, 4] + list(range(8, max_bs + 1, 8))
                ))
                self._graph_wrappers: dict[int, object] = {}
                for bs in self._cuda_graph_bs:
                    self._graph_wrappers[bs] = BatchPrefillWithPagedKVCacheWrapper(
                        self._decode_workspace, kv_layout="NHD",
                        use_cuda_graph=True,
                        qo_indptr_buf=torch.zeros(
                            bs + 1, dtype=torch.int32, device=self.device),
                        paged_kv_indptr_buf=torch.zeros(
                            bs + 1, dtype=torch.int32, device=self.device),
                        paged_kv_indices_buf=torch.zeros(
                            bs * max_pages_per_req, dtype=torch.int32, device=self.device),
                        paged_kv_last_page_len_buf=torch.zeros(
                            bs, dtype=torch.int32, device=self.device),
                    )
                # Static buffers (fixed addresses for graph capture)
                self._static_input_ids = torch.zeros(max_bs, dtype=torch.int64, device=self.device)
                self._static_positions = torch.zeros(max_bs, dtype=torch.int64, device=self.device)
                self._static_write_indices = torch.zeros(max_bs, dtype=torch.int64, device=self.device)
                # Dummy page for padding idle slots
                self._dummy_block_id = self.block_manager.allocate()
                self._dummy_flat_idx = self._dummy_block_id * block_size
                self._cuda_graphs: dict[int, tuple] = {}  # bs -> (graph, wrapper, logits_ref)

                print(f"  FlashInfer: enabled (paged attention + CUDA Graph)")
            else:
                print(f"  FlashInfer: not available, using legacy KV copy path")

            self._running = True
            self._loop_task: Optional[asyncio.Task] = None
        else:
            self._lock = asyncio.Lock()
            print("  Mode: naive (single request)")

    async def start(self):
        if not self.naive and self._loop_task is None:
            if self.use_flashinfer:
                self._capture_cuda_graphs()
            self._loop_task = asyncio.create_task(self._engine_loop())

    # ------------------------------------------------------------------
    # CUDA Graph capture / replay
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _capture_cuda_graphs(self):
        """Capture CUDA graphs for decode at various batch sizes."""
        page_size = self.block_manager.block_size
        kv = (self.block_manager.k_cache, self.block_manager.v_cache)

        for bs in self._cuda_graph_bs:
            wrapper = self._graph_wrappers[bs]

            # Fill static buffers with dummy data
            self._static_input_ids[:bs].fill_(0)
            self._static_positions[:bs].fill_(0)
            self._static_write_indices[:bs].fill_(self._dummy_flat_idx)

            # Plan with dummy page table (each request → 1 dummy page)
            qo_indptr = torch.arange(bs + 1, dtype=torch.int32, device=self.device)
            kv_indptr = torch.arange(bs + 1, dtype=torch.int32, device=self.device)
            kv_indices = torch.full((bs,), self._dummy_block_id, dtype=torch.int32, device=self.device)
            lpl = torch.ones(bs, dtype=torch.int32, device=self.device)
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_indptr,
                paged_kv_indices=kv_indices,
                paged_kv_last_page_len=lpl,
                num_qo_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim_qk=self.config.head_dim,
                page_size=page_size,
                causal=True,
                q_data_type=self.dtype,
            )

            # Warm-up runs (let CUDA allocator settle)
            for _ in range(3):
                self.model.forward_packed(
                    input_ids=self._static_input_ids[:bs],
                    positions=self._static_positions[:bs],
                    kv_cache=kv, attn_wrapper=wrapper,
                    write_indices=self._static_write_indices[:bs],
                )
            torch.cuda.synchronize()

            # Capture graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                logits, _ = self.model.forward_packed(
                    input_ids=self._static_input_ids[:bs],
                    positions=self._static_positions[:bs],
                    kv_cache=kv, attn_wrapper=wrapper,
                    write_indices=self._static_write_indices[:bs],
                )
            self._cuda_graphs[bs] = (g, wrapper, logits)

        print(f"  CUDA Graph: captured bs={list(self._cuda_graphs.keys())}")

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

    async def _execute_batch_ragged(self, batch: ScheduleBatch):
        """
        Execute one forward pass for the entire batch using ragged/packed tensors.

        All requests' tokens are concatenated into a single flat tensor.
        cu_seqlens marks boundaries. One forward pass processes everything.

        NOTE: we use `with torch.inference_mode()` instead of the decorator
        form. The decorator only holds the context during coroutine creation;
        once the async function actually runs, grad mode is re-entered and
        autograd silently records every forward pass, retaining intermediate
        tensors indefinitely.
        """
        with torch.inference_mode():
            await self._execute_batch_ragged_inner(batch)

    async def _execute_batch_ragged_inner(self, batch: ScheduleBatch):
        num_layers = self.config.num_hidden_layers

        # Filter active requests
        active = [sr for sr in batch.scheduled
                  if sr.request.status != RequestStatus.FINISHED]
        if not active:
            return

        # ------------------------------------------------------------------
        # Build ragged batch tensors (common for both paths)
        # ------------------------------------------------------------------
        all_input_ids = []
        all_positions = []
        q_lens = []        # number of new tokens per request (Q)
        cache_starts = []  # where cached KV starts for each request

        for sr in active:
            req = sr.request
            num_input = len(sr.input_token_ids)

            if sr.is_prefill:
                start_pos = req.num_prefilled
                positions = list(range(start_pos, start_pos + num_input))
                cache_starts.append(start_pos)
            else:
                pos = req.current_len - 1
                positions = [pos]
                cache_starts.append(pos)
                # Ensure blocks exist for decode
                self.scheduler.ensure_decode_block(req)

            all_input_ids.extend(sr.input_token_ids)
            all_positions.extend(positions)
            q_lens.append(num_input)

        total_q = len(all_input_ids)
        batch_size = len(active)

        input_ids = torch.tensor(all_input_ids, device=self.device)
        positions = torch.tensor(all_positions, device=self.device)

        cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        for i in range(batch_size):
            cu_seqlens_q[i + 1] = cu_seqlens_q[i] + q_lens[i]

        # Logit indices: last token of each request (for efficient lm_head)
        logit_indices = (cu_seqlens_q[1:] - 1).long()

        # Check if this is a pure-decode batch eligible for CUDA Graph
        is_pure_decode = (self.use_flashinfer
                          and all(not sr.is_prefill for sr in active)
                          and batch_size <= max(self._cuda_graphs, default=0))

        if is_pure_decode:
            # ==============================================================
            # CUDA Graph decode path — graph replay, minimal CPU overhead
            # ==============================================================
            block_size = self.block_manager.block_size
            capture_bs = min(b for b in self._cuda_graphs if b >= batch_size)

            kv_indices_list = []
            kv_indptr = [0]
            kv_last_page_lens = []
            write_indices_list = []

            for i, sr in enumerate(active):
                req = sr.request
                bt = self.scheduler.get_block_table(req.request_id)
                total_kv = req.current_len
                kv_indices_list.extend(bt.block_indices)
                kv_indptr.append(kv_indptr[-1] + len(bt.block_indices))
                lpl = total_kv % block_size
                kv_last_page_lens.append(lpl if lpl > 0 else block_size)
                # Write index for this decode token
                pos = cache_starts[i]
                page_idx = pos // block_size
                slot = pos % block_size
                write_indices_list.append(bt.block_indices[page_idx] * block_size + slot)

            # Pad to capture_bs with dummy requests
            for _ in range(capture_bs - batch_size):
                kv_indices_list.append(self._dummy_block_id)
                kv_indptr.append(kv_indptr[-1] + 1)
                kv_last_page_lens.append(1)

            # Copy real data into static buffers
            self._static_input_ids[:batch_size].copy_(input_ids)
            self._static_positions[:batch_size].copy_(positions)
            self._static_write_indices[:batch_size].copy_(
                torch.tensor(write_indices_list, dtype=torch.int64, device=self.device))
            # Pad static buffers
            if capture_bs > batch_size:
                self._static_input_ids[batch_size:capture_bs].fill_(0)
                self._static_positions[batch_size:capture_bs].fill_(0)
                self._static_write_indices[batch_size:capture_bs].fill_(self._dummy_flat_idx)

            # Plan and replay
            g, wrapper, static_logits = self._cuda_graphs[capture_bs]
            qo_indptr = torch.arange(capture_bs + 1, dtype=torch.int32, device=self.device)
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=torch.tensor(kv_indptr, dtype=torch.int32, device=self.device),
                paged_kv_indices=torch.tensor(kv_indices_list, dtype=torch.int32, device=self.device),
                paged_kv_last_page_len=torch.tensor(kv_last_page_lens, dtype=torch.int32, device=self.device),
                num_qo_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim_qk=self.config.head_dim,
                page_size=block_size,
                causal=True,
                q_data_type=self.dtype,
            )
            g.replay()
            logits = static_logits[:batch_size]

        elif self.use_flashinfer:
            # ==============================================================
            # FlashInfer paged attention path — zero-copy KV
            # ==============================================================
            block_size = self.block_manager.block_size

            kv_indices_list = []
            kv_page_counts = [0]
            kv_last_page_lens = []
            write_indices_list = []

            for i, sr in enumerate(active):
                req = sr.request
                bt = self.scheduler.get_block_table(req.request_id)
                start = cache_starts[i]
                num_input = q_lens[i]
                total_kv = start + num_input

                # Page table metadata for FlashInfer
                kv_indices_list.extend(bt.block_indices)
                kv_page_counts.append(kv_page_counts[-1] + len(bt.block_indices))
                lpl = total_kv % block_size
                kv_last_page_lens.append(lpl if lpl > 0 else block_size)

                # Compute flat write indices for KV scatter
                for t in range(num_input):
                    abs_pos = start + t
                    page_idx = abs_pos // block_size
                    slot = abs_pos % block_size
                    flat_idx = bt.block_indices[page_idx] * block_size + slot
                    write_indices_list.append(flat_idx)

            kv_indices = torch.tensor(kv_indices_list, dtype=torch.int32, device=self.device)
            kv_indptr = torch.tensor(kv_page_counts, dtype=torch.int32, device=self.device)
            kv_last_page_len = torch.tensor(kv_last_page_lens, dtype=torch.int32, device=self.device)
            write_indices = torch.tensor(write_indices_list, dtype=torch.int64, device=self.device)

            # Plan FlashInfer wrapper (shared across all layers)
            self._prefill_wrapper.plan(
                qo_indptr=cu_seqlens_q,
                paged_kv_indptr=kv_indptr,
                paged_kv_indices=kv_indices,
                paged_kv_last_page_len=kv_last_page_len,
                num_qo_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim_qk=self.config.head_dim,
                page_size=block_size,
                causal=True,
                q_data_type=self.dtype,
            )

            # Single forward pass — each layer writes KV in-place
            logits, _ = self.model.forward_packed(
                input_ids=input_ids,
                positions=positions,
                kv_cache=(self.block_manager.k_cache, self.block_manager.v_cache),
                attn_wrapper=self._prefill_wrapper,
                write_indices=write_indices,
                logit_indices=logit_indices,
            )
        else:
            # ==============================================================
            # Legacy path (flash_attn or SDPA with KV copy)
            # ==============================================================
            num_kv_heads = self.config.num_key_value_heads
            head_dim = self.config.head_dim

            k_lens = [cache_starts[i] + q_lens[i] for i in range(batch_size)]
            cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
            for i in range(batch_size):
                cu_seqlens_k[i + 1] = cu_seqlens_k[i] + k_lens[i]
            max_seqlen_q = max(q_lens)
            max_seqlen_k = max(k_lens)

            # Assemble cached KV per layer
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
                                k_parts.append(k_cached.transpose(0, 1))
                                v_parts.append(v_cached.transpose(0, 1))
                            else:
                                k_parts.append(torch.zeros(
                                    cached_len, num_kv_heads, head_dim,
                                    dtype=self.dtype, device=self.device))
                                v_parts.append(torch.zeros(
                                    cached_len, num_kv_heads, head_dim,
                                    dtype=self.dtype, device=self.device))
                    if k_parts:
                        cached_kvs.append((torch.cat(k_parts, dim=0),
                                           torch.cat(v_parts, dim=0)))
                    else:
                        cached_kvs.append((None, None))
            else:
                cached_kvs = [(None, None)] * num_layers

            # Forward pass
            logits, new_kvs = self.model.forward_packed(
                input_ids=input_ids,
                positions=positions,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                cached_kvs=cached_kvs,
                logit_indices=logit_indices,
            )

            # Write new KV to paged blocks
            q_offset = 0
            for i, sr in enumerate(active):
                req = sr.request
                num_input = q_lens[i]
                bt = self.scheduler.get_block_table(req.request_id)
                if bt is None:
                    q_offset += num_input
                    continue
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
                        self.block_manager.write_kv(
                            bt.block_indices[block_idx],
                            layer_idx,
                            slot_offset,
                            new_k[t:t + 1].transpose(0, 1),
                            new_v[t:t + 1].transpose(0, 1),
                        )
                q_offset += num_input

        # ------------------------------------------------------------------
        # Update prefill progress & sample tokens (common for both paths)
        # ------------------------------------------------------------------
        for i, sr in enumerate(active):
            req = sr.request
            num_input = q_lens[i]

            if sr.is_prefill:
                req.num_prefilled += num_input

            should_sample = (not sr.is_prefill) or (sr.is_prefill and not req.is_prefilling)

            if should_sample:
                # logits are (batch_size, vocab) thanks to logit_indices
                last_logit = logits[i:i + 1]

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

        # Drop local refs
        del logits, input_ids, positions, cu_seqlens_q

        if _debug_mem_env():
            self._step_count = getattr(self, "_step_count", 0) + 1
            if self._step_count % 10 == 0:
                a = torch.cuda.memory_allocated() / 1024**3
                r = torch.cuda.memory_reserved() / 1024**3
                print(f"[step {self._step_count}] alloc={a:.2f} GB  "
                      f"reserved={r:.2f} GB batch={batch_size} "
                      f"total_q={total_q}", flush=True)

    def _check_finished(self, req: Request, token_id: int) -> bool:
        if not req.sampling_params.ignore_eos and token_id == self.tokenizer.eos_token_id:
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
