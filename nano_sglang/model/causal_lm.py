"""
Unified Llama/Qwen2 model definition from scratch in PyTorch.

Two forward modes:
  1. forward()       — batch mode (batch, seq_len), for naive single-request
  2. forward_packed() — packed/ragged mode (total_tokens,), for continuous batching

Two attention backends:
  1. Naive: torch.matmul + softmax (any GPU)
  2. FlashAttention: flash_attn_varlen_func (SM >= 80, A100/H100)

Automatically selects the best backend based on GPU capability.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


# ---------------------------------------------------------------------------
# FlashAttention availability detection
# ---------------------------------------------------------------------------

_FLASH_ATTN_AVAILABLE = False
_GPU_SM_VERSION = 0

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    pass

_FLASHINFER_AVAILABLE = False
_FLASHINFER_NORM_AVAILABLE = False
try:
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    _FLASHINFER_AVAILABLE = True
    from flashinfer.norm import rmsnorm as _fi_rmsnorm, fused_add_rmsnorm as _fi_fused_add_rmsnorm
    _FLASHINFER_NORM_AVAILABLE = True
except ImportError:
    pass


def get_gpu_sm_version(device_id: int = 0) -> int:
    """Get GPU compute capability (SM version * 10, e.g. A100 = 80)."""
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability(device_id)
    return major * 10 + minor


def can_use_flash_attn(device: str = "cuda") -> bool:
    """Check if FlashAttention can be used on this GPU."""
    if not _FLASH_ATTN_AVAILABLE:
        return False
    if not torch.cuda.is_available():
        return False
    device_id = 0
    if ":" in device:
        device_id = int(device.split(":")[1])
    sm = get_gpu_sm_version(device_id)
    return sm >= 80  # FlashAttention 2 requires SM >= 80 (A100+)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    model_type: str = "llama"
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    torch_dtype: str = "float16"
    attention_bias: bool = False
    sliding_window: Optional[int] = None
    use_sliding_window: bool = False

    def __post_init__(self):
        if self.attention_bias is None:
            self.attention_bias = self.model_type == "qwen2"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_pretrained(cls, model_path: str) -> "ModelConfig":
        with open(Path(model_path) / "config.json") as f:
            raw = json.load(f)
        model_type = raw.get("model_type", "llama")
        return cls(
            model_type=model_type,
            vocab_size=raw.get("vocab_size", 32000),
            hidden_size=raw.get("hidden_size", 4096),
            intermediate_size=raw.get("intermediate_size", 11008),
            num_hidden_layers=raw.get("num_hidden_layers", 32),
            num_attention_heads=raw.get("num_attention_heads", 32),
            num_key_value_heads=raw.get("num_key_value_heads",
                                        raw.get("num_attention_heads", 32)),
            max_position_embeddings=raw.get("max_position_embeddings", 4096),
            rms_norm_eps=raw.get("rms_norm_eps", 1e-5),
            rope_theta=raw.get("rope_theta", 10000.0),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            torch_dtype=raw.get("torch_dtype", "float16"),
            attention_bias=raw.get("attention_bias"),
            sliding_window=raw.get("sliding_window", None),
            use_sliding_window=raw.get("use_sliding_window", False),
        )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _FLASHINFER_NORM_AVAILABLE:
            return _fi_rmsnorm(x, self.weight, self.eps)
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)

    def forward_fused_residual(self, x: torch.Tensor, residual: torch.Tensor):
        """Fused: residual += x; x = rmsnorm(residual). Both modified in-place."""
        if _FLASHINFER_NORM_AVAILABLE:
            _fi_fused_add_rmsnorm(x, residual, self.weight, self.eps)
            return x, residual
        # Fallback
        residual = residual + x
        return self.forward(residual), residual


def precompute_rope_freqs(
    head_dim: int, max_seq_len: int, theta: float = 10000.0,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE using the HuggingFace (Llama/Qwen2) half-split convention:
    dim i is paired with dim i + head_dim/2, not adjacent pairs.

    cos/sin are precomputed as (max_seq_len, head_dim/2); we concat each
    to head_dim via duplication so the broadcast matches the x layout.

    Works for both batched (batch, heads, seq, dim) and flat
    (total_tokens, heads, dim) shapes.
    """
    orig_dtype = x.dtype
    head_dim = x.shape[-1]
    half = head_dim // 2

    # cos[position_ids]: (..., head_dim/2)   →   cat with self → (..., head_dim)
    cos = cos[position_ids]
    sin = sin[position_ids]
    cos = torch.cat([cos, cos], dim=-1).to(orig_dtype)
    sin = torch.cat([sin, sin], dim=-1).to(orig_dtype)

    if x.dim() == 4:
        # position_ids: (batch, seq) → cos: (batch, seq, head_dim)
        # x: (batch, heads, seq, head_dim)
        cos = cos.unsqueeze(1)  # (batch, 1, seq, head_dim)
        sin = sin.unsqueeze(1)
    else:
        # position_ids: (total,) → cos: (total, head_dim)
        # x: (total, heads, head_dim)
        cos = cos.unsqueeze(1)  # (total, 1, head_dim)
        sin = sin.unsqueeze(1)

    # rotate_half(x): [-x2, x1] where x1 = x[..., :half], x2 = x[..., half:]
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rotated * sin).to(orig_dtype)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, use_flash: bool = False):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.use_flash = use_flash
        bias = config.attention_bias

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batched forward: (batch, seq_len, hidden_size)."""
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin, position_ids)
        k = apply_rope(k, cos, sin, position_ids)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_cache[:bsz, :, cache_position:cache_position + seq_len, :] = k
            v_cache[:bsz, :, cache_position:cache_position + seq_len, :] = v
            k = k_cache[:bsz, :, :cache_position + seq_len, :]
            v = v_cache[:bsz, :, :cache_position + seq_len, :]

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)

    def forward_packed(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        # FlashInfer paged attention params (zero-copy path)
        layer_idx: Optional[int] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attn_wrapper: Optional[object] = None,
        write_indices: Optional[torch.Tensor] = None,
        # Legacy attention params (fallback path)
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        cached_k: Optional[torch.Tensor] = None,
        cached_v: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Packed/ragged forward for continuous batching.

        Two paths:
          1. FlashInfer (kv_cache != None): write K/V directly to paged cache
             via index_put, then run paged attention. Zero copy, no clone.
          2. Legacy (kv_cache is None): clone K/V, concat with cached, run
             flash_attn_varlen or SDPA. Returns new_k/new_v for engine to write.
        """
        total_q = hidden_states.shape[0]

        # Project Q/K/V
        q = self.q_proj(hidden_states).view(total_q, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(total_q, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(total_q, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, cos, sin, positions)
        k = apply_rope(k, cos, sin, positions)

        # --- FlashInfer paged attention path ---
        if _FLASHINFER_AVAILABLE and kv_cache is not None:
            return self._flashinfer_paged_attention(
                q, k, v, layer_idx, kv_cache, attn_wrapper, write_indices,
            )

        # --- Legacy path: clone + concat + flash_attn/SDPA ---
        new_k = k.clone()
        new_v = v.clone()

        if cached_k is not None and cached_k.shape[0] > 0:
            k = torch.cat([cached_k, k], dim=0)
            v = torch.cat([cached_v, v], dim=0)

        if self.use_flash and _FLASH_ATTN_AVAILABLE:
            attn_output = self._flash_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
            )
        else:
            attn_output = self._sdpa_packed_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
            )

        output = self.o_proj(attn_output.view(total_q, -1))
        return output, new_k, new_v

    def _flashinfer_paged_attention(
        self, q, k, v, layer_idx, kv_cache, wrapper, write_indices,
    ) -> tuple[torch.Tensor, None, None]:
        """FlashInfer paged attention: write K/V in-place, attend via page table."""
        k_cache, v_cache = kv_cache
        # Batch-write new K/V into paged cache using advanced indexing
        k_flat = k_cache[layer_idx].view(-1, self.num_kv_heads, self.head_dim)
        v_flat = v_cache[layer_idx].view(-1, self.num_kv_heads, self.head_dim)
        k_flat[write_indices] = k
        v_flat[write_indices] = v
        # Run paged attention (reads from cache via page table, set up in plan())
        attn_output = wrapper.run(q, (k_cache[layer_idx], v_cache[layer_idx]))
        output = self.o_proj(attn_output.view(q.shape[0], -1))
        return output, None, None

    def _flash_attention(
        self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    ) -> torch.Tensor:
        """FlashAttention 2 varlen kernel — supports GQA natively."""
        # flash_attn_varlen_func handles GQA when num_heads_q != num_heads_k
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=True,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
        )

    def _sdpa_packed_attention(
        self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    ) -> torch.Tensor:
        """
        Packed attention using torch.nn.functional.scaled_dot_product_attention.

        SDPA auto-dispatches to the best available backend (FlashAttention 2
        kernel if bf16/fp16 and SM >= 80, else mem_efficient, else math).
        We still have to loop per-request because SDPA is dense, not varlen,
        but each call is one fused kernel instead of matmul + softmax + mask.

        GQA is handled natively by SDPA (torch >= 2.5) via enable_gqa=True.
        """
        batch_size = cu_seqlens_q.shape[0] - 1
        outputs = []
        scale = 1.0 / math.sqrt(self.head_dim)

        for i in range(batch_size):
            q_start = cu_seqlens_q[i].item()
            q_end = cu_seqlens_q[i + 1].item()
            k_start = cu_seqlens_k[i].item()
            k_end = cu_seqlens_k[i + 1].item()

            # (seq, heads, dim) → (1, heads, seq, dim) as SDPA expects
            qi = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
            ki = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
            vi = v[k_start:k_end].transpose(0, 1).unsqueeze(0)

            seq_q = q_end - q_start
            seq_k = k_end - k_start

            if seq_q == 1:
                # Decode: attend to all cached tokens, no mask
                out = F.scaled_dot_product_attention(
                    qi, ki, vi, is_causal=False, scale=scale, enable_gqa=True,
                )
            elif seq_q == seq_k:
                # Full prefill, no cached prefix
                out = F.scaled_dot_product_attention(
                    qi, ki, vi, is_causal=True, scale=scale, enable_gqa=True,
                )
            else:
                # Chunked prefill with cached prefix: q attends to all cached
                # tokens + its own past, but not to future q tokens.
                offset = seq_k - seq_q
                mask = torch.zeros(
                    seq_q, seq_k, device=qi.device, dtype=qi.dtype,
                )
                # row r: allowed cols [0, offset + r]  (inclusive)
                row = torch.arange(seq_q, device=qi.device).unsqueeze(1)
                col = torch.arange(seq_k, device=qi.device).unsqueeze(0)
                mask = torch.where(
                    col <= (offset + row), 0.0, float("-inf"),
                ).to(qi.dtype)
                out = F.scaled_dot_product_attention(
                    qi, ki, vi, attn_mask=mask, scale=scale, enable_gqa=True,
                )

            # (1, heads, seq_q, dim) → (seq_q, heads, dim)
            out = out.squeeze(0).transpose(0, 1)
            outputs.append(out)

        return torch.cat(outputs, dim=0)

    def _naive_packed_attention(
        self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    ) -> torch.Tensor:
        """
        Naive attention for packed sequences (old fallback, kept for reference).

        Processes each sequence in the batch separately.
        """
        batch_size = cu_seqlens_q.shape[0] - 1
        outputs = []

        for i in range(batch_size):
            q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
            k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()

            qi = q[q_start:q_end]   # (seq_q, num_heads, head_dim)
            ki = k[k_start:k_end]   # (seq_k, num_kv_heads, head_dim)
            vi = v[k_start:k_end]

            # Expand KV heads for GQA
            if self.num_kv_groups > 1:
                ki = ki.repeat_interleave(self.num_kv_groups, dim=1)
                vi = vi.repeat_interleave(self.num_kv_groups, dim=1)

            # (num_heads, seq_q, head_dim) @ (num_heads, head_dim, seq_k) → (num_heads, seq_q, seq_k)
            qi_t = qi.transpose(0, 1)
            ki_t = ki.transpose(0, 1)
            vi_t = vi.transpose(0, 1)

            scores = torch.matmul(qi_t, ki_t.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Causal mask
            seq_q = q_end - q_start
            seq_k = k_end - k_start
            if seq_q > 1:
                # Prefill: standard causal mask, offset for cached tokens
                offset = seq_k - seq_q
                mask = torch.full((seq_q, seq_k), float("-inf"),
                                  device=scores.device, dtype=scores.dtype)
                for r in range(seq_q):
                    mask[r, :offset + r + 1] = 0.0
                scores = scores + mask.unsqueeze(0)
            # Decode (seq_q == 1): no mask needed, attend to everything

            attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(qi.dtype)
            out_i = torch.matmul(attn, vi_t)  # (num_heads, seq_q, head_dim)
            outputs.append(out_i.transpose(0, 1))  # (seq_q, num_heads, head_dim)

        return torch.cat(outputs, dim=0)  # (total_q, num_heads, head_dim)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, use_flash: bool = False):
        super().__init__()
        self.self_attn = Attention(config, use_flash=use_flash)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin, position_ids,
                kv_cache=None, cache_position=None, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, cos, sin, position_ids, kv_cache, cache_position, attention_mask
        )
        # Fused residual + norm
        hidden_states, residual = self.post_attention_layernorm.forward_fused_residual(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_packed(self, hidden_states, cos, sin, positions,
                       layer_idx=None,
                       kv_cache=None, attn_wrapper=None, write_indices=None,
                       cu_seqlens_q=None, cu_seqlens_k=None,
                       max_seqlen_q=0, max_seqlen_k=0,
                       cached_k=None, cached_v=None):
        """Packed forward. Returns (output, new_k, new_v)."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_k, new_v = self.self_attn.forward_packed(
            hidden_states, cos, sin, positions,
            layer_idx=layer_idx,
            kv_cache=kv_cache, attn_wrapper=attn_wrapper,
            write_indices=write_indices,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            cached_k=cached_k, cached_v=cached_v,
        )
        # Fused: residual += hidden_states; hidden_states = rmsnorm(residual)
        hidden_states, residual = self.post_attention_layernorm.forward_fused_residual(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, new_k, new_v


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig, use_flash: bool = False):
        super().__init__()
        self.config = config
        self.use_flash = use_flash
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, use_flash=use_flash)
             for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.rope_cos, self.rope_sin = precompute_rope_freqs(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.rope_cos = self.rope_cos.to(device)
        self.rope_sin = self.rope_sin.to(device)
        return result

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        cache_position: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batched forward: (batch, seq_len) → (batch, seq_len, vocab_size)."""
        hidden_states = self.embed_tokens(input_ids)

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states = layer(
                hidden_states, self.rope_cos, self.rope_sin,
                position_ids, kv_cache, cache_position, attention_mask,
            )

        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    def forward_packed(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        # FlashInfer params (zero-copy path)
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attn_wrapper: Optional[object] = None,
        write_indices: Optional[torch.Tensor] = None,
        logit_indices: Optional[torch.Tensor] = None,
        # Legacy params (fallback path)
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        cached_kvs: Optional[list[tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = None,
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Packed/ragged forward for continuous batching.

        Two paths:
          1. FlashInfer (kv_cache != None): each layer writes K/V directly to
             the paged cache and runs paged attention. No new_kvs returned.
          2. Legacy (kv_cache is None): each layer returns new_k/new_v for the
             engine to write to blocks.

        logit_indices: if provided, only compute lm_head for these token
            positions (one per request), reducing vocab projection cost.
        """
        hidden_states = self.embed_tokens(input_ids)

        if kv_cache is not None:
            # --- FlashInfer path: K/V written in-place by each layer ---
            for i, layer in enumerate(self.layers):
                hidden_states, _, _ = layer.forward_packed(
                    hidden_states, self.rope_cos, self.rope_sin, positions,
                    layer_idx=i,
                    kv_cache=kv_cache, attn_wrapper=attn_wrapper,
                    write_indices=write_indices,
                )
            new_kvs = None
        else:
            # --- Legacy path ---
            new_kvs = []
            for i, layer in enumerate(self.layers):
                cached_k, cached_v = (None, None)
                if cached_kvs is not None and cached_kvs[i][0] is not None:
                    cached_k, cached_v = cached_kvs[i]

                hidden_states, new_k, new_v = layer.forward_packed(
                    hidden_states, self.rope_cos, self.rope_sin, positions,
                    cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                    cached_k=cached_k, cached_v=cached_v,
                )
                new_kvs.append((new_k, new_v))

        hidden_states = self.norm(hidden_states)
        # Only compute logits for needed positions (last token per request)
        if logit_indices is not None:
            hidden_states = hidden_states[logit_indices]
        logits = self.lm_head(hidden_states)
        return logits, new_kvs


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

HF_TO_NANO = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
}

LAYER_SUFFIXES = [
    "self_attn.q_proj.weight", "self_attn.q_proj.bias",
    "self_attn.k_proj.weight", "self_attn.k_proj.bias",
    "self_attn.v_proj.weight", "self_attn.v_proj.bias",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "input_layernorm.weight", "post_attention_layernorm.weight",
]


def _build_hf_to_nano_map(num_layers: int) -> dict[str, str]:
    mapping = dict(HF_TO_NANO)
    for i in range(num_layers):
        for suffix in LAYER_SUFFIXES:
            mapping[f"model.layers.{i}.{suffix}"] = f"layers.{i}.{suffix}"
    return mapping


def _resolve_model_path(model_path: str) -> Path:
    p = Path(model_path)
    if p.is_dir() and list(p.glob("*.safetensors")):
        return p
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(model_path)
        return Path(local_dir)
    except Exception:
        pass
    return p


def load_model_from_pretrained(
    model_path: str,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
) -> tuple[CausalLM, ModelConfig]:
    resolved_path = _resolve_model_path(model_path)
    config = ModelConfig.from_pretrained(str(resolved_path))

    if dtype is None:
        dtype_map = {
            "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
        }
        dtype = dtype_map.get(config.torch_dtype, torch.float16)

    # Detect FlashAttention support
    use_flash = can_use_flash_attn(device)

    with torch.device("meta"):
        model = CausalLM(config, use_flash=use_flash)

    name_map = _build_hf_to_nano_map(config.num_hidden_layers)

    shard_files = sorted(resolved_path.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {resolved_path}")

    state_dict = {}
    for shard_file in shard_files:
        with safe_open(str(shard_file), framework="pt", device="cpu") as f:
            for hf_name in f.keys():
                if hf_name in name_map:
                    state_dict[name_map[hf_name]] = f.get_tensor(hf_name).to(dtype)

    if config.tie_word_embeddings and "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]

    model.load_state_dict(state_dict, assign=True)

    model.rope_cos, model.rope_sin = precompute_rope_freqs(
        config.head_dim, config.max_position_embeddings, config.rope_theta,
        device=device,
    )

    model = model.to(device)
    model.eval()

    attn_backend = "FlashAttention 2" if use_flash else "torch SDPA"
    print(f"Loaded {config.model_type} model from {resolved_path}")
    print(f"  {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
          f"vocab={config.vocab_size}")
    print(f"  dtype={dtype}, device={device}, attention={attn_backend}")

    return model, config
