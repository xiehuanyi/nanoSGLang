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
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


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
    Apply RoPE. Works for both batched (batch, heads, seq, dim) and
    flat (total_tokens, heads, dim) shapes.
    """
    orig_dtype = x.dtype

    if x.dim() == 4:
        # Batched: (batch, heads, seq, dim)
        cos = cos[position_ids].unsqueeze(1).to(orig_dtype)
        sin = sin[position_ids].unsqueeze(1).to(orig_dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)
    else:
        # Flat: (total_tokens, heads, dim)
        cos = cos[position_ids].unsqueeze(1).to(orig_dtype)  # (total_tokens, 1, dim/2)
        sin = sin[position_ids].unsqueeze(1).to(orig_dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)


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
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        cached_k: Optional[torch.Tensor] = None,
        cached_v: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Packed/ragged forward for continuous batching.

        Args:
            hidden_states: (total_q_tokens, hidden_size)
            positions:     (total_q_tokens,)
            cu_seqlens_q:  (batch_size + 1,) cumulative Q lengths
            cu_seqlens_k:  (batch_size + 1,) cumulative K/V lengths (including cache)
            max_seqlen_q:  max Q sequence length in this batch
            max_seqlen_k:  max K/V sequence length in this batch
            cached_k:      (total_cached_tokens, num_kv_heads, head_dim) or None
            cached_v:      (total_cached_tokens, num_kv_heads, head_dim) or None

        Returns:
            (output, new_k, new_v)
            output: (total_q_tokens, hidden_size)
            new_k:  (total_q_tokens, num_kv_heads, head_dim) — new K to cache
            new_v:  (total_q_tokens, num_kv_heads, head_dim) — new V to cache
        """
        total_q = hidden_states.shape[0]

        # Project Q/K/V
        q = self.q_proj(hidden_states).view(total_q, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(total_q, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(total_q, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, cos, sin, positions)
        k = apply_rope(k, cos, sin, positions)

        # Save new K/V for cache update
        new_k = k.clone()
        new_v = v.clone()

        # Prepend cached K/V if available
        if cached_k is not None and cached_k.shape[0] > 0:
            k = torch.cat([cached_k, k], dim=0)
            v = torch.cat([cached_v, v], dim=0)

        if self.use_flash and _FLASH_ATTN_AVAILABLE:
            attn_output = self._flash_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
            )
        else:
            attn_output = self._naive_packed_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
            )

        output = self.o_proj(attn_output.view(total_q, -1))
        return output, new_k, new_v

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

    def _naive_packed_attention(
        self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    ) -> torch.Tensor:
        """
        Naive attention for packed sequences (fallback when FlashAttention unavailable).

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
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_packed(self, hidden_states, cos, sin, positions,
                       cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       cached_k=None, cached_v=None):
        """Packed forward. Returns (output, new_k, new_v)."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_k, new_v = self.self_attn.forward_packed(
            hidden_states, cos, sin, positions,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            cached_k, cached_v,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
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
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        cached_kvs: Optional[list[tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Packed/ragged forward for continuous batching.

        Args:
            input_ids:   (total_tokens,) flat token ids
            positions:   (total_tokens,) position ids
            cu_seqlens_q: (batch+1,) cumulative Q token counts
            cu_seqlens_k: (batch+1,) cumulative K/V token counts (incl. cache)
            max_seqlen_q: max Q length
            max_seqlen_k: max K/V length
            cached_kvs:  per-layer list of (cached_k, cached_v) flat tensors,
                         or None if no cache

        Returns:
            (logits, new_kvs)
            logits:  (total_tokens, vocab_size)
            new_kvs: per-layer list of (new_k, new_v) for cache update
        """
        hidden_states = self.embed_tokens(input_ids)

        new_kvs = []
        for i, layer in enumerate(self.layers):
            cached_k, cached_v = (None, None)
            if cached_kvs is not None and cached_kvs[i][0] is not None:
                cached_k, cached_v = cached_kvs[i]

            hidden_states, new_k, new_v = layer.forward_packed(
                hidden_states, self.rope_cos, self.rope_sin, positions,
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                cached_k, cached_v,
            )
            new_kvs.append((new_k, new_v))

        hidden_states = self.norm(hidden_states)
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

    attn_backend = "FlashAttention 2" if use_flash else "naive (torch)"
    print(f"Loaded {config.model_type} model from {resolved_path}")
    print(f"  {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
          f"vocab={config.vocab_size}")
    print(f"  dtype={dtype}, device={device}, attention={attn_backend}")

    return model, config
