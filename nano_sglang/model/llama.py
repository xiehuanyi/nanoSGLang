"""
Unified Llama/Qwen2 model definition from scratch in PyTorch.

Supports both Llama and Qwen2.5 architectures:
  - RMSNorm
  - Multi-Head / Grouped-Query Attention with RoPE
  - Optional attention bias (Qwen2)
  - Optional sliding window attention (Qwen2)
  - SwiGLU FFN

Loads weights from HuggingFace safetensors checkpoints.
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
# Config — unified for Llama / Qwen2
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    model_type: str = "llama"  # "llama" or "qwen2"
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
    attention_bias: bool = False       # Qwen2 uses bias in QKV
    sliding_window: Optional[int] = None  # Qwen2 sliding window
    use_sliding_window: bool = False

    def __post_init__(self):
        # Ensure attention_bias is bool (config.json may have null)
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
    orig_dtype = x.dtype
    cos = cos[position_ids].unsqueeze(1).to(orig_dtype)
    sin = sin[position_ids].unsqueeze(1).to(orig_dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
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


class MLP(nn.Module):
    """SwiGLU FFN."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache=None,
        cache_position=None,
        attention_mask=None,
    ) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class CausalLM(nn.Module):
    """Unified causal LM for Llama / Qwen2 architectures."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        hidden_states = self.embed_tokens(input_ids)

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states = layer(
                hidden_states, self.rope_cos, self.rope_sin,
                position_ids, kv_cache, cache_position, attention_mask,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


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
    """Resolve a HF hub name or local path to a local directory with safetensors."""
    p = Path(model_path)
    if p.is_dir() and list(p.glob("*.safetensors")):
        return p
    # Try huggingface_hub snapshot_download
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

    with torch.device("meta"):
        model = CausalLM(config)

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

    # Recompute RoPE (not part of state_dict, still on meta device)
    model.rope_cos, model.rope_sin = precompute_rope_freqs(
        config.head_dim, config.max_position_embeddings, config.rope_theta,
        device=device,
    )

    model = model.to(device)
    model.eval()

    print(f"Loaded {config.model_type} model from {resolved_path}")
    print(f"  {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
          f"vocab={config.vocab_size}")
    print(f"  dtype={dtype}, device={device}")

    return model, config
