"""
Tensor Parallelism — Split model across multiple GPUs using NCCL.

Strategy:
  - Column parallel: split weight along output dim, each GPU computes a shard,
    all-gather the results.
  - Row parallel: split weight along input dim, each GPU computes partial result,
    all-reduce to sum.

For attention:
  - Q, K, V projections: column parallel (split heads across GPUs)
  - O projection: row parallel (each GPU has partial hidden, all-reduce)

For MLP (SwiGLU):
  - gate_proj, up_proj: column parallel
  - down_proj: row parallel

Embedding and LM head:
  - Embedding: each GPU holds full copy (or vocab parallel for very large models)
  - LM head: column parallel + all-gather logits
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist


def init_distributed(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    backend: str = "nccl",
):
    """Initialize distributed process group."""
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size <= 1:
        return rank, world_size

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

    torch.cuda.set_device(rank)
    return rank, world_size


class ColumnParallelLinear(nn.Module):
    """
    Linear layer split along the output dimension.

    Each GPU holds weight of shape (output_size // world_size, input_size).
    After forward, results are gathered if gather_output=True.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = False,
        gather_output: bool = False,
    ):
        super().__init__()
        assert out_features % world_size == 0
        self.out_features_per_rank = out_features // world_size
        self.world_size = world_size
        self.rank = rank
        self.gather_output = gather_output

        self.linear = nn.Linear(in_features, self.out_features_per_rank, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear(x)

        if self.gather_output and self.world_size > 1:
            # All-gather along output dimension
            gathered = [torch.empty_like(output) for _ in range(self.world_size)]
            dist.all_gather(gathered, output)
            output = torch.cat(gathered, dim=-1)

        return output

    def load_weight_shard(self, full_weight: torch.Tensor, full_bias: Optional[torch.Tensor] = None):
        """Load the shard for this rank from a full weight tensor."""
        shard = full_weight[
            self.rank * self.out_features_per_rank:
            (self.rank + 1) * self.out_features_per_rank
        ]
        self.linear.weight.data.copy_(shard)
        if full_bias is not None and self.linear.bias is not None:
            bias_shard = full_bias[
                self.rank * self.out_features_per_rank:
                (self.rank + 1) * self.out_features_per_rank
            ]
            self.linear.bias.data.copy_(bias_shard)


class RowParallelLinear(nn.Module):
    """
    Linear layer split along the input dimension.

    Each GPU holds weight of shape (output_size, input_size // world_size).
    After forward, results are all-reduced.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = False,
        reduce_output: bool = True,
    ):
        super().__init__()
        assert in_features % world_size == 0
        self.in_features_per_rank = in_features // world_size
        self.world_size = world_size
        self.rank = rank
        self.reduce_output = reduce_output

        self.linear = nn.Linear(self.in_features_per_rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear(x)

        if self.reduce_output and self.world_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)

        return output

    def load_weight_shard(self, full_weight: torch.Tensor, full_bias: Optional[torch.Tensor] = None):
        """Load the shard for this rank from a full weight tensor."""
        shard = full_weight[
            :,
            self.rank * self.in_features_per_rank:
            (self.rank + 1) * self.in_features_per_rank,
        ]
        self.linear.weight.data.copy_(shard)
        if full_bias is not None and self.linear.bias is not None:
            self.linear.bias.data.copy_(full_bias)


class TPAttention(nn.Module):
    """
    Tensor-parallel attention: Q/K/V are column-parallel, O is row-parallel.

    Each GPU handles (num_heads // world_size) attention heads.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        world_size: int,
        rank: int,
        bias: bool = False,
    ):
        super().__init__()
        assert num_heads % world_size == 0
        assert num_kv_heads % world_size == 0

        self.num_heads_per_rank = num_heads // world_size
        self.num_kv_heads_per_rank = num_kv_heads // world_size
        self.head_dim = head_dim
        self.world_size = world_size
        self.rank = rank

        self.q_proj = ColumnParallelLinear(
            hidden_size, num_heads * head_dim, world_size, rank, bias=bias
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, world_size, rank, bias=bias
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, world_size, rank, bias=bias
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim, hidden_size, world_size, rank, bias=False,
            reduce_output=True,
        )

    def forward(self, hidden_states, **kwargs):
        # Each GPU computes attention for its own head shard
        # Q: (batch, seq, num_heads_per_rank * head_dim)
        # K/V: (batch, seq, num_kv_heads_per_rank * head_dim)
        # After attention computation, O projection does all-reduce
        raise NotImplementedError(
            "Full TP attention forward requires integration with the main model. "
            "See tensor_parallel_model() for how to replace layers."
        )


class TPMLP(nn.Module):
    """
    Tensor-parallel MLP: gate/up are column-parallel, down is row-parallel.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        world_size: int,
        rank: int,
    ):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, world_size, rank
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, world_size, rank
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, world_size, rank,
            reduce_output=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


def tensor_parallel_model(model, world_size: int, rank: int):
    """
    Replace model layers with tensor-parallel versions.

    This modifies the model in-place, replacing Linear layers in attention
    and MLP with their TP counterparts.
    """
    if world_size <= 1:
        return model

    config = model.config

    for layer in model.layers:
        # Replace MLP
        old_mlp = layer.mlp
        tp_mlp = TPMLP(
            config.hidden_size, config.intermediate_size,
            world_size, rank,
        )
        # Load sharded weights
        tp_mlp.gate_proj.load_weight_shard(old_mlp.gate_proj.weight.data)
        tp_mlp.up_proj.load_weight_shard(old_mlp.up_proj.weight.data)
        tp_mlp.down_proj.load_weight_shard(old_mlp.down_proj.weight.data)
        layer.mlp = tp_mlp

        # Replace attention projections
        attn = layer.self_attn
        old_q = attn.q_proj
        old_k = attn.k_proj
        old_v = attn.v_proj
        old_o = attn.o_proj

        bias = old_q.bias is not None

        new_q = ColumnParallelLinear(
            config.hidden_size, config.num_attention_heads * config.head_dim,
            world_size, rank, bias=bias,
        )
        new_k = ColumnParallelLinear(
            config.hidden_size, config.num_key_value_heads * config.head_dim,
            world_size, rank, bias=bias,
        )
        new_v = ColumnParallelLinear(
            config.hidden_size, config.num_key_value_heads * config.head_dim,
            world_size, rank, bias=bias,
        )
        new_o = RowParallelLinear(
            config.num_attention_heads * config.head_dim, config.hidden_size,
            world_size, rank, reduce_output=True,
        )

        new_q.load_weight_shard(old_q.weight.data, getattr(old_q, 'bias', None) and old_q.bias.data)
        new_k.load_weight_shard(old_k.weight.data, getattr(old_k, 'bias', None) and old_k.bias.data)
        new_v.load_weight_shard(old_v.weight.data, getattr(old_v, 'bias', None) and old_v.bias.data)
        new_o.load_weight_shard(old_o.weight.data)

        attn.q_proj = new_q
        attn.k_proj = new_k
        attn.v_proj = new_v
        attn.o_proj = new_o

        # Update head counts for the attention module
        attn.num_heads = config.num_attention_heads // world_size
        attn.num_kv_heads = config.num_key_value_heads // world_size

    return model
