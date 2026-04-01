"""
Quantization — FP8 and AWQ (Activation-aware Weight Quantization).

Provides utilities to:
  1. Quantize model weights to lower precision (FP8, INT4/AWQ)
  2. Replace nn.Linear layers with quantized versions
  3. Load pre-quantized checkpoints

FP8 (E4M3): supported natively on H100/Ada GPUs via torch.float8_e4m3fn
AWQ (INT4): group-wise quantization with scale/zero-point per group
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# FP8 Quantization
# ---------------------------------------------------------------------------


class FP8Linear(nn.Module):
    """
    Linear layer with FP8 weight storage.

    Weights are stored in float8_e4m3fn format and dequantized to compute dtype
    at runtime. On H100, this can use native FP8 matmul for ~2x speedup.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # FP8 weight + scale factor
        self.register_buffer(
            "weight_fp8",
            torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn),
        )
        self.register_buffer("weight_scale", torch.ones(1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weight
        weight = self.weight_fp8.to(x.dtype) * self.weight_scale
        return F.linear(x, weight, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FP8Linear":
        """Convert a regular Linear to FP8."""
        has_bias = linear.bias is not None
        fp8_linear = cls(linear.in_features, linear.out_features, bias=has_bias)

        # Quantize weight to FP8
        weight = linear.weight.data.float()
        abs_max = weight.abs().max()
        # FP8 E4M3 max value is 448.0
        scale = abs_max / 448.0
        if scale == 0:
            scale = torch.ones(1)

        fp8_weight = (weight / scale).to(torch.float8_e4m3fn)
        fp8_linear.weight_fp8 = fp8_weight
        fp8_linear.weight_scale = scale.to(weight.device)

        if has_bias:
            fp8_linear.bias = nn.Parameter(linear.bias.data.clone())

        return fp8_linear


# ---------------------------------------------------------------------------
# AWQ (INT4) Quantization
# ---------------------------------------------------------------------------


class AWQLinear(nn.Module):
    """
    INT4 group-wise quantized linear layer (AWQ style).

    Weights are packed as INT4 (two values per byte) with per-group
    scale and zero-point factors.

    group_size: number of input features per quantization group (default 128)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = (in_features + group_size - 1) // group_size

        # Packed INT4 weights: each int32 holds 8 INT4 values
        self.register_buffer(
            "qweight",
            torch.zeros(out_features, in_features // 8, dtype=torch.int32),
        )
        # Per-group scales and zeros
        self.register_buffer(
            "scales",
            torch.ones(out_features, self.num_groups, dtype=torch.float16),
        )
        self.register_buffer(
            "zeros",
            torch.zeros(out_features, self.num_groups, dtype=torch.float16),
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def _dequantize(self) -> torch.Tensor:
        """Unpack INT4 weights and dequantize to float16."""
        # Unpack int32 -> 8x int4
        weight = torch.zeros(
            self.out_features, self.in_features,
            dtype=self.scales.dtype, device=self.scales.device,
        )

        for i in range(8):
            # Extract 4-bit values
            int4_vals = (self.qweight >> (i * 4)) & 0xF
            weight[:, i::8] = int4_vals.to(self.scales.dtype)

        # Apply per-group scale and zero-point
        for g in range(self.num_groups):
            start = g * self.group_size
            end = min(start + self.group_size, self.in_features)
            weight[:, start:end] = (
                (weight[:, start:end] - self.zeros[:, g:g + 1])
                * self.scales[:, g:g + 1]
            )

        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize()
        return F.linear(x, weight, self.bias)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, group_size: int = 128
    ) -> "AWQLinear":
        """Quantize a regular Linear to AWQ INT4."""
        has_bias = linear.bias is not None
        awq = cls(linear.in_features, linear.out_features, group_size, has_bias)

        weight = linear.weight.data.float()

        # Per-group quantization
        for g in range(awq.num_groups):
            start = g * group_size
            end = min(start + group_size, linear.in_features)
            group = weight[:, start:end]

            gmin = group.min(dim=1, keepdim=True).values
            gmax = group.max(dim=1, keepdim=True).values

            scale = (gmax - gmin) / 15.0  # INT4: 0-15
            scale = scale.clamp(min=1e-8)
            zero = (-gmin / scale).round()

            awq.scales[:, g] = scale.squeeze().half()
            awq.zeros[:, g] = zero.squeeze().half()

            # Quantize
            quantized = ((group - gmin) / scale).round().clamp(0, 15).to(torch.int32)

            # Pack into int32
            for i in range(end - start):
                col = start + i
                byte_idx = col // 8
                bit_offset = col % 8
                if byte_idx < awq.qweight.shape[1]:
                    awq.qweight[:, byte_idx] |= quantized[:, i] << (bit_offset * 4)

        if has_bias:
            awq.bias = nn.Parameter(linear.bias.data.clone())

        return awq


# ---------------------------------------------------------------------------
# Model quantization utility
# ---------------------------------------------------------------------------


def quantize_model(
    model: nn.Module,
    method: str = "fp8",
    group_size: int = 128,
    skip_modules: Optional[list[str]] = None,
) -> nn.Module:
    """
    Quantize all nn.Linear layers in a model.

    Args:
        model: the model to quantize
        method: "fp8" or "awq"
        group_size: group size for AWQ (ignored for FP8)
        skip_modules: list of module name patterns to skip

    Returns:
        The quantized model (modified in-place)
    """
    if skip_modules is None:
        skip_modules = ["embed_tokens", "lm_head", "norm"]

    for name, module in model.named_modules():
        if any(skip in name for skip in skip_modules):
            continue

        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{child_name}" if name else child_name
                if any(skip in full_name for skip in skip_modules):
                    continue

                if method == "fp8":
                    quantized = FP8Linear.from_linear(child)
                elif method == "awq":
                    quantized = AWQLinear.from_linear(child, group_size)
                else:
                    raise ValueError(f"Unknown quantization method: {method}")

                setattr(module, child_name, quantized)
                print(f"  Quantized {full_name} ({method})")

    return model
