"""
Sampling strategies: greedy, temperature, top-k, top-p (nucleus).

Uses FlashInfer's fused sampling kernels when available for batch-efficient
GPU sampling. Falls back to PyTorch implementation.
"""

import torch
import torch.nn.functional as F

_FLASHINFER_SAMPLING = False
try:
    from flashinfer.sampling import (
        softmax as _fi_softmax,
        sampling_from_probs as _fi_sampling_from_probs,
        top_k_sampling_from_probs as _fi_top_k_sampling,
        top_p_sampling_from_probs as _fi_top_p_sampling,
        top_k_top_p_sampling_from_probs as _fi_top_k_top_p_sampling,
    )
    _FLASHINFER_SAMPLING = True
except ImportError:
    pass


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    Sample a token from logits.

    Args:
        logits: (batch, vocab_size) — logits for the last position
        temperature: sampling temperature. 0 = greedy.
        top_p: nucleus sampling threshold (1.0 = disabled)
        top_k: top-k filtering (0 = disabled)

    Returns:
        (batch,) sampled token ids
    """
    # Greedy
    if temperature == 0 or temperature < 1e-6:
        return logits.argmax(dim=-1)

    if _FLASHINFER_SAMPLING:
        return _sample_flashinfer(logits, temperature, top_p, top_k)

    return _sample_pytorch(logits, temperature, top_p, top_k)


def _sample_flashinfer(logits, temperature, top_p, top_k):
    """Batch sampling using FlashInfer's fused kernels."""
    temps = torch.full((logits.shape[0],), temperature, dtype=torch.float32, device=logits.device)
    probs = _fi_softmax(logits.float(), temps)

    use_top_k = top_k > 0
    use_top_p = top_p < 1.0

    if use_top_k and use_top_p:
        return _fi_top_k_top_p_sampling(probs, top_k_val=top_k, top_p_val=top_p)
    elif use_top_k:
        return _fi_top_k_sampling(probs, top_k_val=top_k)
    elif use_top_p:
        return _fi_top_p_sampling(probs, top_p_val=top_p)
    else:
        return _fi_sampling_from_probs(probs)


def _sample_pytorch(logits, temperature, top_p, top_k):
    """Fallback PyTorch sampling."""
    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_values = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth_values, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
