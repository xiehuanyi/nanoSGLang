"""
Sampling strategies: greedy, temperature, top-k, top-p (nucleus).
"""

import torch
import torch.nn.functional as F


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

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # Zero out everything below the top-k threshold
        kth_values = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth_values, float("-inf"))

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        # Shift right so the first token above threshold is kept
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")

        # Scatter back to original ordering
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
