"""
Speculative Decoding — Propose-Verify cycle.

Uses a small "draft" model to quickly generate K candidate tokens,
then the large "target" model verifies them in a single forward pass.

The verification accepts tokens where the target model agrees with the draft.
Rejected tokens are resampled from the corrected distribution.

Benefits: each step generates up to K+1 tokens with only 1 target model forward.
Trade-off: requires an additional small model and its KV cache.

Reference: "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
"""

import torch
import torch.nn.functional as F
from typing import Optional

from nano_sglang.engine.sampling import sample_token


class SpeculativeDecoder:
    """
    Speculative decoding with a draft + target model pair.

    Usage:
        spec = SpeculativeDecoder(draft_model, target_model, num_speculative=5)
        accepted_tokens = spec.speculative_step(
            input_ids, position_ids, draft_kv_caches, target_kv_caches, ...
        )
    """

    def __init__(
        self,
        draft_model,
        target_model,
        num_speculative: int = 5,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.num_speculative = num_speculative
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    @torch.inference_mode()
    def speculative_step(
        self,
        input_ids: torch.Tensor,        # (1, 1) last accepted token
        start_pos: int,
        draft_kv_caches,
        target_kv_caches,
        make_mask_fn,                    # fn(seq_len, full_len) -> mask
    ) -> list[int]:
        """
        Run one speculative decoding step.

        1. Draft model generates K candidate tokens autoregressively
        2. Target model scores all K+1 positions in one forward pass
        3. Accept/reject based on probability comparison

        Returns: list of accepted token ids (1 to K+1 tokens)
        """
        device = input_ids.device
        draft_tokens = []
        draft_probs = []

        # ---- Step 1: Draft model generates K candidates ----
        cur_input = input_ids
        cur_pos = start_pos

        for _ in range(self.num_speculative):
            position_ids = torch.tensor([[cur_pos]], device=device)
            attn_mask = make_mask_fn(1, cur_pos + 1)

            logits = self.draft_model(
                input_ids=cur_input,
                position_ids=position_ids,
                kv_caches=draft_kv_caches,
                cache_position=cur_pos,
                attention_mask=attn_mask,
            )

            # Get draft probability distribution
            if self.temperature > 0:
                probs = F.softmax(logits[:, -1, :] / self.temperature, dim=-1)
            else:
                probs = F.softmax(logits[:, -1, :], dim=-1)

            # Sample from draft
            token = sample_token(
                logits[:, -1, :],
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            token_id = token.item()

            draft_tokens.append(token_id)
            draft_probs.append(probs[0, token_id].item())

            cur_input = token.unsqueeze(0)
            cur_pos += 1

        # ---- Step 2: Target model scores all positions at once ----
        # Build input: [last_accepted, draft_0, draft_1, ..., draft_{K-1}]
        verify_ids = torch.tensor(
            [input_ids[0, 0].item()] + draft_tokens, device=device
        ).unsqueeze(0)

        verify_len = len(draft_tokens) + 1
        position_ids = torch.arange(
            start_pos, start_pos + verify_len, device=device
        ).unsqueeze(0)

        attn_mask = make_mask_fn(verify_len, start_pos + verify_len)

        target_logits = self.target_model(
            input_ids=verify_ids,
            position_ids=position_ids,
            kv_caches=target_kv_caches,
            cache_position=start_pos,
            attention_mask=attn_mask,
        )

        # ---- Step 3: Accept/reject ----
        accepted = []
        for i, draft_token in enumerate(draft_tokens):
            # Target probability for the draft token at position i
            if self.temperature > 0:
                target_probs = F.softmax(
                    target_logits[:, i, :] / self.temperature, dim=-1
                )
            else:
                target_probs = F.softmax(target_logits[:, i, :], dim=-1)

            p_target = target_probs[0, draft_token].item()
            p_draft = draft_probs[i]

            # Acceptance criterion: accept with probability min(1, p_target / p_draft)
            if p_draft > 0:
                accept_prob = min(1.0, p_target / p_draft)
            else:
                accept_prob = 1.0

            r = torch.rand(1).item()
            if r < accept_prob:
                accepted.append(draft_token)
            else:
                # Reject: sample from corrected distribution
                # p_corrected = max(0, p_target - p_draft) / sum(max(0, p_target - p_draft))
                corrected = torch.clamp(target_probs[0] - F.softmax(
                    target_logits[:, i, :] / max(self.temperature, 1e-6), dim=-1
                )[0] * (p_draft / max(p_target, 1e-10)), min=0)

                if corrected.sum() > 0:
                    corrected = corrected / corrected.sum()
                    resampled = torch.multinomial(corrected, 1).item()
                else:
                    resampled = sample_token(
                        target_logits[:, i, :],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                    ).item()

                accepted.append(resampled)
                break  # Stop at first rejection

        # If all K were accepted, sample one bonus token from the last position
        if len(accepted) == self.num_speculative:
            bonus = sample_token(
                target_logits[:, -1, :],
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            ).item()
            accepted.append(bonus)

        return accepted

    def reset_draft_cache(self, draft_kv_caches, accepted_len: int, start_pos: int):
        """
        After rejection, we need to roll back the draft KV cache
        to only include accepted tokens.

        For naive KV cache, this is a no-op since we just overwrite.
        For paged cache, we'd free the extra blocks.
        """
        # With naive cache, nothing to do — positions beyond accepted_len
        # will be overwritten in the next step.
        pass
