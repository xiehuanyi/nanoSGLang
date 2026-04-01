"""
Tokenizer wrapper around HuggingFace transformers.

Provides a clean encode/decode interface and handles chat template formatting.
"""

from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Ensure we have pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self, messages: list[dict], add_generation_prompt: bool = True
    ) -> list[int]:
        """Convert chat messages to token ids using the model's chat template."""
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=True
        )
