"""
Structured Output — FSM-based token masking for JSON schema compliance.

Implements a finite state machine that tracks the generation state
and produces a token mask to constrain the model's output to valid JSON
conforming to a given schema.

States:
  START         → expecting '{'
  OBJECT_KEY    → expecting '"key"'
  COLON         → expecting ':'
  VALUE         → expecting a value (string, number, bool, null, object, array)
  STRING_VALUE  → inside a string value
  NUMBER_VALUE  → inside a number
  COMMA_OR_END  → expecting ',' or '}'
  ARRAY_VALUE   → inside an array
  DONE          → valid JSON completed
"""

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import torch


class FSMState(Enum):
    START = auto()
    OBJECT_OPEN = auto()
    OBJECT_KEY = auto()
    COLON = auto()
    VALUE = auto()
    STRING_VALUE = auto()
    STRING_ESCAPE = auto()
    NUMBER_VALUE = auto()
    COMMA_OR_END = auto()
    ARRAY_OPEN = auto()
    ARRAY_VALUE = auto()
    ARRAY_COMMA_OR_END = auto()
    BOOL_TRUE = auto()
    BOOL_FALSE = auto()
    NULL_VALUE = auto()
    DONE = auto()


@dataclass
class JSONSchema:
    """Simplified JSON schema representation."""
    type: str = "object"  # object, array, string, number, integer, boolean, null
    properties: dict = field(default_factory=dict)  # name -> JSONSchema
    required: list = field(default_factory=list)
    items: Optional["JSONSchema"] = None  # For arrays
    enum: Optional[list] = None  # Allowed values

    @classmethod
    def from_dict(cls, d: dict) -> "JSONSchema":
        schema = cls(
            type=d.get("type", "object"),
            required=d.get("required", []),
            enum=d.get("enum"),
        )
        if "properties" in d:
            schema.properties = {
                k: cls.from_dict(v) for k, v in d["properties"].items()
            }
        if "items" in d:
            schema.items = cls.from_dict(d["items"])
        return schema


class JSONConstrainedDecoder:
    """
    FSM that guides token-by-token generation to produce valid JSON.

    Usage:
        decoder = JSONConstrainedDecoder(tokenizer, schema)
        for each step:
            mask = decoder.get_token_mask(vocab_size)
            logits = logits + mask  # apply mask (additive, -inf blocks tokens)
            token = sample(logits)
            decoder.advance(token)
    """

    def __init__(self, tokenizer, schema: JSONSchema):
        self.tokenizer = tokenizer
        self.schema = schema

        # Build a map: token_id -> decoded string
        self._build_token_map()

        self.state = FSMState.START
        self.state_stack: list[FSMState] = []
        self.generated_text = ""
        self.depth = 0
        self.current_key: Optional[str] = None
        self.expected_keys: list[str] = list(schema.properties.keys())
        self.key_index = 0
        self.finished = False

    def _build_token_map(self):
        """Pre-decode all tokens for fast lookup."""
        self.token_strings: dict[int, str] = {}
        for i in range(self.tokenizer.vocab_size):
            try:
                s = self.tokenizer.decode([i], skip_special_tokens=True)
                if s:
                    self.token_strings[i] = s
            except Exception:
                pass

    def get_token_mask(self, vocab_size: int, device: str = "cpu") -> torch.Tensor:
        """
        Return an additive mask: 0.0 for allowed tokens, -inf for blocked.
        """
        mask = torch.full((vocab_size,), float("-inf"), device=device)

        if self.finished:
            # Allow EOS
            mask[self.tokenizer.eos_token_id] = 0.0
            return mask

        allowed_prefixes = self._get_allowed_prefixes()

        for token_id, token_str in self.token_strings.items():
            for prefix in allowed_prefixes:
                if prefix.startswith(token_str) or token_str.startswith(prefix):
                    mask[token_id] = 0.0
                    break

        # Always ensure at least one token is allowed
        if (mask == float("-inf")).all():
            mask[self.tokenizer.eos_token_id] = 0.0

        return mask

    def _get_allowed_prefixes(self) -> list[str]:
        """Get list of allowed string prefixes based on current FSM state."""
        if self.state == FSMState.START:
            return ["{"]
        elif self.state == FSMState.OBJECT_OPEN:
            if self.key_index < len(self.expected_keys):
                key = self.expected_keys[self.key_index]
                return [f'"{key}"']
            return ["}"]
        elif self.state == FSMState.OBJECT_KEY:
            return [":"]
        elif self.state == FSMState.COLON:
            return [": "]
        elif self.state == FSMState.VALUE:
            return ['"', "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "-", "true", "false", "null", "{", "["]
        elif self.state == FSMState.STRING_VALUE:
            return []  # Allow any printable character
        elif self.state == FSMState.COMMA_OR_END:
            if self.key_index < len(self.expected_keys):
                return [", "]
            return ["}"]
        elif self.state == FSMState.DONE:
            return []
        return []

    def advance(self, token_id: int):
        """Advance the FSM state with a generated token."""
        token_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
        self.generated_text += token_str

        # Simple state machine transitions
        for ch in token_str:
            self._advance_char(ch)

    def _advance_char(self, ch: str):
        if self.state == FSMState.START:
            if ch == "{":
                self.state = FSMState.OBJECT_OPEN
                self.depth += 1
        elif self.state == FSMState.OBJECT_OPEN:
            if ch == '"':
                self.state = FSMState.OBJECT_KEY
            elif ch == '}':
                self.depth -= 1
                if self.depth <= 0:
                    self.state = FSMState.DONE
                    self.finished = True
                else:
                    self.state = FSMState.COMMA_OR_END
        elif self.state == FSMState.OBJECT_KEY:
            if ch == '"':
                self.state = FSMState.COLON
                self.key_index += 1
        elif self.state == FSMState.COLON:
            if ch == ':':
                self.state = FSMState.VALUE
        elif self.state == FSMState.VALUE:
            if ch == '"':
                self.state = FSMState.STRING_VALUE
            elif ch in '0123456789-':
                self.state = FSMState.NUMBER_VALUE
            elif ch == '{':
                self.depth += 1
                self.state = FSMState.OBJECT_OPEN
            elif ch == '[':
                self.state = FSMState.ARRAY_OPEN
            elif ch in 'tfn':
                self.state = FSMState.STRING_VALUE  # simplified
        elif self.state == FSMState.STRING_VALUE:
            if ch == '\\':
                self.state = FSMState.STRING_ESCAPE
            elif ch == '"':
                self.state = FSMState.COMMA_OR_END
        elif self.state == FSMState.STRING_ESCAPE:
            self.state = FSMState.STRING_VALUE
        elif self.state == FSMState.NUMBER_VALUE:
            if ch in ',}]':
                self.state = FSMState.COMMA_OR_END
                self._advance_char(ch)  # re-process
        elif self.state == FSMState.COMMA_OR_END:
            if ch == ',':
                self.state = FSMState.OBJECT_OPEN
            elif ch == '}':
                self.depth -= 1
                if self.depth <= 0:
                    self.state = FSMState.DONE
                    self.finished = True
                else:
                    self.state = FSMState.COMMA_OR_END
            elif ch == ']':
                self.state = FSMState.COMMA_OR_END
        elif self.state == FSMState.ARRAY_OPEN:
            if ch == ']':
                self.state = FSMState.COMMA_OR_END
            else:
                self.state = FSMState.VALUE
                self._advance_char(ch)

    def is_valid(self) -> bool:
        """Check if the generated text is valid JSON."""
        try:
            json.loads(self.generated_text)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
