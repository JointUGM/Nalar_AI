import math
from functools import cached_property
from typing import Any

import tiktoken


class TokenCounter:
    """Token ruler.

    `count` uses cl100k_base, the tokenizer of text-embedding-3-small, which is the right ruler
    for chunk sizes and embedding limits. Claude tokenizes differently, so Claude budgets use
    `claude_estimate`, a conservative 1.35x factor.
    """

    CLAUDE_SAFETY_FACTOR = 1.35

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name

    @cached_property
    def _encoding(self) -> Any:
        return tiktoken.get_encoding(self._encoding_name)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))

    def claude_estimate(self, text: str) -> int:
        return math.ceil(self.count(text) * self.CLAUDE_SAFETY_FACTOR)
