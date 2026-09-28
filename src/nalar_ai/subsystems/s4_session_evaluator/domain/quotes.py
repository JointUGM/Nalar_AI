"""Quotes must be real (design doc §7 rule 3, decision S4-D10)."""

import re
import unicodedata

from nalar_ai.shared.text import normalize_key


def find_quote(answer: str, quote: str) -> str | None:
    """The span of `answer` that `quote` copies, or None when the quote is not in it.

    Words must appear in order and whole. Case, punctuation and whitespace between words are
    ignored, because models tidy them. The returned span comes from the answer itself, so
    what gets stored is always the student's own text.
    """
    words = normalize_key(quote).split()
    if not words:
        return None
    pattern = r"(?<!\w)" + r"[\W_]+".join(re.escape(word) for word in words) + r"(?!\w)"
    match = re.search(pattern, unicodedata.normalize("NFKC", answer), re.IGNORECASE)
    return match.group(0) if match else None
