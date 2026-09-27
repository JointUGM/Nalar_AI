import re
import unicodedata

_INLINE_SPACE = re.compile(r"[ \t  -​]+")
_NON_WORD = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    """NFKC, remove soft hyphens, collapse inline whitespace, trim each line."""
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    lines = [_INLINE_SPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def normalize_key(text: str) -> str:
    """Comparison key: case-folded, punctuation removed, single spaces."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _NON_WORD.sub(" ", text).replace("_", " ")
    return " ".join(text.split())


def fence_untrusted(tag: str, body: str) -> str:
    """Wrap untrusted text (teacher material, prior output) in a tag block it can't escape."""
    pattern = re.compile(rf"<\s*/?\s*{re.escape(tag)}\s*>", re.IGNORECASE)
    safe = pattern.sub(lambda m: m.group(0).replace("<", "‹").replace(">", "›"), body)
    return f"<{tag}>\n{safe}\n</{tag}>"
