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


def contains_phrase(text_key: str, phrase_key: str) -> bool:
    """Whole-word phrase match on two normalize_key() outputs ("gaya" does not match "gayanya")."""
    return bool(phrase_key) and f" {phrase_key} " in f" {text_key} "


def contains_stem(text_key: str, phrase_key: str) -> bool:
    """Like contains_phrase, but each word may carry affixes: "gesekannya" and "bergesekan"
    both contain "gesekan". For answer terms, where blocking too much is safe and missing
    an Indonesian affixed form is a leak."""
    words = phrase_key.split()
    if not words:
        return False
    pattern = r"(?<!\S)" + r"\s+".join(rf"\S*{re.escape(word)}\S*" for word in words) + r"(?!\S)"
    return re.search(pattern, text_key) is not None
