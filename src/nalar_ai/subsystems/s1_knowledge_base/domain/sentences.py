"""Indonesian-aware sentence splitting, used only to split oversized paragraphs."""

import re

_ABBREVIATIONS = frozenset(
    {
        "dll",
        "dsb",
        "dst",
        "dkk",
        "mis",
        "misal",
        "no",
        "hlm",
        "gb",
        "gbr",
        "tab",
        "yth",
        "sdr",
        "kg",
        "km",
        "cm",
        "mm",
        "dr",
        "prof",
        "ir",
        "st",
        "jl",
        "vs",
    }
)
_BOUNDARY = re.compile(r"([.!?][\"”’)]*)\s+(?=[\"“(]?[A-Z0-9])")
_LAST_WORD = re.compile(r"(\w+)[.!?][\"”’)]*$")


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        candidate = text[start : match.end(1)]
        last_word = _LAST_WORD.search(candidate)
        if last_word and last_word.group(1).casefold() in _ABBREVIATIONS:
            continue
        sentences.append(candidate.strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [sentence for sentence in sentences if sentence]
