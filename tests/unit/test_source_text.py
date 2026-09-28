from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# UTF-8 text decoded as cp1252 and re-encoded as UTF-8: a section sign gains a leading
# A-circumflex (0xC2), and an ellipsis becomes a-circumflex + euro sign (0xE2 0x20AC).
MOJIBAKE = (chr(0xC2), chr(0xE2) + chr(0x20AC))


def test_no_source_file_contains_mojibake() -> None:
    bad = [
        str(path.relative_to(ROOT))
        for folder in ("src", "tests", "evals", "scripts")
        for path in (ROOT / folder).rglob("*")
        if path.suffix in {".py", ".prompt", ".yaml"}
        and any(mark in path.read_text(encoding="utf-8") for mark in MOJIBAKE)
    ]
    assert bad == []
