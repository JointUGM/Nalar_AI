from nalar_ai.subsystems.s1_knowledge_base.domain.headings import HeadingDetector, body_font_size
from nalar_ai.subsystems.s1_knowledge_base.domain.models import TextLine
from tests.support.builders import line

BODY = "Gaya gesek selalu berlawanan arah dengan gerak benda di atas permukaan."


def _lines() -> list[TextLine]:
    return [
        line("Bab 1 Gaya dan Gerak", y=80, size=18, bold=True),
        line("A. Gaya", y=110, size=14, bold=True),
        line(BODY, y=140),
        line(BODY, y=160),
        line("1. Gaya Gesek", y=190, bold=True),
        line(BODY + " Lanjutan.", y=210, bold=True),
        line("Perhatikan gambar berikut:", y=240, bold=True),
        line("2025", y=260, size=20),
    ]


def test_body_size_is_the_size_with_most_characters() -> None:
    assert body_font_size(_lines()) == 11.0


def test_levels_by_size_then_bold() -> None:
    lines = _lines()
    detector = HeadingDetector(lines)
    assert detector.body_size == 11.0
    assert detector.level(lines[0]) == 1
    assert detector.level(lines[1]) == 2
    assert detector.level(lines[2]) is None
    assert detector.level(lines[4]) == 3  # bold at body size = deepest level
    assert detector.level(lines[5]) is None  # bold sentence ending with a period
    assert detector.level(lines[6]) is None  # bold lead-in ending with a colon
    assert detector.level(lines[7]) is None  # no letters
