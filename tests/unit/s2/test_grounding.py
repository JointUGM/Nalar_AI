from uuid import UUID

from nalar_ai.shared.enums import ChunkKind
from nalar_ai.subsystems.s2_mission_designer.domain.grounding import (
    SourceParagraph,
    select_paragraphs,
)


def paragraph(
    n: int, kind: ChunkKind = ChunkKind.EXAMPLE, content: str | None = None
) -> SourceParagraph:
    return SourceParagraph(UUID(int=n), kind, content or str(n), "", n, n)


def test_every_target_gets_a_first_paragraph_before_second_and_reading_order_is_restored():
    c1, c2 = UUID(int=10), UUID(int=20)
    paragraphs = [paragraph(1, ChunkKind.EXPLANATION), paragraph(2), paragraph(3)]
    chosen, missing = select_paragraphs(
        {c1: [UUID(int=1), UUID(int=2)], c2: [UUID(int=3)]}, paragraphs, lambda _: 10, 20, 3
    )
    assert [p.id for p in chosen] == [UUID(int=2), UUID(int=3)]
    assert missing == ()


def test_over_budget_and_restricted_sources_report_ungrounded_targets():
    c1, c2 = UUID(int=10), UUID(int=20)
    chosen, missing = select_paragraphs(
        {c1: [UUID(int=1)], c2: [UUID(int=2)]},
        [paragraph(1, ChunkKind.EXERCISE), paragraph(2)],
        lambda _: 30,
        20,
        3,
    )
    assert chosen == () and missing == (c1, c2)


def test_shared_text_is_not_repeated_in_the_grounding_context():
    chosen, missing = select_paragraphs(
        {UUID(int=10): [UUID(int=1), UUID(int=2)]},
        [paragraph(1, content="Sama."), paragraph(2, content="sama")],
        lambda _: 10,
        100,
        3,
    )
    assert len(chosen) == 1 and missing == ()
