from nalar_ai.shared.enums import ChunkKind
from nalar_ai.subsystems.s1_knowledge_base.domain.chunker import (
    ChunkingPolicy,
    chunk_paragraphs,
    embedding_text,
)
from nalar_ai.subsystems.s1_knowledge_base.domain.structure import Paragraph

POLICY = ChunkingPolicy(max_tokens=10, min_tail_tokens=3, hard_max_tokens=12)
PATH = ("Bab 1", "A. Gaya")


def words(text: str) -> int:
    return len(text.split())


def para(
    text: str,
    *,
    page: int = 1,
    path: tuple[str, ...] = PATH,
    kind: ChunkKind = ChunkKind.EXPLANATION,
) -> Paragraph:
    return Paragraph(text=text, page_start=page, page_end=page, heading_path=path, kind=kind)


def test_short_paragraphs_under_one_heading_pack_together() -> None:
    (chunk,) = chunk_paragraphs(
        [para("satu dua tiga empat", page=3), para("lima enam tujuh", page=4)], words, POLICY
    )
    assert chunk.content == "satu dua tiga empat\n\nlima enam tujuh"
    assert (chunk.page_start, chunk.page_end) == (3, 4)
    assert chunk.heading_path == "Bab 1 > A. Gaya"
    assert chunk.token_count == 7
    assert chunk.local_index == 0


def test_chunks_never_cross_headings_or_kinds() -> None:
    chunks = chunk_paragraphs(
        [
            para("a b"),
            para("c d", path=("Bab 1", "B. Gerak")),
            para("e f", path=("Bab 1", "B. Gerak"), kind=ChunkKind.EXERCISE),
        ],
        words,
        POLICY,
    )
    assert [(c.heading_path, c.kind, c.content) for c in chunks] == [
        ("Bab 1 > A. Gaya", ChunkKind.EXPLANATION, "a b"),
        ("Bab 1 > B. Gerak", ChunkKind.EXPLANATION, "c d"),
        ("Bab 1 > B. Gerak", ChunkKind.EXERCISE, "e f"),
    ]
    assert [c.local_index for c in chunks] == [0, 1, 2]


def test_long_paragraph_is_split_at_sentences() -> None:
    text = "Satu dua tiga empat lima. Enam tujuh delapan sembilan sepuluh. Sebelas dua belas tiga belas."
    chunks = chunk_paragraphs([para(text)], words, POLICY)
    assert [c.content for c in chunks] == [
        "Satu dua tiga empat lima. Enam tujuh delapan sembilan sepuluh.",
        "Sebelas dua belas tiga belas.",
    ]
    assert all(c.token_count <= POLICY.max_tokens for c in chunks)


def test_sentence_longer_than_the_limit_is_windowed() -> None:
    text = " ".join(f"k{i}" for i in range(25))
    chunks = chunk_paragraphs([para(text)], words, POLICY)
    assert [c.token_count for c in chunks] == [10, 10, 5]


def test_tiny_tail_merges_into_previous_chunk() -> None:
    chunks = chunk_paragraphs([para("a b c d e f g h i"), para("j k")], words, POLICY)
    assert [c.token_count for c in chunks] == [11]
    assert chunks[0].content == "a b c d e f g h i\n\nj k"


def test_hashes_are_deterministic_and_include_heading_and_kind() -> None:
    first = chunk_paragraphs([para("a b c")], words, POLICY)[0]
    again = chunk_paragraphs([para("a b c")], words, POLICY)[0]
    other_kind = chunk_paragraphs([para("a b c", kind=ChunkKind.SUMMARY)], words, POLICY)[0]
    assert first.content_sha256 == again.content_sha256
    assert first.content_sha256 != other_kind.content_sha256
    assert len(first.content_sha256) == 64


def test_embedding_text_prefixes_the_heading_path() -> None:
    assert (
        embedding_text("Bab 1 > A. Gaya", "Perhatikan gambar.")
        == "Bab 1 > A. Gaya\n\nPerhatikan gambar."
    )
