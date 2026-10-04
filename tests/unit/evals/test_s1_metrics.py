import json
from pathlib import Path

import pytest

from evals.s1.dataset import KindLabel, load_gold
from evals.s1.metrics import kind_report, match_recall, recall_at_k
from nalar_ai.platform.embeddings.fakes import HashingEmbedder
from nalar_ai.shared.enums import ChunkKind


def _vec(text: str) -> list[float]:
    return HashingEmbedder.vector(text, 256)


def test_match_recall_uses_names_then_similarity() -> None:
    gold = ["Gaya gesek", "Inersia benda diam", "Tekanan hidrostatis"]
    predicted = ["gaya gesek", "inersia pada benda diam"]
    report = match_recall(
        gold, predicted, [_vec(g) for g in gold], [_vec(p) for p in predicted], 0.7
    )
    assert (report.matched, report.total) == (2, 3)
    assert report.missing == ("Tekanan hidrostatis",)
    assert report.recall == pytest.approx(2 / 3)


def test_kind_report_counts_restricted_recall_separately() -> None:
    labels = [
        KindLabel(4, "Gaya gesek melawan", ChunkKind.EXPLANATION),
        KindLabel(9, "1. Apa yang dimaksud", ChunkKind.EXERCISE),
        KindLabel(9, "Jawab: gaya", ChunkKind.ANSWER_KEY),
    ]
    chunks = [
        (4, 4, "Gaya gesek melawan gerak benda.", ChunkKind.EXPLANATION),
        (9, 9, "1. Apa yang dimaksud dengan gaya?", ChunkKind.EXERCISE),
        (9, 9, "Jawab: gaya adalah dorongan.", ChunkKind.EXPLANATION),
    ]
    report = kind_report(labels, chunks)
    assert report.accuracy == pytest.approx(2 / 3)
    assert report.restricted_recall == pytest.approx(1 / 2)
    assert report.unmatched == ()


def test_recall_at_k() -> None:
    assert recall_at_k([["a", "b", "c"], ["x", "y", "z"]], [{"c"}, {"q"}], k=3) == 0.5


def test_gold_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(
            {
                "topic": "Gaya dan Gerak",
                "concepts": [{"name": "Gaya gesek", "description": "Melawan gerak."}],
                "misconceptions": [{"concept": "Gaya gesek", "statement": "Gaya bisa habis"}],
                "kind_labels": [{"page": 4, "text_prefix": "Gaya gesek", "kind": "explanation"}],
            }
        ),
        encoding="utf-8",
    )
    gold = load_gold(path)
    assert gold.topic == "Gaya dan Gerak"
    assert gold.concept_texts() == ["Gaya gesek: Melawan gerak."]
    assert gold.misconception_statements() == ["Gaya bisa habis"]
    assert gold.kind_labels == (KindLabel(4, "Gaya gesek", ChunkKind.EXPLANATION),)


def test_unfound_restricted_labels_count_as_misses() -> None:
    labels = [KindLabel(9, "Kunci Jawaban nomor 1", ChunkKind.ANSWER_KEY)]
    chunks = [(4, 4, "Gaya gesek melawan gerak benda.", ChunkKind.EXPLANATION)]
    report = kind_report(labels, chunks)
    assert report.accuracy == 0.0
    assert report.restricted_recall == 0.0
    assert len(report.unmatched) == 1


def test_empty_recall_evidence_is_not_perfect() -> None:
    assert match_recall([], [], [], [], threshold=0.85).recall == 0.0
    assert recall_at_k([], [], k=3) == 0.0


def test_no_kind_labels_supply_no_accuracy_or_restricted_evidence() -> None:
    report = kind_report([], [])
    assert report.accuracy == 0.0
    assert report.restricted_recall == 0.0


def test_kind_accuracy_includes_unmatched_labels() -> None:
    labels = [
        KindLabel(4, "Gaya gesek melawan", ChunkKind.EXPLANATION),
        KindLabel(9, "Kunci Jawaban nomor 1", ChunkKind.ANSWER_KEY),
    ]
    chunks = [(4, 4, "Gaya gesek melawan gerak benda.", ChunkKind.EXPLANATION)]
    report = kind_report(labels, chunks)
    assert report.accuracy == 0.5
    assert report.restricted_recall == 0.0
    assert report.unmatched == (labels[1],)


def test_nonrestricted_labels_cannot_establish_restricted_recall() -> None:
    labels = [KindLabel(4, "Gaya gesek melawan", ChunkKind.EXPLANATION)]
    chunks = [(4, 4, "Gaya gesek melawan gerak benda.", ChunkKind.EXPLANATION)]
    report = kind_report(labels, chunks)
    assert report.accuracy == 1.0
    assert report.restricted_recall == 0.0
