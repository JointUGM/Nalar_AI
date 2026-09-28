from dataclasses import fields, replace
from uuid import UUID

from nalar_ai.shared.enums import ProbeStrategy as M
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import (
    BankQuestion,
    MisconceptionRef,
    WriterView,
    validate_pack,
)
from tests.support.s3 import M_HABIS, REFERENCE, T_GESEK, T_LEMBAM, make_pack

OUTSIDER = UUID(int=0x99)


def test_the_sample_pack_is_valid() -> None:
    assert validate_pack(make_pack()) == []


def test_writer_view_has_no_field_that_could_carry_the_answer() -> None:
    names = {f.name for f in fields(WriterView)}
    assert names == {"target_ids", "misconception_ids", "question_bank"}
    pack = make_pack()
    view = pack.writer_view()
    assert view.misconception_ids == tuple(m.id for m in pack.misconceptions)
    assert REFERENCE not in repr(view)
    assert "Gaya gesek" not in repr(view)  # concept names stay out
    assert "Gesekan hanya" not in repr(view)  # wrong-idea statements stay out too


def test_every_target_needs_a_question_for_every_move() -> None:
    pack = make_pack()
    bank = tuple(
        q for q in pack.question_bank if not (q.concept_id == T_GESEK and q.move is M.TRANSFER)
    )
    (problem,) = validate_pack(replace(pack, question_bank=bank))
    assert problem == f"target {T_GESEK} has no approved question for: transfer"


def test_links_must_stay_inside_the_targets() -> None:
    pack = make_pack()
    stray_misconception = MisconceptionRef(UUID(int=0x55), OUTSIDER, "x")
    stray_question = BankQuestion("stray", OUTSIDER, M.TRANSFER, "Kenapa?")
    crossed = BankQuestion("crossed", T_GESEK, M.COUNTER_EXAMPLE, "Kenapa?", M_HABIS)
    problems = validate_pack(
        replace(
            pack,
            misconceptions=(*pack.misconceptions, stray_misconception),
            question_bank=(*pack.question_bank, stray_question, crossed),
        )
    )
    assert problems == [
        f"misconception {UUID(int=0x55)} points to a concept that is not a target",
        "question stray points to a concept that is not a target",
        "question crossed names a misconception of another concept",
    ]


def test_duplicates_budget_and_answer_terms_are_checked() -> None:
    pack = make_pack()
    problems = validate_pack(
        replace(
            pack,
            targets=(*pack.targets, pack.targets[0]),
            question_bank=(*pack.question_bank, pack.question_bank[0]),
            max_probes=1,
            answer_terms=(),
        )
    )
    assert "target concept ids are not unique" in problems
    assert "question bank ids are not unique" in problems
    assert "max_probes is smaller than the number of targets" in problems
    assert "the pack has no answer_terms for the leak guard" in problems


def test_lookup_helpers() -> None:
    pack = make_pack()
    assert pack.target_ids == (T_GESEK, T_LEMBAM)
    habis = pack.misconception(M_HABIS)
    assert habis is not None and habis.concept_id == T_LEMBAM
    assert pack.misconception(OUTSIDER) is None
