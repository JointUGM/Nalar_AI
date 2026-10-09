import asyncio
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from uuid import UUID

from nalar_ai.platform.http.context_pack import (
    BankQuestionIn,
    ContextPackIn,
    MisconceptionIn,
    TargetConceptIn,
)
from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.platform.tokens import TokenCounter
from nalar_ai.shared.context_pack import validate_pack
from nalar_ai.shared.enums import ProbeStrategy, RetrievalPath, RetrievalSource
from nalar_ai.shared.errors import InvalidInputError, OutputValidationError
from nalar_ai.shared.provenance import RetrievalRef, UsageLedger
from nalar_ai.shared.question_guard import GuardLexicon, check_text
from nalar_ai.shared.text import contains_phrase, fence_untrusted, normalize_key, stem_spans
from nalar_ai.subsystems.s2_mission_designer.application.policy import DesignerPolicy
from nalar_ai.subsystems.s2_mission_designer.application.schemas import (
    BankDraft,
    CoreDraft,
    CriticDraft,
    GenerateMissionIn,
    GenerateMissionOut,
    QuestionRepairs,
    SelectionDraft,
    SelectTargetsIn,
    SelectTargetsOut,
)
from nalar_ai.subsystems.s2_mission_designer.domain.grounding import (
    SourceParagraph,
    select_paragraphs,
)


def _data(tag: str, value: object) -> str:
    return fence_untrusted(tag, json.dumps(value, ensure_ascii=False, default=str))


def _unique(ids: Sequence[UUID], name: str) -> None:
    if len(set(ids)) != len(ids):
        raise InvalidInputError(f"{name} ids are not unique")


def _terms(body: GenerateMissionIn, core: CoreDraft) -> list[str]:
    terms = [*(t.name for t in body.targets), *core.answer_terms]
    result: dict[str, str] = {}
    for term in terms:
        key = normalize_key(term)
        if key:
            result.setdefault(key, term.strip())
    return list(result.values())


class SelectTargetsUseCase:
    def __init__(self, llm: LLMGateway, module: DesignerPolicy) -> None:
        self._llm, self._module = llm, module

    async def execute(self, body: SelectTargetsIn, ledger: UsageLedger) -> SelectTargetsOut:
        _unique([c.id for c in body.concepts], "concept")
        aliases = {f"c{i}": c for i, c in enumerate(body.concepts, 1)}

        def check(out: SelectionDraft) -> list[str]:
            if len(set(out.targets)) != len(out.targets) or any(
                t not in aliases for t in out.targets
            ):
                return ["targets must be unique supplied c aliases"]
            return []

        out = await self._llm.generate(
            prompt_id="s2.select_targets",
            variables={
                "objective": fence_untrusted("objective", body.learning_objective),
                "concepts": _data(
                    "concepts",
                    [
                        {"id": a, "name": c.name, "description": c.description}
                        for a, c in aliases.items()
                    ],
                ),
            },
            output_model=SelectionDraft,
            semantic_check=check,
            ledger=ledger,
            retrieval=tuple(
                RetrievalRef(
                    RetrievalSource.CONCEPT,
                    c.id,
                    RetrievalPath.TEACHER_CONFIRMED,
                    i,
                )
                for i, c in enumerate(body.concepts, 1)
            ),
            policy=self._module.generation_policy,
        )
        return SelectTargetsOut(target_concept_ids=[aliases[a].id for a in out.targets])


class DesignMissionUseCase:
    def __init__(self, llm: LLMGateway, module: DesignerPolicy, tokens: TokenCounter) -> None:
        self._llm, self._module, self._tokens = llm, module, tokens

    async def execute(self, body: GenerateMissionIn, ledger: UsageLedger) -> GenerateMissionOut:
        targets = {f"c{i}": c for i, c in enumerate(body.targets, 1)}
        wrong = {f"m{i}": m for i, m in enumerate(body.misconceptions, 1)}
        _unique([c.id for c in body.targets], "target")
        _unique([m.id for m in body.misconceptions], "misconception")
        _unique([p.id for p in body.paragraphs], "paragraph")
        target_ids = {c.id for c in body.targets}
        if any(m.concept_id not in target_ids for m in body.misconceptions):
            raise InvalidInputError("misconception is not on a selected target")
        links = {
            c.id: list(
                dict.fromkeys(
                    [
                        *c.source_chunk_ids,
                        *(
                            p
                            for m in body.misconceptions
                            if m.concept_id == c.id
                            for p in m.source_chunk_ids
                        ),
                    ]
                )
            )
            for c in body.targets
        }
        paragraphs, ungrounded = select_paragraphs(
            links,
            [
                SourceParagraph(p.id, p.kind, p.content, p.heading_path, p.page_start, p.page_end)
                for p in body.paragraphs
            ],
            self._tokens.claude_estimate,
            self._module.token_budget,
            self._module.per_target,
        )
        material_refs = tuple(
            RetrievalRef(
                RetrievalSource.MATERIAL_CHUNK,
                p.id,
                RetrievalPath.LINK,
                i,
            )
            for i, p in enumerate(paragraphs, 1)
        )

        def refs(aliases: Sequence[str]) -> tuple[RetrievalRef, ...]:
            rows = [
                *(
                    (RetrievalSource.CONCEPT, targets[a].id, RetrievalPath.TEACHER_CONFIRMED)
                    for a in aliases
                ),
                *((r.source, r.id, r.path) for r in material_refs),
            ]
            return tuple(
                RetrievalRef(source, id_, path, i) for i, (source, id_, path) in enumerate(rows, 1)
            )

        retrieval = refs(list(targets))
        context = {
            "objective": body.learning_objective,
            "forbidden_verdict_words": list(self._module.question_guard.verdict_terms),
            "targets": [
                {"id": a, "name": c.name, "description": c.description} for a, c in targets.items()
            ],
            "misconceptions": [
                {
                    "id": a,
                    "target": next(t for t, c in targets.items() if c.id == m.concept_id),
                    "statement": m.statement,
                    "correct_understanding": m.correct_understanding,
                    "counter_examples": m.counter_examples,
                }
                for a, m in wrong.items()
            ],
            "teacher_material": [
                {
                    "id": str(p.id),
                    "page_start": p.page_start,
                    "page_end": p.page_end,
                    "heading_path": p.heading_path,
                    "text": p.content,
                }
                for p in paragraphs
            ],
        }

        async def core_call(feedback: str) -> CoreDraft:
            return await self._llm.generate(
                prompt_id="s2.generate_core",
                variables={"context": _data("teacher_context", context), "feedback": feedback},
                output_model=CoreDraft,
                ledger=ledger,
                retrieval=retrieval,
                semantic_check=lambda c: self._core_problems(body, c),
                # Live 2026-10-09: about half of anchors use an ordinary blocked word ("tepat
                # di", "hampir sama"); one named-word repair still failed 1 in 8, so allow two.
                policy=replace(self._module.generation_policy, max_repairs=2),
            )

        core = await core_call("(none)")

        async def bank_call(alias: str, feedback: str) -> BankDraft:
            bank_retrieval = refs([alias])
            bank_context = {
                **context,
                "targets": [
                    {
                        "id": alias,
                        "name": targets[alias].name,
                        "description": targets[alias].description,
                    }
                ],
                "misconceptions": [
                    {
                        "id": a,
                        "statement": m.statement,
                        "correct_understanding": m.correct_understanding,
                        "counter_examples": m.counter_examples,
                    }
                    for a, m in wrong.items()
                    if m.concept_id == targets[alias].id
                ],
                "target": alias,
                "anchor_problem": core.anchor_problem,
                "reference_reasoning": core.reference_reasoning,
                "answer_terms": _terms(body, core),
                "forbidden_verdict_words": list(self._module.question_guard.verdict_terms),
            }
            policy = replace(self._module.generation_policy, max_repairs=0)
            # Repairs run on the fast tier; one retry there is far cheaper than losing the job.
            repair_policy = self._module.generation_policy
            variables = {"context": _data("teacher_context", bank_context), "feedback": feedback}
            try:
                draft = await self._llm.generate(
                    prompt_id="s2.generate_bank",
                    variables=variables,
                    output_model=BankDraft,
                    ledger=ledger,
                    retrieval=bank_retrieval,
                    policy=policy,
                )
            except OutputValidationError:
                repair_feedback = (
                    "Return a complete concise JSON bank with exactly sixteen questions."
                )
            else:
                problems = self._bank_problems(draft, _terms(body, core))
                if not problems:
                    return draft
                repair_feedback = (
                    _data("previous_bank", draft.model_dump()) + "\n" + "\n".join(problems)
                )
                if all(p.startswith("questions.") for p in problems):
                    indices = {
                        i
                        for i, q in enumerate(draft.questions)
                        if check_text(
                            q.text, "", (), _terms(body, core), self._module.question_guard
                        )
                    }

                    def repaired_bank(out: QuestionRepairs) -> BankDraft:
                        questions = list(draft.questions)
                        for replacement in out.replacements:
                            if replacement.index in indices:
                                questions[replacement.index] = questions[
                                    replacement.index
                                ].model_copy(update={"text": replacement.text})
                        return BankDraft(questions=questions)

                    def repair_check(out: QuestionRepairs) -> list[str]:
                        # Extra indices are dropped by repaired_bank, so they cannot touch an
                        # accepted question; only a missing one is a failure.
                        if not indices <= {r.index for r in out.replacements}:
                            return [f"replace exactly these question indices: {sorted(indices)}"]
                        return self._bank_problems(repaired_bank(out), _terms(body, core))

                    try:
                        repairs = await self._llm.generate(
                            prompt_id="s2.repair_questions",
                            variables={
                                **variables,
                                "feedback": repair_feedback
                                + f"\nReplace ONLY indices {sorted(indices)}.",
                            },
                            output_model=QuestionRepairs,
                            ledger=ledger,
                            retrieval=bank_retrieval,
                            policy=repair_policy,
                            semantic_check=repair_check,
                        )
                    except OutputValidationError:
                        pass  # Fall back to the guarded whole-bank repair below.
                    else:
                        return repaired_bank(repairs)
            return await self._llm.generate(
                prompt_id="s2.repair_bank",
                variables={**variables, "feedback": repair_feedback},
                output_model=BankDraft,
                ledger=ledger,
                retrieval=bank_retrieval,
                policy=repair_policy,
                semantic_check=lambda b: self._bank_problems(b, _terms(body, core)),
            )

        async def banks_for(aliases: Sequence[str], feedback: str) -> dict[str, BankDraft]:
            # Let every started call settle: sibling failures must not lose paid invocations.
            results = await asyncio.gather(
                *(bank_call(a, feedback) for a in aliases),
                return_exceptions=True,
            )
            banks: dict[str, BankDraft] = {}
            for alias, result in zip(aliases, results, strict=True):
                if isinstance(result, BaseException):
                    raise result
                banks[alias] = result
            return banks

        banks = await banks_for(list(targets), "(none)")
        for attempt in range(2):

            def critic_check(out: CriticDraft, core: CoreDraft = core) -> list[str]:
                problems: list[str] = []
                for issue in out.issues:
                    if issue.component == "bank":
                        if issue.target not in banks:
                            problems.append(f"bank target must be one of {', '.join(banks)}")
                            continue
                        texts = [q.text for q in banks[issue.target].questions]
                    elif issue.target is not None:
                        problems.append("non-bank issues must have target null")
                        continue
                    elif issue.component == "rubric":
                        texts = [s for levels in core.rubric.model_dump().values() for s in levels]
                    else:
                        texts = [getattr(core, issue.component)]
                    if not any(issue.quote in text for text in texts):
                        problems.append(f"quote must occur verbatim in {issue.component}")
                return problems

            critic = await self._llm.generate(
                prompt_id="s2.critic",
                variables={
                    "context": _data("teacher_context", context),
                    "draft": _data(
                        "draft",
                        {
                            "student_visible": {
                                "anchor_problem": core.anchor_problem,
                                "banks": {a: b.model_dump() for a, b in banks.items()},
                            },
                            "hidden_teacher_reference": {
                                "reference_reasoning": core.reference_reasoning,
                                "rubric": core.rubric.model_dump(),
                                "answer_terms": _terms(body, core),
                            },
                        },
                    ),
                },
                output_model=CriticDraft,
                ledger=ledger,
                retrieval=retrieval,
                semantic_check=critic_check,
                policy=self._module.critic_policy,
            )
            if not critic.issues:
                break
            if attempt == 1:
                raise OutputValidationError(
                    "mission critic rejected the repaired draft",
                    details={"components": [i.component for i in critic.issues]},
                )
            feedback = _data("critic_feedback", [i.model_dump() for i in critic.issues])
            parts = {i.component for i in critic.issues if i.component != "bank"}
            if parts:
                fresh = await core_call(feedback)
                # Preserve accepted fields and regenerate banks only when their inputs change.
                core = core.model_copy(
                    update={
                        **{p: getattr(fresh, p) for p in parts},
                        "answer_terms": list(
                            dict.fromkeys([*core.answer_terms, *fresh.answer_terms])
                        )[:40]
                        if "reference_reasoning" in parts
                        else core.answer_terms,
                    }
                )
                problems = self._core_problems(body, core)
                if problems:
                    raise OutputValidationError(
                        "repaired core failed the leak guard", details={"errors": problems}
                    )
            redo = (
                list(targets)
                if parts & {"anchor_problem", "reference_reasoning"}
                else list(
                    dict.fromkeys(
                        i.target
                        for i in critic.issues
                        if i.component == "bank" and i.target is not None
                    )
                )
            )
            banks.update(await banks_for(redo, feedback))
        questions = [
            BankQuestionIn(
                id=f"{alias}-{q.move.value}-{i}",
                concept_id=targets[alias].id,
                move=q.move,
                misconception_id=None,
                text=q.text,
            )
            for alias, bank in banks.items()
            for i, q in enumerate(bank.questions, 1)
        ]
        pack = ContextPackIn(
            pack_version=1,
            anchor_problem=core.anchor_problem,
            reference_reasoning=core.reference_reasoning,
            max_probes=body.max_probes,
            max_duration_minutes=body.max_duration_minutes,
            targets=[
                TargetConceptIn(id=c.id, name=c.name, description=c.description)
                for c in body.targets
            ],
            misconceptions=[
                MisconceptionIn(
                    id=m.id,
                    concept_id=m.concept_id,
                    statement=m.statement,
                    detection_cues=list(m.detection_cues),
                )
                for m in body.misconceptions
            ],
            question_bank=questions,
            # Stem matching keeps a bounded prefix of a long approved title protective in S3.
            answer_terms=list(dict.fromkeys(t[:100] for t in _terms(body, core))),
        )
        problems = validate_pack(pack.to_pack())
        if problems:
            raise OutputValidationError(
                "generated context pack is invalid", details={"errors": problems}
            )
        return GenerateMissionOut(
            context_pack=pack,
            rubric=core.rubric,
            source_chunk_ids=[p.id for p in paragraphs],
            ungrounded_concept_ids=list(ungrounded),
            probe_plan={
                move.value: [q.id for q in questions if q.move == move] for move in ProbeStrategy
            },
        )

    def _core_problems(self, body: GenerateMissionIn, core: CoreDraft) -> list[str]:
        return _guard_problems(
            "anchor_problem", core.anchor_problem, _terms(body, core), self._module.anchor_guard
        )

    def _bank_problems(self, bank: BankDraft, terms: list[str]) -> list[str]:
        counts = Counter(q.move for q in bank.questions)
        problems = [
            f"questions: need two distinct questions for {m.value}"
            for m in ProbeStrategy
            if counts[m] < 2
        ]
        if len({normalize_key(q.text) for q in bank.questions}) != len(bank.questions):
            problems.append("questions: duplicate wording")
        for i, q in enumerate(bank.questions):
            problems += _guard_problems(
                f"questions.{i}", q.text, terms, self._module.question_guard
            )
        return problems


def _guard_problems(path: str, text: str, terms: list[str], lexicon: GuardLexicon) -> list[str]:
    """Repair feedback that names every offending word: a bare "blocked_verdict" leaves the
    model guessing, and ordinary words ("tepat di", "hampir sama") are easy to miss."""
    blocked = check_text(text, "", (), terms, lexicon)
    if not blocked:
        return []
    key = normalize_key(text)
    verdicts = [w for w in lexicon.verdict_terms if contains_phrase(key, w)]
    hidden = [t for t in terms if stem_spans(key, normalize_key(t))]
    return [
        f"{path}: {blocked.value}",
        *(
            [f"{path}: omit these verdict words entirely: {', '.join(verdicts)}"]
            if verdicts
            else []
        ),
        *(
            [f"{path}: omit these hidden answer terms entirely: {', '.join(hidden)}"]
            if hidden
            else []
        ),
    ]
