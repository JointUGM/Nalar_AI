from collections.abc import Mapping
from typing import Any
from uuid import UUID

from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.platform.tokens import TokenCounter
from nalar_ai.shared.context_pack import validate_pack
from nalar_ai.shared.enums import RetrievalPath, RetrievalSource
from nalar_ai.shared.errors import InvalidInputError, OutputValidationError
from nalar_ai.shared.provenance import RetrievalRef, UsageLedger
from nalar_ai.subsystems.s2_mission_designer.application.design_mission import (
    DesignMissionUseCase,
    _data,
)
from nalar_ai.subsystems.s2_mission_designer.application.policy import DesignerPolicy
from nalar_ai.subsystems.s2_mission_designer.application.schemas import (
    BankDraft,
    CoreDraft,
    CriticDraft,
    GenerateMissionOut,
    QuestionTextPatch,
    ReviseMissionIn,
    ReviseMissionOut,
    RubricPatch,
)
from nalar_ai.subsystems.s2_mission_designer.domain.revision import revision_scope


def field_changes(base: Mapping[str, Any], result: Mapping[str, Any]) -> list[str]:
    changed: list[str] = []
    for field in sorted(set(base) | set(result)):
        old, new = base.get(field), result.get(field)
        if old == new:
            continue
        if isinstance(old, dict) and isinstance(new, dict):
            changed.extend(f"{field}.{p}" for p in field_changes(old, new))
        else:
            changed.append(field)
    return changed


class ReviseMissionUseCase:
    def __init__(self, llm: LLMGateway, module: DesignerPolicy, tokens: TokenCounter) -> None:
        self._llm, self._module = llm, module
        self._designer = DesignMissionUseCase(llm, module, tokens)

    async def execute(self, body: ReviseMissionIn, ledger: UsageLedger) -> ReviseMissionOut:
        base, inp = body.base.generation, body.generation_input
        pack = base.context_pack
        if validate_pack(pack.to_pack()):
            raise InvalidInputError("base mission is invalid")
        if len({f.id for f in body.feedback}) != len(body.feedback):
            raise InvalidInputError("feedback ids must be unique")
        bank_ids = {q.id for q in pack.question_bank}
        if any(not set(f.question_ids) <= bank_ids for f in body.feedback):
            raise InvalidInputError("feedback references unknown questions")
        goal_changed = body.base.learning_objective != inp.learning_objective or [
            t.id for t in pack.targets
        ] != [t.id for t in inp.targets]
        scope = revision_scope(goal_changed, [f.component for f in body.feedback])
        if not scope:
            raise InvalidInputError("revision makes no change")
        feedback = [f.model_dump(mode="json") for f in body.feedback]
        revision = {
            "base": base.model_dump(mode="json"),
            "feedback": feedback,
            "previous_objective": body.base.learning_objective,
        }
        if "anchor_problem" in scope:
            result = await self._designer.execute(inp, ledger, revision=revision)
        else:
            if {t.id for t in inp.targets} != {t.id for t in pack.targets}:
                raise InvalidInputError("partial revision cannot change target ids")
            result = await self._partial(body, ledger, scope)
        if validate_pack(result.context_pack.to_pack()):
            raise OutputValidationError("revised context pack is invalid")
        return ReviseMissionOut(
            generation=result,
            effective_scope=list(scope),
            changed_fields=field_changes(
                base.model_dump(mode="json"), result.model_dump(mode="json")
            ),
        )

    async def _partial(
        self, body: ReviseMissionIn, ledger: UsageLedger, scope: tuple[str, ...]
    ) -> GenerateMissionOut:
        base, inp = body.base.generation, body.generation_input
        pack = base.context_pack
        refs = tuple(
            RetrievalRef(RetrievalSource.CONCEPT, t.id, RetrievalPath.TEACHER_CONFIRMED, i)
            for i, t in enumerate(inp.targets, 1)
        )
        refs += tuple(
            RetrievalRef(RetrievalSource.MATERIAL_CHUNK, p, RetrievalPath.LINK, len(refs) + i)
            for i, p in enumerate(base.source_chunk_ids, 1)
        )
        context = {
            "objective": inp.learning_objective,
            "targets": [
                {**t.model_dump(mode="json"), "id": f"c{i}", "concept_id": str(t.id)}
                for i, t in enumerate(inp.targets, 1)
            ],
            "misconceptions": [m.model_dump(mode="json") for m in inp.misconceptions],
            "feedback": [f.model_dump(mode="json") for f in body.feedback],
            "effective_scope": list(scope),
            "base": base.model_dump(mode="json"),
            "forbidden_verdict_words": list(self._module.question_guard.verdict_terms),
        }
        result = base.model_copy(deep=True)
        repair: list[dict[str, Any]] = []
        for attempt in range(2):
            variables = {
                "context": _data("teacher_context", context),
                "feedback": _data("critic_feedback", repair),
            }
            if "rubric" in scope:
                patched = await self._llm.generate(
                    prompt_id="s2.revise_rubric",
                    variables=variables,
                    output_model=RubricPatch,
                    ledger=ledger,
                    retrieval=refs,
                    policy=self._module.generation_policy,
                )
                result.rubric = patched.rubric
            if "bank" in scope:
                items = [f for f in body.feedback if f.component == "bank"]
                requested = (
                    {q.id for q in pack.question_bank}
                    if any(not f.question_ids for f in items)
                    else {id_ for f in items for id_ in f.question_ids}
                )
                for target in pack.targets:
                    selected = [
                        q
                        for q in pack.question_bank
                        if q.concept_id == target.id and q.id in requested
                    ]
                    if not selected:
                        continue
                    required = {q.id for q in selected}

                    def check_patch(
                        p: QuestionTextPatch,
                        required: set[str] = required,
                        target_id: UUID = target.id,
                    ) -> list[str]:
                        ids = [r.id for r in p.replacements]
                        if set(ids) != required or len(ids) != len(required):
                            return ["replacements must contain each requested id exactly once"]
                        replacements = {r.id: r.text for r in p.replacements}
                        bank = BankDraft(
                            questions=[
                                {"move": q.move, "text": replacements.get(q.id, q.text)}
                                for q in pack.question_bank
                                if q.concept_id == target_id
                            ]
                        )
                        return self._designer._bank_problems(bank, list(pack.answer_terms))

                    patched_q = await self._llm.generate(
                        prompt_id="s2.revise_questions",
                        variables={
                            **variables,
                            "questions": _data(
                                "selected_questions", [q.model_dump(mode="json") for q in selected]
                            ),
                        },
                        output_model=QuestionTextPatch,
                        ledger=ledger,
                        retrieval=refs,
                        policy=self._module.generation_policy,
                        semantic_check=check_patch,
                    )
                    texts = {r.id: r.text for r in patched_q.replacements}
                    result.context_pack.question_bank = [
                        q.model_copy(update={"text": texts[q.id]}) if q.id in texts else q
                        for q in result.context_pack.question_bank
                    ]
            core = CoreDraft(
                anchor_problem=result.context_pack.anchor_problem,
                reference_reasoning=result.context_pack.reference_reasoning,
                rubric=result.rubric,
                answer_terms=list(result.context_pack.answer_terms),
            )
            problems = self._designer._core_problems(inp, core)
            for approved_target in inp.targets:
                bank = BankDraft(
                    questions=[
                        {"move": q.move, "text": q.text}
                        for q in result.context_pack.question_bank
                        if q.concept_id == approved_target.id
                    ]
                )
                problems += self._designer._bank_problems(
                    bank, list(result.context_pack.answer_terms)
                )
            if problems:
                raise OutputValidationError(
                    "revised mission failed guards", details={"errors": problems}
                )

            def critic_check(out: CriticDraft) -> list[str]:
                errors: list[str] = []
                for issue in out.issues:
                    if issue.component == "rubric":
                        texts = [
                            v for levels in result.rubric.model_dump().values() for v in levels
                        ]
                    elif issue.component == "bank":
                        aliases = {f"c{i}": t.id for i, t in enumerate(inp.targets, 1)}
                        if issue.target not in aliases:
                            errors.append("bank issue must name a valid target alias")
                            continue
                        texts = [
                            q.text
                            for q in result.context_pack.question_bank
                            if q.concept_id == aliases[issue.target]
                        ]
                    else:
                        texts = [getattr(result.context_pack, issue.component)]
                    if not any(issue.quote in text for text in texts):
                        errors.append("critic quote must occur in its exact component")
                return errors

            critic = await self._llm.generate(
                prompt_id="s2.revision_critic",
                variables={
                    "context": _data("teacher_context", context),
                    "draft": _data(
                        "draft",
                        {
                            "student_visible": {
                                "anchor_problem": result.context_pack.anchor_problem,
                                "banks": {
                                    f"c{i}": [
                                        q.model_dump(mode="json")
                                        for q in result.context_pack.question_bank
                                        if q.concept_id == t.id
                                    ]
                                    for i, t in enumerate(inp.targets, 1)
                                },
                            },
                            "hidden_teacher_reference": {
                                "reference_reasoning": core.reference_reasoning,
                                "rubric": result.rubric.model_dump(),
                                "answer_terms": core.answer_terms,
                            },
                        },
                    ),
                },
                output_model=CriticDraft,
                ledger=ledger,
                retrieval=refs,
                policy=self._module.critic_policy,
                semantic_check=critic_check,
            )
            if not critic.issues:
                return result
            blocking = [i for i in critic.issues if i.severity == "blocking"]
            if any(i.component not in scope for i in blocking):
                raise OutputValidationError(
                    "REVISION_SCOPE_CONFLICT",
                    details={"components": [i.component for i in blocking]},
                )
            if attempt == 1:
                if blocking:
                    raise OutputValidationError("revision critic rejected repaired draft")
                return result
            repair = [i.model_dump(mode="json") for i in critic.issues if i.component in scope]
        raise OutputValidationError("revision did not pass review")
