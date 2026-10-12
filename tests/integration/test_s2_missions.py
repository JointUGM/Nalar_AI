import json
from typing import Any
from uuid import UUID

import pytest

from nalar_ai.shared.enums import ProbeStrategy
from tests.support.auth import SERVICE_KEY

C1, C2, P1, P2 = [str(UUID(int=i)) for i in range(1, 5)]
HEADERS = {"X-Service-Key": SERVICE_KEY}


def mission_input() -> dict[str, Any]:
    return {
        "learning_objective": "Menjelaskan gerak benda dalam situasi sehari-hari",
        "targets": [
            {"id": C1, "name": "Gesekan", "source_chunk_ids": [P1]},
            {"id": C2, "name": "Inersia", "source_chunk_ids": [P2]},
        ],
        "paragraphs": [
            {"id": P1, "kind": "example", "content": "Rahasia sumber pertama."},
            {"id": P2, "kind": "explanation", "content": "Rahasia sumber kedua."},
        ],
    }


def core() -> dict[str, Any]:
    return {
        "anchor_problem": "Rani menggelindingkan bola di halaman. Apa yang terjadi sesudah tangannya dilepas, dan mengapa?",
        "reference_reasoning": "Gesekan melawan gerak, sedangkan inersia mempertahankan gerak.",
        "rubric": {
            d: [f"{d} tingkat {i} pada bola" for i in range(5)]
            for d in ("claim", "evidence", "mechanism", "transfer")
        },
        "answer_terms": ["gesekan", "inersia"],
    }


def bank() -> dict[str, Any]:
    return {
        "questions": [
            {
                "move": m.value,
                "text": f"Apa alasanmu tentang kejadian pada bola untuk situasi {m.value.replace('_', ' ')} {i}?",
                "misconception": None,
            }
            for m in ProbeStrategy
            for i in range(2)
        ]
    }


def queue_mission(llm: Any, critic: dict[str, Any] | None = None) -> None:
    llm.queue(
        *(json.dumps(x) for x in [core(), {"issues": []}, bank(), bank(), critic or {"issues": []}])
    )


def test_generated_pack_is_compatible_with_s3_and_s4_and_has_link_provenance(client, llm):
    queue_mission(llm)
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text
    payload = response.json()
    result, pack = payload["result"], payload["result"]["context_pack"]
    assert len(pack["question_bank"]) == 32
    assert len({q["id"] for q in pack["question_bank"]}) == 32
    assert set(result["source_chunk_ids"]) == {P1, P2}
    assert result["ungrounded_concept_ids"] == []
    assert "paragraphs" not in pack and "teacher_material" not in pack
    assert "Rahasia sumber" not in json.dumps(pack)
    assert len(payload["invocations"]) == 5
    for record in payload["invocations"]:
        concepts = {r["id"] for r in record["retrieval"] if r["source"] == "concept"}
        sources = {r["id"] for r in record["retrieval"] if r["source"] == "material_chunk"}
        assert sources == ({P1} if concepts == {C1} else {P2} if concepts == {C2} else {P1, P2})
        assert (
            concepts in ({C1}, {C2})
            if "generate_bank" in record["prompt_version"]
            else concepts == {C1, C2}
        )
    from nalar_ai.platform.http.context_pack import ContextPackIn
    from nalar_ai.shared.context_pack import validate_pack
    from nalar_ai.subsystems.s4_session_evaluator.api.evaluate import EvaluationPackIn, RubricIn

    assert validate_pack(ContextPackIn.model_validate(pack).to_pack()) == []
    EvaluationPackIn.model_validate(pack)
    RubricIn.model_validate(result["rubric"])


def test_target_selection_repairs_unknown_alias_and_never_returns_invented_concept(client, llm):
    body = {"learning_objective": "Gerak", "concepts": mission_input()["targets"]}
    llm.queue('{"targets":["c1","invented"]}', '{"targets":["c1","c2"]}')
    response = client.post("/v1/s2/targets/select", json=body, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["result"]["target_concept_ids"] == [C1, C2]
    assert len(response.json()["invocations"]) == 2


def test_pending_concepts_and_duplicate_targets_are_rejected_before_spending(client, llm):
    body = mission_input()
    body["targets"][0]["review_status"] = "pending"
    assert client.post("/v1/s2/missions/generate", json=body, headers=HEADERS).status_code == 422
    body = mission_input()
    body["targets"][1]["id"] = C1
    assert client.post("/v1/s2/missions/generate", json=body, headers=HEADERS).status_code == 422
    assert llm.requests == []


def test_answer_keys_and_unlinked_paragraphs_never_enter_a_prompt(client, llm):
    body = mission_input()
    body["paragraphs"][0]["kind"] = "answer_key"
    body["paragraphs"].append({"id": str(UUID(int=9)), "kind": "example", "content": "UNLINKED"})
    queue_mission(llm)
    response = client.post("/v1/s2/missions/generate", json=body, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["result"]["ungrounded_concept_ids"] == [C1]
    assert response.json()["result"]["source_chunk_ids"] == [P2]
    assert all(
        "Rahasia sumber pertama" not in b.text and "UNLINKED" not in b.text
        for r in llm.requests
        for b in r.blocks
    )


def test_core_answer_leak_fails_closed_with_every_invocation(client, llm):
    leaked = core()
    leaked["anchor_problem"] = "Gesekan memperlambat bola, mengapa?"
    llm.queue(json.dumps(leaked), json.dumps(leaked), json.dumps(leaked))
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 502
    assert len(response.json()["invocations"]) == 3


def test_critic_repairs_only_the_broken_bank_then_rechecks(client, llm):
    queue_mission(
        llm,
        {
            "issues": [
                {
                    "component": "bank",
                    "target": "c2",
                    "problem": "Pertanyaan terlalu rumit",
                    "quote": bank()["questions"][0]["text"],
                }
            ]
        },
    )
    repaired = {
        "replacements": [{"index": 0, "text": "Apa alasanmu tentang gerak bola di halaman?"}]
    }
    llm.queue(json.dumps(repaired), '{"issues":[]}')
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text
    assert len(response.json()["invocations"]) == 7
    assert sum("generate_core" in i["prompt_version"] for i in response.json()["invocations"]) == 1
    questions = response.json()["result"]["context_pack"]["question_bank"]
    assert [q["text"] for q in questions[:16]] == [q["text"] for q in bank()["questions"]]
    repaired_bank = [q["text"] for q in questions[16:]]
    assert repaired_bank[0] == repaired["replacements"][0]["text"]
    assert repaired_bank[1:] == [q["text"] for q in bank()["questions"]][1:]
    repair_request = next(r for r in llm.requests if "specified rejected Socratic" in r.system)
    assert "Pertanyaan terlalu rumit" in repair_request.blocks[-1].text


def test_critic_rejection_after_repair_returns_no_package(client, llm, caplog):
    issue = {
        "issues": [
            {
                "component": "bank",
                "target": "c1",
                "problem": "Masih memberi jawaban",
                "quote": bank()["questions"][0]["text"],
            }
        ]
    }
    queue_mission(llm, issue)
    repaired = {"replacements": [{"index": 0, "text": bank()["questions"][0]["text"]}]}
    llm.queue(json.dumps(repaired), json.dumps(issue))
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 502
    assert "result" not in response.json()
    assert len(response.json()["invocations"]) == 7
    diagnostic = next(
        record.message for record in caplog.records if "stage=critic_after_repair" in record.message
    )
    assert "components=bank" in diagnostic
    assert "blocking_issues=1" in diagnostic
    assert response.json()["request_id"] in diagnostic
    assert "s2.critic@v5" in diagnostic
    assert issue["issues"][0]["problem"] not in diagnostic
    assert issue["issues"][0]["quote"] not in diagnostic


def test_invalid_bank_gets_one_fast_repair_without_weakening_the_guard(client, llm):
    rejected = bank()
    rejected["questions"][0]["text"] = "Mengapa jawabanmu benar?"
    replacement = {"replacements": [{"index": 0, "text": bank()["questions"][0]["text"]}]}
    llm.queue(
        *(
            json.dumps(x)
            for x in [core(), {"issues": []}, rejected, replacement, bank(), {"issues": []}]
        )
    )
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text
    records = response.json()["invocations"]
    repair = [i for i in records if "repair_questions" in i["prompt_version"]]
    assert len(repair) == 1 and repair[0]["model"] == "gpt-5.4-mini"
    questions = response.json()["result"]["context_pack"]["question_bank"]
    assert [q["text"] for q in questions[:16]] == [q["text"] for q in bank()["questions"]]
    assert all(
        q["misconception_id"] is None
        for q in response.json()["result"]["context_pack"]["question_bank"]
    )


def test_sibling_bank_failures_settle_and_preserve_all_paid_attempts(client, llm):
    rejected = bank()
    rejected["questions"][0]["text"] = "Mengapa jawabanmu benar?"
    # Per target: bank, narrow repair + retry, whole-bank repair + retry.
    llm.queue(*(json.dumps(x) for x in [core(), {"issues": []}, *[rejected] * 10]))
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ai_output_invalid"
    assert len(response.json()["invocations"]) == 12


def test_core_rubric_repair_happens_before_banks_are_generated(client, llm):
    issue = {
        "issues": [
            {
                "component": "rubric",
                "target": None,
                "problem": "Perjelas deskriptor",
                "quote": "claim tingkat 0 pada bola",
            }
        ]
    }
    llm.queue(
        *(
            json.dumps(x)
            for x in [core(), issue, core(), {"issues": []}, bank(), bank(), {"issues": []}]
        )
    )
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text
    records = response.json()["invocations"]
    assert sum("generate_bank" in i["prompt_version"] for i in records) == 2


def test_bank_generation_receives_only_its_target_wrong_ideas(client, llm):
    body = mission_input()
    body["misconceptions"] = [
        {
            "id": str(UUID(int=i + 10)),
            "concept_id": c,
            "statement": f"Pernyataan keliru unik {i}",
            "correct_understanding": "Penjelasan untuk guru",
        }
        for i, c in enumerate((C1, C2))
    ]
    queue_mission(llm)
    response = client.post("/v1/s2/missions/generate", json=body, headers=HEADERS)
    assert response.status_code == 200, response.text
    bank_contexts = [r.blocks[-1].text for r in llm.requests if "single target c alias" in r.system]
    assert len(bank_contexts) == 2
    assert "Pernyataan keliru unik 1" not in bank_contexts[0]
    assert "Pernyataan keliru unik 0" not in bank_contexts[1]
    assert "Rahasia sumber kedua" not in bank_contexts[0]
    assert "Rahasia sumber pertama" not in bank_contexts[1]
    assert "reference_reasoning" not in bank_contexts[0]
    assert '"objective"' not in bank_contexts[0]
    assert '"anchor_problem"' not in bank_contexts[0]


def test_anchor_presupposing_the_assessed_relationship_fails_closed(client, llm):
    leaked = core()
    leaked["anchor_problem"] = (
        "Rani mengganti tempo bunyi. Mengapa perubahan tempo tidak mengubah waktu tempuh bunyi?"
    )
    llm.queue(*(json.dumps(leaked) for _ in range(3)))
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 502
    assert len(response.json()["invocations"]) == 3
    assert "answer_relationship" in response.json()["error"]["details"]["errors"][0]


@pytest.mark.parametrize(
    "text",
    [
        "Jika jarak tetap, mengapa kilatan selalu mendahului bunyi guntur?",
        "Kilatan datang lebih awal dari suara. Apa hal paling sederhana yang membedakan keduanya?",
        "Apa penyebab jeda kilatan dan guntur bertambah saat pengamat menjauh?",
        "Penonton tribun belakang merasakan jeda lebih lama daripada penonton depan. Apa alasannya?",
        "Teman jauh melihat kembang api sebelum mendengar ledakannya. Apa alasannya?",
        "Bagaimana menjelaskan keterlambatan suara dibanding cahaya?",
    ],
)
def test_explicit_light_sound_relationships_are_rejected_before_critic(client, llm, text):
    leaked = core()
    leaked["anchor_problem"] = text
    llm.queue(*(json.dumps(leaked) for _ in range(3)))
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 502
    assert len(response.json()["invocations"]) == 3


def test_anchor_relationship_feedback_repairs_before_banks_are_written(client, llm):
    leaked = core()
    leaked["anchor_problem"] = (
        "Rani mengganti tempo bunyi. Mengapa perubahan tempo tidak mengubah waktu tempuh bunyi?"
    )
    llm.queue(
        *(json.dumps(x) for x in [leaked, core(), {"issues": []}, bank(), bank(), {"issues": []}])
    )
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text
    assert "answer_relationship" in llm.requests[1].blocks[-1].text


def test_core_audit_rejects_semantic_leaks_before_any_bank_spend(client, llm):
    unsafe = core()
    unsafe["anchor_problem"] = (
        "Tali Rani bergerak. Waktu tempuhnya sama meski tangan dipercepat. Apa alasanmu?"
    )
    issue = {
        "issues": [
            {
                "component": "anchor_problem",
                "category": "answer_leak",
                "severity": "minor",
                "problem": "Hubungan diberikan",
                "quote": "Waktu tempuhnya sama meski tangan dipercepat.",
            }
        ]
    }
    llm.queue(*(json.dumps(x) for x in [unsafe, issue, unsafe, issue]))
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 502
    records = response.json()["invocations"]
    assert len(records) == 4
    assert not any("bank" in r["prompt_version"] for r in records)
    assert "core_safety_feedback" in llm.requests[2].blocks[-1].text


def test_core_audit_repairs_a_scene_then_audits_the_new_reference(client, llm):
    unsafe = core()
    unsafe["anchor_problem"] = (
        "Tali Rani bergerak. Waktu tempuhnya sama meski tangan dipercepat. Apa alasanmu?"
    )
    issue = {
        "issues": [
            {
                "component": "anchor_problem",
                "category": "answer_leak",
                "problem": "Hubungan diberikan",
                "quote": "Waktu tempuhnya sama meski tangan dipercepat.",
            }
        ]
    }
    llm.queue(
        *(
            json.dumps(x)
            for x in [unsafe, issue, core(), {"issues": []}, bank(), bank(), {"issues": []}]
        )
    )
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text
    assert len(response.json()["invocations"]) == 7


def test_prediction_about_a_relationship_is_allowed(client, llm):
    candidate = core()
    candidate["anchor_problem"] = (
        "Rani mengganti tempo bunyi. Apakah waktu tempuh tetap sama, dan apa alasanmu?"
    )
    llm.queue(*(json.dumps(x) for x in [candidate, {"issues": []}, bank(), bank(), {"issues": []}]))
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text


def test_shape_feedback_explains_trailing_instruction_in_anchor(client, llm):
    malformed = core()
    malformed["anchor_problem"] += " Jelaskan alasanmu."
    llm.queue(
        *(
            json.dumps(x)
            for x in [malformed, core(), {"issues": []}, bank(), bank(), {"issues": []}]
        )
    )
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text
    assert "the final character must be ?" in llm.requests[1].blocks[-1].text


def test_presupposing_bank_question_is_repaired_at_its_exact_index(client, llm):
    rejected = bank()
    rejected["questions"][0]["text"] = "Mengapa perubahan tempo tidak mengubah waktu tempuh?"
    replacement = {"replacements": [{"index": 0, "text": bank()["questions"][0]["text"]}]}
    llm.queue(
        *(
            json.dumps(x)
            for x in [core(), {"issues": []}, rejected, replacement, bank(), {"issues": []}]
        )
    )
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text
    repair = next(r for r in llm.requests if "specified rejected Socratic" in r.system)
    assert "Replace ONLY indices [0]" in repair.blocks[-1].text


@pytest.mark.parametrize("path", ["/v1/s2/targets/select", "/v1/s2/missions/generate"])
def test_s2_requires_the_service_key(client, path):
    assert client.post(path, json={}).status_code == 401


def test_explicit_style_issues_go_to_teacher_review_without_regeneration(client, llm):
    # Live 2026-10-09 (jobs ea36201a, 2139127e): the second pass found only wording nits and
    # the job died; minor issues are left to the teacher's mandatory review.
    issue = {
        "issues": [
            {
                "component": "bank",
                "target": "c1",
                "severity": "minor",
                "category": "style",
                "problem": "Perjelas deskriptor",
                "quote": bank()["questions"][0]["text"],
            }
        ]
    }
    queue_mission(llm, issue)
    llm.queue(json.dumps(core()), '{"issues":[]}', json.dumps(issue))
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text
    records = response.json()["invocations"]
    assert sum("critic" in i["prompt_version"] for i in records) == 1
    assert len(records) == 5


@pytest.mark.parametrize(
    "category", ["scientific_error", "answer_leak", "target_mismatch", "unspecified"]
)
def test_safety_category_is_blocking_even_when_critic_labels_it_minor(client, llm, category):
    issue = {
        "issues": [
            {
                "component": "bank",
                "target": "c1",
                "category": category,
                "severity": "minor",
                "problem": "Temuan substantif",
                "quote": bank()["questions"][0]["text"],
            }
        ]
    }
    if category != "target_mismatch":
        issue["coverage"] = [
            {"target": "c1", "question_ids": ["q0"]},
            {"target": "c2", "question_ids": ["q0"]},
        ]
    queue_mission(llm, issue)
    llm.queue(
        json.dumps({"replacements": [{"index": 0, "text": bank()["questions"][0]["text"]}]}),
        json.dumps(issue),
    )
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 502
    assert len(response.json()["invocations"]) == 7


def test_supporting_question_nit_does_not_erase_confirmed_bank_coverage(client, llm):
    issue = {
        "coverage": [
            {"target": "c1", "question_ids": ["q1"]},
            {"target": "c2", "question_ids": ["q0"]},
        ],
        "issues": [
            {
                "component": "bank",
                "target": "c1",
                "category": "target_mismatch",
                "severity": "blocking",
                "problem": "Pertanyaan pendukung",
                "quote": bank()["questions"][0]["text"],
            }
        ],
    }
    queue_mission(llm, issue)
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text
    assert len(response.json()["invocations"]) == 5


def test_fabricated_coverage_ids_fail_closed(client, llm):
    issue = {
        "coverage": [
            {"target": "c1", "question_ids": ["invented"]},
            {"target": "c2", "question_ids": ["q0"]},
        ],
        "issues": [],
    }
    queue_mission(llm, issue)
    llm.queue(json.dumps(issue))
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 502
    assert len(response.json()["invocations"]) == 6


def test_anchor_repair_replaces_coupled_reference_and_rubric(client, llm):
    issue = {
        "issues": [
            {
                "component": "anchor_problem",
                "problem": "Ubah situasi",
                "quote": "Rani menggelindingkan bola di halaman.",
            }
        ]
    }
    fresh = core()
    fresh["anchor_problem"] = (
        "Rani mendorong kotak di lantai. Bagaimana kamu menjelaskan kejadian ini?"
    )
    fresh["reference_reasoning"] = "Gesekan dan inersia menjelaskan kejadian pada kotak."
    fresh["rubric"]["claim"][0] = "Tidak memberi klaim tentang kotak."
    llm.queue(
        *(
            json.dumps(x)
            for x in [core(), issue, fresh, {"issues": []}, bank(), bank(), {"issues": []}]
        )
    )
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["context_pack"]["reference_reasoning"] == fresh["reference_reasoning"]
    assert result["rubric"] == fresh["rubric"]


def test_critic_issue_without_severity_is_blocking(client, llm):
    issue = {
        "issues": [
            {
                "component": "rubric",
                "target": None,
                "problem": "Deskriptor salah secara ilmiah",
                "quote": "claim tingkat 0 pada bola",
            }
        ]
    }
    llm.queue(*(json.dumps(x) for x in [core(), issue, core(), issue]))
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ai_output_invalid"


def test_critic_target_on_a_non_bank_issue_is_ignored_not_fatal(client, llm):
    # Live 2026-10-09 (job c8a1ffc8): the critic tagged a rubric issue "c1" twice and the job died.
    issue = {
        "issues": [
            {
                "component": "rubric",
                "target": "c1",
                "problem": "Perjelas deskriptor",
                "quote": "claim tingkat 0 pada bola",
            }
        ]
    }
    llm.queue(
        *(
            json.dumps(x)
            for x in [core(), issue, core(), {"issues": []}, bank(), bank(), {"issues": []}]
        )
    )
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text
    records = response.json()["invocations"]
    assert sum("audit_core" in i["prompt_version"] for i in records) == 2
    assert sum("generate_core" in i["prompt_version"] for i in records) == 2


def test_critic_fabricated_evidence_fails_closed_and_keeps_invocations(client, llm):
    issue = {
        "issues": [
            {
                "component": "bank",
                "target": "c1",
                "problem": "Istilah bocor",
                "quote": "inersia",
            }
        ]
    }
    queue_mission(llm, issue)
    llm.queue(json.dumps(issue))
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 502
    assert len(response.json()["invocations"]) == 6
    assert "result" not in response.json()


def test_narrow_repair_cannot_change_an_accepted_question(client, llm):
    rejected = bank()
    rejected["questions"][0]["text"] = "Mengapa jawabanmu benar?"
    wrong = {"replacements": [{"index": 1, "text": bank()["questions"][0]["text"]}]}
    llm.queue(
        *(
            json.dumps(x)
            for x in [core(), {"issues": []}, rejected, wrong, wrong, rejected, rejected, bank()]
        )
    )
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ai_output_invalid"
    assert len(response.json()["invocations"]) == 8
    assert "result" not in response.json()


def test_narrow_repair_ignores_extra_indices_and_keeps_accepted_questions(client, llm):
    # Live 2026-10-09 (job fccb9551): the fast model returned the asked index plus another.
    rejected = bank()
    rejected["questions"][0]["text"] = "Mengapa jawabanmu benar?"
    extra = {
        "replacements": [
            {"index": 0, "text": bank()["questions"][0]["text"]},
            {"index": 5, "text": "Mengapa jawabanmu benar?"},
        ]
    }
    llm.queue(
        *(json.dumps(x) for x in [core(), {"issues": []}, rejected, extra, bank(), {"issues": []}])
    )
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text
    questions = response.json()["result"]["context_pack"]["question_bank"]
    assert [q["text"] for q in questions[:16]] == [q["text"] for q in bank()["questions"]]


def test_failed_narrow_repair_falls_back_to_a_guarded_whole_bank_repair(client, llm):
    # Live 2026-10-09 (job 0d3fe93a): the narrow repair kept a forbidden word and the job died.
    rejected = bank()
    rejected["questions"][0]["text"] = "Mengapa jawabanmu benar?"
    still_bad = {"replacements": [{"index": 0, "text": "Apakah itu benar?"}]}
    llm.queue(
        *(
            json.dumps(x)
            for x in [
                core(),
                {"issues": []},
                rejected,
                still_bad,
                still_bad,
                bank(),
                bank(),
                {"issues": []},
            ]
        )
    )
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text
    records = response.json()["invocations"]
    assert sum("repair_bank" in i["prompt_version"] for i in records) == 1
    questions = response.json()["result"]["context_pack"]["question_bank"]
    assert not any("benar" in q["text"] for q in questions)


def test_anchor_repair_names_every_blocked_word(client, llm):
    # Live 2026-10-09 (job 186ffe1f): "hampir sama", "tepat di jalur" and the model's own
    # term "merambat" blocked the anchor, and an unnamed "blocked_verdict" repair failed again.
    blocked = core()
    blocked["anchor_problem"] = (
        "Bola Rani berhenti tepat di tepi halaman yang bergesekan dengan rumput. Mengapa?"
    )
    llm.queue(json.dumps(blocked))
    queue_mission(llm)
    response = client.post("/v1/s2/missions/generate", json=mission_input(), headers=HEADERS)
    assert response.status_code == 200, response.text
    repair = llm.requests[1].blocks[-1].text
    assert "omit these verdict words entirely: tepat" in repair
    assert "omit these hidden answer terms entirely: Gesekan" in repair


def test_long_approved_titles_fit_the_contract_and_remain_guarded_in_s3(client, llm):
    from nalar_ai.shared.question_guard import GuardLexicon, check_text

    body = mission_input()
    title = "Konsep dengan judul panjang " * 5
    body["targets"][0]["name"] = title
    queue_mission(llm)
    response = client.post("/v1/s2/missions/generate", json=body, headers=HEADERS)
    assert response.status_code == 200, response.text
    terms = response.json()["result"]["context_pack"]["answer_terms"]
    assert title.strip()[:100] in terms
    assert (
        check_text(
            f"Mengapa {title} memengaruhi bola?",
            "",
            (),
            terms,
            GuardLexicon.build(verdict_terms=[], max_chars=500, max_sentences=3),
        )
        is not None
    )
