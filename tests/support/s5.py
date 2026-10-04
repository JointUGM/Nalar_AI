"""A scored Gaya dan Gerak class, one student's results and scripted S5 model replies."""

import json
from dataclasses import replace
from typing import Any
from uuid import UUID

from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.platform.prompts.registry import PromptRegistry
from nalar_ai.shared.enums import ConceptOutcome
from nalar_ai.subsystems.s5_insight_synthesizer.application.policy import SynthesizerPolicy
from nalar_ai.subsystems.s5_insight_synthesizer.domain.counts import (
    ClassCounts,
    ConceptCount,
    MisconceptionCount,
)
from nalar_ai.subsystems.s5_insight_synthesizer.domain.insight import InsightLimits
from nalar_ai.subsystems.s5_insight_synthesizer.domain.numbers import NumberLexicon
from nalar_ai.subsystems.s5_insight_synthesizer.domain.summary import (
    ParentConcept,
    ParentSummaryInput,
)
from nalar_ai.subsystems.s5_insight_synthesizer.infrastructure.config_loader import (
    load_synthesizer_config,
)
from nalar_ai.subsystems.s5_insight_synthesizer.module import PROMPTS_DIR
from tests.support.s1 import MODELS

C_GESEK = UUID("00000000-0000-4000-8000-00000000c001")
C_LEMBAM = UUID("00000000-0000-4000-8000-00000000c002")
M_HABIS = UUID("00000000-0000-4000-8000-00000000e001")  # m1
M_BERAT = UUID("00000000-0000-4000-8000-00000000e002")  # m2
M_DIAM = UUID("00000000-0000-4000-8000-00000000e003")  # m3, held by no student


def make_counts(**changes: Any) -> ClassCounts:
    counts = ClassCounts(
        mission_title="Gaya dan Gerak",
        total=28,
        incomplete=3,
        concepts=(
            ConceptCount(
                C_GESEK,
                "Gaya gesek",
                mastered=9,
                developing=7,
                not_observed=2,
                misconceptions=(
                    MisconceptionCount(M_HABIS, "Gaya bisa habis seperti bensin", 10, 4),
                ),
            ),
            ConceptCount(
                C_LEMBAM,
                "Kelembaman",
                mastered=12,
                developing=8,
                not_observed=2,
                misconceptions=(
                    MisconceptionCount(M_BERAT, "Benda berat selalu jatuh lebih cepat", 6, 1),
                    MisconceptionCount(M_DIAM, "Benda diam tidak punya gaya", 0, 0),
                ),
            ),
        ),
    )
    return replace(counts, **changes)


LEXICON = NumberLexicon.build(
    number_words=["satu", "dua", "tiga", "setengah", "kedua"],
    count_claims=["sebagian besar", "semua siswa", "kebanyakan"],
)
LIMITS = InsightLimits(
    narrative_max_words=120,
    explanation_max_words=50,
    name_max_words=8,
    max_clusters=6,
    max_suggestions=3,
    suggestion_max_words=60,
    suggestion_max_chars=600,
)

NARRATIVE = (
    "{{count:m1}} dari {{class:total}} siswa masih berpikir gaya bisa habis; "
    "{{resolved:m1}} siswa berubah pikiran selama sesi. Pada kelembaman, {{count:m2}} siswa "
    "masih mengira benda berat jatuh lebih cepat."
)
CLUSTERS: list[dict[str, Any]] = [
    {
        "name": "Gaya dianggap seperti bahan bakar",
        "explanation": "Siswa membayangkan gaya tersimpan di dalam benda; {{count:m1}} siswa "
        "memegang ide ini.",
        "misconceptions": ["m1"],
    },
    {
        "name": "Berat menentukan kecepatan jatuh",
        "explanation": "Siswa mengaitkan berat dengan kecepatan jatuh.",
        "misconceptions": ["m2"],
    },
]


def insight_reply(**changes: Any) -> str:
    return json.dumps(
        {
            "narrative": NARRATIVE,
            "clusters": CLUSTERS,
            "suggestions": ["Minta siswa menjelaskan gaya gesek melalui prediksi dan pengamatan."],
            **changes,
        }
    )


def make_parent_input(**changes: Any) -> ParentSummaryInput:
    data = ParentSummaryInput(
        mission_title="Gaya dan Gerak",
        concepts=(
            ParentConcept("Gaya gesek", ConceptOutcome.MASTERED, None, False),
            ParentConcept(
                "Kelembaman",
                ConceptOutcome.MISCONCEPTION,
                "Benda berat selalu jatuh lebih cepat",
                False,
            ),
        ),
        evaluation_summary="Siswa menjelaskan gesekan dengan contoh lantai dan es.",
    )
    return replace(data, **changes)


SUMMARY = (
    "Ananda sudah memahami bahwa gesekan membuat benda yang meluncur perlahan berhenti, dan "
    "Ananda bisa menjelaskannya dengan contoh lantai dan es. Ananda masih mengembangkan "
    "pemahaman tentang kelembaman: Ananda masih mengira benda yang lebih berat selalu jatuh "
    "lebih cepat. Ini langkah yang wajar dalam belajar sains. Di rumah, Anda bisa mengajak "
    "Ananda menjatuhkan selembar kertas dan sebuah buku bersama-sama, lalu mengulanginya dengan "
    "kertas yang sudah diremas. Tanyakan apa yang Ananda lihat dan mengapa hasilnya berbeda. "
    "Dengarkan alasannya tanpa terburu-buru memberi jawaban."
)


def summary_reply(text: str = SUMMARY) -> str:
    return json.dumps({"summary": text})


CONFIG = load_synthesizer_config()


def make_gateway(llm: ScriptedLLM) -> LLMGateway:
    return LLMGateway(
        port=llm,
        prompts=PromptRegistry.from_directories([PROMPTS_DIR]),
        models=MODELS,
        base_delay_s=0.0,
    )


def make_policy() -> SynthesizerPolicy:
    return SynthesizerPolicy(insight_timeout_s=5.0, summary_timeout_s=5.0, max_attempts=2)
