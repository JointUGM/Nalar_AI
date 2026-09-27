"""Run S1 on one chapter with the real gateway and score it against a gold set.

Costs real money (about US$0.5 per chapter). Needs a configured .env.

    uv run python -m evals.s1.run --pdf path/to/bab1.pdf --pages 1-30 \
        --title "Bab 1 Gaya dan Gerak" --gold evals/s1/datasets/gaya_dan_gerak.json
"""

import argparse
import asyncio
import json
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evals.s1.dataset import load_gold
from evals.s1.metrics import kind_report, match_recall
from nalar_ai.container import build_container
from nalar_ai.settings import get_settings
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s1_knowledge_base.application.chunk_section import (
    ChunkSectionCommand,
    ChunkSectionUseCase,
)
from nalar_ai.subsystems.s1_knowledge_base.application.extract_concepts import (
    ExtractConceptsCommand,
    ExtractConceptsUseCase,
    concept_text,
)
from nalar_ai.subsystems.s1_knowledge_base.application.generate_misconceptions import (
    ConceptForMisconceptions,
    GenerateMisconceptionsCommand,
    GenerateMisconceptionsUseCase,
)
from nalar_ai.subsystems.s1_knowledge_base.application.inputs import ChunkRef

GATES = {
    "concept_recall": 0.80,
    "misconception_recall": 0.70,
    "kind_accuracy": 0.90,
    "restricted_recall": 1.0,
}
MATCH_THRESHOLD = 0.85


async def run(args: argparse.Namespace, pdf: bytes) -> dict[str, Any]:
    settings = get_settings()
    container = build_container(settings)
    gold = load_gold(Path(args.gold))
    start_page, end_page = (int(part) for part in args.pages.split("-"))
    ledger = UsageLedger(f"eval-{uuid.uuid4().hex[:8]}", max_cost_usd=args.budget)
    started = time.perf_counter()
    try:
        chunked = await ChunkSectionUseCase(
            reader=container.s1.reader,
            lexicon=container.s1.lexicon,
            embeddings=container.embeddings,
            count_tokens=container.tokens.count,
            policy=container.s1.chunking,
        ).execute(
            ChunkSectionCommand(
                pdf=pdf,
                page_start=start_page,
                page_end=end_page,
                title=args.title,
                next_title=args.next_title,
            ),
            ledger,
        )
        refs = tuple(
            ChunkRef(
                uuid.uuid4(),
                c.draft.heading_path,
                c.draft.kind,
                c.draft.content,
                c.draft.page_start,
                c.draft.page_end,
            )
            for c in chunked.chunks
        )
        extracted = await ExtractConceptsUseCase(
            llm=container.llm,
            embeddings=container.embeddings,
            estimate_tokens=container.tokens.claude_estimate,
            max_section_tokens=settings.s1_max_section_claude_tokens,
            duplicate_threshold=settings.s1_dedup_link_threshold,
        ).execute(ExtractConceptsCommand(args.subject, args.phase, args.title, refs), ledger)
        generated = await GenerateMisconceptionsUseCase(
            llm=container.llm,
            embeddings=container.embeddings,
            estimate_tokens=container.tokens.claude_estimate,
            evidence_budget_tokens=settings.s1_evidence_pack_claude_tokens,
            duplicate_threshold=settings.s1_dedup_link_threshold,
        ).execute(
            GenerateMisconceptionsCommand(
                args.subject,
                args.phase,
                refs,
                tuple(
                    ConceptForMisconceptions(c.key, c.name, c.description, c.source_chunk_ids)
                    for c in extracted.concepts
                ),
            ),
            ledger,
        )

        predicted_concepts = [concept_text(c.name, c.description) for c in extracted.concepts]
        predicted_misconceptions = [m.statement for m in generated.misconceptions]
        gold_concepts = gold.concept_texts()
        gold_misconceptions = gold.misconception_statements()
        vectors = await container.embeddings.embed(
            [*gold_concepts, *gold_misconceptions], tag="query", ledger=ledger
        )
        gold_concept_vectors = vectors[: len(gold_concepts)]
        gold_misconception_vectors = vectors[len(gold_concepts) :]
        concept_report = match_recall(
            gold_concepts,
            predicted_concepts,
            gold_concept_vectors,
            [c.embedding for c in extracted.concepts],
            MATCH_THRESHOLD,
        )
        misconception_report = match_recall(
            gold_misconceptions,
            predicted_misconceptions,
            gold_misconception_vectors,
            [m.embedding for m in generated.misconceptions],
            MATCH_THRESHOLD,
        )
        kinds = kind_report(
            gold.kind_labels,
            [
                (c.draft.page_start, c.draft.page_end, c.draft.content, c.draft.kind)
                for c in chunked.chunks
            ],
        )
    finally:
        await container.aclose()

    return {
        "topic": gold.topic,
        "scores": {
            "concept_recall": concept_report.recall,
            "misconception_recall": misconception_report.recall,
            "kind_accuracy": kinds.accuracy,
            "restricted_recall": kinds.restricted_recall,
        },
        "gates": GATES,
        "missing_concepts": concept_report.missing,
        "missing_misconceptions": misconception_report.missing,
        "unmatched_kind_labels": [asdict(label) for label in kinds.unmatched],
        "failed_concepts": [asdict(f) for f in generated.failed],
        "cost_usd": round(ledger.total_cost_usd, 4),
        "calls": len(ledger.records),
        "seconds": round(time.perf_counter() - started, 1),
    }


def _write_and_score(report: dict[str, Any], out: str) -> int:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"s1-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    scores: dict[str, float] = report["scores"]
    for name, floor in GATES.items():
        print(
            f"{'PASS' if scores[name] >= floor else 'FAIL'}  {name:<22} {scores[name]:.2f} (gate {floor:.2f})"
        )
    print(
        f"cost US${report['cost_usd']}  calls {report['calls']}  {report['seconds']}s  report {out_path}"
    )
    return 0 if all(scores[name] >= floor for name, floor in GATES.items()) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="S1 eval gate")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--pages", required=True, help="inclusive range, e.g. 1-30")
    parser.add_argument("--title", required=True)
    parser.add_argument("--next-title", default=None)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--subject", default="Ilmu Pengetahuan Alam")
    parser.add_argument("--phase", default="D")
    parser.add_argument("--budget", type=float, default=2.0, help="US$ cap for the whole run")
    parser.add_argument("--out", default="evals/s1/runs")
    args = parser.parse_args()
    report = asyncio.run(run(args, Path(args.pdf).read_bytes()))
    sys.exit(_write_and_score(report, args.out))


if __name__ == "__main__":
    main()
