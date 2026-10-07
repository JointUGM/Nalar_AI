"""Score CP drafts against a human gold set. Running it costs real money (about US$0.10 per document).

    uv run python -m evals.s1.cp_draft --pdf cp_ipa.pdf --pages cp_ipa_pages.json \
        --gold evals/s1/datasets/cp_draft/cp_ipa_2022.json

The gold file is copied by a human from the official PDF, never written by a model (see the
datasets README).

Production sends the backend's own extracted, header-stripped pages and publishes only statements
that pass its verbatim check. To measure that, pass --pages: a JSON object {"1": "page text", ...}
exported from the backend's `national_reference_pages` for the same PDF. Without it the pages come
from PyMuPDF, whose text differs from the backend's (hyphenation, table order), so recall measured
that way can be higher than what production keeps. Either way a statement counts only if it is
verbatim in the pages it cites, the same rule the backend applies. The eval skips the backend's
candidate-page filter, so it also sends the non-CP pages.
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from nalar_ai.container import build_container
from nalar_ai.settings import get_settings
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s1_knowledge_base.application.draft_curriculum import (
    DraftCurriculumCommand,
    DraftCurriculumUseCase,
)

GATES = {"recall": 0.90, "precision": 0.95, "invented_metadata": 0.0}
WINDOW_CHARS = 12_000  # the backend's default reference_draft_window_chars


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _verbatim(text: str, start: int, end: int, pages: dict[int, str]) -> bool:
    """The backend's verify_source: every cited page has text and the quote occurs in them."""
    if start < 1 or end < start or any(not pages.get(p, "").strip() for p in range(start, end + 1)):
        return False
    value = _key(text)
    return bool(value) and value in _key(" ".join(pages[p] for p in range(start, end + 1)))


def score(
    gold: dict[str, Any], drafts: list[dict[str, Any]], pages: dict[int, str]
) -> dict[str, float]:
    found = {
        _key(s["text"])
        for d in drafts
        for subject in d["subjects"]
        for element in subject["elements"]
        for s in element["statements"]
        if _verbatim(s["text"], s["page_start"], s["page_end"], pages)
    }
    found = {text for text in found if not any(text != other and text in other for other in found)}
    expected = {_key(s) for s in gold["statements"]}
    hits = len(found & expected)
    whole = _key(" ".join(pages.values()))
    invented = any(
        quote is not None and (gold[field] is None or _key(quote["text"]) not in whole)
        for d in drafts
        for field in ("decree_code", "effective_on")
        for quote in [d[field]]
    )
    return {
        "recall": hits / len(expected) if expected else 1.0,
        "precision": hits / len(found) if found else 1.0,
        "invented_metadata": float(invented),
    }


def _windows(pages: dict[int, str]) -> list[dict[int, str]]:
    windows: list[dict[int, str]] = [{}]
    for number, text in pages.items():
        if windows[-1] and sum(map(len, windows[-1].values())) + len(text) > WINDOW_CHARS:
            windows.append({})
        windows[-1][number] = text
    return [window for window in windows if window]


async def _run(
    name: str, data: bytes, given: dict[int, str] | None, gold: dict[str, Any], budget: float
) -> int:
    settings = get_settings()
    container = build_container(settings)
    try:
        if given is None:
            print(
                "no --pages: using PyMuPDF text, not the backend's (see the module docstring)",
                file=sys.stderr,
            )
            reader = container.s1.reader
            pages = {
                page.number: "\n".join(line.text for line in page.lines)
                for page in reader.read_pages(data, 1, reader.inspect(data).page_count)
            }
        else:
            pages = given
        use_case = DraftCurriculumUseCase(
            llm=container.llm, timeout_s=settings.s1_cp_draft_timeout_seconds
        )
        ledger = UsageLedger("eval-cp-draft", budget)
        drafts = [
            (await use_case.execute(DraftCurriculumCommand(name, window), ledger)).model_dump()
            for window in _windows(pages)
        ]
    finally:
        await container.aclose()
    result = score(gold, drafts, pages)
    spend = {
        "cost_usd": round(ledger.total_cost_usd, 4),
        "calls": len(ledger.records),
        "max_output_tokens": max((r.output_tokens for r in ledger.records), default=0),
        "max_latency_ms": max((r.latency_ms for r in ledger.records), default=0),
    }
    print(json.dumps(result | spend, indent=2))
    failed = [
        key
        for key, value in result.items()
        if (value > GATES[key] if key == "invented_metadata" else value < GATES[key])
    ]
    print("GATE", "FAIL " + ", ".join(failed) if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CP draft eval gate")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pages", type=Path, help="the backend's extracted pages as JSON")
    parser.add_argument("--budget", type=float, default=1.0, help="US$ cap for this document")
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    given = (
        {int(n): text for n, text in json.loads(args.pages.read_text(encoding="utf-8")).items()}
        if args.pages
        else None
    )
    raise SystemExit(
        asyncio.run(_run(args.pdf.stem, args.pdf.read_bytes(), given, gold, args.budget))
    )
