from evals.s1.cp_draft import score


def test_score_counts_exact_statements_and_invented_metadata() -> None:
    gold = {"decree_code": None, "effective_on": None, "statements": ["A satu.", "B dua."]}
    drafts = [
        {
            "decree_code": {"text": "999/H/KR/2099", "page": 1},
            "effective_on": None,
            "subjects": [
                {
                    "name": "IPA",
                    "phase": "D",
                    "elements": [
                        {
                            "element": "E",
                            "text": "A satu. C tiga.",
                            "page_start": 1,
                            "page_end": 1,
                            "statements": [
                                {"text": "A  satu.", "page_start": 1, "page_end": 1},
                                {"text": "C tiga.", "page_start": 1, "page_end": 1},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    result = score(gold, drafts, {1: "A satu. C tiga."})
    assert result == {"recall": 0.5, "precision": 0.5, "invented_metadata": 1.0}


def _draft(*statements: str, end: int = 1) -> dict[str, object]:
    return {
        "decree_code": None,
        "effective_on": None,
        "subjects": [
            {
                "name": "IPA",
                "phase": "D",
                "elements": [
                    {
                        "element": "E",
                        "text": "x",
                        "page_start": 1,
                        "page_end": 1,
                        "statements": [
                            {"text": s, "page_start": 1, "page_end": end} for s in statements
                        ],
                    }
                ],
            }
        ],
    }


def test_statements_the_backend_would_drop_are_not_counted_as_found() -> None:
    gold = {"decree_code": None, "effective_on": None, "statements": ["A satu."]}
    result = score(gold, [_draft("A satu.", "B parafrasa.")], {1: "A satu."})
    assert result["precision"] == 1.0 and result["recall"] == 1.0


def test_a_fragment_of_another_found_statement_is_not_counted_separately() -> None:
    gold = {"decree_code": None, "effective_on": None, "statements": ["A satu dua."]}
    result = score(gold, [_draft("A satu", "A satu dua.")], {1: "A satu dua."})
    assert result["precision"] == 1.0


def test_a_statement_across_a_continued_table_header_counts_like_the_backend() -> None:
    # The backend's verify_source (D-CPD-14, D-CPD-15): hyphen line breaks and a table header
    # that tops continued pages are layout, not content.
    gold = {
        "decree_code": None,
        "effective_on": None,
        "statements": ["A langkah-langkah satu dua."],
    }
    draft = _draft("A langkah-langkah satu dua.", end=2)
    pages = {1: "Elemen Fase D\nA langkah-\nlangkah satu", 2: "Elemen Fase D\ndua.", 3: "Isi."}
    assert score(gold, [draft], pages)["recall"] == 1.0
