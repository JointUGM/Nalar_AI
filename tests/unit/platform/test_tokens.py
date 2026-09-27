from nalar_ai.platform.tokens import TokenCounter


def test_counts_with_cl100k_and_estimates_claude_conservatively() -> None:
    counter = TokenCounter()
    text = "Benda berhenti karena gaya gesek."
    assert counter.count(text) == 11
    assert counter.claude_estimate(text) == 15
    assert counter.count("") == 0
