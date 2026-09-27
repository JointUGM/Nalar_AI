import pytest

from nalar_ai.shared.vectors import cosine


def test_cosine_basics() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_rejects_mismatched_dimensions() -> None:
    with pytest.raises(ValueError, match="dimension"):
        cosine([1.0], [1.0, 2.0])
