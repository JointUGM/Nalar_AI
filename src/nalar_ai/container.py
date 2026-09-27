from dataclasses import dataclass

from nalar_ai.settings import Settings


@dataclass(frozen=True, slots=True)
class Container:
    """Composition root: every long-lived object the app needs, built once at startup."""

    settings: Settings


def build_container(settings: Settings) -> Container:
    return Container(settings=settings)
