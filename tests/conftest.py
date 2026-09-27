from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from nalar_ai.container import Container, build_container
from nalar_ai.main import create_app
from nalar_ai.settings import Settings
from tests.support.auth import SERVICE_KEY


@pytest.fixture
def settings() -> Settings:
    # model_construct skips the environment, so a developer's .env never leaks into tests.
    return Settings.model_construct(service_key=SecretStr(SERVICE_KEY), llm_provider="fake")


@pytest.fixture
def container(settings: Settings) -> Container:
    return build_container(settings)


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    with TestClient(create_app(container)) as test_client:
        yield test_client
