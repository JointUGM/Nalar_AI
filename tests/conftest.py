from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from nalar_ai.container import Container, build_container
from nalar_ai.main import create_app
from nalar_ai.platform.embeddings.fakes import HashingEmbedder
from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.settings import Settings
from tests.support.auth import SERVICE_KEY


@pytest.fixture
def settings() -> Settings:
    # model_construct skips the environment, so a developer's .env never leaks into tests.
    return Settings.model_construct(service_key=SecretStr(SERVICE_KEY), llm_provider="fake")


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def container(settings: Settings, llm: ScriptedLLM, embedder: HashingEmbedder) -> Container:
    return build_container(settings, llm_port=llm, embedding_port=embedder)


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    with TestClient(create_app(container)) as test_client:
        yield test_client
