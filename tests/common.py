import os
from typing import AsyncIterable, Iterable
import pytest
from aleph_alpha_client import AlephAlphaClient, AsyncClient
from aleph_alpha_client.aleph_alpha_client import Client

from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel


@pytest.fixture(scope="session")
def client() -> Iterable[AlephAlphaClient]:
    api_url = os.environ.get("TEST_API_URL")
    if api_url is None:
        raise ValueError(
            "Test parameters could not be read from .env. Make sure to create a .env file with the key TEST_API_URL."
        )

    username = os.environ.get("TEST_USERNAME")
    password = os.environ.get("TEST_PASSWORD")
    token = os.environ.get("TEST_TOKEN")
    if username is not None and password is not None:
        token = None
    elif token is not None:
        username = None
        password = None
    else:
        raise ValueError(
            "Test parameters could not be read from .env. Make sure to create a .env file with either the key TEST_TOKEN or the keys TEST_USERNAME and TEST_PASSWORD."
        )

    client = AlephAlphaClient(
        host=api_url, token=token, email=username, password=password
    )

    yield client


@pytest.fixture(scope="session")
def sync_client() -> Iterable[Client]:
    token = os.environ["TEST_TOKEN"]
    client = Client(token, host=os.environ["TEST_API_URL"])
    yield client


@pytest.fixture()
async def async_client() -> AsyncIterable[AsyncClient]:
    token = os.environ["TEST_TOKEN"]
    async with AsyncClient(token, host=os.environ["TEST_API_URL"]) as client:
        yield client


@pytest.fixture(scope="session")
def model_name() -> str:
    model = os.environ.get("TEST_MODEL")
    if model is None:
        raise ValueError(
            "Test parameters could not be read from .env. Make sure to create a .env file with the key TEST_MODEL."
        )
    return model


@pytest.fixture(scope="session")
def checkpoint_name() -> str:
    checkpoint = os.environ.get("TEST_CHECKPOINT")
    if checkpoint is None:
        raise ValueError(
            "Test parameters could not be read from .env. Make sure to create a .env file with the key TEST_CHECKPOINT."
        )
    return checkpoint


@pytest.fixture(scope="session")
def qa_checkpoint_name() -> str:
    checkpoint = os.environ.get("TEST_CHECKPOINT_QA")
    if checkpoint is None:
        raise ValueError(
            "Test parameters could not be read from .env. Make sure to create a .env file with the key TEST_CHECKPOINT_QA"
        )
    return checkpoint


@pytest.fixture(scope="session")
def summarization_checkpoint_name() -> str:
    checkpoint = os.environ.get("TEST_CHECKPOINT_SUMMARIZATION")
    if checkpoint is None:
        raise ValueError(
            "Test parameters could not be read from .env. Make sure to create a .env file with the key TEST_CHECKPOINT_SUMMARIZATION"
        )
    return checkpoint


@pytest.fixture(scope="session")
def model(client: AlephAlphaClient, model_name: str) -> AlephAlphaModel:
    return AlephAlphaModel(client, model_name)


@pytest.fixture(scope="session")
def luminous_base(client: AlephAlphaClient) -> AlephAlphaModel:
    return AlephAlphaModel(client, "luminous-base")


@pytest.fixture(scope="session")
def luminous_extended(client: AlephAlphaClient) -> AlephAlphaModel:
    return AlephAlphaModel(client, "luminous-extended")
