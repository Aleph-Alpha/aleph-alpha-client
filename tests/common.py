import os
from typing import Iterable
import pytest
from aleph_alpha_client import AlephAlphaClient

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
def model_name() -> str:
    model = os.environ.get("TEST_MODEL")
    if model is None:
        raise ValueError(
            "Test parameters could not be read from .env. Make sure to create a .env file with the key TEST_MODEL."
        )
    return model


@pytest.fixture(scope="session")
def model(client: AlephAlphaClient, model_name: str) -> AlephAlphaModel:
    return AlephAlphaModel(client, model_name)


@pytest.fixture(scope="session")
def luminous_base(client: AlephAlphaClient) -> AlephAlphaModel:
    return AlephAlphaModel(client, "luminous-base")


@pytest.fixture(scope="session")
def luminous_extended(client: AlephAlphaClient) -> AlephAlphaModel:
    return AlephAlphaModel(client, "luminous-extended")
