import os
from typing import AsyncIterable
import pytest
from aleph_alpha_client import AsyncClient, Client


@pytest.fixture(scope="session")
def sync_client() -> Client:
    return Client(
        token=get_env_var("TEST_TOKEN"),
        host=get_env_var("TEST_API_URL"),
        total_retries=2,
    )


@pytest.fixture()
async def async_client() -> AsyncIterable[AsyncClient]:
    async with AsyncClient(
        token=get_env_var("TEST_TOKEN"),
        host=get_env_var("TEST_API_URL"),
        total_retries=2,
    ) as client:
        yield client


@pytest.fixture(scope="session")
def model_name() -> str:
    return "luminous-base"


def get_env_var(env_var: str) -> str:
    value = os.environ.get(env_var)
    if value is None:
        raise ValueError(
            f"Test parameters could not be read from .env. Make sure to create a .env file with the key {env_var}"
        )
    return value
