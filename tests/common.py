import os
from pathlib import Path
from typing import AsyncIterable
import pytest
from aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.prompt import Image


@pytest.fixture(scope="session")
def sync_client() -> Client:
    return Client(
        token=get_env_var("TEST_TOKEN"),
        host=get_env_var("TEST_API_URL"),
        # This will retry after [0.0, 0.5, 1.0, 2.0, 4.0] seconds
        total_retries=5,
    )


@pytest.fixture()
async def async_client() -> AsyncIterable[AsyncClient]:
    async with AsyncClient(
        token=get_env_var("TEST_TOKEN"),
        host=get_env_var("TEST_API_URL"),
        # This will retry after [0.0, 0.5, 1.0, 2.0, 4.0] seconds
        total_retries=5,
    ) as client:
        yield client


@pytest.fixture(scope="session")
def model_name() -> str:
    return "luminous-base"


@pytest.fixture(scope="session")
def prompt_image() -> Image:
    image_source_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    return Image.from_file(image_source_path)


def get_env_var(env_var: str) -> str:
    value = os.environ.get(env_var)
    if value is None:
        raise ValueError(
            f"Test parameters could not be read from .env. Make sure to create a .env file with the key {env_var}"
        )
    return value
