import pytest
from AlephAlphaClient import AlephAlphaClient
from dotenv import dotenv_values

@pytest.fixture
def client():
    config = dotenv_values(".env")

    api_url = config.get("TEST_API_URL")
    model = config.get("TEST_MODEL")
    token = config.get("TEST_TOKEN")

    if any([v is None for v in [api_url, model, token]]):
        raise ValueError("Test parameters could not be read from .env. Make sure to create a .env file with the keys TEST_API_URL, TEST_MODEL, TEST_TOKEN")

    client = AlephAlphaClient(
            host=api_url,
            token=token
        )
    client.test_model = model
    yield client

