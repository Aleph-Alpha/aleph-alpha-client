import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient, AsyncClient, Client
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.tokenization import TokenizationRequest

from tests.common import (
    sync_client,
    client,
    model_name,
    model,
    checkpoint_name,
    async_client,
)


# AsyncClient


@pytest.mark.needs_api
async def test_can_tokenize_with_async_client(
    async_client: AsyncClient, model_name: str
):
    request = TokenizationRequest(prompt="hello", token_ids=True, tokens=True)

    response = await async_client.tokenize(request, model=model_name)
    assert response.tokens and len(response.tokens) == 1
    assert response.token_ids and len(response.token_ids) == 1


@pytest.mark.needs_api
async def test_can_tokenize_with_async_client_with_checkpoint(
    async_client: AsyncClient, checkpoint_name: str
):
    request = TokenizationRequest(prompt="hello", token_ids=True, tokens=True)

    response = await async_client.tokenize(request, checkpoint=checkpoint_name)
    assert response.tokens and len(response.tokens) == 1
    assert response.token_ids and len(response.token_ids) == 1


# Client


@pytest.mark.needs_api
def test_tokenize(sync_client: Client, model_name: str):
    response = sync_client.tokenize(
        request=TokenizationRequest("Hello", tokens=True, token_ids=True),
        model=model_name,
    )

    assert response.tokens and len(response.tokens) == 1
    assert response.token_ids and len(response.token_ids) == 1


@pytest.mark.needs_api
def test_tokenize_against_checkpoint(sync_client: Client, checkpoint_name: str):
    response = sync_client.tokenize(
        request=TokenizationRequest("Hello", tokens=True, token_ids=True),
        checkpoint=checkpoint_name,
    )

    assert response.tokens and len(response.tokens) == 1
    assert response.token_ids and len(response.token_ids) == 1


# ALephAlphaClient


@pytest.mark.needs_api
def test_tokenize_with_client_against_model(client: AlephAlphaClient, model_name: str):
    response = client.tokenize(model_name, prompt="Hello", tokens=True, token_ids=True)

    assert len(response["tokens"]) == 1
    assert len(response["token_ids"]) == 1


@pytest.mark.needs_api
def test_tokenize_with_client_against_checkpoint(
    client: AlephAlphaClient, checkpoint_name: str
):
    response = client.tokenize(
        model=None,
        prompt="Hello",
        tokens=True,
        token_ids=True,
        checkpoint=checkpoint_name,
    )

    assert len(response["tokens"]) == 1
    assert len(response["token_ids"]) == 1
